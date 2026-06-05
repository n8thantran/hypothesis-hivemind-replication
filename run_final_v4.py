"""
Final experiment runner v4 - strategic approach.
Runs experiments in priority order, saves results incrementally.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import sys

from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices
from train_eval import train_and_evaluate, evaluate

device = 'cuda'
NUM_CLASSES = 100
CHANNEL = 3
IM_SIZE = (32, 32)

RESULTS_PATH = 'results/results_v4.json'

def save_results(results):
    os.makedirs('results', exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}

def model_fn():
    return ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE)

def eval_config(images, labels, test_images, test_labels,
                label_type, soft_labels=None, num_runs=3, epochs=300):
    accs = []
    for run in range(num_runs):
        acc = train_and_evaluate(
            images, labels, test_images, test_labels,
            model_fn, num_classes=NUM_CLASSES, device=device,
            label_type=label_type, soft_labels=soft_labels,
            epochs=epochs, batch_size=256, seed=run, verbose=False
        )
        accs.append(acc)
    return np.mean(accs), np.std(accs)

def get_teacher():
    ckpt = torch.load('teacher_final.pt', weights_only=False)
    return ckpt['state_dict'], ckpt['accuracy']

def get_soft_labels_for_indices(indices, full_sl):
    return full_sl[indices]

def get_soft_labels_for_images(images, teacher_state):
    model = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
    model.load_state_dict({k: v.to(device) for k, v in teacher_state.items()})
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), 512):
            batch = images[i:i+512].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)

# ============================================================
# Coreset Methods
# ============================================================
def random_select(train_labels, ipc, seed=42):
    np.random.seed(seed)
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    selected = []
    for c in range(NUM_CLASSES):
        perm = np.random.permutation(len(class_indices[c]))[:ipc]
        selected.extend([class_indices[c][p] for p in perm])
    return selected

def kcenters_select(train_images, train_labels, ipc, teacher_state, seed=42):
    """K-centers using K-means clustering in feature space, select nearest-to-centroid."""
    from sklearn.cluster import MiniBatchKMeans
    
    np.random.seed(seed)
    
    model = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
    model.load_state_dict({k: v.to(device) for k, v in teacher_state.items()})
    model.eval()
    
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    
    # Extract features
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(device)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    all_features = torch.cat(all_features, dim=0).numpy()
    
    selected = []
    for c in range(NUM_CLASSES):
        indices = np.array(class_indices[c])
        features = all_features[indices]
        
        if len(indices) <= ipc:
            selected.extend(indices.tolist())
            continue
        
        kmeans = MiniBatchKMeans(n_clusters=ipc, random_state=seed, n_init=3, 
                                 max_iter=300, batch_size=min(1000, len(features)))
        kmeans.fit(features)
        
        # For each cluster, select the sample nearest to centroid
        for k in range(ipc):
            cluster_mask = kmeans.labels_ == k
            cluster_features = features[cluster_mask]
            cluster_indices = indices[cluster_mask]
            
            if len(cluster_features) == 0:
                # Fallback: random sample
                selected.append(np.random.choice(indices))
                continue
            
            dists = np.linalg.norm(cluster_features - kmeans.cluster_centers_[k], axis=1)
            selected.append(int(cluster_indices[np.argmin(dists)]))
    
    return selected

# ============================================================
# DM Distillation
# ============================================================
def dm_distill(train_images, train_labels, ipc=10, iterations=20000,
               lr_img=1.0, batch_real=256, save_path=None):
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        return data['images'], data['labels']
    
    print(f"DM distillation: IPC={ipc}, {iterations} iters, batch_real={batch_real}")
    torch.manual_seed(0)
    np.random.seed(0)
    
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images.append(train_images[indices[p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    real_by_class = {}
    for c in range(NUM_CLASSES):
        real_by_class[c] = train_images[class_indices[c]]
    
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    
    for it in range(iterations):
        net = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        
        loss = torch.tensor(0.0, device=device)
        
        for c in range(NUM_CLASSES):
            n_c = len(real_by_class[c])
            perm = torch.randperm(n_c)[:batch_real]
            real_batch = real_by_class[c][perm].to(device)
            
            syn_mask = syn_labels == c
            syn_batch = syn_images[syn_mask]
            
            seed_aug = int(time.time() * 1000) % (2**31)
            torch.manual_seed(seed_aug)
            real_aug = DiffAugment(real_batch, strategy=dsa_strategy)
            torch.manual_seed(seed_aug)
            syn_aug = DiffAugment(syn_batch, strategy=dsa_strategy)
            
            with torch.no_grad():
                real_feat = net.embed(real_aug)
            syn_feat = net.embed(syn_aug)
            
            loss += torch.sum((real_feat.mean(0) - syn_feat.mean(0)) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        del net
        
        if (it + 1) % 2000 == 0:
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.6f}")
    
    result_images = syn_images.detach().cpu()
    result_labels = syn_labels.cpu()
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels

# ============================================================
# DC Distillation
# ============================================================
def dc_distill(train_images, train_labels, ipc=10, iterations=1000,
               lr_img=1.0, batch_real=256, save_path=None):
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        return data['images'], data['labels']
    
    if ipc <= 1:
        outer_loops, inner_loops = 1, 1
    elif ipc <= 10:
        outer_loops, inner_loops = 10, 50
    elif ipc <= 50:
        outer_loops, inner_loops = 50, 10
    else:
        outer_loops, inner_loops = 50, 10
    
    print(f"DC distillation: IPC={ipc}, {iterations} iters, {outer_loops}x{inner_loops} loops")
    torch.manual_seed(0)
    np.random.seed(0)
    
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images.append(train_images[indices[p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    
    real_by_class = {}
    for c in range(NUM_CLASSES):
        real_by_class[c] = train_images[class_indices[c]]
    
    for it in range(iterations):
        net = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        net.train()
        optimizer_net = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        
        # Pre-compute BN stats on real data
        net.eval()
        with torch.no_grad():
            real_sample = []
            for c in range(NUM_CLASSES):
                perm = torch.randperm(len(real_by_class[c]))[:2]
                real_sample.append(real_by_class[c][perm])
            real_sample = torch.cat(real_sample, dim=0).to(device)
            net(real_sample)
        net.train()
        
        for ol in range(outer_loops):
            loss = torch.tensor(0.0, device=device)
            
            seed_aug = int(time.time() * 1000) % (2**31)
            
            for c in range(NUM_CLASSES):
                n_c = len(real_by_class[c])
                perm = torch.randperm(n_c)[:batch_real]
                real_batch = real_by_class[c][perm].to(device)
                real_labs = torch.full((len(perm),), c, dtype=torch.long, device=device)
                
                torch.manual_seed(seed_aug + c)
                real_aug = DiffAugment(real_batch, strategy=dsa_strategy)
                out_real = net(real_aug)
                loss_real = criterion(out_real, real_labs)
                gw_real = torch.autograd.grad(loss_real, net.parameters(), create_graph=False)
                gw_real = [g.detach().clone() for g in gw_real]
                
                syn_mask = syn_labels == c
                syn_batch = syn_images[syn_mask]
                syn_labs = torch.full((syn_batch.shape[0],), c, dtype=torch.long, device=device)
                
                torch.manual_seed(seed_aug + c)
                syn_aug = DiffAugment(syn_batch, strategy=dsa_strategy)
                out_syn = net(syn_aug)
                loss_syn = criterion(out_syn, syn_labs)
                gw_syn = torch.autograd.grad(loss_syn, net.parameters(), create_graph=True)
                
                for gwr, gws in zip(gw_real, gw_syn):
                    shape = gwr.shape
                    if len(shape) == 4:
                        gwr_flat = gwr.reshape(shape[0], -1)
                        gws_flat = gws.reshape(shape[0], -1)
                    elif len(shape) == 2:
                        gwr_flat = gwr
                        gws_flat = gws
                    elif len(shape) == 1:
                        continue
                    else:
                        continue
                    
                    cos_sim = F.cosine_similarity(gwr_flat, gws_flat, dim=1)
                    loss += torch.sum(1 - cos_sim)
            
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            
            # Inner loop: train network on synthetic data
            if ol < outer_loops - 1:
                for il in range(inner_loops):
                    syn_aug_net = DiffAugment(syn_images.detach(), strategy=dsa_strategy)
                    out_net = net(syn_aug_net)
                    loss_net = criterion(out_net, syn_labels)
                    optimizer_net.zero_grad()
                    loss_net.backward()
                    optimizer_net.step()
        
        del net
        
        if (it + 1) % 100 == 0:
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.4f}")
    
    result_images = syn_images.detach().cpu()
    result_labels = syn_labels.cpu()
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels

# ============================================================
# TM Distillation
# ============================================================
def train_experts(train_images, train_labels, test_images, test_labels,
                  num_experts=10, expert_epochs=50,
                  save_dir='/workspace/expert_traj_v4'):
    os.makedirs(save_dir, exist_ok=True)
    
    existing = [f for f in os.listdir(save_dir) if f.startswith('expert_')]
    if len(existing) >= num_experts:
        print(f"  {len(existing)} experts already exist in {save_dir}")
        return save_dir
    
    n_train = len(train_images)
    batch_size = 256
    criterion = nn.CrossEntropyLoss()
    
    for exp_idx in range(num_experts):
        save_path = os.path.join(save_dir, f'expert_{exp_idx}.pt')
        if os.path.exists(save_path):
            print(f"  Expert {exp_idx} already exists")
            continue
        
        print(f"  Training expert {exp_idx+1}/{num_experts}...")
        torch.manual_seed(exp_idx * 1000)
        
        model = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        trajectory = [{k: v.cpu().clone() for k, v in model.state_dict().items()}]
        
        model.train()
        for epoch in range(expert_epochs):
            perm = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i+batch_size]
                imgs = train_images[idx].to(device)
                labs = train_labels[idx].to(device)
                imgs = DiffAugment(imgs, strategy='crop_scale_rotate')
                
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labs)
                loss.backward()
                optimizer.step()
            
            trajectory.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            
            if (epoch + 1) % 10 == 0:
                acc = evaluate(model, test_images, test_labels, device)
                print(f"    Epoch {epoch+1}/{expert_epochs}, Acc: {acc:.2f}%")
        
        torch.save(trajectory, save_path)
        print(f"    Expert {exp_idx} saved ({len(trajectory)} checkpoints)")
    
    return save_dir

def tm_distill(train_images, train_labels, ipc=10,
               expert_dir='/workspace/expert_traj_v4',
               iterations=5000, lr_img=1000.0, syn_steps=30,
               expert_epochs=3, max_start_epoch=25,
               save_path=None):
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        return data['images'], data['labels']
    
    print(f"TM distillation: IPC={ipc}, {iterations} iters, syn_steps={syn_steps}")
    torch.manual_seed(0)
    np.random.seed(0)
    
    expert_files = sorted([f for f in os.listdir(expert_dir) if f.startswith('expert_')])
    expert_trajectories = []
    for f in expert_files:
        traj = torch.load(os.path.join(expert_dir, f), map_location='cpu', weights_only=False)
        expert_trajectories.append(traj)
    print(f"  Loaded {len(expert_trajectories)} experts, each with {len(expert_trajectories[0])} checkpoints")
    
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images.append(train_images[indices[p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    syn_lr = torch.tensor(0.01, device=device, requires_grad=True)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=1e-5, momentum=0.5)
    
    criterion = nn.CrossEntropyLoss()
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    
    for it in range(iterations):
        exp_idx = np.random.randint(len(expert_trajectories))
        traj = expert_trajectories[exp_idx]
        max_start = min(max_start_epoch, len(traj) - expert_epochs - 1)
        if max_start < 1:
            max_start = 1
        start_epoch = np.random.randint(0, max_start)
        
        start_params = traj[start_epoch]
        target_params = traj[start_epoch + expert_epochs]
        
        student = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        student.load_state_dict({k: v.to(device) for k, v in start_params.items()})
        student.train()
        
        # Synthetic training steps
        n_syn = len(syn_images)
        for step in range(syn_steps):
            perm = torch.randperm(n_syn, device=device)
            batch_imgs = syn_images[perm]
            batch_labels = syn_labels[perm]
            batch_imgs_aug = DiffAugment(batch_imgs, strategy=dsa_strategy)
            
            out = student(batch_imgs_aug)
            loss_s = criterion(out, batch_labels)
            
            grads = torch.autograd.grad(loss_s, student.parameters(), create_graph=True)
            with torch.no_grad():
                for param, grad in zip(student.parameters(), grads):
                    param.sub_(syn_lr * grad)
        
        # Trajectory matching loss
        trainable_student = []
        trainable_target = []
        for name, param in student.named_parameters():
            trainable_student.append(param.reshape(-1))
            if name in target_params:
                trainable_target.append(target_params[name].to(device).reshape(-1))
        
        flat_student = torch.cat(trainable_student)
        flat_target = torch.cat(trainable_target)
        
        # Normalized matching (from MTT paper)
        student_norm = flat_student / (torch.norm(flat_student) + 1e-6)
        target_norm = flat_target / (torch.norm(flat_target) + 1e-6)
        loss = torch.sum((student_norm - target_norm) ** 2)
        
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        loss.backward()
        optimizer_img.step()
        optimizer_lr.step()
        
        with torch.no_grad():
            syn_lr.clamp_(min=1e-6)
        
        del student
        
        if (it + 1) % 500 == 0:
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.6f}, lr: {syn_lr.item():.6f}")
    
    result_images = syn_images.detach().cpu()
    result_labels = syn_labels.cpu()
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', 'coreset', 'dm10', 'dm50', 'dc10', 'dc50', 
                               'tm_experts', 'tm10', 'tm50', 'eval_existing'])
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("Loading data...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    teacher_state, teacher_acc = get_teacher()
    print(f"Teacher accuracy: {teacher_acc:.2f}%")
    
    full_sl = torch.load('soft_labels_final.pt', weights_only=False)
    
    results = load_results()
    
    def run_eval(key, images, labels, soft_labels=None, num_runs=3):
        """Run HL and SL evaluation for a config."""
        if f'{key}_HL' in results and f'{key}_SL' in results:
            print(f"  {key} already evaluated")
            return
        
        if f'{key}_HL' not in results:
            print(f"  Evaluating {key} HL...")
            hl_m, hl_s = eval_config(images, labels, test_images, test_labels, 'hard', num_runs=num_runs)
            results[f'{key}_HL'] = {'mean': round(hl_m, 2), 'std': round(hl_s, 2)}
            save_results(results)
            print(f"    HL: {hl_m:.2f}±{hl_s:.2f}")
        
        if f'{key}_SL' not in results:
            if soft_labels is None:
                soft_labels = get_soft_labels_for_images(images, teacher_state)
            print(f"  Evaluating {key} SL...")
            sl_m, sl_s = eval_config(images, labels, test_images, test_labels, 'soft', soft_labels, num_runs=num_runs)
            results[f'{key}_SL'] = {'mean': round(sl_m, 2), 'std': round(sl_s, 2)}
            save_results(results)
            print(f"    SL: {sl_m:.2f}±{sl_s:.2f}")
    
    # ============================================================
    # Phase: Coreset methods
    # ============================================================
    if args.phase in ['all', 'coreset']:
        for ipc in [10, 50]:
            # Random
            key = f'Random_IPC{ipc}'
            if f'{key}_HL' not in results or f'{key}_SL' not in results:
                print(f"\n{'='*60}\nRandom IPC={ipc}\n{'='*60}")
                selected = random_select(train_labels, ipc)
                imgs = train_images[selected]
                labs = train_labels[selected]
                sl = full_sl[selected]
                run_eval(key, imgs, labs, sl)
            
            # K-centers (K-means in feature space)
            key = f'Kcenters_IPC{ipc}'
            if f'{key}_HL' not in results or f'{key}_SL' not in results:
                print(f"\n{'='*60}\nK-centers IPC={ipc}\n{'='*60}")
                selected = kcenters_select(train_images, train_labels, ipc, teacher_state)
                imgs = train_images[selected]
                labs = train_labels[selected]
                sl = full_sl[selected]
                run_eval(key, imgs, labs, sl)
    
    # ============================================================
    # Phase: DM
    # ============================================================
    if args.phase in ['all', 'dm10']:
        key = 'DM_IPC10'
        if f'{key}_HL' not in results or f'{key}_SL' not in results:
            print(f"\n{'='*60}\nDM IPC=10 (20000 iters)\n{'='*60}")
            imgs, labs = dm_distill(train_images, train_labels, ipc=10,
                                   iterations=20000, batch_real=256,
                                   save_path='distilled_dm_ipc10_v4.pt')
            run_eval(key, imgs, labs)
    
    if args.phase in ['all', 'dm50']:
        key = 'DM_IPC50'
        if f'{key}_HL' not in results or f'{key}_SL' not in results:
            print(f"\n{'='*60}\nDM IPC=50 (10000 iters)\n{'='*60}")
            imgs, labs = dm_distill(train_images, train_labels, ipc=50,
                                   iterations=10000, batch_real=256,
                                   save_path='distilled_dm_ipc50_v4.pt')
            run_eval(key, imgs, labs)
    
    # ============================================================
    # Phase: DC
    # ============================================================
    if args.phase in ['all', 'dc10']:
        key = 'DC_IPC10'
        if f'{key}_HL' not in results or f'{key}_SL' not in results:
            print(f"\n{'='*60}\nDC IPC=10 (1000 iters, 10x50)\n{'='*60}")
            imgs, labs = dc_distill(train_images, train_labels, ipc=10,
                                   iterations=1000, batch_real=256,
                                   save_path='distilled_dc_ipc10_v4.pt')
            run_eval(key, imgs, labs)
    
    if args.phase in ['all', 'dc50']:
        key = 'DC_IPC50'
        if f'{key}_HL' not in results or f'{key}_SL' not in results:
            print(f"\n{'='*60}\nDC IPC=50 (500 iters, 50x10)\n{'='*60}")
            imgs, labs = dc_distill(train_images, train_labels, ipc=50,
                                   iterations=500, batch_real=256,
                                   save_path='distilled_dc_ipc50_v4.pt')
            run_eval(key, imgs, labs)
    
    # ============================================================
    # Phase: TM
    # ============================================================
    if args.phase in ['all', 'tm_experts']:
        print(f"\n{'='*60}\nTraining TM experts\n{'='*60}")
        train_experts(train_images, train_labels, test_images, test_labels,
                     num_experts=10, expert_epochs=50)
    
    if args.phase in ['all', 'tm10']:
        key = 'TM_IPC10'
        if f'{key}_HL' not in results or f'{key}_SL' not in results:
            # Make sure experts exist
            expert_dir = '/workspace/expert_traj_v4'
            if not os.path.exists(expert_dir) or len(os.listdir(expert_dir)) < 3:
                # Use existing experts
                expert_dir = '/workspace/expert_trajectories'
            
            print(f"\n{'='*60}\nTM IPC=10 (5000 iters)\n{'='*60}")
            imgs, labs = tm_distill(train_images, train_labels, ipc=10,
                                   expert_dir=expert_dir,
                                   iterations=5000, syn_steps=30,
                                   save_path='distilled_tm_ipc10_v4.pt')
            run_eval(key, imgs, labs)
    
    if args.phase in ['all', 'tm50']:
        key = 'TM_IPC50'
        if f'{key}_HL' not in results or f'{key}_SL' not in results:
            expert_dir = '/workspace/expert_traj_v4'
            if not os.path.exists(expert_dir) or len(os.listdir(expert_dir)) < 3:
                expert_dir = '/workspace/expert_trajectories'
            
            print(f"\n{'='*60}\nTM IPC=50 (3000 iters)\n{'='*60}")
            imgs, labs = tm_distill(train_images, train_labels, ipc=50,
                                   expert_dir=expert_dir,
                                   iterations=3000, syn_steps=20,
                                   save_path='distilled_tm_ipc50_v4.pt')
            run_eval(key, imgs, labs)
    
    # ============================================================
    # Phase: Evaluate existing distilled data
    # ============================================================
    if args.phase == 'eval_existing':
        for method, ipc in [('dm', 10), ('dm', 50), ('dc', 10), ('dc', 50), ('tm', 10), ('tm', 50)]:
            key = f'{method.upper()}_IPC{ipc}'
            fpath = f'distilled_{method}_ipc{ipc}_v4.pt'
            if not os.path.exists(fpath):
                fpath = f'distilled_{method}_ipc{ipc}.pt'
            if os.path.exists(fpath):
                data = torch.load(fpath, weights_only=False)
                imgs = data['images']
                labs = data['labels']
                print(f"\n{key}: {imgs.shape}")
                run_eval(key, imgs, labs)
    
    # ============================================================
    # Print final table
    # ============================================================
    elapsed = time.time() - start_time
    print(f"\n\nTotal time: {elapsed/60:.1f} min")
    print("\n" + "=" * 90)
    print("RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("=" * 90)
    
    paper = {
        'Random_IPC10': (18.64, 33.43), 'Random_IPC50': (34.66, 45.39),
        'Kcenters_IPC10': (25.04, 34.70), 'Kcenters_IPC50': (38.64, 46.24),
        'DM_IPC10': (29.23, 26.13), 'DM_IPC50': (42.32, 43.46),
        'DC_IPC10': (28.42, 23.54), 'DC_IPC50': (30.56, 33.46),
        'TM_IPC10': (38.18, 37.60), 'TM_IPC50': (46.32, 46.26),
    }
    
    print(f"{'Method':<15} {'IPC':>4}  {'HL (ours)':>14}  {'HL (paper)':>10}  {'SL (ours)':>14}  {'SL (paper)':>10}")
    print("-" * 90)
    
    for key in ['Random_IPC10', 'Random_IPC50', 'Kcenters_IPC10', 'Kcenters_IPC50',
                'DM_IPC10', 'DM_IPC50', 'DC_IPC10', 'DC_IPC50', 'TM_IPC10', 'TM_IPC50']:
        method = key.rsplit('_', 1)[0]
        ipc = key.rsplit('IPC', 1)[1]
        p_hl, p_sl = paper.get(key, (0, 0))
        
        hl_str = "N/A"
        sl_str = "N/A"
        if f'{key}_HL' in results:
            r = results[f'{key}_HL']
            hl_str = f"{r['mean']:>5.2f}±{r['std']:.2f}"
        if f'{key}_SL' in results:
            r = results[f'{key}_SL']
            sl_str = f"{r['mean']:>5.2f}±{r['std']:.2f}"
        
        print(f"{method:<15} {ipc:>4}  {hl_str:>14}  {p_hl:>10.2f}  {sl_str:>14}  {p_sl:>10.2f}")
