"""
Strategic experiment runner - prioritizes what can be done well in limited time.
Phase 1: Teacher + Soft Labels (~10 min)
Phase 2: Coreset methods (Random, K-centers) evaluation (~20 min)
Phase 3: DM distillation + evaluation (~40 min)
Phase 4: DC distillation + evaluation (~20 min)
Phase 5: TM distillation + evaluation (~30 min)
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

def load_data():
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    return train_images, train_labels, test_images, test_labels

# ============================================================
# Teacher Training
# ============================================================
def train_teacher(train_images, train_labels, test_images, test_labels,
                  epochs=200, batch_size=256, save_path='teacher_final.pt'):
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, weights_only=False)
        if ckpt.get('accuracy', 0) >= 55.0:
            print(f"Teacher loaded: {ckpt['accuracy']:.2f}%")
            return ckpt['state_dict'], ckpt['accuracy']
    
    print(f"Training teacher for {epochs} epochs...")
    model = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    n_train = len(train_images)
    best_acc = 0
    best_state = None
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = train_images[idx].to(device)
            labs = train_labels[idx].to(device)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            acc = evaluate(model, test_images, test_labels, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1}/{epochs}, Acc: {acc:.2f}%, Best: {best_acc:.2f}%")
    
    torch.save({'state_dict': best_state, 'accuracy': best_acc}, save_path)
    return best_state, best_acc

# ============================================================
# Soft Labels
# ============================================================
def generate_soft_labels(teacher_state, train_images, save_path='soft_labels_final.pt'):
    if os.path.exists(save_path):
        return torch.load(save_path, weights_only=False)
    
    model = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
    model.load_state_dict({k: v.to(device) for k, v in teacher_state.items()})
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    soft_labels = torch.cat(all_logits, dim=0)
    torch.save(soft_labels, save_path)
    return soft_labels

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
    """K-centers using K-means in teacher feature space."""
    from sklearn.cluster import KMeans
    
    np.random.seed(seed)
    
    model = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
    model.load_state_dict({k: v.to(device) for k, v in teacher_state.items()})
    model.eval()
    
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    
    # Extract all features
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
        
        kmeans = KMeans(n_clusters=ipc, random_state=seed, n_init=3, max_iter=100)
        kmeans.fit(features)
        
        for k in range(ipc):
            cluster_mask = kmeans.labels_ == k
            cluster_features = features[cluster_mask]
            cluster_indices = indices[cluster_mask]
            
            if len(cluster_features) == 0:
                selected.append(np.random.choice(indices))
                continue
            
            dists = np.linalg.norm(cluster_features - kmeans.cluster_centers_[k], axis=1)
            selected.append(cluster_indices[np.argmin(dists)])
    
    return selected

# ============================================================
# DM Distillation
# ============================================================
def dm_distill(train_images, train_labels, ipc=10, iterations=20000,
               lr_img=1.0, batch_real=256, save_path=None):
    """Distribution Matching with proper hyperparameters."""
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        return data['images'], data['labels']
    
    print(f"DM distillation: IPC={ipc}, {iterations} iters, batch_real={batch_real}")
    torch.manual_seed(0)
    np.random.seed(0)
    
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    
    # Initialize from real images
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
    
    # Pre-organize real images by class
    real_by_class = {}
    for c in range(NUM_CLASSES):
        real_by_class[c] = train_images[class_indices[c]]
    
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    
    for it in range(iterations):
        # New random network each iteration
        net = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        net.eval()
        
        # Freeze network
        for p in net.parameters():
            p.requires_grad = False
        
        loss = torch.tensor(0.0, device=device)
        
        # Process all classes
        for c in range(NUM_CLASSES):
            # Sample real data
            n_c = len(real_by_class[c])
            perm = torch.randperm(n_c)[:batch_real]
            real_batch = real_by_class[c][perm].to(device)
            
            # Synthetic data for this class
            syn_mask = syn_labels == c
            syn_batch = syn_images[syn_mask]
            
            # Apply DSA
            real_aug = DiffAugment(real_batch, strategy=dsa_strategy)
            syn_aug = DiffAugment(syn_batch, strategy=dsa_strategy)
            
            # Get embeddings
            with torch.no_grad():
                real_feat = net.embed(real_aug)
            syn_feat = net.embed(syn_aug)
            
            # Mean matching loss
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
def dc_distill(train_images, train_labels, ipc=10, 
               outer_loops=None, inner_loops=None,
               lr_img=1.0, batch_real=256, save_path=None):
    """Gradient Matching (DC) with proper hyperparameters."""
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        return data['images'], data['labels']
    
    # Standard DC loop settings from DCBench
    if outer_loops is None:
        if ipc == 1: outer_loops, inner_loops = 1, 1
        elif ipc == 10: outer_loops, inner_loops = 10, 50
        elif ipc == 50: outer_loops, inner_loops = 50, 10
        else: outer_loops, inner_loops = 10, 50
    
    total_iterations = 1000  # Standard DC uses 1000 iterations
    
    print(f"DC distillation: IPC={ipc}, {total_iterations} iters, {outer_loops}x{inner_loops} loops")
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
    
    for it in range(total_iterations):
        # New random network
        net = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        net.train()
        optimizer_net = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        
        # Compute BN statistics on real data
        with torch.no_grad():
            for module in net.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean.zero_()
                    module.running_var.fill_(1)
            
            # Run a batch of real data through to set BN stats
            real_sample = []
            for c in range(NUM_CLASSES):
                perm = torch.randperm(len(real_by_class[c]))[:2]
                real_sample.append(real_by_class[c][perm])
            real_sample = torch.cat(real_sample, dim=0).to(device)
            net(real_sample)
        
        for ol in range(outer_loops):
            loss = torch.tensor(0.0, device=device)
            
            for c in range(NUM_CLASSES):
                # Real gradient
                n_c = len(real_by_class[c])
                perm = torch.randperm(n_c)[:batch_real]
                real_batch = real_by_class[c][perm].to(device)
                real_labs = torch.full((len(perm),), c, dtype=torch.long, device=device)
                
                real_aug = DiffAugment(real_batch, strategy=dsa_strategy)
                out_real = net(real_aug)
                loss_real = criterion(out_real, real_labs)
                gw_real = torch.autograd.grad(loss_real, net.parameters(), create_graph=False)
                gw_real = [g.detach().clone() for g in gw_real]
                
                # Synthetic gradient
                syn_mask = syn_labels == c
                syn_batch = syn_images[syn_mask]
                syn_labs = torch.full((syn_batch.shape[0],), c, dtype=torch.long, device=device)
                
                syn_aug = DiffAugment(syn_batch, strategy=dsa_strategy)
                out_syn = net(syn_aug)
                loss_syn = criterion(out_syn, syn_labs)
                gw_syn = torch.autograd.grad(loss_syn, net.parameters(), create_graph=True)
                
                # Cosine distance (DC's default "ours" metric)
                for gwr, gws in zip(gw_real, gw_syn):
                    shape = gwr.shape
                    if len(shape) == 4:  # Conv
                        gwr = gwr.reshape(shape[0], -1)
                        gws = gws.reshape(shape[0], -1)
                    elif len(shape) == 1:  # Bias/BN - skip
                        continue
                    elif len(shape) == 2:  # Linear
                        pass
                    else:
                        continue
                    
                    cos_sim = F.cosine_similarity(gwr, gws, dim=1)
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
            print(f"  Iter {it+1}/{total_iterations}, Loss: {loss.item():.4f}")
    
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
                  save_dir='/workspace/expert_traj_final'):
    os.makedirs(save_dir, exist_ok=True)
    
    existing = [f for f in os.listdir(save_dir) if f.startswith('expert_')]
    if len(existing) >= num_experts:
        print(f"  {len(existing)} experts already exist")
        return save_dir
    
    n_train = len(train_images)
    batch_size = 256
    criterion = nn.CrossEntropyLoss()
    
    for exp_idx in range(num_experts):
        save_path = os.path.join(save_dir, f'expert_{exp_idx}.pt')
        if os.path.exists(save_path):
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
                # Expert uses crop_scale_rotate augmentation (from TM paper)
                imgs = DiffAugment(imgs, strategy='crop_scale_rotate')
                
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labs)
                loss.backward()
                optimizer.step()
            
            trajectory.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        
        torch.save(trajectory, save_path)
        acc = evaluate(model, test_images, test_labels, device)
        print(f"    Expert {exp_idx} final acc: {acc:.2f}%")
    
    return save_dir

def tm_distill(train_images, train_labels, ipc=10,
               expert_dir='/workspace/expert_traj_final',
               iterations=5000, lr_img=1000.0, syn_steps=30,
               expert_epochs=3, max_start_epoch=25,
               save_path=None):
    """Trajectory Matching distillation."""
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        return data['images'], data['labels']
    
    print(f"TM distillation: IPC={ipc}, {iterations} iters")
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Load experts
    expert_files = sorted([f for f in os.listdir(expert_dir) if f.startswith('expert_')])
    expert_trajectories = []
    for f in expert_files:
        traj = torch.load(os.path.join(expert_dir, f), map_location='cpu', weights_only=False)
        expert_trajectories.append(traj)
    print(f"  Loaded {len(expert_trajectories)} experts")
    
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
        
        # Trajectory matching loss (normalized)
        loss = torch.tensor(0.0, device=device)
        target_dict = {k: v.to(device) for k, v in target_params.items()}
        
        flat_student = torch.cat([p.reshape(-1) for p in student.parameters()])
        flat_target = torch.cat([target_dict[k].reshape(-1) for k in student.state_dict().keys() if k in target_dict and 'num_batches_tracked' not in k and 'running' not in k])
        
        # Only match trainable params
        trainable_student = torch.cat([p.reshape(-1) for p in student.parameters()])
        trainable_target_list = []
        for name, param in student.named_parameters():
            if name in target_dict:
                trainable_target_list.append(target_dict[name].reshape(-1))
        trainable_target = torch.cat(trainable_target_list)
        
        # Normalized matching
        loss = torch.sum((trainable_student / (torch.norm(trainable_student) + 1e-6) - 
                         trainable_target / (torch.norm(trainable_target) + 1e-6)) ** 2)
        
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
# Evaluation
# ============================================================
def eval_config(images, labels, test_images, test_labels, 
                label_type, soft_labels=None, num_runs=3):
    model_fn = lambda: ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE)
    accs = []
    for run in range(num_runs):
        acc = train_and_evaluate(
            images, labels, test_images, test_labels,
            model_fn, num_classes=NUM_CLASSES, device=device,
            label_type=label_type, soft_labels=soft_labels,
            epochs=300, batch_size=256, seed=run, verbose=False
        )
        accs.append(acc)
    return np.mean(accs), np.std(accs)

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='all',
                       help='teacher, coreset, dm, dc, tm, eval, all')
    parser.add_argument('--ipc', type=int, default=0, help='0=both 10 and 50')
    args = parser.parse_args()
    
    start_time = time.time()
    
    train_images, train_labels, test_images, test_labels = load_data()
    
    results = {}
    results_path = 'results/results_final.json'
    os.makedirs('results', exist_ok=True)
    
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    
    # Phase 1: Teacher
    if args.phase in ['all', 'teacher']:
        teacher_state, teacher_acc = train_teacher(
            train_images, train_labels, test_images, test_labels,
            epochs=200, save_path='teacher_final.pt'
        )
    else:
        ckpt = torch.load('teacher_final.pt', weights_only=False)
        teacher_state = ckpt['state_dict']
    
    full_sl = generate_soft_labels(teacher_state, train_images, save_path='soft_labels_final.pt')
    
    ipcs = [10, 50] if args.ipc == 0 else [args.ipc]
    
    # Phase 2: Coreset evaluation
    if args.phase in ['all', 'coreset']:
        for ipc in ipcs:
            # Random
            key = f'random_ipc{ipc}'
            if key not in results:
                print(f"\n{'='*60}\nRandom IPC={ipc}\n{'='*60}")
                selected = random_select(train_labels, ipc)
                imgs = train_images[selected]
                labs = train_labels[selected]
                sl = full_sl[selected]
                
                hl_m, hl_s = eval_config(imgs, labs, test_images, test_labels, 'hard', num_runs=3)
                sl_m, sl_s = eval_config(imgs, labs, test_images, test_labels, 'soft', sl, num_runs=3)
                results[key] = {'hl_mean': hl_m, 'hl_std': hl_s, 'sl_mean': sl_m, 'sl_std': sl_s}
                print(f"  HL: {hl_m:.2f}±{hl_s:.2f}, SL: {sl_m:.2f}±{sl_s:.2f}")
                
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
            # K-centers
            key = f'kcenters_ipc{ipc}'
            if key not in results:
                print(f"\n{'='*60}\nK-centers IPC={ipc}\n{'='*60}")
                selected = kcenters_select(train_images, train_labels, ipc, teacher_state)
                imgs = train_images[selected]
                labs = train_labels[selected]
                sl = full_sl[selected]
                
                hl_m, hl_s = eval_config(imgs, labs, test_images, test_labels, 'hard', num_runs=3)
                sl_m, sl_s = eval_config(imgs, labs, test_images, test_labels, 'soft', sl, num_runs=3)
                results[key] = {'hl_mean': hl_m, 'hl_std': hl_s, 'sl_mean': sl_m, 'sl_std': sl_s}
                print(f"  HL: {hl_m:.2f}±{hl_s:.2f}, SL: {sl_m:.2f}±{sl_s:.2f}")
                
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
    
    # Phase 3: DM
    if args.phase in ['all', 'dm']:
        for ipc in ipcs:
            key = f'dm_ipc{ipc}'
            if key not in results:
                print(f"\n{'='*60}\nDM IPC={ipc}\n{'='*60}")
                iters = 20000 if ipc == 10 else 10000
                imgs, labs = dm_distill(train_images, train_labels, ipc=ipc,
                                       iterations=iters, batch_real=256,
                                       save_path=f'distilled_dm_ipc{ipc}_final.pt')
                sl = get_soft_labels_for_images(imgs, teacher_state)
                
                hl_m, hl_s = eval_config(imgs, labs, test_images, test_labels, 'hard', num_runs=3)
                sl_m, sl_s = eval_config(imgs, labs, test_images, test_labels, 'soft', sl, num_runs=3)
                results[key] = {'hl_mean': hl_m, 'hl_std': hl_s, 'sl_mean': sl_m, 'sl_std': sl_s}
                print(f"  HL: {hl_m:.2f}±{hl_s:.2f}, SL: {sl_m:.2f}±{sl_s:.2f}")
                
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
    
    # Phase 4: DC
    if args.phase in ['all', 'dc']:
        for ipc in ipcs:
            key = f'dc_ipc{ipc}'
            if key not in results:
                print(f"\n{'='*60}\nDC IPC={ipc}\n{'='*60}")
                imgs, labs = dc_distill(train_images, train_labels, ipc=ipc,
                                       save_path=f'distilled_dc_ipc{ipc}_final.pt')
                sl = get_soft_labels_for_images(imgs, teacher_state)
                
                hl_m, hl_s = eval_config(imgs, labs, test_images, test_labels, 'hard', num_runs=3)
                sl_m, sl_s = eval_config(imgs, labs, test_images, test_labels, 'soft', sl, num_runs=3)
                results[key] = {'hl_mean': hl_m, 'hl_std': hl_s, 'sl_mean': sl_m, 'sl_std': sl_s}
                print(f"  HL: {hl_m:.2f}±{hl_s:.2f}, SL: {sl_m:.2f}±{sl_s:.2f}")
                
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
    
    # Phase 5: TM
    if args.phase in ['all', 'tm']:
        # Train experts first
        print(f"\n{'='*60}\nTraining TM experts\n{'='*60}")
        train_experts(train_images, train_labels, test_images, test_labels,
                     num_experts=10, expert_epochs=50)
        
        for ipc in ipcs:
            key = f'tm_ipc{ipc}'
            if key not in results:
                print(f"\n{'='*60}\nTM IPC={ipc}\n{'='*60}")
                imgs, labs = tm_distill(train_images, train_labels, ipc=ipc,
                                       iterations=5000,
                                       save_path=f'distilled_tm_ipc{ipc}_final.pt')
                sl = get_soft_labels_for_images(imgs, teacher_state)
                
                hl_m, hl_s = eval_config(imgs, labs, test_images, test_labels, 'hard', num_runs=3)
                sl_m, sl_s = eval_config(imgs, labs, test_images, test_labels, 'soft', sl, num_runs=3)
                results[key] = {'hl_mean': hl_m, 'hl_std': hl_s, 'sl_mean': sl_m, 'sl_std': sl_s}
                print(f"  HL: {hl_m:.2f}±{hl_s:.2f}, SL: {sl_m:.2f}±{sl_s:.2f}")
                
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
    
    # Print table
    elapsed = time.time() - start_time
    print(f"\n\nTotal time: {elapsed/60:.1f} min")
    print("\n" + "=" * 90)
    print("RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("=" * 90)
    
    paper = {
        'random_ipc10': (18.64, 33.43), 'random_ipc50': (34.66, 45.39),
        'kcenters_ipc10': (25.04, 34.70), 'kcenters_ipc50': (38.64, 46.24),
        'dm_ipc10': (29.23, 26.13), 'dm_ipc50': (42.32, 43.46),
        'dc_ipc10': (28.42, 23.54), 'dc_ipc50': (30.56, 33.46),
        'tm_ipc10': (38.18, 37.60), 'tm_ipc50': (46.32, 46.26),
    }
    
    print(f"{'Method':<12} {'IPC':>4}  {'HL (ours)':>14}  {'HL (paper)':>10}  {'SL (ours)':>14}  {'SL (paper)':>10}")
    print("-" * 90)
    
    for key in ['random_ipc10', 'random_ipc50', 'kcenters_ipc10', 'kcenters_ipc50',
                'dm_ipc10', 'dm_ipc50', 'dc_ipc10', 'dc_ipc50', 'tm_ipc10', 'tm_ipc50']:
        if key in results:
            r = results[key]
            method = key.rsplit('_', 1)[0]
            ipc = key.rsplit('ipc', 1)[1]
            p_hl, p_sl = paper.get(key, (0, 0))
            print(f"{method:<12} {ipc:>4}  {r['hl_mean']:>5.2f}±{r['hl_std']:.2f}     {p_hl:>5.2f}      {r['sl_mean']:>5.2f}±{r['sl_std']:.2f}     {p_sl:>5.2f}")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
