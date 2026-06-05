"""
Final pipeline: Distill + Evaluate all 20 experiments for Table small_scale_c100.
Methods: Random, K-centers, DM, DC, TM
IPC: 10, 50
Labels: HL, SL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import sys
from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices, random_select
from train_eval import train_and_evaluate, run_experiment

device = 'cuda'
NUM_RUNS = 3
RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# CORESET METHODS
# ============================================================

def feature_kcenters(train_images, train_labels, num_classes, ipc, device='cuda'):
    """K-centers selection in feature space using a pretrained model."""
    print(f"Computing K-centers (feature space) IPC={ipc}...")
    
    # Load teacher model for feature extraction
    teacher_path = '/workspace/teacher_final.pt'
    if os.path.exists(teacher_path):
        model = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        model.load_state_dict(torch.load(teacher_path, map_location=device))
        model.eval()
    else:
        # Use random model if no teacher
        model = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        model.eval()
    
    # Extract features
    features_list = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(device)
            feat = model.embed(batch)
            features_list.append(feat.cpu())
    features = torch.cat(features_list, dim=0)
    
    # K-centers per class
    selected = []
    class_indices = get_class_indices(train_labels, num_classes)
    
    for c in range(num_classes):
        idx = class_indices[c]
        class_feat = features[idx]
        n = len(idx)
        
        # Greedy k-centers
        chosen = []
        # Start with the point closest to class mean
        mean_feat = class_feat.mean(0)
        dists_to_mean = torch.cdist(class_feat.unsqueeze(0), mean_feat.unsqueeze(0).unsqueeze(0)).squeeze()
        first = dists_to_mean.argmin().item()
        chosen.append(first)
        
        # Min distance to any chosen center
        min_dists = torch.cdist(class_feat.unsqueeze(0), class_feat[first].unsqueeze(0).unsqueeze(0)).squeeze()
        
        for _ in range(1, ipc):
            # Pick the point with maximum min-distance to chosen centers
            farthest = min_dists.argmax().item()
            chosen.append(farthest)
            new_dists = torch.cdist(class_feat.unsqueeze(0), class_feat[farthest].unsqueeze(0).unsqueeze(0)).squeeze()
            min_dists = torch.minimum(min_dists, new_dists)
        
        for j in chosen:
            selected.append(idx[j])
    
    del model
    torch.cuda.empty_cache()
    return selected


# ============================================================
# DISTRIBUTION MATCHING (DM)
# ============================================================

def distill_dm(train_images, train_labels, num_classes=100, ipc=10, 
               iterations=10000, lr_img=1.0, batch_real=64, device='cuda', seed=0):
    """Distribution Matching distillation."""
    print(f"\n{'='*60}")
    print(f"DM Distillation: IPC={ipc}, {iterations} iterations")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels, num_classes)
    
    # Initialize from real images
    syn_images = []
    syn_labels = []
    for c in range(num_classes):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images.append(train_images[indices[p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    # Pre-load real images by class on GPU
    real_by_class = [train_images[class_indices[c]].to(device) for c in range(num_classes)]
    
    start_time = time.time()
    for it in range(iterations):
        net = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        net.eval()
        
        loss = torch.tensor(0.0, device=device)
        
        # Sample real images
        real_batch = []
        for c in range(num_classes):
            n = real_by_class[c].shape[0]
            perm = torch.randperm(n, device=device)[:batch_real]
            real_batch.append(real_by_class[c][perm])
        
        all_real = torch.cat(real_batch, dim=0)
        all_real_aug = DiffAugment(all_real, strategy='color_crop_cutout_flip_scale_rotate')
        
        with torch.no_grad():
            all_real_feat = net.embed(all_real_aug)
        
        all_syn_aug = DiffAugment(syn_images, strategy='color_crop_cutout_flip_scale_rotate')
        all_syn_feat = net.embed(all_syn_aug)
        
        for c in range(num_classes):
            real_mean = all_real_feat[c * batch_real:(c + 1) * batch_real].mean(0)
            syn_mask = syn_labels == c
            syn_mean = all_syn_feat[syn_mask].mean(0)
            loss += torch.sum((real_mean - syn_mean) ** 2)
        
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        
        del net
        
        if (it + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (it + 1) * (iterations - it - 1)
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.2f}, "
                  f"Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")
    
    # Clean up GPU memory
    for t in real_by_class:
        del t
    torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"DM IPC={ipc} complete in {total_time:.0f}s")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# DATASET CONDENSATION (DC) - Gradient Matching
# ============================================================

def distill_dc(train_images, train_labels, num_classes=100, ipc=10,
               outer_loops=100, inner_loops=50, lr_img=1.0, device='cuda', seed=0):
    """DC: Dataset Condensation via Gradient Matching."""
    print(f"\n{'='*60}")
    print(f"DC Distillation: IPC={ipc}, outer={outer_loops}, inner={inner_loops}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels, num_classes)
    
    # Initialize from real images
    syn_images = []
    syn_labels = []
    for c in range(num_classes):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images.append(train_images[indices[p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    # Pre-load real images by class on GPU
    real_by_class = [train_images[class_indices[c]].to(device) for c in range(num_classes)]
    
    criterion = nn.CrossEntropyLoss()
    batch_real = 256
    
    start_time = time.time()
    for outer in range(outer_loops):
        net = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        net.train()
        optimizer_net = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        
        for inner in range(inner_loops):
            loss = torch.tensor(0.0, device=device)
            
            # For each class, match gradients
            # Sample a subset of classes for efficiency
            class_order = np.random.permutation(num_classes)
            
            for c in class_order:
                # Real gradient
                n = real_by_class[c].shape[0]
                perm = torch.randperm(n, device=device)[:batch_real]
                real_imgs = real_by_class[c][perm]
                real_imgs_aug = DiffAugment(real_imgs, strategy='color_crop_cutout_flip_scale_rotate')
                real_labels = torch.full((real_imgs_aug.shape[0],), c, dtype=torch.long, device=device)
                
                out_real = net(real_imgs_aug)
                loss_real = criterion(out_real, real_labels)
                grad_real = torch.autograd.grad(loss_real, net.parameters(), create_graph=False)
                grad_real = [g.detach() for g in grad_real]
                
                # Synthetic gradient
                syn_mask = syn_labels == c
                syn_imgs_c = syn_images[syn_mask]
                syn_imgs_aug = DiffAugment(syn_imgs_c, strategy='color_crop_cutout_flip_scale_rotate')
                syn_labels_c = torch.full((syn_imgs_aug.shape[0],), c, dtype=torch.long, device=device)
                
                out_syn = net(syn_imgs_aug)
                loss_syn = criterion(out_syn, syn_labels_c)
                grad_syn = torch.autograd.grad(loss_syn, net.parameters(), create_graph=True)
                
                # Gradient matching loss (cosine distance)
                for g_real, g_syn in zip(grad_real, grad_syn):
                    g_real_flat = g_real.flatten()
                    g_syn_flat = g_syn.flatten()
                    cos_sim = F.cosine_similarity(g_real_flat.unsqueeze(0), g_syn_flat.unsqueeze(0))
                    loss += (1 - cos_sim)
            
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            
            # Update network on synthetic data
            net.train()
            syn_aug = DiffAugment(syn_images.detach(), strategy='color_crop_cutout_flip_scale_rotate')
            out = net(syn_aug)
            loss_net = criterion(out, syn_labels)
            optimizer_net.zero_grad()
            loss_net.backward()
            optimizer_net.step()
        
        del net, optimizer_net
        torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        eta = elapsed / (outer + 1) * (outer_loops - outer - 1)
        print(f"  Outer {outer+1}/{outer_loops}, Loss: {loss.item():.4f}, "
              f"Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")
    
    for t in real_by_class:
        del t
    torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"DC IPC={ipc} complete in {total_time:.0f}s")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# TRAJECTORY MATCHING (TM)
# ============================================================

def train_expert(train_images, train_labels, num_classes=100, epochs=50, 
                 device='cuda', seed=0):
    """Train an expert model and save trajectory."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    trajectory = [model.state_dict()]
    
    n = len(train_images)
    batch_size = 256
    
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            imgs = train_images[idx].to(device)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            labels = train_labels[idx].to(device)
            
            out = model(imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        trajectory.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
    
    return trajectory


def distill_tm(train_images, train_labels, num_classes=100, ipc=10,
               iterations=5000, lr_img=1000.0, n_experts=3, expert_epochs=50,
               M=2, N=10, device='cuda', seed=0):
    """TM: Trajectory Matching distillation."""
    print(f"\n{'='*60}")
    print(f"TM Distillation: IPC={ipc}, {iterations} iterations")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Train expert trajectories
    expert_dir = '/workspace/expert_trajectories_final'
    os.makedirs(expert_dir, exist_ok=True)
    
    trajectories = []
    for e in range(n_experts):
        expert_path = os.path.join(expert_dir, f'expert_{e}.pt')
        if os.path.exists(expert_path):
            print(f"  Loading expert {e} from {expert_path}")
            traj = torch.load(expert_path, map_location='cpu')
        else:
            print(f"  Training expert {e}...")
            traj = train_expert(train_images, train_labels, num_classes, 
                              expert_epochs, device, seed=seed+e)
            torch.save(traj, expert_path)
        trajectories.append(traj)
    
    class_indices = get_class_indices(train_labels, num_classes)
    
    # Initialize from real images
    syn_images = []
    syn_labels = []
    for c in range(num_classes):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images.append(train_images[indices[p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=iterations)
    
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for it in range(iterations):
        # Sample random expert and starting point
        expert_idx = np.random.randint(len(trajectories))
        traj = trajectories[expert_idx]
        max_start = len(traj) - N - 1
        if max_start < 1:
            max_start = 1
        start_epoch = np.random.randint(0, max_start)
        
        # Load starting parameters
        start_params = traj[start_epoch]
        target_params = traj[min(start_epoch + N, len(traj) - 1)]
        
        # Create student model from start params
        student = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        student.load_state_dict({k: v.to(device) for k, v in start_params.items()})
        student.train()
        
        student_optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
        
        # Train student on synthetic data for M steps
        for m in range(M):
            perm = torch.randperm(len(syn_images))
            syn_aug = DiffAugment(syn_images[perm], strategy='color_crop_cutout_flip_scale_rotate')
            out = student(syn_aug)
            loss_student = criterion(out, syn_labels[perm])
            student_optimizer.zero_grad()
            loss_student.backward()
            student_optimizer.step()
        
        # Compute trajectory matching loss
        target_params_flat = torch.cat([v.to(device).flatten() for v in target_params.values()])
        student_params_flat = torch.cat([p.flatten() for p in student.parameters()])
        
        # Normalized parameter distance
        loss = torch.sum((target_params_flat - student_params_flat) ** 2) / torch.sum(target_params_flat ** 2 + 1e-6)
        
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        scheduler.step()
        
        del student, student_optimizer
        
        if (it + 1) % 500 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (it + 1) * (iterations - it - 1)
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.6f}, "
                  f"Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")
    
    torch.cuda.empty_cache()
    total_time = time.time() - start_time
    print(f"TM IPC={ipc} complete in {total_time:.0f}s")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_soft_labels(images, labels, teacher_logits_path='/workspace/soft_labels_final.pt',
                         train_images_full=None, train_labels_full=None):
    """
    Generate soft labels for a subset of images.
    For distilled images, use teacher to generate new soft labels.
    For coreset (real images), look up from pre-computed logits.
    """
    # For coreset methods, we need to find which indices these images correspond to
    # For DD methods, we need to run the teacher on the synthetic images
    
    teacher_path = '/workspace/teacher_final.pt'
    if not os.path.exists(teacher_path):
        print("WARNING: No teacher model found, using uniform soft labels")
        return torch.zeros(len(images), 100)
    
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    model.load_state_dict(torch.load(teacher_path, map_location=device))
    model.eval()
    
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(images), 512):
            batch = images[i:i+512].to(device)
            logits = model(batch)
            logits_list.append(logits.cpu())
    
    del model
    torch.cuda.empty_cache()
    return torch.cat(logits_list, dim=0)


def main():
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Load pre-computed soft labels for full training set
    soft_labels_full = None
    if os.path.exists('/workspace/soft_labels_final.pt'):
        soft_labels_full = torch.load('/workspace/soft_labels_final.pt', map_location='cpu')
        print(f"Loaded full soft labels: {soft_labels_full.shape}")
    
    model_fn = lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))
    
    all_results = {}
    
    # ============================================================
    # 1. RANDOM CORESET
    # ============================================================
    for ipc in [10, 50]:
        print(f"\n{'='*60}")
        print(f"RANDOM IPC={ipc}")
        print(f"{'='*60}")
        
        selected = random_select(train_labels, ipc=ipc, seed=0)
        sub_images = train_images[selected]
        sub_labels = train_labels[selected]
        
        # HL
        print(f"\n--- Random IPC={ipc} HL ---")
        mean_acc, std_acc = run_experiment(
            sub_images, sub_labels, test_images, test_labels,
            model_fn, label_type='hard', epochs=300, num_runs=NUM_RUNS, verbose=True
        )
        all_results[f'Random_IPC{ipc}_HL'] = {'mean': mean_acc, 'std': std_acc}
        
        # SL
        print(f"\n--- Random IPC={ipc} SL ---")
        if soft_labels_full is not None:
            sub_soft = soft_labels_full[selected]
        else:
            sub_soft = generate_soft_labels(sub_images, sub_labels)
        
        mean_acc, std_acc = run_experiment(
            sub_images, sub_labels, test_images, test_labels,
            model_fn, label_type='soft', soft_labels=sub_soft,
            epochs=300, num_runs=NUM_RUNS, verbose=True
        )
        all_results[f'Random_IPC{ipc}_SL'] = {'mean': mean_acc, 'std': std_acc}
        
        # Save intermediate
        with open(os.path.join(RESULTS_DIR, 'results_final.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # ============================================================
    # 2. K-CENTERS CORESET
    # ============================================================
    for ipc in [10, 50]:
        print(f"\n{'='*60}")
        print(f"K-CENTERS IPC={ipc}")
        print(f"{'='*60}")
        
        selected = feature_kcenters(train_images, train_labels, 100, ipc, device)
        sub_images = train_images[selected]
        sub_labels = train_labels[selected]
        
        # HL
        print(f"\n--- K-centers IPC={ipc} HL ---")
        mean_acc, std_acc = run_experiment(
            sub_images, sub_labels, test_images, test_labels,
            model_fn, label_type='hard', epochs=300, num_runs=NUM_RUNS, verbose=True
        )
        all_results[f'Kcenter_IPC{ipc}_HL'] = {'mean': mean_acc, 'std': std_acc}
        
        # SL
        print(f"\n--- K-centers IPC={ipc} SL ---")
        if soft_labels_full is not None:
            sub_soft = soft_labels_full[selected]
        else:
            sub_soft = generate_soft_labels(sub_images, sub_labels)
        
        mean_acc, std_acc = run_experiment(
            sub_images, sub_labels, test_images, test_labels,
            model_fn, label_type='soft', soft_labels=sub_soft,
            epochs=300, num_runs=NUM_RUNS, verbose=True
        )
        all_results[f'Kcenter_IPC{ipc}_SL'] = {'mean': mean_acc, 'std': std_acc}
        
        with open(os.path.join(RESULTS_DIR, 'results_final.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # ============================================================
    # 3. DM DISTILLATION
    # ============================================================
    for ipc in [10, 50]:
        dm_path = f'/workspace/distilled_dm_v2_ipc{ipc}.pt'
        
        if os.path.exists(dm_path):
            print(f"\nLoading existing DM IPC={ipc} from {dm_path}")
            data = torch.load(dm_path, map_location='cpu')
            dm_images, dm_labels = data['images'], data['labels']
        else:
            dm_images, dm_labels = distill_dm(
                train_images, train_labels, ipc=ipc,
                iterations=10000, lr_img=1.0, device=device
            )
            torch.save({'images': dm_images, 'labels': dm_labels}, dm_path)
        
        # HL
        print(f"\n--- DM IPC={ipc} HL ---")
        mean_acc, std_acc = run_experiment(
            dm_images, dm_labels, test_images, test_labels,
            model_fn, label_type='hard', epochs=300, num_runs=NUM_RUNS, verbose=True
        )
        all_results[f'DM_IPC{ipc}_HL'] = {'mean': mean_acc, 'std': std_acc}
        
        # SL
        print(f"\n--- DM IPC={ipc} SL ---")
        dm_soft = generate_soft_labels(dm_images, dm_labels)
        mean_acc, std_acc = run_experiment(
            dm_images, dm_labels, test_images, test_labels,
            model_fn, label_type='soft', soft_labels=dm_soft,
            epochs=300, num_runs=NUM_RUNS, verbose=True
        )
        all_results[f'DM_IPC{ipc}_SL'] = {'mean': mean_acc, 'std': std_acc}
        
        with open(os.path.join(RESULTS_DIR, 'results_final.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # ============================================================
    # 4. DC DISTILLATION
    # ============================================================
    for ipc in [10, 50]:
        dc_path = f'/workspace/distilled_dc_v2_ipc{ipc}.pt'
        
        if os.path.exists(dc_path):
            print(f"\nLoading existing DC IPC={ipc} from {dc_path}")
            data = torch.load(dc_path, map_location='cpu')
            dc_images, dc_labels = data['images'], data['labels']
        else:
            # DC is expensive - use fewer iterations for IPC 50
            outer = 50 if ipc == 10 else 30
            inner = 10  # Reduced for speed
            dc_images, dc_labels = distill_dc(
                train_images, train_labels, ipc=ipc,
                outer_loops=outer, inner_loops=inner, lr_img=1.0, device=device
            )
            torch.save({'images': dc_images, 'labels': dc_labels}, dc_path)
        
        # HL
        print(f"\n--- DC IPC={ipc} HL ---")
        mean_acc, std_acc = run_experiment(
            dc_images, dc_labels, test_images, test_labels,
            model_fn, label_type='hard', epochs=300, num_runs=NUM_RUNS, verbose=True
        )
        all_results[f'DC_IPC{ipc}_HL'] = {'mean': mean_acc, 'std': std_acc}
        
        # SL
        print(f"\n--- DC IPC={ipc} SL ---")
        dc_soft = generate_soft_labels(dc_images, dc_labels)
        mean_acc, std_acc = run_experiment(
            dc_images, dc_labels, test_images, test_labels,
            model_fn, label_type='soft', soft_labels=dc_soft,
            epochs=300, num_runs=NUM_RUNS, verbose=True
        )
        all_results[f'DC_IPC{ipc}_SL'] = {'mean': mean_acc, 'std': std_acc}
        
        with open(os.path.join(RESULTS_DIR, 'results_final.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # ============================================================
    # 5. TM DISTILLATION
    # ============================================================
    for ipc in [10, 50]:
        tm_path = f'/workspace/distilled_tm_v2_ipc{ipc}.pt'
        
        if os.path.exists(tm_path):
            print(f"\nLoading existing TM IPC={ipc} from {tm_path}")
            data = torch.load(tm_path, map_location='cpu')
            tm_images, tm_labels = data['images'], data['labels']
        else:
            tm_images, tm_labels = distill_tm(
                train_images, train_labels, ipc=ipc,
                iterations=3000, lr_img=1000.0, n_experts=3,
                expert_epochs=50, M=2, N=10, device=device
            )
            torch.save({'images': tm_images, 'labels': tm_labels}, tm_path)
        
        # HL
        print(f"\n--- TM IPC={ipc} HL ---")
        mean_acc, std_acc = run_experiment(
            tm_images, tm_labels, test_images, test_labels,
            model_fn, label_type='hard', epochs=300, num_runs=NUM_RUNS, verbose=True
        )
        all_results[f'TM_IPC{ipc}_HL'] = {'mean': mean_acc, 'std': std_acc}
        
        # SL
        print(f"\n--- TM IPC={ipc} SL ---")
        tm_soft = generate_soft_labels(tm_images, tm_labels)
        mean_acc, std_acc = run_experiment(
            tm_images, tm_labels, test_images, test_labels,
            model_fn, label_type='soft', soft_labels=tm_soft,
            epochs=300, num_runs=NUM_RUNS, verbose=True
        )
        all_results[f'TM_IPC{ipc}_SL'] = {'mean': mean_acc, 'std': std_acc}
        
        with open(os.path.join(RESULTS_DIR, 'results_final.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # ============================================================
    # GENERATE FINAL TABLE
    # ============================================================
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS TABLE (Table small_scale_c100)")
    print("=" * 80)
    
    paper_results = {
        'Random_IPC10_HL': 18.64, 'Random_IPC10_SL': 33.43,
        'Random_IPC50_HL': 34.66, 'Random_IPC50_SL': 45.39,
        'Kcenter_IPC10_HL': 25.04, 'Kcenter_IPC10_SL': 34.70,
        'Kcenter_IPC50_HL': 38.64, 'Kcenter_IPC50_SL': 46.24,
        'DM_IPC10_HL': 29.23, 'DM_IPC10_SL': 26.13,
        'DM_IPC50_HL': 42.32, 'DM_IPC50_SL': 43.46,
        'DC_IPC10_HL': 28.42, 'DC_IPC10_SL': 23.54,
        'DC_IPC50_HL': 30.56, 'DC_IPC50_SL': 33.46,
        'TM_IPC10_HL': 38.18, 'TM_IPC10_SL': 37.60,
        'TM_IPC50_HL': 46.32, 'TM_IPC50_SL': 46.26,
    }
    
    header = f"{'Method':<12} {'IPC':>4} {'HL (ours)':>12} {'HL (paper)':>12} {'SL (ours)':>12} {'SL (paper)':>12}"
    print(header)
    print("-" * len(header))
    
    for method in ['Random', 'Kcenter', 'DM', 'DC', 'TM']:
        for ipc in [10, 50]:
            hl_key = f'{method}_IPC{ipc}_HL'
            sl_key = f'{method}_IPC{ipc}_SL'
            
            hl_ours = f"{all_results[hl_key]['mean']:.2f}±{all_results[hl_key]['std']:.2f}" if hl_key in all_results else "N/A"
            sl_ours = f"{all_results[sl_key]['mean']:.2f}±{all_results[sl_key]['std']:.2f}" if sl_key in all_results else "N/A"
            hl_paper = f"{paper_results.get(hl_key, 'N/A')}"
            sl_paper = f"{paper_results.get(sl_key, 'N/A')}"
            
            print(f"{method:<12} {ipc:>4} {hl_ours:>12} {hl_paper:>12} {sl_ours:>12} {sl_paper:>12}")
    
    # Save table to file
    with open(os.path.join(RESULTS_DIR, 'table_final.txt'), 'w') as f:
        f.write(header + '\n')
        f.write("-" * len(header) + '\n')
        for method in ['Random', 'Kcenter', 'DM', 'DC', 'TM']:
            for ipc in [10, 50]:
                hl_key = f'{method}_IPC{ipc}_HL'
                sl_key = f'{method}_IPC{ipc}_SL'
                hl_ours = f"{all_results[hl_key]['mean']:.2f}±{all_results[hl_key]['std']:.2f}" if hl_key in all_results else "N/A"
                sl_ours = f"{all_results[sl_key]['mean']:.2f}±{all_results[sl_key]['std']:.2f}" if sl_key in all_results else "N/A"
                hl_paper = f"{paper_results.get(hl_key, 'N/A')}"
                sl_paper = f"{paper_results.get(sl_key, 'N/A')}"
                f.write(f"{method:<12} {ipc:>4} {hl_ours:>12} {hl_paper:>12} {sl_ours:>12} {sl_paper:>12}\n")
    
    print(f"\nResults saved to {RESULTS_DIR}/results_final.json and {RESULTS_DIR}/table_final.txt")


if __name__ == '__main__':
    main()
