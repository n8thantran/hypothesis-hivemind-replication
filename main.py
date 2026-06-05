"""
Main experiment script for replicating Table: small_scale_c100 from
"Rethinking Dataset Distillation: Hard Truths About Soft Labels"

CIFAR-100, ConvNet-D3, IPC {10, 50}
Methods: Random, K-centers, DM, DC, TM
Settings: Hard Label (HL) and Soft Label (SL)

Paper's evaluation hyperparameters:
- HL: 300 epochs, SGD lr=0.01, StepLR@epoch151 (gamma=0.5), batch=256, DSA, CE loss
- SL: 300 epochs, AdamW lr=1e-3, Cosine scheduler, batch=256, DSA, KL-Div(T=20) loss
"""

import os
import sys
import json
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ============================================================
# TEACHER MODEL
# ============================================================
def train_teacher(train_images, train_labels, test_images, test_labels, 
                  epochs=200, device='cuda'):
    """Train a ConvNet-D3 teacher on full CIFAR-100 training set."""
    print("Training teacher model...")
    model = get_convnet_d3(num_classes=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset = TensorDataset(train_images, train_labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    best_acc = 0
    best_state = None
    
    for ep in range(epochs):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # Apply DSA augmentation during teacher training
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            acc = evaluate_model(model, test_images, test_labels, device)
            print(f"  Teacher epoch {ep+1}/{epochs}: test acc = {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_state)
    print(f"  Best teacher accuracy: {best_acc:.2f}%")
    return model


def generate_soft_labels(teacher, train_images, device='cuda', batch_size=512):
    """Generate soft labels from teacher for all training images."""
    print("Generating soft labels from teacher...")
    teacher.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(train_images), batch_size):
            batch = train_images[i:i+batch_size].to(device)
            logits = teacher(batch)
            all_logits.append(logits.cpu())
    logits = torch.cat(all_logits, dim=0)
    # Store logits (not softmax) - we'll apply temperature during training
    return logits


def evaluate_model(model, test_images, test_labels, device='cuda', batch_size=256):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            imgs = test_images[i:i+batch_size].to(device)
            labels = test_labels[i:i+batch_size].to(device)
            out = model(imgs)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


# ============================================================
# EVALUATION (Student Training)
# ============================================================
def train_student_hl(images, labels, test_images, test_labels, 
                     epochs=300, device='cuda', seed=0):
    """Train student with Hard Labels.
    Paper: 300 epochs, SGD lr=0.01, StepLR@151 (gamma=0.5), batch=256, DSA, CE loss
    """
    set_seed(seed)
    model = get_convnet_d3(num_classes=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    n = len(images)
    batch_size = min(256, n)
    
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            imgs = images[idx].to(device)
            labs = labels[idx].to(device)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            out = model(imgs)
            loss = F.cross_entropy(out, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    acc = evaluate_model(model, test_images, test_labels, device)
    return acc


def train_student_sl(images, soft_logits, test_images, test_labels,
                     epochs=300, temperature=20, device='cuda', seed=0):
    """Train student with Soft Labels.
    Paper: 300 epochs, AdamW lr=1e-3, Cosine scheduler, batch=256, DSA, KL-Div(T=20)
    """
    set_seed(seed)
    model = get_convnet_d3(num_classes=100).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n = len(images)
    batch_size = min(256, n)
    T = temperature
    
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            imgs = images[idx].to(device)
            target_logits = soft_logits[idx].to(device)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            student_logits = model(imgs)
            # KL-Div with temperature T
            log_p = F.log_softmax(student_logits / T, dim=1)
            q = F.softmax(target_logits / T, dim=1)
            loss = F.kl_div(log_p, q, reduction='batchmean') * (T * T)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    acc = evaluate_model(model, test_images, test_labels, device)
    return acc


# ============================================================
# CORESET METHODS
# ============================================================
def select_random(train_images, train_labels, ipc, seed=42):
    """Random subset selection."""
    set_seed(seed)
    n_classes = 100
    indices = []
    for c in range(n_classes):
        class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_idx))[:ipc]
        indices.append(class_idx[perm])
    indices = torch.cat(indices)
    return train_images[indices], train_labels[indices], indices


def select_k_centers(train_images, train_labels, ipc, device='cuda', seed=42):
    """K-centers coreset selection using feature space (DeepCore style).
    
    1. Train a model on full data
    2. Extract features
    3. For each class, run K-center greedy to select diverse representative samples
    """
    set_seed(seed)
    print("K-centers: training feature extractor...")
    
    # Train a model for feature extraction (short training, just for features)
    model = get_convnet_d3(num_classes=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    dataset = TensorDataset(train_images, train_labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    for ep in range(50):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    # Extract features
    print("K-centers: extracting features...")
    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(device)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    features = torch.cat(all_features, dim=0)
    
    # K-center greedy per class
    print("K-centers: running K-center greedy algorithm...")
    n_classes = 100
    indices = []
    for c in range(n_classes):
        class_mask = (train_labels == c)
        class_idx = class_mask.nonzero(as_tuple=True)[0]
        class_feat = features[class_idx]
        
        # Normalize features
        class_feat = F.normalize(class_feat, dim=1)
        
        # K-center greedy: iteratively select point that is farthest from selected set
        # This maximizes minimum distance to selected set (coverage)
        n = len(class_feat)
        if n <= ipc:
            indices.append(class_idx)
            continue
        
        # Start with the point closest to the mean (most representative)
        mean_feat = class_feat.mean(dim=0, keepdim=True)
        dists_to_mean = torch.cdist(class_feat, mean_feat).squeeze()
        first = dists_to_mean.argmin().item()
        
        selected = [first]
        # Distance of each point to nearest selected point
        min_dists = torch.cdist(class_feat, class_feat[first:first+1]).squeeze()
        
        for _ in range(ipc - 1):
            # Select point with maximum distance to nearest selected point
            # Set already selected to -inf
            min_dists_copy = min_dists.clone()
            for s in selected:
                min_dists_copy[s] = -1
            next_idx = min_dists_copy.argmax().item()
            selected.append(next_idx)
            # Update min distances
            new_dists = torch.cdist(class_feat, class_feat[next_idx:next_idx+1]).squeeze()
            min_dists = torch.minimum(min_dists, new_dists)
        
        indices.append(class_idx[torch.tensor(selected)])
    
    indices = torch.cat(indices)
    return train_images[indices], train_labels[indices], indices


# ============================================================
# DISTILLATION METHODS
# ============================================================
def distill_dm(train_images, train_labels, ipc, n_iter=20000, device='cuda', seed=42):
    """Distribution Matching (DM) distillation.
    Matches mean of features between synthetic and real data.
    """
    set_seed(seed)
    n_classes = 100
    n_ch, im_h, im_w = 3, 32, 32
    
    # Initialize synthetic data from real data
    syn_images = []
    syn_labels = []
    for c in range(n_classes):
        class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_idx))[:ipc]
        syn_images.append(train_images[class_idx[perm]].clone())
        syn_labels.append(torch.full((ipc,), c, dtype=torch.long))
    
    syn_images = torch.cat(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.cat(syn_labels).to(device)
    
    optimizer = optim.SGD([syn_images], lr=1.0, momentum=0.5)
    
    # Organize real data by class  
    real_by_class = {}
    for c in range(n_classes):
        class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        real_by_class[c] = train_images[class_idx]
    
    for it in range(n_iter):
        # Sample a new model every iteration (fresh random model for feature extraction)
        model = get_convnet_d3(num_classes=100).to(device)
        model.eval()
        
        loss = torch.tensor(0.0, device=device)
        
        # For each class, match feature distributions
        for c in range(n_classes):
            # Real data sample
            real_c = real_by_class[c]
            real_idx = torch.randperm(len(real_c))[:min(256, len(real_c))]
            real_batch = real_c[real_idx].to(device)
            real_batch = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate')
            
            # Synthetic data for class c
            syn_c = syn_images[syn_labels == c]
            syn_c_aug = DiffAugment(syn_c, strategy='color_crop_cutout_flip_scale_rotate')
            
            # Get features
            with torch.no_grad():
                real_feat = model.embed(real_batch)
            syn_feat = model.embed(syn_c_aug)
            
            # Match mean
            loss += torch.mean((real_feat.mean(0) - syn_feat.mean(0)) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 2000 == 0:
            print(f"  DM iter {it+1}/{n_iter}: loss = {loss.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_dc(train_images, train_labels, ipc, n_iter=5000, device='cuda', seed=42):
    """Dataset Condensation via Gradient Matching (DC/GM).
    Matches gradients between synthetic and real data.
    """
    set_seed(seed)
    n_classes = 100
    
    # Initialize synthetic data from real data
    syn_images = []
    syn_labels = []
    for c in range(n_classes):
        class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_idx))[:ipc]
        syn_images.append(train_images[class_idx[perm]].clone())
        syn_labels.append(torch.full((ipc,), c, dtype=torch.long))
    
    syn_images = torch.cat(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.cat(syn_labels).to(device)
    
    optimizer = optim.SGD([syn_images], lr=1.0, momentum=0.5)
    
    # Organize real data by class
    real_by_class = {}
    for c in range(n_classes):
        class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        real_by_class[c] = train_images[class_idx]
    
    for it in range(n_iter):
        model = get_convnet_d3(num_classes=100).to(device)
        model.train()
        
        loss = torch.tensor(0.0, device=device)
        
        for c in range(n_classes):
            # Real data gradient
            real_c = real_by_class[c]
            real_idx = torch.randperm(len(real_c))[:min(256, len(real_c))]
            real_batch = real_c[real_idx].to(device)
            real_batch = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate')
            real_out = model(real_batch)
            real_loss = F.cross_entropy(real_out, torch.full((len(real_batch),), c, device=device, dtype=torch.long))
            real_grads = torch.autograd.grad(real_loss, model.parameters(), create_graph=False)
            
            # Synthetic data gradient
            syn_c = syn_images[syn_labels == c]
            syn_c_aug = DiffAugment(syn_c, strategy='color_crop_cutout_flip_scale_rotate')
            syn_out = model(syn_c_aug)
            syn_loss_val = F.cross_entropy(syn_out, torch.full((len(syn_c_aug),), c, device=device, dtype=torch.long))
            syn_grads = torch.autograd.grad(syn_loss_val, model.parameters(), create_graph=True)
            
            # Match gradients (cosine distance)
            for rg, sg in zip(real_grads, syn_grads):
                rg_flat = rg.detach().flatten()
                sg_flat = sg.flatten()
                cos_sim = F.cosine_similarity(rg_flat.unsqueeze(0), sg_flat.unsqueeze(0))
                loss += (1 - cos_sim)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 1000 == 0:
            print(f"  DC iter {it+1}/{n_iter}: loss = {loss.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_tm(train_images, train_labels, ipc, expert_epochs=100, 
               n_iter=5000, device='cuda', seed=42):
    """Trajectory Matching (TM) distillation.
    Match training trajectories of models on synthetic vs real data.
    """
    set_seed(seed)
    n_classes = 100
    
    # Step 1: Generate expert trajectories on real data
    print("  TM: generating expert trajectories...")
    expert_trajectories = []
    for exp_i in range(5):  # 5 expert trajectories
        set_seed(seed + exp_i + 100)
        model = get_convnet_d3(num_classes=100).to(device)
        trajectory = [copy.deepcopy(model.state_dict())]
        
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        dataset = TensorDataset(train_images, train_labels)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        for ep in range(expert_epochs):
            model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
                out = model(imgs)
                loss = F.cross_entropy(out, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            trajectory.append(copy.deepcopy(model.state_dict()))
        
        expert_trajectories.append(trajectory)
        print(f"    Expert {exp_i}: done ({expert_epochs} epochs)")
    
    # Step 2: Initialize synthetic data
    syn_images = []
    syn_labels = []
    for c in range(n_classes):
        class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_idx))[:ipc]
        syn_images.append(train_images[class_idx[perm]].clone())
        syn_labels.append(torch.full((ipc,), c, dtype=torch.long))
    
    syn_images = torch.cat(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.cat(syn_labels).to(device)
    
    syn_optimizer = optim.SGD([syn_images], lr=100.0, momentum=0.5)
    
    # Step 3: TM optimization
    print("  TM: optimizing synthetic data...")
    for it in range(n_iter):
        # Sample expert trajectory and time step
        exp_idx = np.random.randint(len(expert_trajectories))
        traj = expert_trajectories[exp_idx]
        max_t = len(traj) - 2  # need t and t+N
        if max_t < 1:
            continue
        t = np.random.randint(0, max_t)
        
        # Starting parameters
        start_params = traj[t]
        target_params = traj[min(t + 2, len(traj) - 1)]  # N=2 steps ahead
        
        # Initialize student from start_params
        student = get_convnet_d3(num_classes=100).to(device)
        student.load_state_dict(start_params)
        student_opt = optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
        
        # Train student on synthetic data for a few steps
        for _ in range(2):
            perm = torch.randperm(len(syn_images))
            batch_size = min(256, len(syn_images))
            for start in range(0, len(syn_images), batch_size):
                idx = perm[start:start+batch_size]
                imgs = syn_images[idx]
                labs = syn_labels[idx]
                imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
                out = student(imgs)
                loss = F.cross_entropy(out, labs)
                student_opt.zero_grad()
                loss.backward()
                student_opt.step()
        
        # Match student params to target params
        match_loss = torch.tensor(0.0, device=device)
        for (name, sp), (_, tp) in zip(student.named_parameters(), 
                                        [(n, p) for n, p in zip(
                                            target_params.keys(), 
                                            target_params.values())]):
            if name in target_params:
                tp_val = target_params[name].to(device)
                sp_flat = sp.flatten()
                tp_flat = tp_val.flatten()
                match_loss += torch.mean((sp_flat - tp_flat) ** 2) / (torch.mean(tp_flat ** 2) + 1e-6)
        
        syn_optimizer.zero_grad()
        match_loss.backward()
        syn_optimizer.step()
        
        if (it + 1) % 1000 == 0:
            print(f"  TM iter {it+1}/{n_iter}: loss = {match_loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_experiment(method, ipc, train_images, train_labels, test_images, test_labels,
                   teacher_logits, device='cuda', n_runs=3, skip_sl=False, skip_hl=False):
    """Run full experiment for a method/IPC combo."""
    print(f"\n{'='*60}")
    print(f"Method: {method}, IPC: {ipc}")
    print(f"{'='*60}")
    
    # Get dataset
    if method == 'random':
        images, labels, indices = select_random(train_images, train_labels, ipc)
        sl_logits = teacher_logits[indices] if teacher_logits is not None else None
    elif method == 'k_centers':
        images, labels, indices = select_k_centers(train_images, train_labels, ipc, device)
        sl_logits = teacher_logits[indices] if teacher_logits is not None else None
    elif method == 'dm':
        images, labels = distill_dm(train_images, train_labels, ipc, 
                                     n_iter=20000 if ipc == 10 else 10000, device=device)
        sl_logits = None  # Will need teacher to generate
    elif method == 'dc':
        images, labels = distill_dc(train_images, train_labels, ipc,
                                     n_iter=5000 if ipc == 10 else 3000, device=device)
        sl_logits = None
    elif method == 'tm':
        images, labels = distill_tm(train_images, train_labels, ipc,
                                     n_iter=5000, device=device)
        sl_logits = None
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # For DD methods, get soft labels from teacher
    if sl_logits is None and teacher_logits is not None:
        # Relabel synthetic images with teacher
        print("  Generating soft labels for synthetic data from teacher...")
        teacher_model = torch.load('teacher_model.pt', weights_only=False)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        with torch.no_grad():
            sl_logits = teacher_model(images.to(device)).cpu()
    
    print(f"  Dataset: {images.shape}, labels: {labels.shape}")
    
    results = {'method': method, 'ipc': ipc, 'hl_accs': [], 'sl_accs': []}
    
    # HL evaluation
    if not skip_hl:
        for run in range(n_runs):
            acc = train_student_hl(images, labels, test_images, test_labels, 
                                   epochs=300, device=device, seed=run)
            results['hl_accs'].append(acc)
            print(f"  HL run {run}: {acc:.2f}%")
    
    # SL evaluation
    if not skip_sl and sl_logits is not None:
        for run in range(n_runs):
            acc = train_student_sl(images, sl_logits, test_images, test_labels,
                                   epochs=300, temperature=20, device=device, seed=run)
            results['sl_accs'].append(acc)
            print(f"  SL run {run}: {acc:.2f}%")
    
    if results['hl_accs']:
        results['hl_mean'] = np.mean(results['hl_accs'])
        results['hl_std'] = np.std(results['hl_accs'])
        print(f"  HL: {results['hl_mean']:.2f} ± {results['hl_std']:.2f}")
    
    if results['sl_accs']:
        results['sl_mean'] = np.mean(results['sl_accs'])
        results['sl_std'] = np.std(results['sl_accs'])
        print(f"  SL: {results['sl_mean']:.2f} ± {results['sl_std']:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='random',
                       choices=['random', 'k_centers', 'dm', 'dc', 'tm', 'all'])
    parser.add_argument('--ipc', type=int, default=10, choices=[10, 50])
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--skip_sl', action='store_true')
    parser.add_argument('--skip_hl', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = args.device
    
    # Load data
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Get or train teacher for soft labels
    teacher_logits_path = 'teacher_logits.pt'
    teacher_model_path = 'teacher_model.pt'
    
    if os.path.exists(teacher_logits_path):
        print("Loading existing teacher logits...")
        teacher_logits = torch.load(teacher_logits_path, weights_only=True)
    else:
        print("Need to train teacher...")
        teacher = train_teacher(train_images, train_labels, test_images, test_labels,
                               epochs=300, device=device)
        torch.save(teacher, teacher_model_path)
        teacher_logits = generate_soft_labels(teacher, train_images, device)
        torch.save(teacher_logits, teacher_logits_path)
    
    # Run experiments
    if args.method == 'all':
        methods = ['random', 'k_centers', 'dm', 'dc', 'tm']
    else:
        methods = [args.method]
    
    all_results = {}
    for method in methods:
        key = f"{method}_ipc{args.ipc}"
        result = run_experiment(method, args.ipc, train_images, train_labels,
                               test_images, test_labels, teacher_logits,
                               device=device, n_runs=args.n_runs,
                               skip_sl=args.skip_sl, skip_hl=args.skip_hl)
        all_results[key] = result
        
        # Save incrementally
        os.makedirs('results', exist_ok=True)
        with open('results/results_main.json', 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<15} {'IPC':<6} {'HL':<20} {'SL':<20}")
    print("-"*60)
    for key, res in all_results.items():
        hl_str = f"{res.get('hl_mean',0):.2f}±{res.get('hl_std',0):.2f}" if res.get('hl_accs') else "N/A"
        sl_str = f"{res.get('sl_mean',0):.2f}±{res.get('sl_std',0):.2f}" if res.get('sl_accs') else "N/A"
        print(f"{res['method']:<15} {res['ipc']:<6} {hl_str:<20} {sl_str:<20}")
    
    return all_results


if __name__ == '__main__':
    main()
