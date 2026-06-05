#!/usr/bin/env python
"""
Complete experiment pipeline for Table 1 (tab:small_scale_c100).
CIFAR-100, ConvNet-D3, IPC 10 and 50.

Methods: Random, K-centers, DM, DC, TM
Settings: Hard Label (HL) and Soft Label (SL)

Paper hyperparameters (Table tab:stage3_hyper):
- HL: 300 epochs, SGD lr=0.01 momentum=0.9, StepLR@epoch151(gamma=0.5), batch=256, DSA, CE
- SL: 300 epochs, AdamW lr=1e-3, Cosine scheduler, batch=256, DSA, KL-Div(T=20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import copy
import argparse
import sys

from convnet import get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors

DSA_STRATEGY = 'color_crop_cutout_flip_scale_rotate'


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================
def evaluate_model(model, test_images, test_labels, device='cuda', batch_size=512):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            imgs = test_images[i:i+batch_size].to(device)
            labels = test_labels[i:i+batch_size].to(device)
            out = model(imgs)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_and_eval_hl(syn_images, syn_labels, test_images, test_labels,
                      epochs=300, device='cuda', seed=0):
    """Hard Label evaluation: 300ep, SGD lr=0.01, StepLR@151, DSA, CE"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    n = len(syn_images)
    batch_size = min(256, n)
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            imgs = syn_images[idx].to(device)
            labels = syn_labels[idx].to(device)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return evaluate_model(model, test_images, test_labels, device)


def train_and_eval_sl(syn_images, syn_soft_logits, test_images, test_labels,
                      epochs=300, device='cuda', seed=0, T=20):
    """Soft Label evaluation: 300ep, AdamW lr=1e-3, Cosine, DSA, KL-Div(T=20)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n = len(syn_images)
    batch_size = min(256, n)
    soft_targets = F.softmax(syn_soft_logits / T, dim=1)
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            imgs = syn_images[idx].to(device)
            targets = soft_targets[idx].to(device)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            optimizer.zero_grad()
            out = model(imgs)
            log_probs = F.log_softmax(out / T, dim=1)
            loss = F.kl_div(log_probs, targets, reduction='batchmean') * (T * T)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return evaluate_model(model, test_images, test_labels, device)


# ============================================================
# TEACHER + SOFT LABELS
# ============================================================
def train_teacher(train_images, train_labels, test_images, test_labels,
                  epochs=300, device='cuda'):
    """Train teacher with DSA augmentation on full CIFAR-100."""
    print("Training teacher model (300 epochs with DSA)...")
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True,
                                         num_workers=0, drop_last=False)
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            acc = evaluate_model(model, test_images, test_labels, device)
            print(f"  Epoch {epoch+1}: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_state)
    print(f"  Best teacher: {best_acc:.2f}%")
    return model


def get_soft_logits(model, images, device='cuda'):
    """Get teacher logits for images."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), 512):
            batch = images[i:i+512].to(device)
            all_logits.append(model(batch).cpu())
    return torch.cat(all_logits, dim=0)


# ============================================================
# CORESET METHODS
# ============================================================
def select_random(train_images, train_labels, ipc, seed=0):
    np.random.seed(seed)
    indices = []
    for c in range(100):
        class_idx = (train_labels == c).nonzero(as_tuple=True)[0].numpy()
        chosen = np.random.choice(class_idx, size=ipc, replace=False)
        indices.extend(chosen.tolist())
    indices = sorted(indices)
    return train_images[indices], train_labels[indices], indices


def select_k_centers(train_images, train_labels, ipc, feature_model, device='cuda'):
    """K-centers using pretrained features + greedy farthest-first traversal."""
    print("  Computing K-centers selection...")
    
    # Extract features using teacher model
    feature_model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(device)
            feat = feature_model.embed(batch)
            all_features.append(feat.cpu())
    features = torch.cat(all_features, dim=0)
    features = F.normalize(features, dim=1)
    
    selected_indices = []
    for c in range(100):
        class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        class_feat = features[class_idx]
        n = len(class_feat)
        
        if n <= ipc:
            selected_indices.extend(class_idx.tolist())
            continue
        
        # Start with point closest to class mean
        mean_feat = class_feat.mean(dim=0, keepdim=True)
        dists_to_mean = torch.cdist(class_feat, mean_feat).squeeze()
        first = dists_to_mean.argmin().item()
        
        chosen = [first]
        min_dists = torch.cdist(class_feat, class_feat[first:first+1]).squeeze()
        
        for _ in range(ipc - 1):
            farthest = min_dists.argmax().item()
            chosen.append(farthest)
            new_dists = torch.cdist(class_feat, class_feat[farthest:farthest+1]).squeeze()
            min_dists = torch.minimum(min_dists, new_dists)
        
        selected_indices.extend(class_idx[chosen].tolist())
    
    selected_indices = sorted(selected_indices)
    return train_images[selected_indices], train_labels[selected_indices], selected_indices


# ============================================================
# DISTILLATION: DM
# ============================================================
def distill_dm(train_images, train_labels, ipc, n_iters=20000, device='cuda'):
    """Distribution Matching distillation."""
    print(f"  DM distillation (IPC={ipc}, iters={n_iters})...")
    num_classes = 100
    
    # Initialize from real images
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=device)
    syn_labels = torch.arange(num_classes, device=device).repeat_interleave(ipc)
    
    with torch.no_grad():
        for c in range(num_classes):
            class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(class_idx))[:ipc]
            syn_images[c*ipc:(c+1)*ipc] = train_images[class_idx[perm]].to(device)
    syn_images = syn_images.detach().requires_grad_(True)
    
    optimizer = torch.optim.SGD([syn_images], lr=1.0, momentum=0.5)
    
    for it in range(n_iters):
        model = get_convnet_d3().to(device)
        model.train()
        
        loss_total = torch.tensor(0.0, device=device)
        
        # Process classes in random batches for efficiency
        class_order = np.random.permutation(num_classes)
        
        for c in class_order:
            class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(class_idx))[:min(256, len(class_idx))]
            real_batch = train_images[class_idx[perm]].to(device)
            real_batch = DiffAugment(real_batch, strategy=DSA_STRATEGY)
            
            syn_batch = syn_images[c*ipc:(c+1)*ipc]
            syn_batch_aug = DiffAugment(syn_batch, strategy=DSA_STRATEGY)
            
            with torch.no_grad():
                real_feat = model.embed(real_batch).mean(0)
            syn_feat = model.embed(syn_batch_aug).mean(0)
            
            loss_total += torch.sum((real_feat - syn_feat) ** 2)
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        if (it + 1) % 5000 == 0:
            print(f"    Iter {it+1}/{n_iters}, loss: {loss_total.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# DISTILLATION: DC (gradient matching)
# ============================================================
def distill_dc(train_images, train_labels, ipc, n_iters=5000, device='cuda'):
    """Dataset Condensation via gradient matching."""
    print(f"  DC distillation (IPC={ipc}, iters={n_iters})...")
    num_classes = 100
    
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=device)
    syn_labels = torch.arange(num_classes, device=device).repeat_interleave(ipc)
    
    with torch.no_grad():
        for c in range(num_classes):
            class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(class_idx))[:ipc]
            syn_images[c*ipc:(c+1)*ipc] = train_images[class_idx[perm]].to(device)
    syn_images = syn_images.detach().requires_grad_(True)
    
    optimizer = torch.optim.SGD([syn_images], lr=1.0, momentum=0.5)
    
    for it in range(n_iters):
        model = get_convnet_d3().to(device)
        model.train()
        
        loss_total = torch.tensor(0.0, device=device)
        
        # Sample a subset of classes per iteration for efficiency
        n_classes_per_iter = min(10, num_classes)
        classes = np.random.choice(num_classes, n_classes_per_iter, replace=False)
        
        for c in classes:
            class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(class_idx))[:256]
            real_batch = train_images[class_idx[perm]].to(device)
            real_labels_batch = train_labels[class_idx[perm]].to(device)
            real_batch = DiffAugment(real_batch, strategy=DSA_STRATEGY)
            
            real_out = model(real_batch)
            real_loss = F.cross_entropy(real_out, real_labels_batch)
            real_grads = torch.autograd.grad(real_loss, model.parameters(), create_graph=False)
            
            syn_batch = syn_images[c*ipc:(c+1)*ipc]
            syn_lab = syn_labels[c*ipc:(c+1)*ipc]
            syn_batch_aug = DiffAugment(syn_batch, strategy=DSA_STRATEGY)
            
            syn_out = model(syn_batch_aug)
            syn_loss = F.cross_entropy(syn_out, syn_lab)
            syn_grads = torch.autograd.grad(syn_loss, model.parameters(), create_graph=True)
            
            for rg, sg in zip(real_grads, syn_grads):
                rg_flat = rg.detach().flatten()
                sg_flat = sg.flatten()
                loss_total += (1 - F.cosine_similarity(rg_flat.unsqueeze(0), sg_flat.unsqueeze(0)))
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        if (it + 1) % 1000 == 0:
            print(f"    Iter {it+1}/{n_iters}, loss: {loss_total.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# DISTILLATION: TM (trajectory matching)
# ============================================================
def distill_tm(train_images, train_labels, ipc, n_iters=5000,
               n_experts=5, expert_epochs=50, device='cuda'):
    """Trajectory Matching distillation."""
    print(f"  TM distillation (IPC={ipc}, iters={n_iters})...")
    num_classes = 100
    
    # Train expert trajectories
    print("    Training expert trajectories...")
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    expert_trajectories = []
    for exp_id in range(n_experts):
        torch.manual_seed(exp_id * 1000)
        model = get_convnet_d3().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        trajectory = [copy.deepcopy(model.state_dict())]
        for epoch in range(expert_epochs):
            model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(imgs), labels)
                loss.backward()
                optimizer.step()
            trajectory.append(copy.deepcopy(model.state_dict()))
        expert_trajectories.append(trajectory)
        print(f"    Expert {exp_id+1}/{n_experts} done")
    
    # Match trajectories
    print("    Matching trajectories...")
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=device)
    syn_labels = torch.arange(num_classes, device=device).repeat_interleave(ipc)
    
    with torch.no_grad():
        for c in range(num_classes):
            class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(class_idx))[:ipc]
            syn_images[c*ipc:(c+1)*ipc] = train_images[class_idx[perm]].to(device)
    syn_images = syn_images.detach().requires_grad_(True)
    
    syn_optimizer = torch.optim.SGD([syn_images], lr=1000.0, momentum=0.5)
    M = 2  # student steps
    student_lr = 0.01
    
    for it in range(n_iters):
        exp_id = np.random.randint(n_experts)
        trajectory = expert_trajectories[exp_id]
        max_start = max(1, len(trajectory) - M - 1)
        start_epoch = np.random.randint(0, max_start)
        
        start_params = trajectory[start_epoch]
        target_params = trajectory[min(start_epoch + M, len(trajectory) - 1)]
        
        student = get_convnet_d3().to(device)
        student.load_state_dict(start_params)
        student.train()
        student_opt = torch.optim.SGD(student.parameters(), lr=student_lr, momentum=0.9)
        
        for step in range(M):
            perm = torch.randperm(len(syn_images))
            idx = perm[:min(256, len(syn_images))]
            imgs = syn_images[idx]
            labels = syn_labels[idx]
            imgs_aug = DiffAugment(imgs, strategy=DSA_STRATEGY)
            student_opt.zero_grad()
            loss = F.cross_entropy(student(imgs_aug), labels)
            loss.backward()
            student_opt.step()
        
        match_loss = torch.tensor(0.0, device=device)
        target_model = get_convnet_d3().to(device)
        target_model.load_state_dict(target_params)
        
        for sp, tp in zip(student.parameters(), target_model.parameters()):
            match_loss += F.mse_loss(sp, tp.detach(), reduction='sum')
        
        n_params = sum(p.numel() for p in student.parameters())
        match_loss = match_loss / n_params
        
        syn_optimizer.zero_grad()
        match_loss.backward()
        syn_optimizer.step()
        
        if (it + 1) % 1000 == 0:
            print(f"    Iter {it+1}/{n_iters}, loss: {match_loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', 
                        default=['random', 'k_centers', 'dm', 'dc', 'tm'])
    parser.add_argument('--ipcs', nargs='+', type=int, default=[10, 50])
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--dm_iters', type=int, default=20000)
    parser.add_argument('--dc_iters', type=int, default=5000)
    parser.add_argument('--tm_iters', type=int, default=5000)
    parser.add_argument('--tm_experts', type=int, default=5)
    parser.add_argument('--skip_distill', action='store_true')
    parser.add_argument('--results_file', type=str, default='results/final_results.json')
    args = parser.parse_args()
    
    device = 'cuda'
    os.makedirs('results', exist_ok=True)
    
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Get or train teacher
    teacher_path = 'teacher_final_v3.pt'
    logits_path = 'teacher_logits_final_v3.pt'
    
    if os.path.exists(teacher_path) and os.path.exists(logits_path):
        print("Loading existing teacher...")
        teacher = get_convnet_d3().to(device)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=False))
        all_soft_logits = torch.load(logits_path, map_location='cpu', weights_only=False)
        acc = evaluate_model(teacher, test_images, test_labels, device)
        print(f"  Teacher accuracy: {acc:.2f}%")
    else:
        teacher = train_teacher(train_images, train_labels, test_images, test_labels,
                                epochs=300, device=device)
        torch.save(teacher.state_dict(), teacher_path)
        all_soft_logits = get_soft_logits(teacher, train_images, device)
        torch.save(all_soft_logits, logits_path)
    
    all_results = {}
    
    # Load existing results if any
    if os.path.exists(args.results_file):
        with open(args.results_file) as f:
            all_results = json.load(f)
        print(f"  Loaded {len(all_results)} existing results")
    
    for ipc in args.ipcs:
        for method in args.methods:
            key_base = f"{method}_ipc{ipc}"
            print(f"\n{'='*60}")
            print(f"  {method.upper()} IPC={ipc}")
            print(f"{'='*60}")
            
            # ---- Get data ----
            if method == 'random':
                syn_imgs, syn_labels, idx = select_random(train_images, train_labels, ipc, seed=42)
                syn_logits = all_soft_logits[idx]
                
            elif method == 'k_centers':
                syn_imgs, syn_labels, idx = select_k_centers(
                    train_images, train_labels, ipc, teacher, device)
                syn_logits = all_soft_logits[idx]
                
            elif method in ['dm', 'dc', 'tm']:
                distilled_path = f'distilled_{method}_ipc{ipc}_final.pt'
                
                if args.skip_distill and os.path.exists(distilled_path):
                    print(f"  Loading existing {method} data from {distilled_path}")
                    data = torch.load(distilled_path, map_location='cpu', weights_only=False)
                    syn_imgs, syn_labels = data['images'], data['labels']
                else:
                    if method == 'dm':
                        syn_imgs, syn_labels = distill_dm(
                            train_images, train_labels, ipc, args.dm_iters, device)
                    elif method == 'dc':
                        syn_imgs, syn_labels = distill_dc(
                            train_images, train_labels, ipc, args.dc_iters, device)
                    elif method == 'tm':
                        syn_imgs, syn_labels = distill_tm(
                            train_images, train_labels, ipc, args.tm_iters,
                            args.tm_experts, device=device)
                    torch.save({'images': syn_imgs, 'labels': syn_labels}, distilled_path)
                
                # Generate soft labels from teacher
                syn_logits = get_soft_logits(teacher, syn_imgs, device)
            
            print(f"  Data shape: {syn_imgs.shape}")
            
            # ---- Evaluate HL ----
            hl_key = f"{key_base}_hl"
            if hl_key not in all_results:
                print(f"\n  Evaluating HL...")
                hl_accs = []
                for run in range(args.n_runs):
                    acc = train_and_eval_hl(syn_imgs, syn_labels, test_images, test_labels,
                                            device=device, seed=run*100+42)
                    hl_accs.append(acc)
                    print(f"    HL run {run+1}/{args.n_runs}: {acc:.2f}%")
                all_results[hl_key] = {
                    'mean': float(np.mean(hl_accs)),
                    'std': float(np.std(hl_accs)),
                    'accs': hl_accs
                }
                print(f"  >> HL: {np.mean(hl_accs):.2f} ± {np.std(hl_accs):.2f}")
            else:
                print(f"  HL already computed: {all_results[hl_key]['mean']:.2f}")
            
            # ---- Evaluate SL ----
            sl_key = f"{key_base}_sl"
            if sl_key not in all_results:
                print(f"\n  Evaluating SL...")
                sl_accs = []
                for run in range(args.n_runs):
                    acc = train_and_eval_sl(syn_imgs, syn_logits, test_images, test_labels,
                                            device=device, seed=run*100+42)
                    sl_accs.append(acc)
                    print(f"    SL run {run+1}/{args.n_runs}: {acc:.2f}%")
                all_results[sl_key] = {
                    'mean': float(np.mean(sl_accs)),
                    'std': float(np.std(sl_accs)),
                    'accs': sl_accs
                }
                print(f"  >> SL: {np.mean(sl_accs):.2f} ± {np.std(sl_accs):.2f}")
            else:
                print(f"  SL already computed: {all_results[sl_key]['mean']:.2f}")
            
            # Save after each method
            with open(args.results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved to {args.results_file}")
    
    # Print final table
    print_table(all_results)


def print_table(results):
    paper = {
        'random_ipc10_hl': 18.64, 'random_ipc10_sl': 33.43,
        'random_ipc50_hl': 34.66, 'random_ipc50_sl': 45.39,
        'k_centers_ipc10_hl': 25.04, 'k_centers_ipc10_sl': 34.70,
        'k_centers_ipc50_hl': 38.64, 'k_centers_ipc50_sl': 46.24,
        'dm_ipc10_hl': 29.23, 'dm_ipc10_sl': 26.13,
        'dm_ipc50_hl': 42.32, 'dm_ipc50_sl': 43.46,
        'dc_ipc10_hl': 28.42, 'dc_ipc10_sl': 23.54,
        'dc_ipc50_hl': 30.56, 'dc_ipc50_sl': 33.46,
        'tm_ipc10_hl': 38.18, 'tm_ipc10_sl': 37.60,
        'tm_ipc50_hl': 46.32, 'tm_ipc50_sl': 46.26,
    }
    
    print("\n" + "="*90)
    print("RESULTS: Table 1 (CIFAR-100, ConvNet-D3)")
    print("="*90)
    header = f"{'Method':<12} {'IPC':>4} | {'HL (ours)':>14} {'HL (paper)':>12} | {'SL (ours)':>14} {'SL (paper)':>12}"
    print(header)
    print("-" * len(header))
    
    for method in ['random', 'k_centers', 'dm', 'dc', 'tm']:
        for ipc in [10, 50]:
            hl_key = f"{method}_ipc{ipc}_hl"
            sl_key = f"{method}_ipc{ipc}_sl"
            
            hl_str = f"{results[hl_key]['mean']:.2f}±{results[hl_key]['std']:.2f}" if hl_key in results else "N/A"
            sl_str = f"{results[sl_key]['mean']:.2f}±{results[sl_key]['std']:.2f}" if sl_key in results else "N/A"
            hl_paper = f"{paper.get(hl_key, 'N/A')}"
            sl_paper = f"{paper.get(sl_key, 'N/A')}"
            
            name = method.replace('_', '-')
            print(f"{name:<12} {ipc:>4} | {hl_str:>14} {hl_paper:>12} | {sl_str:>14} {sl_paper:>12}")
    
    print("="*90)


if __name__ == '__main__':
    main()
