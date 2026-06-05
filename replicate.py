#!/usr/bin/env python3
"""
Replicate Table 1 (small_scale_c100) from:
"Rethinking Dataset Distillation: Hard Truths About Soft Labels"

CIFAR-100, ConvNet-D3, IPC 10 & 50
Methods: Random, K-centers, DC, DM, TM
Settings: HL (hard labels) and SL (soft labels)

Evaluation hyperparameters (from paper Table tab:stage3_hyper):
  HL: 300 epochs, SGD lr=0.01, StepLR@151 (halve), batch=256, DSA aug, CE loss
  SL: 300 epochs, AdamW lr=1e-3, Cosine scheduler, batch=256, DSA aug, KL-Div(T=20)
"""

import os
import sys
import json
import time
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms

from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment, ParamDiffAug

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD  = [0.2675, 0.2565, 0.2761]

# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────
def get_cifar100(data_dir='./data'):
    """Load CIFAR-100 as normalized tensors."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])
    train_ds = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
    
    # Pre-load all data to GPU
    train_images = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    train_labels = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    test_images  = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    test_labels  = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    
    return train_images, train_labels, test_images, test_labels


# ──────────────────────────────────────────────────────────────
# Teacher model for soft labels
# ──────────────────────────────────────────────────────────────
def train_teacher(train_images, train_labels, test_images, test_labels, 
                  epochs=300, lr=0.01, save_path='teacher_model.pt'):
    """Train ConvNet-D3 teacher on full CIFAR-100."""
    if os.path.exists(save_path):
        print(f"Loading existing teacher from {save_path}")
        model = get_convnet_d3().to(DEVICE)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.eval()
        # Quick test
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(test_images.to(DEVICE), test_labels.to(DEVICE)),
                                     batch_size=256, shuffle=False)
            correct = total = 0
            for x, y in test_loader:
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        print(f"Teacher accuracy: {100.*correct/total:.2f}%")
        return model
    
    print("Training teacher model on full CIFAR-100...")
    model = get_convnet_d3().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    ds = TensorDataset(train_images, train_labels)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    param_aug = ParamDiffAug()
    
    best_acc = 0
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate', param=param_aug)
            out = model(x)
            loss = F.cross_entropy(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (ep + 1) % 50 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_loader = DataLoader(TensorDataset(test_images.to(DEVICE), test_labels.to(DEVICE)),
                                         batch_size=256, shuffle=False)
                correct = total = 0
                for x, y in test_loader:
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += len(y)
            acc = 100.*correct/total
            print(f"  Teacher epoch {ep+1}/{epochs}: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)
    
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    print(f"Teacher trained. Best acc: {best_acc:.2f}%")
    return model


def generate_soft_labels(teacher, train_images, temperature=20.0, save_path='soft_labels_t20.pt'):
    """Generate soft labels from teacher with given temperature."""
    if os.path.exists(save_path):
        print(f"Loading cached soft labels from {save_path}")
        return torch.load(save_path, map_location='cpu')
    
    print("Generating soft labels...")
    teacher.eval()
    all_soft = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            x = train_images[i:i+256].to(DEVICE)
            logits = teacher(x) / temperature
            soft = F.softmax(logits, dim=1)
            all_soft.append(soft.cpu())
    soft_labels = torch.cat(all_soft, dim=0)
    torch.save(soft_labels, save_path)
    print(f"Soft labels generated: {soft_labels.shape}")
    return soft_labels


# ──────────────────────────────────────────────────────────────
# Coreset selection
# ──────────────────────────────────────────────────────────────
def select_random(train_labels, ipc, num_classes=100, seed=0):
    """Random balanced subset: ipc images per class."""
    rng = np.random.RandomState(seed)
    indices = []
    for c in range(num_classes):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0].numpy()
        chosen = rng.choice(cls_idx, size=ipc, replace=False)
        indices.extend(chosen.tolist())
    return indices


def select_kcenter(train_images, train_labels, ipc, num_classes=100, seed=0):
    """
    K-Center coreset selection using feature embeddings.
    DeepCore approach: train a model, extract features, run K-center greedy per class.
    K-center greedy: iteratively pick the point furthest from the current selected set.
    """
    print("Running K-Center selection with feature embeddings...")
    
    # Train a quick feature extractor (just use a fresh model trained briefly)
    model = get_convnet_d3().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    ds = TensorDataset(train_images, train_labels)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    param_aug = ParamDiffAug()
    
    # Train for enough epochs to get meaningful features
    model.train()
    for ep in range(50):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate', param=param_aug)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Extract features
    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            x = train_images[i:i+256].to(DEVICE)
            feat = model.embed(x)
            all_features.append(feat.cpu())
    features = torch.cat(all_features, dim=0)
    
    # K-center greedy per class
    rng = np.random.RandomState(seed)
    indices = []
    for c in range(num_classes):
        cls_mask = (train_labels == c).numpy()
        cls_idx = np.where(cls_mask)[0]
        cls_feat = features[cls_idx].numpy()
        
        # Normalize features
        norms = np.linalg.norm(cls_feat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        cls_feat = cls_feat / norms
        
        # K-center greedy: pick the point that maximizes minimum distance to selected set
        n = len(cls_idx)
        selected = [rng.randint(0, n)]  # Start with random point
        
        # Compute distances from first selected point to all others
        min_dist = np.full(n, np.inf)
        
        for _ in range(ipc - 1):
            # Update min distances
            last = cls_feat[selected[-1]]
            d = np.sum((cls_feat - last[None, :]) ** 2, axis=1)
            min_dist = np.minimum(min_dist, d)
            min_dist[selected] = -1  # Exclude already selected
            
            # Pick the point with maximum minimum distance
            next_idx = np.argmax(min_dist)
            selected.append(next_idx)
        
        for s in selected:
            indices.append(cls_idx[s])
    
    print(f"K-Center selected {len(indices)} samples")
    return indices


# ──────────────────────────────────────────────────────────────
# Dataset Distillation Methods
# ──────────────────────────────────────────────────────────────

def distill_dm(train_images, train_labels, ipc, num_classes=100, 
               num_iters=20000, lr_img=1.0, seed=0):
    """Distribution Matching (DM) distillation."""
    print(f"Distilling with DM (IPC={ipc}, iters={num_iters})...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize synthetic data
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=DEVICE, requires_grad=True)
    syn_labels = torch.arange(num_classes, device=DEVICE).repeat_interleave(ipc)
    
    # Initialize from real data
    with torch.no_grad():
        for c in range(num_classes):
            cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(cls_idx))[:ipc]
            syn_images.data[c*ipc:(c+1)*ipc] = train_images[cls_idx[perm]].to(DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    param_aug = ParamDiffAug()
    
    # Organize training data by class
    class_data = {}
    for c in range(num_classes):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        class_data[c] = train_images[cls_idx].to(DEVICE)
    
    for it in range(num_iters):
        model = get_convnet_d3().to(DEVICE)
        model.train()
        
        loss = torch.tensor(0.0, device=DEVICE)
        
        for c in range(num_classes):
            # Real images for this class
            idx_real = torch.randperm(len(class_data[c]))[:256]
            real_batch = class_data[c][idx_real]
            
            # Synthetic images for this class
            syn_batch = syn_images[c*ipc:(c+1)*ipc]
            
            # Apply DSA
            seed_aug = int(time.time() * 1000) % 100000
            real_aug = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate', 
                                   seed=seed_aug, param=param_aug)
            syn_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate', 
                                  seed=seed_aug, param=param_aug)
            
            # Feature matching
            with torch.no_grad():
                real_feat = model.embed(real_aug).mean(0)
            syn_feat = model.embed(syn_aug).mean(0)
            
            loss += torch.sum((real_feat - syn_feat) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 2000 == 0:
            print(f"  DM iter {it+1}/{num_iters}, loss={loss.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_dc(train_images, train_labels, ipc, num_classes=100,
               outer_iters=1000, inner_iters=1, lr_img=1.0, lr_net=0.01, seed=0):
    """Dataset Condensation via Gradient Matching (DC)."""
    print(f"Distilling with DC (IPC={ipc}, outer_iters={outer_iters})...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize synthetic data
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=DEVICE, requires_grad=True)
    syn_labels = torch.arange(num_classes, device=DEVICE).repeat_interleave(ipc)
    
    # Initialize from real data
    with torch.no_grad():
        for c in range(num_classes):
            cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(cls_idx))[:ipc]
            syn_images.data[c*ipc:(c+1)*ipc] = train_images[cls_idx[perm]].to(DEVICE)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    param_aug = ParamDiffAug()
    
    class_data = {}
    for c in range(num_classes):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        class_data[c] = train_images[cls_idx].to(DEVICE)
    
    for it in range(outer_iters):
        model = get_convnet_d3().to(DEVICE)
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        loss_total = torch.tensor(0.0, device=DEVICE)
        
        for c in range(num_classes):
            # Real data for class c
            idx_real = torch.randperm(len(class_data[c]))[:256]
            real_batch = class_data[c][idx_real]
            real_labels_batch = torch.full((len(real_batch),), c, device=DEVICE, dtype=torch.long)
            
            # Synthetic data for class c
            syn_batch = syn_images[c*ipc:(c+1)*ipc]
            syn_labels_batch = torch.full((ipc,), c, device=DEVICE, dtype=torch.long)
            
            # DSA
            seed_aug = int(time.time() * 1000) % 100000
            real_aug = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate',
                                   seed=seed_aug, param=param_aug)
            syn_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate',
                                  seed=seed_aug, param=param_aug)
            
            # Gradient matching
            out_real = model(real_aug)
            loss_real = criterion(out_real, real_labels_batch)
            grad_real = torch.autograd.grad(loss_real, model.parameters(), create_graph=False)
            
            out_syn = model(syn_aug)
            loss_syn = criterion(out_syn, syn_labels_batch)
            grad_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)
            
            # Match gradients
            for g_r, g_s in zip(grad_real, grad_syn):
                g_r = g_r.detach()
                # Gradient matching loss (from DC paper)
                loss_total += torch.sum((g_r - g_s) ** 2) / (torch.sum(g_r ** 2) + 1e-8)
        
        optimizer_img.zero_grad()
        loss_total.backward()
        optimizer_img.step()
        
        if (it + 1) % 200 == 0:
            print(f"  DC iter {it+1}/{outer_iters}, loss={loss_total.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_tm(train_images, train_labels, ipc, num_classes=100,
               num_iters=1000, lr_img=0.01, seed=0, 
               expert_dir='expert_trajectories'):
    """Trajectory Matching (TM) distillation."""
    print(f"Distilling with TM (IPC={ipc}, iters={num_iters})...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # First, train expert trajectories if not available
    expert_file = os.path.join(expert_dir, 'experts.pt')
    if not os.path.exists(expert_file):
        os.makedirs(expert_dir, exist_ok=True)
        print("  Training expert trajectories...")
        num_experts = 20  
        expert_epochs = 50
        
        all_trajectories = []
        ds = TensorDataset(train_images, train_labels)
        param_aug = ParamDiffAug()
        
        for exp_i in range(num_experts):
            model = get_convnet_d3().to(DEVICE)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            loader = DataLoader(ds, batch_size=256, shuffle=True)
            
            trajectory = [copy.deepcopy(model.state_dict())]
            
            for ep in range(expert_epochs):
                model.train()
                for x, y in loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate', param=param_aug)
                    loss = F.cross_entropy(model(x), y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                trajectory.append(copy.deepcopy(model.state_dict()))
            
            # Save only to CPU to save memory
            cpu_traj = [{k: v.cpu() for k, v in sd.items()} for sd in trajectory]
            all_trajectories.append(cpu_traj)
            print(f"    Expert {exp_i+1}/{num_experts} done")
        
        torch.save(all_trajectories, expert_file)
        print(f"  Expert trajectories saved to {expert_file}")
    else:
        print(f"  Loading expert trajectories from {expert_file}")
    
    all_trajectories = torch.load(expert_file, map_location='cpu')
    
    # Initialize synthetic data
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=DEVICE, requires_grad=True)
    syn_labels = torch.arange(num_classes, device=DEVICE).repeat_interleave(ipc)
    
    # Initialize from real data
    with torch.no_grad():
        for c in range(num_classes):
            cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(cls_idx))[:ipc]
            syn_images.data[c*ipc:(c+1)*ipc] = train_images[cls_idx[perm]].to(DEVICE)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    param_aug = ParamDiffAug()
    
    num_experts = len(all_trajectories)
    expert_epochs = len(all_trajectories[0]) - 1
    
    for it in range(num_iters):
        # Sample random expert and starting epoch
        exp_idx = np.random.randint(num_experts)
        start_epoch = np.random.randint(0, max(1, expert_epochs - 2))
        
        # Get expert parameters at start and end
        expert_start = all_trajectories[exp_idx][start_epoch]
        expert_end = all_trajectories[exp_idx][min(start_epoch + 2, expert_epochs)]
        
        # Load student model with expert start params
        student = get_convnet_d3().to(DEVICE)
        student.load_state_dict({k: v.to(DEVICE) for k, v in expert_start.items()})
        student.train()
        
        # Train student on synthetic data for a few steps
        student_opt = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
        
        for _ in range(30):  # Matching window
            seed_aug = int(time.time() * 1000) % 100000
            syn_aug = DiffAugment(syn_images, strategy='color_crop_cutout_flip_scale_rotate',
                                  seed=seed_aug, param=param_aug)
            loss = F.cross_entropy(student(syn_aug), syn_labels)
            student_opt.zero_grad()
            loss.backward()
            student_opt.step()
        
        # Trajectory matching loss: distance between student params and expert end params
        tm_loss = torch.tensor(0.0, device=DEVICE)
        expert_end_device = {k: v.to(DEVICE) for k, v in expert_end.items()}
        
        param_norm = torch.tensor(0.0, device=DEVICE)
        param_diff = torch.tensor(0.0, device=DEVICE)
        for (name, p_student), (_, p_expert) in zip(student.named_parameters(), 
                                                      expert_end_device.items()):
            param_diff += torch.sum((p_student - p_expert) ** 2)
            param_norm += torch.sum(p_expert ** 2)
        
        tm_loss = param_diff / (param_norm + 1e-8)
        
        optimizer_img.zero_grad()
        tm_loss.backward()
        optimizer_img.step()
        
        if (it + 1) % 200 == 0:
            print(f"  TM iter {it+1}/{num_iters}, loss={tm_loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_hl(images, labels, test_images, test_labels, 
                epochs=300, lr=0.01, batch_size=256, seed=0):
    """
    Hard Label evaluation per paper:
    300 epochs, SGD lr=0.01, StepLR@151 (halve), batch=256, DSA, CE loss
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    # Move data to device
    images_dev = images.to(DEVICE)
    labels_dev = labels.to(DEVICE)
    test_images_dev = test_images.to(DEVICE)
    test_labels_dev = test_labels.to(DEVICE)
    
    param_aug = ParamDiffAug()
    n = len(images_dev)
    
    for ep in range(epochs):
        model.train()
        
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            x = images_dev[idx]
            y = labels_dev[idx]
            
            x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate', param=param_aug)
            
            out = model(x)
            loss = F.cross_entropy(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    # Test
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(test_images_dev), 256):
            x = test_images_dev[start:start+256]
            y = test_labels_dev[start:start+256]
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    
    acc = 100. * correct / total
    return acc


def evaluate_sl(images, soft_labels, test_images, test_labels,
                temperature=20.0, epochs=300, lr=1e-3, batch_size=256, seed=0):
    """
    Soft Label evaluation per paper:
    300 epochs, AdamW lr=1e-3, Cosine scheduler, batch=256, DSA, KL-Div(T=20)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    images_dev = images.to(DEVICE)
    soft_dev = soft_labels.to(DEVICE)
    test_images_dev = test_images.to(DEVICE)
    test_labels_dev = test_labels.to(DEVICE)
    
    param_aug = ParamDiffAug()
    n = len(images_dev)
    
    for ep in range(epochs):
        model.train()
        
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            x = images_dev[idx]
            q = soft_dev[idx]  # soft label targets
            
            x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate', param=param_aug)
            
            logits = model(x)
            # KL-Div with temperature
            log_p = F.log_softmax(logits / temperature, dim=1)
            loss = F.kl_div(log_p, q, reduction='batchmean') * (temperature ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    # Test
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(test_images_dev), 256):
            x = test_images_dev[start:start+256]
            y = test_labels_dev[start:start+256]
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    
    acc = 100. * correct / total
    return acc


def run_evaluation(images, labels, soft_labels, test_images, test_labels, 
                   label_type, num_runs=3, **kwargs):
    """Run evaluation multiple times and return stats."""
    accs = []
    for run in range(num_runs):
        seed = run * 42 + 7
        if label_type == 'hl':
            acc = evaluate_hl(images, labels, test_images, test_labels, seed=seed, **kwargs)
        else:
            acc = evaluate_sl(images, soft_labels, test_images, test_labels, seed=seed, **kwargs)
        accs.append(acc)
        print(f"    Run {run+1}/{num_runs}: {acc:.2f}%")
    
    mean = np.mean(accs)
    std = np.std(accs)
    print(f"    -> {mean:.2f} ± {std:.2f}")
    return {'mean': round(mean, 2), 'std': round(std, 2), 'accs': [round(a, 2) for a in accs]}


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', 
                        default=['random', 'kcenter', 'dm', 'dc', 'tm'],
                        help='Methods to run')
    parser.add_argument('--ipcs', nargs='+', type=int, default=[10, 50])
    parser.add_argument('--settings', nargs='+', default=['hl', 'sl'])
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--dm_iters', type=int, default=20000)
    parser.add_argument('--dc_iters', type=int, default=1000)
    parser.add_argument('--tm_iters', type=int, default=1000)
    parser.add_argument('--output', type=str, default='results/replicate_results.json')
    args = parser.parse_args()
    
    os.makedirs('results', exist_ok=True)
    
    # Load data
    print("=" * 60)
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Train teacher & generate soft labels
    teacher = train_teacher(train_images, train_labels, test_images, test_labels,
                           epochs=300, save_path='teacher_model.pt')
    all_soft_labels = generate_soft_labels(teacher, train_images, temperature=20.0,
                                           save_path='soft_labels_t20.pt')
    
    results = {}
    
    for method in args.methods:
        for ipc in args.ipcs:
            print("\n" + "=" * 60)
            print(f"Method: {method}, IPC: {ipc}")
            print("=" * 60)
            
            # Get/distill data
            if method == 'random':
                indices = select_random(train_labels, ipc, seed=42)
                images = train_images[indices]
                labels = train_labels[indices]
                soft = all_soft_labels[indices]
                is_distilled = False
                
            elif method == 'kcenter':
                indices = select_kcenter(train_images, train_labels, ipc, seed=42)
                images = train_images[indices]
                labels = train_labels[indices]
                soft = all_soft_labels[indices]
                is_distilled = False
                
            elif method == 'dm':
                cache_file = f'distilled/dm_ipc{ipc}.pt'
                if os.path.exists(cache_file):
                    data = torch.load(cache_file, map_location='cpu')
                    images, labels = data['images'], data['labels']
                else:
                    os.makedirs('distilled', exist_ok=True)
                    images, labels = distill_dm(train_images, train_labels, ipc,
                                                num_iters=args.dm_iters)
                    torch.save({'images': images, 'labels': labels}, cache_file)
                is_distilled = True
                
            elif method == 'dc':
                cache_file = f'distilled/dc_ipc{ipc}.pt'
                if os.path.exists(cache_file):
                    data = torch.load(cache_file, map_location='cpu')
                    images, labels = data['images'], data['labels']
                else:
                    os.makedirs('distilled', exist_ok=True)
                    images, labels = distill_dc(train_images, train_labels, ipc,
                                                outer_iters=args.dc_iters)
                    torch.save({'images': images, 'labels': labels}, cache_file)
                is_distilled = True
                
            elif method == 'tm':
                cache_file = f'distilled/tm_ipc{ipc}.pt'
                if os.path.exists(cache_file):
                    data = torch.load(cache_file, map_location='cpu')
                    images, labels = data['images'], data['labels']
                else:
                    os.makedirs('distilled', exist_ok=True)
                    images, labels = distill_tm(train_images, train_labels, ipc,
                                                num_iters=args.tm_iters)
                    torch.save({'images': images, 'labels': labels}, cache_file)
                is_distilled = True
            
            # For distilled data with SL, generate soft labels from teacher
            if is_distilled and 'sl' in args.settings:
                teacher.eval()
                with torch.no_grad():
                    soft_chunks = []
                    for i in range(0, len(images), 256):
                        x = images[i:i+256].to(DEVICE)
                        logits = teacher(x) / 20.0
                        soft_chunks.append(F.softmax(logits, dim=1).cpu())
                    soft = torch.cat(soft_chunks, dim=0)
            elif not is_distilled:
                pass  # soft already set from all_soft_labels[indices]
            
            # Evaluate
            for setting in args.settings:
                key = f"{method}_ipc{ipc}_{setting}"
                print(f"\n  Evaluating: {key}")
                
                if setting == 'hl':
                    result = run_evaluation(images, labels, None, 
                                           test_images, test_labels, 
                                           'hl', num_runs=args.num_runs)
                else:
                    result = run_evaluation(images, labels, soft,
                                           test_images, test_labels,
                                           'sl', num_runs=args.num_runs)
                
                results[key] = result
                
                # Save intermediate results
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  Saved to {args.output}")
    
    # Print final table
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("=" * 80)
    print(f"{'Method':<12} {'IPC':<6} {'HL (ours)':<20} {'SL (ours)':<20}")
    print("-" * 60)
    
    paper_results = {
        'dm_ipc10': {'hl': '29.23±0.26', 'sl': '26.13±0.10'},
        'dm_ipc50': {'hl': '42.32±0.37', 'sl': '43.46±0.18'},
        'dc_ipc10': {'hl': '28.42±0.29', 'sl': '23.54±0.31'},
        'dc_ipc50': {'hl': '30.56±0.56', 'sl': '33.46±0.38'},
        'tm_ipc10': {'hl': '38.18±0.42', 'sl': '37.60±0.25'},
        'tm_ipc50': {'hl': '46.32±0.26', 'sl': '46.26±0.30'},
        'random_ipc10': {'hl': '18.64±0.25', 'sl': '33.43±0.18'},
        'random_ipc50': {'hl': '34.66±0.41', 'sl': '45.39±0.23'},
        'kcenter_ipc10': {'hl': '25.04±0.30', 'sl': '34.70±0.27'},
        'kcenter_ipc50': {'hl': '38.64±0.43', 'sl': '46.24±0.12'},
    }
    
    for method in args.methods:
        for ipc in args.ipcs:
            hl_key = f"{method}_ipc{ipc}_hl"
            sl_key = f"{method}_ipc{ipc}_sl"
            paper_key = f"{method}_ipc{ipc}"
            
            hl_str = sl_str = "N/A"
            if hl_key in results:
                r = results[hl_key]
                hl_str = f"{r['mean']:.2f}±{r['std']:.2f}"
            if sl_key in results:
                r = results[sl_key]
                sl_str = f"{r['mean']:.2f}±{r['std']:.2f}"
            
            paper_hl = paper_results.get(paper_key, {}).get('hl', 'N/A')
            paper_sl = paper_results.get(paper_key, {}).get('sl', 'N/A')
            
            print(f"{method:<12} {ipc:<6} {hl_str:<20} {sl_str:<20}")
            print(f"{'(paper)':<12} {'':<6} {paper_hl:<20} {paper_sl:<20}")
    
    # Save table to file
    with open('results/table1_replicated.txt', 'w') as f:
        f.write("CIFAR-100, ConvNet-D3 - Replicated Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Method':<12} {'IPC':<6} {'HL (ours)':<20} {'HL (paper)':<20} {'SL (ours)':<20} {'SL (paper)':<20}\n")
        f.write("-" * 100 + "\n")
        for method in args.methods:
            for ipc in args.ipcs:
                hl_key = f"{method}_ipc{ipc}_hl"
                sl_key = f"{method}_ipc{ipc}_sl"
                paper_key = f"{method}_ipc{ipc}"
                
                hl_str = sl_str = "N/A"
                if hl_key in results:
                    r = results[hl_key]
                    hl_str = f"{r['mean']:.2f}±{r['std']:.2f}"
                if sl_key in results:
                    r = results[sl_key]
                    sl_str = f"{r['mean']:.2f}±{r['std']:.2f}"
                
                paper_hl = paper_results.get(paper_key, {}).get('hl', 'N/A')
                paper_sl = paper_results.get(paper_key, {}).get('sl', 'N/A')
                
                f.write(f"{method:<12} {ipc:<6} {hl_str:<20} {paper_hl:<20} {sl_str:<20} {paper_sl:<20}\n")


if __name__ == '__main__':
    main()
