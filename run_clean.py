#!/usr/bin/env python3
"""
Clean implementation of the paper's Table 1 (small_scale_c100):
CIFAR-100, ConvNet-D3, IPC 10 and 50.

Methods: Random, K-centers, DM, DC, TM
Settings: Hard Label (HL) and Soft Label (SL)

Evaluation hyperparameters from paper's Table (tab:stage3_hyper):
- HL: 300 epochs, SGD lr=0.01, StepLR@151 (gamma=0.5), batch=256, DSA, CE loss
- SL: 300 epochs, AdamW lr=1e-3, Cosine scheduler, batch=256, DSA, KL-Div(T=20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import time
import argparse
from collections import defaultdict

from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


###############################################################################
# 1. Evaluation Functions (matching paper exactly)
###############################################################################

def evaluate_hl(images, labels, test_images, test_labels, num_runs=3, epochs=300, seed=0):
    """
    Hard Label evaluation.
    300 epochs, SGD lr=0.01 momentum=0.9, StepLR@151 gamma=0.5, batch=256, DSA, CE loss.
    """
    accs = []
    for run in range(num_runs):
        torch.manual_seed(seed + run)
        np.random.seed(seed + run)
        
        model = get_convnet_d3().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
        
        n = len(images)
        batch_size = min(256, n)
        
        for epoch in range(epochs):
            model.train()
            # Shuffle
            perm = torch.randperm(n)
            total_loss = 0
            num_batches = 0
            
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                x = images[idx].to(DEVICE)
                y = labels[idx].to(DEVICE)
                
                # DSA augmentation
                x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate')
                
                optimizer.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            
            if (epoch + 1) % 100 == 0:
                acc = test_accuracy(model, test_images, test_labels)
                print(f"  [HL Run {run+1}] Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}, Test Acc: {acc:.2f}%")
        
        acc = test_accuracy(model, test_images, test_labels)
        accs.append(acc)
        print(f"  [HL Run {run+1}] Final: {acc:.2f}%")
    
    return np.mean(accs), np.std(accs), accs


def evaluate_sl(images, soft_labels, test_images, test_labels, num_runs=3, epochs=300, seed=0, temperature=20):
    """
    Soft Label evaluation.
    300 epochs, AdamW lr=1e-3, Cosine scheduler, batch=256, DSA, KL-Div(T=20).
    soft_labels should be logits (not probabilities).
    """
    accs = []
    for run in range(num_runs):
        torch.manual_seed(seed + run)
        np.random.seed(seed + run)
        
        model = get_convnet_d3().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        n = len(images)
        batch_size = min(256, n)
        T = temperature
        
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            total_loss = 0
            num_batches = 0
            
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                x = images[idx].to(DEVICE)
                sl = soft_labels[idx].to(DEVICE)
                
                # DSA augmentation
                x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate')
                
                optimizer.zero_grad()
                out = model(x)
                
                # KL-Div loss with temperature
                log_student = F.log_softmax(out / T, dim=1)
                teacher_prob = F.softmax(sl / T, dim=1)
                loss = F.kl_div(log_student, teacher_prob, reduction='batchmean') * (T * T)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            
            if (epoch + 1) % 100 == 0:
                acc = test_accuracy(model, test_images, test_labels)
                print(f"  [SL Run {run+1}] Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}, Test Acc: {acc:.2f}%")
        
        acc = test_accuracy(model, test_images, test_labels)
        accs.append(acc)
        print(f"  [SL Run {run+1}] Final: {acc:.2f}%")
    
    return np.mean(accs), np.std(accs), accs


def test_accuracy(model, test_images, test_labels, batch_size=256):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            x = test_images[i:i+batch_size].to(DEVICE)
            y = test_labels[i:i+batch_size].to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return 100.0 * correct / total


###############################################################################
# 2. K-Centers with pretrained features
###############################################################################

def train_teacher_model(train_images, train_labels, epochs=200, device='cuda'):
    """Train a ConvNet teacher on full CIFAR-100."""
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n = len(train_images)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            x = train_images[idx].to(device)
            y = train_labels[idx].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            acc = 0
            model.eval()
            with torch.no_grad():
                correct = 0
                for i in range(0, n, 256):
                    x = train_images[i:i+256].to(device)
                    y = train_labels[i:i+256].to(device)
                    out = model(x)
                    correct += (out.argmax(1) == y).sum().item()
            print(f"  Teacher epoch {epoch+1}/{epochs}, Train acc: {100*correct/n:.2f}%")
    
    return model


def extract_features(images, model, device='cuda'):
    """Extract features from pretrained model."""
    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    return torch.cat(all_features, dim=0).numpy()


def k_centers_select(features, labels, ipc, num_classes=100, seed=0):
    """
    K-Centers greedy selection using pretrained features.
    For each class, select IPC points using farthest-first traversal.
    """
    np.random.seed(seed)
    class_indices = get_class_indices(labels, num_classes)
    selected = []
    
    for c in range(num_classes):
        indices = np.array(class_indices[c])
        feats = features[indices]
        
        # Normalize features for better distance computation
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        feats_norm = feats / norms
        
        # Greedy farthest-first traversal
        chosen = []
        first = np.random.randint(len(indices))
        chosen.append(first)
        
        # Initialize distances
        min_dists = np.full(len(indices), np.inf)
        
        for _ in range(ipc - 1):
            last = chosen[-1]
            dists = np.sum((feats_norm - feats_norm[last:last+1]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[chosen] = -1  # Exclude already chosen
            next_idx = np.argmax(min_dists)
            chosen.append(next_idx)
        
        selected.extend([int(indices[c_idx]) for c_idx in chosen])
    
    return sorted(selected)


###############################################################################
# 3. Soft label generation
###############################################################################

def generate_teacher_soft_labels(train_images, train_labels, device='cuda', 
                                  teacher_path=None, num_teachers=1, epochs=200):
    """Generate soft labels from teacher model(s)."""
    if teacher_path and os.path.exists(teacher_path):
        print(f"Loading existing soft labels from {teacher_path}")
        return torch.load(teacher_path, map_location='cpu')
    
    all_logits = []
    for t in range(num_teachers):
        print(f"Training teacher {t+1}/{num_teachers}...")
        model = train_teacher_model(train_images, train_labels, epochs=epochs, device=device)
        
        model.eval()
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(train_images), 256):
                batch = train_images[i:i+256].to(device)
                logits = model(batch)
                logits_list.append(logits.cpu())
        all_logits.append(torch.cat(logits_list, dim=0))
    
    avg_logits = torch.stack(all_logits).mean(dim=0)
    
    if teacher_path:
        torch.save(avg_logits, teacher_path)
        print(f"Saved soft labels to {teacher_path}")
    
    return avg_logits


###############################################################################
# 4. Dataset Distillation Methods
###############################################################################

def distill_dm(train_images, train_labels, ipc, num_classes=100, 
               iterations=20000, lr_img=1.0, device='cuda'):
    """Distribution Matching (DM) distillation."""
    print(f"DM distillation: IPC={ipc}, iterations={iterations}")
    
    class_indices = get_class_indices(train_labels, num_classes)
    
    # Initialize synthetic data
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=device, requires_grad=True)
    syn_labels = torch.arange(num_classes, device=device).repeat_interleave(ipc)
    
    # Initialize from real data
    with torch.no_grad():
        for c in range(num_classes):
            indices = class_indices[c]
            chosen = np.random.choice(indices, size=ipc, replace=False)
            syn_images[c*ipc:(c+1)*ipc] = train_images[chosen].to(device)
    
    syn_images = syn_images.detach().requires_grad_(True)
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    for it in range(iterations):
        model = get_convnet_d3().to(device)
        model.train()
        
        loss = torch.tensor(0.0, device=device)
        
        for c in range(num_classes):
            # Real data for this class
            real_idx = np.random.choice(class_indices[c], size=min(256, len(class_indices[c])), replace=False)
            real_batch = train_images[real_idx].to(device)
            
            # Synthetic data for this class
            syn_batch = syn_images[c*ipc:(c+1)*ipc]
            
            # Match features
            with torch.no_grad():
                real_feat = model.embed(real_batch)
            syn_feat = model.embed(syn_batch)
            
            loss += torch.mean((real_feat.mean(0) - syn_feat.mean(0)) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 1000 == 0:
            print(f"  DM iter {it+1}/{iterations}, loss: {loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_dc(train_images, train_labels, ipc, num_classes=100,
               outer_iterations=1000, inner_steps=1, lr_img=1.0, lr_net=0.01, device='cuda'):
    """Dataset Condensation (DC) via gradient matching."""
    print(f"DC distillation: IPC={ipc}, outer_iters={outer_iterations}")
    
    class_indices = get_class_indices(train_labels, num_classes)
    
    # Initialize synthetic data from real
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=device)
    syn_labels = torch.arange(num_classes, device=device).repeat_interleave(ipc)
    
    with torch.no_grad():
        for c in range(num_classes):
            indices = class_indices[c]
            chosen = np.random.choice(indices, size=ipc, replace=False)
            syn_images[c*ipc:(c+1)*ipc] = train_images[chosen].to(device)
    
    syn_images = syn_images.detach().requires_grad_(True)
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    for it in range(outer_iterations):
        model = get_convnet_d3().to(device)
        model.train()
        
        loss = torch.tensor(0.0, device=device)
        
        for c in range(num_classes):
            # Real gradient
            real_idx = np.random.choice(class_indices[c], size=min(256, len(class_indices[c])), replace=False)
            real_batch = train_images[real_idx].to(device)
            real_labels_batch = train_labels[real_idx].to(device)
            
            out_real = model(real_batch)
            loss_real = F.cross_entropy(out_real, real_labels_batch)
            grad_real = torch.autograd.grad(loss_real, model.parameters(), create_graph=False)
            
            # Synthetic gradient
            syn_batch = syn_images[c*ipc:(c+1)*ipc]
            syn_labels_batch = syn_labels[c*ipc:(c+1)*ipc]
            
            out_syn = model(syn_batch)
            loss_syn = F.cross_entropy(out_syn, syn_labels_batch)
            grad_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)
            
            # Match gradients
            for g_real, g_syn in zip(grad_real, grad_syn):
                # Cosine distance
                g_real_flat = g_real.reshape(-1)
                g_syn_flat = g_syn.reshape(-1)
                loss += 1 - F.cosine_similarity(g_real_flat.unsqueeze(0), g_syn_flat.unsqueeze(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 200 == 0:
            print(f"  DC iter {it+1}/{outer_iterations}, loss: {loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_tm(train_images, train_labels, ipc, num_classes=100,
               expert_epochs=3, syn_steps=30, expert_dir='expert_trajectories',
               iterations=1000, lr_img=1000.0, device='cuda'):
    """Trajectory Matching (TM) distillation."""
    print(f"TM distillation: IPC={ipc}, iterations={iterations}")
    
    class_indices = get_class_indices(train_labels, num_classes)
    
    # First, generate expert trajectories if not available
    expert_path = os.path.join(expert_dir, 'expert_0.pt')
    if not os.path.exists(expert_path):
        os.makedirs(expert_dir, exist_ok=True)
        print("  Generating expert trajectories...")
        trajectories = []
        for e in range(10):  # 10 expert trajectories
            model = get_convnet_d3().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            
            trajectory = [model.state_dict()]
            n = len(train_images)
            
            for epoch in range(expert_epochs):
                perm = torch.randperm(n)
                for i in range(0, n, 256):
                    idx = perm[i:i+256]
                    x = train_images[idx].to(device)
                    y = train_labels[idx].to(device)
                    x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate')
                    optimizer.zero_grad()
                    out = model(x)
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()
                trajectory.append(model.state_dict())
            
            trajectories.append(trajectory)
            print(f"    Expert {e+1}/10 done")
        
        torch.save(trajectories, expert_path)
    else:
        print(f"  Loading expert trajectories from {expert_path}")
        trajectories = torch.load(expert_path, map_location='cpu')
    
    # Initialize synthetic data
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=device)
    syn_labels = torch.arange(num_classes, device=device).repeat_interleave(ipc)
    
    with torch.no_grad():
        for c in range(num_classes):
            indices = class_indices[c]
            chosen = np.random.choice(indices, size=ipc, replace=False)
            syn_images[c*ipc:(c+1)*ipc] = train_images[chosen].to(device)
    
    syn_images = syn_images.detach().requires_grad_(True)
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    for it in range(iterations):
        # Pick random expert trajectory and starting point
        traj_idx = np.random.randint(len(trajectories))
        traj = trajectories[traj_idx]
        start_epoch = np.random.randint(0, len(traj) - 1)
        
        # Load starting parameters
        start_params = traj[start_epoch]
        target_params = traj[min(start_epoch + 1, len(traj) - 1)]
        
        # Create student model from start params
        student = get_convnet_d3().to(device)
        student.load_state_dict({k: v.to(device) for k, v in start_params.items()})
        student.train()
        
        student_optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
        
        # Train student on synthetic data for syn_steps
        for step in range(syn_steps):
            student_optimizer.zero_grad()
            x = DiffAugment(syn_images, strategy='color_crop_cutout_flip_scale_rotate')
            out = student(x)
            loss = F.cross_entropy(out, syn_labels)
            loss.backward()
            student_optimizer.step()
        
        # Match student params to target params
        param_loss = torch.tensor(0.0, device=device)
        for (name, p_student), (_, p_target) in zip(student.named_parameters(), 
                                                       [(k, v.to(device)) for k, v in target_params.items() if 'num_batches_tracked' not in k]):
            if p_student.requires_grad:
                param_loss += F.mse_loss(p_student, p_target, reduction='sum')
        
        optimizer.zero_grad()
        param_loss.backward()
        optimizer.step()
        
        if (it + 1) % 200 == 0:
            print(f"  TM iter {it+1}/{iterations}, param_loss: {param_loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


###############################################################################
# 5. Main experiment runner
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='all', 
                        choices=['random', 'kcenter', 'dm', 'dc', 'tm', 'all', 'coresets', 'dd'])
    parser.add_argument('--ipc', type=int, default=10, choices=[10, 50])
    parser.add_argument('--mode', type=str, default='both', choices=['hl', 'sl', 'both'])
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--dm_iters', type=int, default=20000)
    parser.add_argument('--dc_iters', type=int, default=1000)
    parser.add_argument('--tm_iters', type=int, default=1000)
    parser.add_argument('--skip_distill', action='store_true', help='Skip distillation, use existing .pt files')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Running: method={args.method}, IPC={args.ipc}, mode={args.mode}")
    print("=" * 60)
    
    # Load data
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Load or generate soft labels
    soft_labels_path = '/workspace/soft_labels_clean.pt'
    if os.path.exists(soft_labels_path):
        print(f"Loading soft labels from {soft_labels_path}")
        all_soft_labels = torch.load(soft_labels_path, map_location='cpu')
    else:
        print("Generating soft labels from teacher...")
        all_soft_labels = generate_teacher_soft_labels(
            train_images, train_labels, device=DEVICE,
            teacher_path=soft_labels_path, num_teachers=1, epochs=200
        )
    
    # Check teacher quality
    teacher_preds = all_soft_labels.argmax(dim=1)
    teacher_acc = (teacher_preds == train_labels).float().mean().item()
    print(f"Teacher train accuracy: {teacher_acc*100:.2f}%")
    
    results = {}
    
    # Determine which methods to run
    if args.method == 'all':
        methods = ['random', 'kcenter', 'dm', 'dc', 'tm']
    elif args.method == 'coresets':
        methods = ['random', 'kcenter']
    elif args.method == 'dd':
        methods = ['dm', 'dc', 'tm']
    else:
        methods = [args.method]
    
    # For K-centers, we need a pretrained model
    teacher_model = None
    features = None
    if 'kcenter' in methods:
        teacher_model_path = '/workspace/teacher_for_kcenter.pt'
        if os.path.exists(teacher_model_path):
            print("Loading pretrained teacher for K-centers...")
            teacher_model = get_convnet_d3().to(DEVICE)
            teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=DEVICE))
        else:
            print("Training teacher for K-centers feature extraction...")
            teacher_model = train_teacher_model(train_images, train_labels, epochs=200, device=DEVICE)
            torch.save(teacher_model.state_dict(), teacher_model_path)
        
        print("Extracting features for K-centers...")
        features = extract_features(train_images, teacher_model, device=DEVICE)
        print(f"Features shape: {features.shape}")
    
    for method in methods:
        key = f"{method}_ipc{args.ipc}"
        print(f"\n{'='*60}")
        print(f"Method: {method}, IPC: {args.ipc}")
        print(f"{'='*60}")
        
        if method == 'random':
            indices = []
            np.random.seed(42)
            class_idx = get_class_indices(train_labels, 100)
            for c in range(100):
                chosen = np.random.choice(class_idx[c], size=args.ipc, replace=False)
                indices.extend(chosen.tolist())
            indices = sorted(indices)
            
            images = train_images[indices]
            labels = train_labels[indices]
            soft_labs = all_soft_labels[indices]
            
        elif method == 'kcenter':
            indices = k_centers_select(features, train_labels, args.ipc, seed=42)
            images = train_images[indices]
            labels = train_labels[indices]
            soft_labs = all_soft_labels[indices]
            
        elif method in ['dm', 'dc', 'tm']:
            # Check for existing distilled data
            distilled_path = f'/workspace/distilled_{method}_ipc{args.ipc}.pt'
            
            if os.path.exists(distilled_path) and args.skip_distill:
                print(f"Loading existing distilled data from {distilled_path}")
                data = torch.load(distilled_path, map_location='cpu')
                if isinstance(data, dict):
                    images = data['images']
                    labels = data['labels']
                else:
                    images = data
                    labels = torch.arange(100).repeat_interleave(args.ipc)
            else:
                if method == 'dm':
                    images, labels = distill_dm(train_images, train_labels, args.ipc,
                                                iterations=args.dm_iters, device=DEVICE)
                elif method == 'dc':
                    images, labels = distill_dc(train_images, train_labels, args.ipc,
                                                outer_iterations=args.dc_iters, device=DEVICE)
                elif method == 'tm':
                    images, labels = distill_tm(train_images, train_labels, args.ipc,
                                                iterations=args.tm_iters, device=DEVICE)
                
                torch.save({'images': images, 'labels': labels}, distilled_path)
            
            # Generate soft labels for distilled data using teacher
            if teacher_model is None:
                teacher_model_path = '/workspace/teacher_for_kcenter.pt'
                if os.path.exists(teacher_model_path):
                    teacher_model = get_convnet_d3().to(DEVICE)
                    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=DEVICE))
                else:
                    teacher_model = train_teacher_model(train_images, train_labels, epochs=200, device=DEVICE)
                    torch.save(teacher_model.state_dict(), teacher_model_path)
            
            teacher_model.eval()
            with torch.no_grad():
                soft_labs_list = []
                for i in range(0, len(images), 256):
                    batch = images[i:i+256].to(DEVICE)
                    logits = teacher_model(batch)
                    soft_labs_list.append(logits.cpu())
                soft_labs = torch.cat(soft_labs_list, dim=0)
        
        result = {'method': method, 'ipc': args.ipc}
        
        # HL evaluation
        if args.mode in ['hl', 'both']:
            print(f"\n--- HL Evaluation ---")
            hl_mean, hl_std, hl_accs = evaluate_hl(
                images, labels, test_images, test_labels, 
                num_runs=args.num_runs, epochs=300
            )
            result['hl_mean'] = hl_mean
            result['hl_std'] = hl_std
            result['hl_accs'] = hl_accs
            print(f"HL: {hl_mean:.2f} ± {hl_std:.2f}")
        
        # SL evaluation
        if args.mode in ['sl', 'both']:
            print(f"\n--- SL Evaluation ---")
            sl_mean, sl_std, sl_accs = evaluate_sl(
                images, soft_labs, test_images, test_labels,
                num_runs=args.num_runs, epochs=300
            )
            result['sl_mean'] = sl_mean
            result['sl_std'] = sl_std
            result['sl_accs'] = sl_accs
            print(f"SL: {sl_mean:.2f} ± {sl_std:.2f}")
        
        results[key] = result
        
        # Save intermediate results
        os.makedirs('/workspace/results', exist_ok=True)
        with open('/workspace/results/results_clean.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Method':<12} {'IPC':<6} {'HL':<20} {'SL':<20}")
    print("-" * 60)
    for key, r in results.items():
        hl_str = f"{r.get('hl_mean', 0):.2f} ± {r.get('hl_std', 0):.2f}" if 'hl_mean' in r else "N/A"
        sl_str = f"{r.get('sl_mean', 0):.2f} ± {r.get('sl_std', 0):.2f}" if 'sl_mean' in r else "N/A"
        print(f"{r['method']:<12} {r['ipc']:<6} {hl_str:<20} {sl_str:<20}")
    
    return results


if __name__ == '__main__':
    main()
