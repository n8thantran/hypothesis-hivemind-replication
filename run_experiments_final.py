"""
Final experiment runner for CIFAR-100 small-scale DD evaluation.
Reproduces Table: tab:small_scale_c100 from the paper.

Methods: DM, DC, TM (DD), Random, K-centers (Coresets)
Settings: HL (hard label), SL (soft label)
IPC: 10, 50
Architecture: ConvNet-D3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import sys
from convnet import get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices, random_select

DEVICE = 'cuda'
DSA_STRATEGY = 'color_crop_cutout_flip_scale_rotate'
NUM_CLASSES = 100
NUM_EVAL_RUNS = 3  # Paper uses 5, we use 3 for speed


def evaluate_hl(syn_images, syn_labels, test_images, test_labels, seed=0):
    """Evaluate with Hard Labels: 300 epochs, SGD lr=0.01, StepLR@151, DSA, CE loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    n = len(syn_images)
    batch_size = min(256, n)
    
    for epoch in range(300):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            imgs = syn_images[idx].to(DEVICE)
            labels = syn_labels[idx].to(DEVICE)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), 256):
            imgs = test_images[i:i+256].to(DEVICE)
            labels = test_labels[i:i+256].to(DEVICE)
            out = model(imgs)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def evaluate_sl(syn_images, soft_labels, test_images, test_labels, seed=0, temperature=20.0):
    """Evaluate with Soft Labels: 300 epochs, AdamW lr=1e-3, Cosine, DSA, KL-Div(T=20)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    
    n = len(syn_images)
    batch_size = min(256, n)
    T = temperature
    
    # Precompute soft targets
    soft_targets = F.softmax(soft_labels / T, dim=-1)
    
    for epoch in range(300):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            imgs = syn_images[idx].to(DEVICE)
            targets = soft_targets[idx].to(DEVICE)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            
            optimizer.zero_grad()
            out = model(imgs)
            log_probs = F.log_softmax(out / T, dim=-1)
            loss = F.kl_div(log_probs, targets, reduction='batchmean') * (T * T)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), 256):
            imgs = test_images[i:i+256].to(DEVICE)
            labels = test_labels[i:i+256].to(DEVICE)
            out = model(imgs)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def multi_run_eval(eval_fn, *args, num_runs=NUM_EVAL_RUNS, **kwargs):
    """Run evaluation multiple times and return mean ± std."""
    accs = []
    for run in range(num_runs):
        acc = eval_fn(*args, seed=run, **kwargs)
        accs.append(acc)
        print(f"  Run {run}: {acc:.2f}%")
    mean = np.mean(accs)
    std = np.std(accs)
    print(f"  Mean: {mean:.2f} ± {std:.2f}")
    return mean, std


def get_soft_labels_for_indices(all_soft_labels, indices, train_labels):
    """Get soft labels for selected indices."""
    return all_soft_labels[indices]


def k_centers_select(train_images, train_labels, ipc, seed=0):
    """K-centers coreset selection using features from a trained model."""
    print("  Computing K-centers selection...")
    torch.manual_seed(seed)
    
    # Train a quick model to extract features
    model = get_convnet_d3().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Quick training on full data (50 epochs)
    n = len(train_images)
    for epoch in range(50):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, 256):
            end = min(start + 256, n)
            idx = perm[start:end]
            imgs = train_images[idx].to(DEVICE)
            labels = train_labels[idx].to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
    
    # Extract features
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, n, 256):
            imgs = train_images[i:i+256].to(DEVICE)
            feat = model.embed(imgs)
            features.append(feat.cpu())
    features = torch.cat(features, dim=0)  # [N, feat_dim]
    
    # K-centers per class
    class_indices = get_class_indices(train_labels)
    selected = []
    
    for c in range(NUM_CLASSES):
        c_indices = class_indices[c]
        c_features = features[c_indices]  # [nc, feat_dim]
        
        if len(c_indices) <= ipc:
            selected.extend(c_indices[:ipc])
            continue
        
        # K-centers greedy algorithm (facility location)
        # Start with the point closest to the mean (most representative)
        mean_feat = c_features.mean(dim=0)
        dists_to_mean = torch.cdist(c_features.unsqueeze(0), mean_feat.unsqueeze(0).unsqueeze(0)).squeeze()
        first = dists_to_mean.argmin().item()
        
        chosen = [first]
        # Min distance from each point to any chosen center
        min_dists = torch.cdist(c_features.unsqueeze(0), c_features[chosen].unsqueeze(0)).squeeze(0).min(dim=1).values
        
        for _ in range(ipc - 1):
            # Pick the point with maximum min-distance to chosen set
            next_idx = min_dists.argmax().item()
            chosen.append(next_idx)
            # Update min distances
            new_dists = torch.cdist(c_features.unsqueeze(0), c_features[next_idx:next_idx+1].unsqueeze(0)).squeeze()
            min_dists = torch.min(min_dists, new_dists)
        
        selected.extend([c_indices[i] for i in chosen])
    
    return selected


def distill_dm(train_images, train_labels, ipc, num_iters=20000):
    """Distribution Matching distillation."""
    print(f"  Distilling DM IPC={ipc} for {num_iters} iters...")
    
    class_indices = get_class_indices(train_labels)
    
    # Initialize synthetic data
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        c_idx = class_indices[c]
        perm = torch.randperm(len(c_idx))[:ipc]
        syn_images.append(train_images[c_idx[perm]].clone())
        syn_labels.extend([c] * ipc)
    
    syn_images = torch.cat(syn_images, dim=0).to(DEVICE).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=1.0, momentum=0.5)
    
    for it in range(num_iters):
        # Sample a random model
        model = get_convnet_d3().to(DEVICE)
        model.eval()
        
        loss = torch.tensor(0.0, device=DEVICE)
        
        for c in range(NUM_CLASSES):
            # Real features
            c_idx = class_indices[c]
            real_idx = c_idx[torch.randperm(len(c_idx))[:min(256, len(c_idx))]]
            real_imgs = train_images[real_idx].to(DEVICE)
            real_imgs_aug = DiffAugment(real_imgs, strategy=DSA_STRATEGY)
            
            # Synthetic features
            syn_mask = syn_labels == c
            syn_imgs = syn_images[syn_mask]
            syn_imgs_aug = DiffAugment(syn_imgs, strategy=DSA_STRATEGY)
            
            with torch.no_grad():
                real_feat = model.embed(real_imgs_aug)
            syn_feat = model.embed(syn_imgs_aug)
            
            loss += torch.mean((real_feat.mean(0) - syn_feat.mean(0)) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 2000 == 0:
            print(f"    Iter {it+1}/{num_iters}, loss: {loss.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_dc(train_images, train_labels, ipc, num_iters=5000):
    """Gradient Matching (DC) distillation."""
    print(f"  Distilling DC IPC={ipc} for {num_iters} iters...")
    
    class_indices = get_class_indices(train_labels)
    
    # Initialize synthetic data
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        c_idx = class_indices[c]
        perm = torch.randperm(len(c_idx))[:ipc]
        syn_images.append(train_images[c_idx[perm]].clone())
        syn_labels.extend([c] * ipc)
    
    syn_images = torch.cat(syn_images, dim=0).to(DEVICE).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=1.0, momentum=0.5)
    
    for it in range(num_iters):
        model = get_convnet_d3().to(DEVICE)
        model.train()
        
        loss = torch.tensor(0.0, device=DEVICE)
        
        for c in range(NUM_CLASSES):
            # Real gradient
            c_idx = class_indices[c]
            real_idx = c_idx[torch.randperm(len(c_idx))[:256]]
            real_imgs = train_images[real_idx].to(DEVICE)
            real_labels = train_labels[real_idx].to(DEVICE)
            real_imgs_aug = DiffAugment(real_imgs, strategy=DSA_STRATEGY)
            
            real_out = model(real_imgs_aug)
            real_loss = F.cross_entropy(real_out, real_labels)
            real_grads = torch.autograd.grad(real_loss, model.parameters(), create_graph=False)
            real_grads = [g.detach() for g in real_grads]
            
            # Synthetic gradient
            syn_mask = syn_labels == c
            syn_imgs = syn_images[syn_mask]
            syn_labs = syn_labels[syn_mask]
            syn_imgs_aug = DiffAugment(syn_imgs, strategy=DSA_STRATEGY)
            
            syn_out = model(syn_imgs_aug)
            syn_loss = F.cross_entropy(syn_out, syn_labs)
            syn_grads = torch.autograd.grad(syn_loss, model.parameters(), create_graph=True)
            
            # Match gradients
            for rg, sg in zip(real_grads, syn_grads):
                loss += 1 - F.cosine_similarity(rg.flatten().unsqueeze(0), 
                                                 sg.flatten().unsqueeze(0)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 1000 == 0:
            print(f"    Iter {it+1}/{num_iters}, loss: {loss.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_tm(train_images, train_labels, ipc, num_iters=5000, expert_dir='expert_trajectories'):
    """Trajectory Matching distillation."""
    print(f"  Distilling TM IPC={ipc} for {num_iters} iters...")
    
    # First, generate expert trajectories if not available
    expert_path = f'{expert_dir}/expert_0.pt'
    if not os.path.exists(expert_path):
        print("  Generating expert trajectories...")
        os.makedirs(expert_dir, exist_ok=True)
        generate_expert_trajectories(train_images, train_labels, expert_dir)
    
    # Load expert trajectories
    expert_data = torch.load(expert_path, map_location='cpu', weights_only=False)
    expert_params = expert_data['params']  # List of parameter snapshots
    print(f"  Loaded {len(expert_params)} expert checkpoints")
    
    class_indices = get_class_indices(train_labels)
    
    # Initialize synthetic data
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        c_idx = class_indices[c]
        perm = torch.randperm(len(c_idx))[:ipc]
        syn_images.append(train_images[c_idx[perm]].clone())
        syn_labels.extend([c] * ipc)
    
    syn_images = torch.cat(syn_images, dim=0).to(DEVICE).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=1.0, momentum=0.5)
    
    # TM hyperparameters
    syn_lr = torch.tensor(0.01, device=DEVICE, requires_grad=True)
    lr_optimizer = torch.optim.SGD([syn_lr], lr=1e-5, momentum=0.5)
    
    num_expert_steps = len(expert_params) - 1
    match_steps = 50  # Number of student steps to match
    
    for it in range(num_iters):
        # Sample starting point from expert trajectory
        start_idx = np.random.randint(0, max(1, num_expert_steps - match_steps))
        end_idx = min(start_idx + match_steps, num_expert_steps)
        
        # Get target parameters
        target_params = expert_params[end_idx]
        
        # Initialize student from expert starting point
        student = get_convnet_d3().to(DEVICE)
        start_params = expert_params[start_idx]
        with torch.no_grad():
            for p, sp in zip(student.parameters(), start_params):
                p.copy_(sp.to(DEVICE))
        
        # Train student on synthetic data for match_steps
        student.train()
        for step in range(end_idx - start_idx):
            perm = torch.randperm(len(syn_images))[:256]
            imgs = syn_images[perm]
            labels = syn_labels[perm]
            imgs_aug = DiffAugment(imgs, strategy=DSA_STRATEGY)
            
            out = student(imgs_aug)
            loss_student = F.cross_entropy(out, labels)
            
            grads = torch.autograd.grad(loss_student, student.parameters(), create_graph=True)
            with torch.no_grad():
                for p, g in zip(student.parameters(), grads):
                    p.sub_(syn_lr * g)
        
        # Match student params to expert target params
        loss = torch.tensor(0.0, device=DEVICE)
        for p, tp in zip(student.parameters(), target_params):
            loss += F.mse_loss(p, tp.to(DEVICE), reduction='sum')
        
        # Normalize by target param norm
        target_norm = sum(tp.to(DEVICE).norm()**2 for tp in target_params)
        loss = loss / (target_norm + 1e-8)
        
        optimizer.zero_grad()
        lr_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_optimizer.step()
        
        if (it + 1) % 500 == 0:
            print(f"    Iter {it+1}/{num_iters}, loss: {loss.item():.6f}, syn_lr: {syn_lr.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def generate_expert_trajectories(train_images, train_labels, save_dir, num_experts=1, num_epochs=100):
    """Generate expert trajectories for TM."""
    for exp_id in range(num_experts):
        torch.manual_seed(exp_id)
        model = get_convnet_d3().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        params_history = []
        params_history.append([p.detach().cpu().clone() for p in model.parameters()])
        
        n = len(train_images)
        for epoch in range(num_epochs):
            model.train()
            perm = torch.randperm(n)
            for start in range(0, n, 256):
                end = min(start + 256, n)
                idx = perm[start:end]
                imgs = train_images[idx].to(DEVICE)
                labels = train_labels[idx].to(DEVICE)
                imgs_aug = DiffAugment(imgs, strategy=DSA_STRATEGY)
                
                optimizer.zero_grad()
                out = model(imgs_aug)
                loss = F.cross_entropy(out, labels)
                loss.backward()
                optimizer.step()
            
            params_history.append([p.detach().cpu().clone() for p in model.parameters()])
            
            if (epoch + 1) % 20 == 0:
                model.eval()
                correct = 0
                with torch.no_grad():
                    for i in range(0, len(train_images), 256):
                        imgs = train_images[i:i+256].to(DEVICE)
                        labels = train_labels[i:i+256].to(DEVICE)
                        out = model(imgs)
                        correct += (out.argmax(1) == labels).sum().item()
                print(f"    Expert {exp_id}, epoch {epoch+1}: train acc = {100*correct/n:.1f}%")
                model.train()
        
        torch.save({'params': params_history}, f'{save_dir}/expert_{exp_id}.pt')
        print(f"  Saved expert trajectory {exp_id} with {len(params_history)} checkpoints")


def generate_teacher_soft_labels(train_images, train_labels, test_images, test_labels):
    """Train a teacher and generate soft labels for all training images."""
    print("Training teacher model for soft labels...")
    torch.manual_seed(42)
    
    model = get_convnet_d3().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    n = len(train_images)
    for epoch in range(200):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, 256):
            end = min(start + 256, n)
            idx = perm[start:end]
            imgs = train_images[idx].to(DEVICE)
            labels = train_labels[idx].to(DEVICE)
            # Standard augmentation for teacher
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, len(test_images), 256):
                    imgs = test_images[i:i+256].to(DEVICE)
                    labels = test_labels[i:i+256].to(DEVICE)
                    out = model(imgs)
                    correct += (out.argmax(1) == labels).sum().item()
            print(f"  Teacher epoch {epoch+1}: test acc = {100*correct/len(test_labels):.2f}%")
            model.train()
    
    # Generate soft labels (logits)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, n, 256):
            imgs = train_images[i:i+256].to(DEVICE)
            logits = model(imgs)
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    
    return all_logits


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['random', 'kcenter', 'dm', 'dc', 'tm'])
    parser.add_argument('--ipcs', nargs='+', type=int, default=[10, 50])
    parser.add_argument('--settings', nargs='+', default=['hl', 'sl'])
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--dm_iters', type=int, default=20000)
    parser.add_argument('--dc_iters', type=int, default=5000)
    parser.add_argument('--tm_iters', type=int, default=5000)
    args = parser.parse_args()
    
    global NUM_EVAL_RUNS
    NUM_EVAL_RUNS = args.num_runs
    
    print("Loading CIFAR-100 data...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Generate or load soft labels
    soft_labels_path = 'soft_labels_teacher.pt'
    if os.path.exists(soft_labels_path):
        print("Loading existing soft labels...")
        all_soft_labels = torch.load(soft_labels_path, weights_only=True)
    else:
        all_soft_labels = generate_teacher_soft_labels(train_images, train_labels, test_images, test_labels)
        torch.save(all_soft_labels, soft_labels_path)
        print(f"Saved soft labels to {soft_labels_path}")
    
    results = {}
    
    for method in args.methods:
        for ipc in args.ipcs:
            key = f"{method}_ipc{ipc}"
            print(f"\n{'='*60}")
            print(f"Method: {method}, IPC: {ipc}")
            print(f"{'='*60}")
            
            # Get synthetic/selected data
            if method == 'random':
                indices = random_select(train_labels, ipc, seed=0)
                syn_images = train_images[indices]
                syn_labels = train_labels[indices]
                soft_labels = all_soft_labels[indices]
                
            elif method == 'kcenter':
                indices = k_centers_select(train_images, train_labels, ipc, seed=0)
                syn_images = train_images[indices]
                syn_labels = train_labels[indices]
                soft_labels = all_soft_labels[indices]
                
            elif method == 'dm':
                cache_path = f'distilled_{method}_ipc{ipc}_final.pt'
                if os.path.exists(cache_path):
                    data = torch.load(cache_path, weights_only=True)
                    syn_images = data['images']
                    syn_labels = data['labels']
                else:
                    syn_images, syn_labels = distill_dm(train_images, train_labels, ipc, 
                                                         num_iters=args.dm_iters)
                    torch.save({'images': syn_images, 'labels': syn_labels}, cache_path)
                # For DD methods, generate soft labels from teacher on synthetic images
                soft_labels = generate_soft_labels_for_synthetic(syn_images, all_soft_labels, train_images)
                
            elif method == 'dc':
                cache_path = f'distilled_{method}_ipc{ipc}_final.pt'
                if os.path.exists(cache_path):
                    data = torch.load(cache_path, weights_only=True)
                    syn_images = data['images']
                    syn_labels = data['labels']
                else:
                    syn_images, syn_labels = distill_dc(train_images, train_labels, ipc,
                                                         num_iters=args.dc_iters)
                    torch.save({'images': syn_images, 'labels': syn_labels}, cache_path)
                soft_labels = generate_soft_labels_for_synthetic(syn_images, all_soft_labels, train_images)
                
            elif method == 'tm':
                cache_path = f'distilled_{method}_ipc{ipc}_final.pt'
                if os.path.exists(cache_path):
                    data = torch.load(cache_path, weights_only=True)
                    syn_images = data['images']
                    syn_labels = data['labels']
                else:
                    syn_images, syn_labels = distill_tm(train_images, train_labels, ipc,
                                                         num_iters=args.tm_iters)
                    torch.save({'images': syn_images, 'labels': syn_labels}, cache_path)
                soft_labels = generate_soft_labels_for_synthetic(syn_images, all_soft_labels, train_images)
            
            print(f"Data: {syn_images.shape}, Labels: {syn_labels.shape}")
            
            result = {}
            
            # HL evaluation
            if 'hl' in args.settings:
                print(f"\n--- HL Evaluation ---")
                mean, std = multi_run_eval(evaluate_hl, syn_images, syn_labels, 
                                           test_images, test_labels, num_runs=args.num_runs)
                result['hl_mean'] = mean
                result['hl_std'] = std
            
            # SL evaluation
            if 'sl' in args.settings:
                print(f"\n--- SL Evaluation ---")
                mean, std = multi_run_eval(evaluate_sl, syn_images, soft_labels,
                                           test_images, test_labels, num_runs=args.num_runs)
                result['sl_mean'] = mean
                result['sl_std'] = std
            
            results[key] = result
            
            # Save intermediate results
            os.makedirs('results', exist_ok=True)
            with open('results/results_final_v6.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved intermediate results")
    
    # Print final table
    print_results_table(results, args.methods, args.ipcs)
    
    return results


def generate_soft_labels_for_synthetic(syn_images, all_soft_labels_unused, train_images_unused):
    """Generate soft labels for synthetic images using a teacher model."""
    # For synthetic images, we need to run them through the teacher
    # Load or train teacher
    teacher_path = 'teacher_model.pt'
    if os.path.exists(teacher_path):
        teacher = get_convnet_d3().to(DEVICE)
        teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
    else:
        # Use the soft labels we already have - they're from the teacher
        # For synthetic images, we need to actually run inference
        # Train a quick teacher
        train_images, train_labels, _, _ = get_cifar100_tensors()
        teacher = get_convnet_d3().to(DEVICE)
        optimizer = torch.optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        n = len(train_images)
        for epoch in range(200):
            teacher.train()
            perm = torch.randperm(n)
            for start in range(0, n, 256):
                end = min(start + 256, n)
                idx = perm[start:end]
                imgs = train_images[idx].to(DEVICE)
                labels = train_labels[idx].to(DEVICE)
                optimizer.zero_grad()
                out = teacher(imgs)
                loss = F.cross_entropy(out, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
        torch.save(teacher.state_dict(), teacher_path)
    
    teacher.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(syn_images), 256):
            imgs = syn_images[i:i+256].to(DEVICE)
            out = teacher(imgs)
            logits.append(out.cpu())
    return torch.cat(logits, dim=0)


def print_results_table(results, methods, ipcs):
    """Print results in a nice table format."""
    print(f"\n{'='*80}")
    print(f"RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print(f"{'='*80}")
    
    # Paper target values
    targets = {
        'dm_ipc10': {'hl': 29.23, 'sl': 26.13},
        'dm_ipc50': {'hl': 42.32, 'sl': 43.46},
        'dc_ipc10': {'hl': 28.42, 'sl': 23.54},
        'dc_ipc50': {'hl': 30.56, 'sl': 33.46},
        'tm_ipc10': {'hl': 38.18, 'sl': 37.60},
        'tm_ipc50': {'hl': 46.32, 'sl': 46.26},
        'random_ipc10': {'hl': 18.64, 'sl': 33.43},
        'random_ipc50': {'hl': 34.66, 'sl': 45.39},
        'kcenter_ipc10': {'hl': 25.04, 'sl': 34.70},
        'kcenter_ipc50': {'hl': 38.64, 'sl': 46.24},
    }
    
    header = f"{'Method':<12} {'IPC':>4} | {'HL (ours)':>14} {'HL (paper)':>14} | {'SL (ours)':>14} {'SL (paper)':>14}"
    print(header)
    print("-" * len(header))
    
    for method in methods:
        for ipc in ipcs:
            key = f"{method}_ipc{ipc}"
            if key not in results:
                continue
            r = results[key]
            t = targets.get(key, {})
            
            hl_str = f"{r.get('hl_mean', 0):.2f}±{r.get('hl_std', 0):.2f}" if 'hl_mean' in r else "N/A"
            sl_str = f"{r.get('sl_mean', 0):.2f}±{r.get('sl_std', 0):.2f}" if 'sl_mean' in r else "N/A"
            hl_target = f"{t.get('hl', 0):.2f}" if 'hl' in t else "N/A"
            sl_target = f"{t.get('sl', 0):.2f}" if 'sl' in t else "N/A"
            
            print(f"{method:<12} {ipc:>4} | {hl_str:>14} {hl_target:>14} | {sl_str:>14} {sl_target:>14}")


if __name__ == '__main__':
    main()
