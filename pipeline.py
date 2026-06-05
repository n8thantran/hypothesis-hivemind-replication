"""
Complete pipeline for replicating Table 3 (small_scale_c100) from the paper.
"Rethinking Dataset Distillation: Hard Truths About Soft Labels"

Target: CIFAR-100, ConvNet-D3, IPC 10 and 50
Methods: DM, DC, TM, Random, K-centers
Settings: HL (hard label) and SL (fixed soft label)

Hyperparameters from Table (tab:stage3_hyper):
- HL: 300 epochs, SGD lr=0.01, StepLR@151 (halve), batch=256, DSA, CE loss
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
from collections import defaultdict

from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices, CIFAR100_MEAN, CIFAR100_STD


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


###############################################################################
# 1. TEACHER MODEL TRAINING (for soft labels)
###############################################################################

def train_teacher(train_images, train_labels, test_images, test_labels,
                  epochs=300, lr=0.01, device=DEVICE, seed=42):
    """Train a ConvNet-D3 teacher on full CIFAR-100."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # Apply DSA augmentation during teacher training too
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            acc = evaluate_model(model, test_images, test_labels, device)
            print(f"  Teacher epoch {epoch+1}/{epochs}: test acc = {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_state)
    print(f"  Best teacher accuracy: {best_acc:.2f}%")
    return model


def generate_soft_labels_from_teacher(teacher, train_images, device=DEVICE):
    """Generate soft label logits from teacher model."""
    teacher.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(device)
            logits = teacher(batch)
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


###############################################################################
# 2. EVALUATION FUNCTIONS
###############################################################################

def evaluate_model(model, test_images, test_labels, device=DEVICE):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), 256):
            imgs = test_images[i:i+256].to(device)
            labels = test_labels[i:i+256].to(device)
            out = model(imgs)
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_and_eval_hl(syn_images, syn_labels, test_images, test_labels,
                      epochs=300, lr=0.01, batch_size=256, device=DEVICE, seed=0):
    """
    Train student with Hard Labels (HL setting).
    SGD, lr=0.01, StepLR@151 (halve), 300 epochs, DSA, CE loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    n = len(syn_images)
    
    for epoch in range(epochs):
        model.train()
        # Shuffle
        perm = torch.randperm(n)
        
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            imgs = syn_images[idx].to(device)
            labels = syn_labels[idx].to(device)
            
            # DSA augmentation
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    acc = evaluate_model(model, test_images, test_labels, device)
    return acc


def train_and_eval_sl(syn_images, syn_soft_logits, test_images, test_labels,
                      epochs=300, lr=1e-3, batch_size=256, T=20, device=DEVICE, seed=0):
    """
    Train student with Soft Labels (SL setting).
    AdamW, lr=1e-3, Cosine scheduler, 300 epochs, DSA, KL-Div(T=20).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n = len(syn_images)
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            imgs = syn_images[idx].to(device)
            soft_targets = syn_soft_logits[idx].to(device)
            
            # DSA augmentation
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(imgs)
            
            # KL-Div loss with temperature T=20
            log_probs = F.log_softmax(out / T, dim=1)
            targets = F.softmax(soft_targets / T, dim=1)
            loss = F.kl_div(log_probs, targets, reduction='batchmean') * (T * T)
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    acc = evaluate_model(model, test_images, test_labels, device)
    return acc


###############################################################################
# 3. CORESET METHODS
###############################################################################

def select_random(labels, ipc, num_classes=100, seed=0):
    """Random coreset: IPC samples per class."""
    np.random.seed(seed)
    class_indices = get_class_indices(labels, num_classes)
    selected = []
    for c in range(num_classes):
        indices = class_indices[c]
        chosen = np.random.choice(indices, size=ipc, replace=False)
        selected.extend(chosen.tolist())
    return sorted(selected)


def select_k_centers(images, labels, ipc, num_classes=100, seed=0, 
                     model=None, device=DEVICE):
    """
    K-Centers coreset selection using DeepCore-style approach.
    Uses pretrained model features + greedy farthest-first traversal.
    """
    np.random.seed(seed)
    class_indices = get_class_indices(labels, num_classes)
    
    # Extract features using model
    if model is not None:
        features = _extract_features_with_model(images, model, device)
    else:
        # Use pixel features as fallback
        features = images.reshape(len(images), -1).numpy()
    
    selected = []
    for c in range(num_classes):
        indices = np.array(class_indices[c])
        feats = features[indices]
        
        # Normalize features for better distance computation
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats = feats / norms
        
        # Greedy farthest-first traversal (K-center)
        # Start from the point closest to the class centroid
        centroid = feats.mean(axis=0, keepdims=True)
        dists_to_centroid = np.sum((feats - centroid) ** 2, axis=1)
        first = np.argmin(dists_to_centroid)
        
        chosen = [first]
        min_dists = np.sum((feats - feats[first:first+1]) ** 2, axis=1)
        
        for _ in range(ipc - 1):
            next_idx = np.argmax(min_dists)
            chosen.append(next_idx)
            new_dists = np.sum((feats - feats[next_idx:next_idx+1]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, new_dists)
        
        selected.extend([int(indices[c_idx]) for c_idx in chosen])
    
    return sorted(selected)


def _extract_features_with_model(images, model, device):
    """Extract features using a model's embed function."""
    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            feat = model.embed(batch)
            all_features.append(feat.cpu().numpy())
    return np.concatenate(all_features, axis=0)


###############################################################################
# 4. DATASET DISTILLATION METHODS
###############################################################################

def distill_dm(train_images, train_labels, ipc, num_classes=100,
               iterations=20000, lr_img=1.0, batch_real=256, device=DEVICE, seed=0):
    """
    Distribution Matching (DM) distillation.
    Matches mean features between real and synthetic data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels, num_classes)
    
    # Initialize synthetic data
    syn_images = []
    syn_labels = []
    for c in range(num_classes):
        indices = class_indices[c]
        chosen = np.random.choice(indices, size=ipc, replace=False)
        syn_images.append(train_images[chosen].clone())
        syn_labels.extend([c] * ipc)
    
    syn_images = torch.cat(syn_images, dim=0).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    print(f"  DM distillation: {iterations} iterations, IPC={ipc}")
    
    for it in range(iterations):
        # Sample a new random model each iteration
        model = get_convnet_d3().to(device)
        model.train()
        
        # For each class, match feature distributions
        loss = torch.tensor(0.0, device=device)
        
        for c in range(num_classes):
            # Real data for this class
            real_indices = class_indices[c]
            real_batch_idx = np.random.choice(real_indices, size=min(batch_real, len(real_indices)), replace=False)
            real_batch = train_images[real_batch_idx].to(device)
            
            # Synthetic data for this class
            syn_idx = torch.where(syn_labels == c)[0]
            syn_batch = syn_images[syn_idx]
            
            # Apply DSA with same seed
            seed_aug = int(torch.randint(0, 100000, (1,)).item())
            real_aug = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate', seed=seed_aug)
            syn_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate', seed=seed_aug)
            
            # Get features
            real_feat = model.embed(real_aug)
            syn_feat = model.embed(syn_aug)
            
            # Match mean features (MMD with mean embedding)
            loss += torch.sum((real_feat.mean(0) - syn_feat.mean(0)) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 2000 == 0:
            print(f"    DM iter {it+1}/{iterations}, loss={loss.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_dc(train_images, train_labels, ipc, num_classes=100,
               outer_loops=10, inner_loops=50, lr_img=1.0, lr_net=0.01,
               batch_real=256, device=DEVICE, seed=0):
    """
    Dataset Condensation (DC) via gradient matching.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels, num_classes)
    
    # Initialize synthetic data from real
    syn_images = []
    syn_labels = []
    for c in range(num_classes):
        indices = class_indices[c]
        chosen = np.random.choice(indices, size=ipc, replace=False)
        syn_images.append(train_images[chosen].clone())
        syn_labels.extend([c] * ipc)
    
    syn_images = torch.cat(syn_images, dim=0).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    total_iters = outer_loops * inner_loops
    print(f"  DC distillation: {outer_loops} outer x {inner_loops} inner = {total_iters} iters, IPC={ipc}")
    
    for ol in range(outer_loops):
        # New random model each outer loop
        model = get_convnet_d3().to(device)
        model.train()
        net_optimizer = torch.optim.SGD(model.parameters(), lr=lr_net, momentum=0.9)
        
        for il in range(inner_loops):
            # Gradient matching loss
            loss = torch.tensor(0.0, device=device)
            
            for c in range(num_classes):
                # Real gradients
                real_indices = class_indices[c]
                real_batch_idx = np.random.choice(real_indices, size=min(batch_real, len(real_indices)), replace=False)
                real_batch = train_images[real_batch_idx].to(device)
                real_labels_batch = train_labels[real_batch_idx].to(device)
                
                seed_aug = int(torch.randint(0, 100000, (1,)).item())
                real_aug = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate', seed=seed_aug)
                
                real_out = model(real_aug)
                real_loss = F.cross_entropy(real_out, real_labels_batch)
                real_grads = torch.autograd.grad(real_loss, model.parameters(), create_graph=False)
                
                # Synthetic gradients
                syn_idx = torch.where(syn_labels == c)[0]
                syn_batch = syn_images[syn_idx]
                syn_labels_batch = syn_labels[syn_idx]
                
                syn_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate', seed=seed_aug)
                
                syn_out = model(syn_aug)
                syn_loss = F.cross_entropy(syn_out, syn_labels_batch)
                syn_grads = torch.autograd.grad(syn_loss, model.parameters(), create_graph=True)
                
                # Match gradients (cosine distance)
                for rg, sg in zip(real_grads, syn_grads):
                    rg = rg.detach()
                    cos_sim = F.cosine_similarity(rg.flatten().unsqueeze(0), sg.flatten().unsqueeze(0))
                    loss += (1 - cos_sim)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update model with synthetic data
            model.train()
            syn_aug_for_update = DiffAugment(syn_images.detach(), strategy='color_crop_cutout_flip_scale_rotate')
            out = model(syn_aug_for_update)
            net_loss = F.cross_entropy(out, syn_labels)
            net_optimizer.zero_grad()
            net_loss.backward()
            net_optimizer.step()
        
        if (ol + 1) % 2 == 0:
            print(f"    DC outer {ol+1}/{outer_loops}, loss={loss.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_tm(train_images, train_labels, ipc, num_classes=100,
               expert_epochs=50, syn_steps=30, expert_lr=0.01,
               lr_img=1000, lr_lr=1e-5, num_experts=100,
               iterations=5000, device=DEVICE, seed=0):
    """
    Trajectory Matching (TM) distillation.
    Matches training trajectories of models trained on real vs synthetic data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels, num_classes)
    
    # Step 1: Generate expert trajectories
    print(f"  TM: Generating {num_experts} expert trajectories...")
    expert_trajectories = []
    
    for exp_idx in range(num_experts):
        model = get_convnet_d3().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=expert_lr, momentum=0.9)
        
        trajectory = [copy.deepcopy(model.state_dict())]
        
        for epoch in range(expert_epochs):
            model.train()
            # Random batch from full dataset
            batch_idx = np.random.choice(len(train_images), size=256, replace=False)
            imgs = train_images[batch_idx].to(device)
            labels = train_labels[batch_idx].to(device)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
            
            trajectory.append(copy.deepcopy(model.state_dict()))
        
        expert_trajectories.append(trajectory)
        
        if (exp_idx + 1) % 20 == 0:
            print(f"    Expert {exp_idx+1}/{num_experts}")
    
    # Step 2: Initialize synthetic data
    syn_images = []
    syn_labels = []
    for c in range(num_classes):
        indices = class_indices[c]
        chosen = np.random.choice(indices, size=ipc, replace=False)
        syn_images.append(train_images[chosen].clone())
        syn_labels.extend([c] * ipc)
    
    syn_images = torch.cat(syn_images, dim=0).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    # Learnable learning rate
    syn_lr = torch.tensor(expert_lr, device=device).requires_grad_(True)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=lr_lr, momentum=0.5)
    
    print(f"  TM: Matching trajectories for {iterations} iterations...")
    
    for it in range(iterations):
        # Pick random expert and starting point
        exp_idx = np.random.randint(num_experts)
        trajectory = expert_trajectories[exp_idx]
        max_start = max(0, len(trajectory) - syn_steps - 1)
        start_epoch = np.random.randint(0, max_start + 1)
        
        # Load starting parameters
        starting_params = trajectory[start_epoch]
        target_params = trajectory[min(start_epoch + syn_steps, len(trajectory) - 1)]
        
        # Create student model from starting point
        student = get_convnet_d3().to(device)
        student.load_state_dict(starting_params)
        student.train()
        
        # Train student on synthetic data for syn_steps
        for step in range(syn_steps):
            # Forward pass on synthetic data
            seed_aug = int(torch.randint(0, 100000, (1,)).item())
            syn_aug = DiffAugment(syn_images, strategy='color_crop_cutout_flip_scale_rotate', seed=seed_aug)
            
            out = student(syn_aug)
            loss = F.cross_entropy(out, syn_labels)
            
            # Manual SGD step (to keep computation graph)
            grads = torch.autograd.grad(loss, student.parameters(), create_graph=True)
            with torch.no_grad():
                for param, grad in zip(student.parameters(), grads):
                    param.sub_(syn_lr * grad)
        
        # Match: student params should be close to target params
        match_loss = torch.tensor(0.0, device=device)
        target_flat = []
        student_flat = []
        
        for (name, param), (_, target) in zip(student.named_parameters(), 
                                                [(n, p) for n, p in zip(target_params.keys(), target_params.values())]):
            target_p = target_params[name].to(device)
            start_p = starting_params[name].to(device)
            
            direction = target_p - start_p
            student_direction = param - start_p
            
            # Normalized parameter matching
            norm = direction.norm() + 1e-6
            match_loss += ((student_direction - direction) / norm).pow(2).sum()
        
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        match_loss.backward()
        optimizer_img.step()
        optimizer_lr.step()
        
        if (it + 1) % 500 == 0:
            print(f"    TM iter {it+1}/{iterations}, loss={match_loss.item():.4f}, lr={syn_lr.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


###############################################################################
# 5. MAIN EXPERIMENT RUNNER
###############################################################################

def run_experiment(method, ipc, setting, train_images, train_labels, 
                   test_images, test_labels, soft_logits=None,
                   teacher_model=None, num_runs=3, device=DEVICE,
                   distilled_data=None):
    """
    Run a single experiment configuration.
    Returns mean accuracy and std.
    """
    accs = []
    
    for run in range(num_runs):
        seed = run * 42 + 1
        
        if distilled_data is not None:
            # Use pre-distilled data
            syn_images, syn_labels = distilled_data
        elif method == 'random':
            indices = select_random(train_labels, ipc, seed=seed)
            syn_images = train_images[indices]
            syn_labels = train_labels[indices]
        elif method == 'k_centers':
            indices = select_k_centers(train_images, train_labels, ipc, seed=seed,
                                       model=teacher_model, device=device)
            syn_images = train_images[indices]
            syn_labels = train_labels[indices]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if setting == 'hl':
            acc = train_and_eval_hl(syn_images, syn_labels, test_images, test_labels,
                                    device=device, seed=seed)
        elif setting == 'sl':
            # Get soft labels for the selected images
            if distilled_data is not None:
                # For DD methods, generate soft labels from teacher
                syn_soft = generate_soft_labels_from_teacher(teacher_model, syn_images.to(device), device)
            else:
                syn_soft = soft_logits[indices] if soft_logits is not None else None
                if syn_soft is None:
                    syn_soft = generate_soft_labels_from_teacher(teacher_model, syn_images.to(device), device)
            
            acc = train_and_eval_sl(syn_images, syn_soft, test_images, test_labels,
                                    device=device, seed=seed)
        
        accs.append(acc)
        print(f"  {method} IPC={ipc} {setting} run {run+1}: {acc:.2f}%")
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    return mean_acc, std_acc, accs


def main():
    """Run all experiments."""
    print("=" * 60)
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Step 1: Train teacher and generate soft labels
    print("\n" + "=" * 60)
    print("Training teacher model...")
    teacher_path = '/workspace/teacher_model.pt'
    soft_labels_path = '/workspace/teacher_soft_logits.pt'
    
    if os.path.exists(teacher_path) and os.path.exists(soft_labels_path):
        print("Loading cached teacher and soft labels...")
        teacher = get_convnet_d3().to(DEVICE)
        teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
        soft_logits = torch.load(soft_labels_path, map_location='cpu')
        acc = evaluate_model(teacher, test_images, test_labels, DEVICE)
        print(f"  Loaded teacher accuracy: {acc:.2f}%")
    else:
        teacher = train_teacher(train_images, train_labels, test_images, test_labels,
                               epochs=300, device=DEVICE)
        torch.save(teacher.state_dict(), teacher_path)
        soft_logits = generate_soft_labels_from_teacher(teacher, train_images, DEVICE)
        torch.save(soft_logits, soft_labels_path)
    
    # Step 2: Run coreset experiments
    results = {}
    
    for method in ['random', 'k_centers']:
        for ipc in [10, 50]:
            for setting in ['hl', 'sl']:
                key = f"{method}_ipc{ipc}_{setting}"
                print(f"\n{'='*60}")
                print(f"Running: {key}")
                
                mean_acc, std_acc, accs = run_experiment(
                    method, ipc, setting, train_images, train_labels,
                    test_images, test_labels, soft_logits=soft_logits,
                    teacher_model=teacher, num_runs=3, device=DEVICE
                )
                
                results[key] = {
                    'method': method, 'ipc': ipc, 'setting': setting,
                    'mean': round(mean_acc, 2), 'std': round(std_acc, 2),
                    'accs': [round(a, 2) for a in accs]
                }
                print(f"  Result: {mean_acc:.2f} ± {std_acc:.2f}")
                
                # Save intermediate results
                os.makedirs('/workspace/results', exist_ok=True)
                with open('/workspace/results/results_pipeline.json', 'w') as f:
                    json.dump(results, f, indent=2)
    
    # Step 3: Run DD experiments
    for dd_method in ['dm', 'dc', 'tm']:
        for ipc in [10, 50]:
            # Check for cached distilled data
            cache_path = f'/workspace/distilled_{dd_method}_ipc{ipc}_v2.pt'
            
            if os.path.exists(cache_path):
                print(f"\nLoading cached {dd_method} IPC={ipc}...")
                data = torch.load(cache_path, map_location='cpu')
                syn_images = data['images']
                syn_labels = data['labels']
            else:
                print(f"\n{'='*60}")
                print(f"Distilling: {dd_method} IPC={ipc}")
                
                if dd_method == 'dm':
                    iters = 20000 if ipc == 10 else 10000
                    syn_images, syn_labels = distill_dm(
                        train_images, train_labels, ipc, iterations=iters, device=DEVICE
                    )
                elif dd_method == 'dc':
                    outer = 20 if ipc == 10 else 10
                    inner = 50
                    syn_images, syn_labels = distill_dc(
                        train_images, train_labels, ipc, 
                        outer_loops=outer, inner_loops=inner, device=DEVICE
                    )
                elif dd_method == 'tm':
                    iters = 5000 if ipc == 10 else 3000
                    syn_images, syn_labels = distill_tm(
                        train_images, train_labels, ipc, iterations=iters, device=DEVICE
                    )
                
                torch.save({'images': syn_images, 'labels': syn_labels}, cache_path)
            
            # Evaluate in both settings
            for setting in ['hl', 'sl']:
                key = f"{dd_method}_ipc{ipc}_{setting}"
                print(f"\n{'='*60}")
                print(f"Evaluating: {key}")
                
                mean_acc, std_acc, accs = run_experiment(
                    dd_method, ipc, setting, train_images, train_labels,
                    test_images, test_labels, soft_logits=soft_logits,
                    teacher_model=teacher, num_runs=3, device=DEVICE,
                    distilled_data=(syn_images, syn_labels)
                )
                
                results[key] = {
                    'method': dd_method, 'ipc': ipc, 'setting': setting,
                    'mean': round(mean_acc, 2), 'std': round(std_acc, 2),
                    'accs': [round(a, 2) for a in accs]
                }
                print(f"  Result: {mean_acc:.2f} ± {std_acc:.2f}")
                
                with open('/workspace/results/results_pipeline.json', 'w') as f:
                    json.dump(results, f, indent=2)
    
    # Print final table
    print_results_table(results)
    
    return results


def print_results_table(results):
    """Print results in paper table format."""
    print("\n" + "=" * 80)
    print("RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("=" * 80)
    print(f"{'Method':<12} {'IPC':>4} {'HL (ours)':>14} {'HL (paper)':>14} {'SL (ours)':>14} {'SL (paper)':>14}")
    print("-" * 80)
    
    paper_results = {
        'dm_ipc10_hl': 29.23, 'dm_ipc10_sl': 26.13,
        'dm_ipc50_hl': 42.32, 'dm_ipc50_sl': 43.46,
        'dc_ipc10_hl': 28.42, 'dc_ipc10_sl': 23.54,
        'dc_ipc50_hl': 30.56, 'dc_ipc50_sl': 33.46,
        'tm_ipc10_hl': 38.18, 'tm_ipc10_sl': 37.60,
        'tm_ipc50_hl': 46.32, 'tm_ipc50_sl': 46.26,
        'random_ipc10_hl': 18.64, 'random_ipc10_sl': 33.43,
        'random_ipc50_hl': 34.66, 'random_ipc50_sl': 45.39,
        'k_centers_ipc10_hl': 25.04, 'k_centers_ipc10_sl': 34.70,
        'k_centers_ipc50_hl': 38.64, 'k_centers_ipc50_sl': 46.24,
    }
    
    for method in ['DM', 'DC', 'TM', 'Random', 'K-centers']:
        for ipc in [10, 50]:
            method_key = method.lower().replace('-', '_')
            hl_key = f"{method_key}_ipc{ipc}_hl"
            sl_key = f"{method_key}_ipc{ipc}_sl"
            
            hl_ours = f"{results[hl_key]['mean']:.2f}±{results[hl_key]['std']:.2f}" if hl_key in results else "N/A"
            sl_ours = f"{results[sl_key]['mean']:.2f}±{results[sl_key]['std']:.2f}" if sl_key in results else "N/A"
            hl_paper = f"{paper_results.get(hl_key, 'N/A')}"
            sl_paper = f"{paper_results.get(sl_key, 'N/A')}"
            
            print(f"{method:<12} {ipc:>4} {hl_ours:>14} {hl_paper:>14} {sl_ours:>14} {sl_paper:>14}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
