"""
Final comprehensive experiment runner for paper replication.
Reproduces Table: tab:small_scale_c100 (CIFAR-100, ConvNet-D3)

Phase 1: Coreset methods (Random, K-centers) - real images
Phase 2: DD methods (DM, DC, TM) - distilled images
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

from convnet import get_convnet_d3, ConvNet
from data_utils import get_cifar100_tensors, random_select, get_class_indices
from evaluate import evaluate_hl, evaluate_sl, evaluate_multiple_runs


def train_teacher(train_images, train_labels, device='cuda', epochs=300, seed=42):
    """Train a teacher model on full CIFAR-100 for soft label generation and feature extraction."""
    from dsa import DiffAugment
    
    teacher_path = '/workspace/teacher_model.pt'
    if os.path.exists(teacher_path):
        print("Loading existing teacher model...")
        model = get_convnet_d3().to(device)
        model.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
        model.eval()
        # Quick accuracy check
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(train_images), 256):
                batch = train_images[i:i+256].to(device)
                labels = train_labels[i:i+256].to(device)
                out = model(batch)
                correct += out.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        print(f"Teacher train accuracy: {100*correct/total:.2f}%")
        return model
    
    print("Training teacher model on full CIFAR-100...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    images_gpu = train_images.to(device)
    labels_gpu = train_labels.to(device)
    n = len(images_gpu)
    batch_size = 256
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch_imgs = images_gpu[idx]
            batch_labels = labels_gpu[idx]
            
            # Use DSA augmentation during teacher training too
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = F.cross_entropy(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i in range(0, n, 256):
                    batch = images_gpu[i:i+256]
                    labels = labels_gpu[i:i+256]
                    out = model(batch)
                    correct += out.argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches:.4f}, Train Acc: {100*correct/total:.2f}%")
    
    torch.save(model.state_dict(), teacher_path)
    print(f"Teacher model saved to {teacher_path}")
    return model


def generate_teacher_soft_labels(teacher, train_images, device='cuda'):
    """Generate soft labels (logits) from teacher model."""
    sl_path = '/workspace/teacher_soft_labels.pt'
    if os.path.exists(sl_path):
        print("Loading existing soft labels...")
        return torch.load(sl_path, map_location='cpu', weights_only=True)
    
    print("Generating soft labels from teacher...")
    teacher.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(device)
            logits = teacher(batch)
            all_logits.append(logits.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    torch.save(logits, sl_path)
    print(f"Soft labels saved: {logits.shape}")
    return logits


def k_centers_select_proper(train_images, train_labels, ipc, teacher_model, 
                             num_classes=100, device='cuda', seed=0):
    """
    K-Centers selection using teacher features + farthest-first traversal.
    This is the DeepCore-style K-centers.
    """
    np.random.seed(seed)
    
    # Extract features using teacher
    print("Extracting features for K-centers...")
    teacher_model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(device)
            feat = teacher_model.embed(batch)
            all_features.append(feat.cpu())
    features = torch.cat(all_features, dim=0).numpy()
    
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    features = features / norms
    
    class_indices = get_class_indices(train_labels, num_classes)
    selected = []
    
    for c in range(num_classes):
        indices = np.array(class_indices[c])
        feats = features[indices]
        
        # Farthest-first traversal (K-centers greedy)
        chosen = []
        # Start with the point closest to the class mean (centroid)
        centroid = feats.mean(axis=0)
        dists_to_centroid = np.sum((feats - centroid) ** 2, axis=1)
        first = np.argmin(dists_to_centroid)
        chosen.append(first)
        
        # Min-distance to chosen set
        min_dists = np.sum((feats - feats[first:first+1]) ** 2, axis=1)
        
        for _ in range(ipc - 1):
            # Select the point farthest from the current chosen set
            next_idx = np.argmax(min_dists)
            chosen.append(next_idx)
            # Update min distances
            new_dists = np.sum((feats - feats[next_idx:next_idx+1]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, new_dists)
        
        selected.extend([int(indices[c_idx]) for c_idx in chosen])
    
    return sorted(selected)


def run_coreset_experiments(train_images, train_labels, test_images, test_labels,
                            teacher_model, soft_labels, device='cuda', num_runs=3):
    """Run all coreset experiments (Random, K-centers) for IPC 10 and 50."""
    results = {}
    
    for ipc in [10, 50]:
        for method in ['random', 'k_centers']:
            print(f"\n{'='*60}")
            print(f"Coreset: {method}, IPC={ipc}")
            print(f"{'='*60}")
            
            # Select coreset
            if method == 'random':
                indices = random_select(train_labels, ipc, seed=0)
            else:
                indices = k_centers_select_proper(train_images, train_labels, ipc, 
                                                   teacher_model, device=device, seed=0)
            
            subset_images = train_images[indices]
            subset_labels = train_labels[indices]
            subset_soft_labels = soft_labels[indices]
            
            print(f"Selected {len(indices)} samples ({ipc} per class)")
            
            # HL evaluation
            print(f"\n--- HL Evaluation ---")
            hl_mean, hl_std, hl_accs = evaluate_multiple_runs(
                subset_images, subset_labels, test_images, test_labels,
                mode='hl', num_runs=num_runs, device=device
            )
            print(f"HL: {hl_mean:.2f} ± {hl_std:.2f}")
            
            # SL evaluation
            print(f"\n--- SL Evaluation ---")
            sl_mean, sl_std, sl_accs = evaluate_multiple_runs(
                subset_images, subset_soft_labels, test_images, test_labels,
                mode='sl', num_runs=num_runs, device=device
            )
            print(f"SL: {sl_mean:.2f} ± {sl_std:.2f}")
            
            key = f"{method}_ipc{ipc}"
            results[key] = {
                'method': method,
                'ipc': ipc,
                'hl_mean': hl_mean,
                'hl_std': hl_std,
                'hl_accs': hl_accs,
                'sl_mean': sl_mean,
                'sl_std': sl_std,
                'sl_accs': sl_accs,
            }
            
            # Save intermediate results
            with open('/workspace/results/results_final_clean.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    return results


def distill_dm(train_images, train_labels, ipc, num_classes=100, 
               device='cuda', iterations=20000, lr=0.01, seed=0):
    """Distribution Matching (DM) distillation."""
    from dsa import DiffAugment
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize synthetic data
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=device, requires_grad=True)
    syn_labels = torch.arange(num_classes, device=device).repeat_interleave(ipc)
    
    # Initialize from class means
    class_indices = get_class_indices(train_labels, num_classes)
    with torch.no_grad():
        for c in range(num_classes):
            idx = class_indices[c]
            # Random subset for initialization
            chosen = np.random.choice(idx, size=min(ipc, len(idx)), replace=False)
            syn_images[c*ipc:(c+1)*ipc] = train_images[chosen].to(device)
    
    syn_images = syn_images.detach().requires_grad_(True)
    optimizer = torch.optim.SGD([syn_images], lr=lr, momentum=0.5)
    
    train_images_gpu = train_images.to(device)
    train_labels_gpu = train_labels.to(device)
    
    for it in range(iterations):
        # Random model for feature extraction
        model = get_convnet_d3().to(device)
        model.eval()
        
        loss = torch.tensor(0.0, device=device)
        
        for c in range(num_classes):
            # Real images for this class
            real_idx = (train_labels_gpu == c).nonzero(as_tuple=True)[0]
            real_batch_idx = real_idx[torch.randperm(len(real_idx), device=device)[:256]]
            real_batch = train_images_gpu[real_batch_idx]
            
            # Synthetic images for this class
            syn_batch = syn_images[c*ipc:(c+1)*ipc]
            
            # Apply DSA
            seed_aug = int(torch.randint(0, 100000, (1,)).item())
            real_aug = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate', seed=seed_aug)
            syn_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate', seed=seed_aug)
            
            with torch.no_grad():
                real_feat = model.embed(real_aug)
                real_mean = real_feat.mean(0)
            
            syn_feat = model.embed(syn_aug)
            syn_mean = syn_feat.mean(0)
            
            loss += torch.sum((real_mean - syn_mean) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 1000 == 0:
            print(f"  DM iter {it+1}/{iterations}, loss: {loss.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_dc(train_images, train_labels, ipc, num_classes=100,
               device='cuda', iterations=5000, lr_img=1.0, lr_net=0.01, seed=0):
    """Dataset Condensation (DC) via gradient matching."""
    from dsa import DiffAugment
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize synthetic data from class means
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=device)
    syn_labels = torch.arange(num_classes, device=device).repeat_interleave(ipc)
    
    class_indices = get_class_indices(train_labels, num_classes)
    with torch.no_grad():
        for c in range(num_classes):
            idx = class_indices[c]
            chosen = np.random.choice(idx, size=min(ipc, len(idx)), replace=False)
            syn_images[c*ipc:(c+1)*ipc] = train_images[chosen].to(device)
    
    syn_images = syn_images.detach().requires_grad_(True)
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    train_images_gpu = train_images.to(device)
    train_labels_gpu = train_labels.to(device)
    
    for it in range(iterations):
        # New random model each iteration
        model = get_convnet_d3().to(device)
        model.train()
        
        # Inner loop: train model on synthetic data for a few steps
        optimizer_net = torch.optim.SGD(model.parameters(), lr=lr_net, momentum=0.9)
        
        loss_total = torch.tensor(0.0, device=device)
        
        for c in range(num_classes):
            # Real gradient
            real_idx = (train_labels_gpu == c).nonzero(as_tuple=True)[0]
            real_batch_idx = real_idx[torch.randperm(len(real_idx), device=device)[:256]]
            real_batch = train_images_gpu[real_batch_idx]
            real_labels_batch = train_labels_gpu[real_batch_idx]
            
            seed_aug = int(torch.randint(0, 100000, (1,)).item())
            real_aug = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate', seed=seed_aug)
            
            real_out = model(real_aug)
            real_loss = F.cross_entropy(real_out, real_labels_batch)
            real_grads = torch.autograd.grad(real_loss, model.parameters(), create_graph=False)
            
            # Synthetic gradient
            syn_batch = syn_images[c*ipc:(c+1)*ipc]
            syn_labels_batch = syn_labels[c*ipc:(c+1)*ipc]
            
            syn_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate', seed=seed_aug)
            
            syn_out = model(syn_aug)
            syn_loss = F.cross_entropy(syn_out, syn_labels_batch)
            syn_grads = torch.autograd.grad(syn_loss, model.parameters(), create_graph=True)
            
            # Gradient matching loss (cosine distance)
            for rg, sg in zip(real_grads, syn_grads):
                rg_flat = rg.flatten()
                sg_flat = sg.flatten()
                cos_sim = F.cosine_similarity(rg_flat.unsqueeze(0), sg_flat.unsqueeze(0))
                loss_total += (1 - cos_sim)
        
        optimizer_img.zero_grad()
        loss_total.backward()
        optimizer_img.step()
        
        if (it + 1) % 500 == 0:
            print(f"  DC iter {it+1}/{iterations}, loss: {loss_total.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_tm(train_images, train_labels, ipc, num_classes=100,
               device='cuda', iterations=5000, lr_img=0.01, seed=0,
               expert_dir='/workspace/expert_trajectories'):
    """Trajectory Matching (TM) distillation."""
    from dsa import DiffAugment
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load or generate expert trajectories
    expert_files = sorted([f for f in os.listdir(expert_dir) if f.endswith('.pt')])
    if not expert_files:
        print("No expert trajectories found! Generating...")
        generate_expert_trajectories(train_images, train_labels, expert_dir, device=device)
        expert_files = sorted([f for f in os.listdir(expert_dir) if f.endswith('.pt')])
    
    print(f"Loading {len(expert_files)} expert trajectories...")
    expert_trajectories = []
    for f in expert_files:
        traj = torch.load(os.path.join(expert_dir, f), map_location='cpu', weights_only=True)
        expert_trajectories.append(traj)
    
    # Initialize synthetic data
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=device)
    syn_labels = torch.arange(num_classes, device=device).repeat_interleave(ipc)
    
    class_indices = get_class_indices(train_labels, num_classes)
    with torch.no_grad():
        for c in range(num_classes):
            idx = class_indices[c]
            chosen = np.random.choice(idx, size=min(ipc, len(idx)), replace=False)
            syn_images[c*ipc:(c+1)*ipc] = train_images[chosen].to(device)
    
    syn_images = syn_images.detach().requires_grad_(True)
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    max_start_epoch = 25  # Sample starting point from first 25 epochs
    syn_steps = 30  # Number of student steps to match
    expert_epochs = 2  # Number of expert epochs to match
    lr_student = 0.01
    
    for it in range(iterations):
        # Sample random expert trajectory and starting point
        traj_idx = np.random.randint(len(expert_trajectories))
        traj = expert_trajectories[traj_idx]
        
        n_epochs = len(traj)
        start_epoch = np.random.randint(0, min(max_start_epoch, n_epochs - expert_epochs))
        
        # Get expert start and target parameters
        expert_start = traj[start_epoch]
        expert_target = traj[min(start_epoch + expert_epochs, n_epochs - 1)]
        
        # Initialize student from expert start
        student = get_convnet_d3().to(device)
        student_params = {k: v.to(device) for k, v in expert_start.items()}
        student.load_state_dict(student_params)
        student.train()
        
        student_opt = torch.optim.SGD(student.parameters(), lr=lr_student, momentum=0.9)
        
        # Train student on synthetic data for syn_steps
        for step in range(syn_steps):
            # Random batch from synthetic data
            perm = torch.randperm(len(syn_images), device=device)[:256]
            batch_imgs = syn_images[perm]
            batch_labels = syn_labels[perm]
            
            seed_aug = int(torch.randint(0, 100000, (1,)).item())
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate', seed=seed_aug)
            
            student_opt.zero_grad()
            out = student(batch_imgs)
            loss = F.cross_entropy(out, batch_labels)
            loss.backward()
            student_opt.step()
        
        # Compute trajectory matching loss
        target_params = {k: v.to(device) for k, v in expert_target.items()}
        
        tm_loss = torch.tensor(0.0, device=device)
        n_params = 0
        for (name, student_p), (_, target_p) in zip(student.named_parameters(), target_params.items()):
            if name in target_params:
                tm_loss += F.mse_loss(student_p, target_params[name], reduction='sum')
                n_params += student_p.numel()
        
        # Normalize
        tm_loss = tm_loss / n_params
        
        optimizer.zero_grad()
        tm_loss.backward()
        optimizer.step()
        
        if (it + 1) % 500 == 0:
            print(f"  TM iter {it+1}/{iterations}, loss: {tm_loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def generate_expert_trajectories(train_images, train_labels, save_dir, 
                                  device='cuda', num_experts=10, epochs=50):
    """Generate expert trajectories for TM."""
    from dsa import DiffAugment
    
    os.makedirs(save_dir, exist_ok=True)
    
    train_images_gpu = train_images.to(device)
    train_labels_gpu = train_labels.to(device)
    n = len(train_images_gpu)
    
    for exp_idx in range(num_experts):
        print(f"Training expert {exp_idx+1}/{num_experts}...")
        torch.manual_seed(exp_idx * 100)
        
        model = get_convnet_d3().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        trajectory = []
        trajectory.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            
            for i in range(0, n, 256):
                idx = perm[i:i+256]
                batch_imgs = train_images_gpu[idx]
                batch_labels = train_labels_gpu[idx]
                
                batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
                
                optimizer.zero_grad()
                out = model(batch_imgs)
                loss = F.cross_entropy(out, batch_labels)
                loss.backward()
                optimizer.step()
            
            trajectory.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        
        torch.save(trajectory, os.path.join(save_dir, f'expert_{exp_idx}.pt'))
        print(f"  Expert {exp_idx+1} saved ({len(trajectory)} checkpoints)")


def run_dd_experiments(train_images, train_labels, test_images, test_labels,
                       soft_labels_all, device='cuda', num_runs=3):
    """Run DD method experiments."""
    results = {}
    
    for method in ['dm', 'dc', 'tm']:
        for ipc in [10, 50]:
            print(f"\n{'='*60}")
            print(f"DD Method: {method.upper()}, IPC={ipc}")
            print(f"{'='*60}")
            
            # Check for existing distilled data
            distilled_path = f'/workspace/distilled_{method}_ipc{ipc}_final.pt'
            
            if os.path.exists(distilled_path):
                print(f"Loading existing distilled data from {distilled_path}")
                data = torch.load(distilled_path, map_location='cpu', weights_only=True)
                syn_images = data['images']
                syn_labels = data['labels']
            else:
                # Distill
                if method == 'dm':
                    iters = 20000 if ipc == 10 else 10000
                    syn_images, syn_labels = distill_dm(
                        train_images, train_labels, ipc, 
                        iterations=iters, device=device
                    )
                elif method == 'dc':
                    iters = 5000 if ipc == 10 else 3000
                    syn_images, syn_labels = distill_dc(
                        train_images, train_labels, ipc,
                        iterations=iters, device=device
                    )
                elif method == 'tm':
                    iters = 5000 if ipc == 10 else 3000
                    syn_images, syn_labels = distill_tm(
                        train_images, train_labels, ipc,
                        iterations=iters, device=device
                    )
                
                # Save
                torch.save({'images': syn_images, 'labels': syn_labels}, distilled_path)
                print(f"Saved distilled data to {distilled_path}")
            
            # Generate soft labels for synthetic data using teacher
            # For DD methods, we need to generate soft labels from the teacher
            teacher_path = '/workspace/teacher_model.pt'
            teacher = get_convnet_d3().to(device)
            teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
            teacher.eval()
            
            with torch.no_grad():
                syn_soft_labels = teacher(syn_images.to(device)).cpu()
            
            # HL evaluation
            print(f"\n--- HL Evaluation ---")
            hl_mean, hl_std, hl_accs = evaluate_multiple_runs(
                syn_images, syn_labels, test_images, test_labels,
                mode='hl', num_runs=num_runs, device=device
            )
            print(f"HL: {hl_mean:.2f} ± {hl_std:.2f}")
            
            # SL evaluation
            print(f"\n--- SL Evaluation ---")
            sl_mean, sl_std, sl_accs = evaluate_multiple_runs(
                syn_images, syn_soft_labels, test_images, test_labels,
                mode='sl', num_runs=num_runs, device=device
            )
            print(f"SL: {sl_mean:.2f} ± {sl_std:.2f}")
            
            key = f"{method}_ipc{ipc}"
            results[key] = {
                'method': method,
                'ipc': ipc,
                'hl_mean': hl_mean,
                'hl_std': hl_std,
                'hl_accs': hl_accs,
                'sl_mean': sl_mean,
                'sl_std': sl_std,
                'sl_accs': sl_accs,
            }
            
            # Save intermediate
            with open('/workspace/results/results_dd_final.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    return results


def print_results_table(results):
    """Print results in paper format."""
    print("\n" + "="*80)
    print("Table: Small-scale DD methods on CIFAR-100 (ConvNet-D3)")
    print("="*80)
    print(f"{'Method':<15} {'IPC':>5} {'HL (ours)':>15} {'HL (paper)':>15} {'SL (ours)':>15} {'SL (paper)':>15}")
    print("-"*80)
    
    paper_results = {
        'dm_ipc10': {'hl': 29.23, 'sl': 26.13},
        'dm_ipc50': {'hl': 42.32, 'sl': 43.46},
        'dc_ipc10': {'hl': 28.42, 'sl': 23.54},
        'dc_ipc50': {'hl': 30.56, 'sl': 33.46},
        'tm_ipc10': {'hl': 38.18, 'sl': 37.60},
        'tm_ipc50': {'hl': 46.32, 'sl': 46.26},
        'random_ipc10': {'hl': 18.64, 'sl': 33.43},
        'random_ipc50': {'hl': 34.66, 'sl': 45.39},
        'k_centers_ipc10': {'hl': 25.04, 'sl': 34.70},
        'k_centers_ipc50': {'hl': 38.64, 'sl': 46.24},
    }
    
    for key in ['dm_ipc10', 'dm_ipc50', 'dc_ipc10', 'dc_ipc50', 'tm_ipc10', 'tm_ipc50',
                'random_ipc10', 'random_ipc50', 'k_centers_ipc10', 'k_centers_ipc50']:
        if key in results:
            r = results[key]
            p = paper_results.get(key, {'hl': 0, 'sl': 0})
            method_name = key.split('_ipc')[0].upper().replace('_', '-')
            ipc = key.split('ipc')[1]
            print(f"{method_name:<15} {ipc:>5} {r['hl_mean']:>7.2f}±{r['hl_std']:.2f} {p['hl']:>7.2f} {r['sl_mean']:>7.2f}±{r['sl_std']:.2f} {p['sl']:>7.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='all', choices=['coreset', 'dd', 'all'])
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    os.makedirs('/workspace/results', exist_ok=True)
    
    # Load data
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Train teacher and generate soft labels
    teacher = train_teacher(train_images, train_labels, device=args.device)
    soft_labels = generate_teacher_soft_labels(teacher, train_images, device=args.device)
    
    all_results = {}
    
    if args.phase in ['coreset', 'all']:
        coreset_results = run_coreset_experiments(
            train_images, train_labels, test_images, test_labels,
            teacher, soft_labels, device=args.device, num_runs=args.num_runs
        )
        all_results.update(coreset_results)
    
    if args.phase in ['dd', 'all']:
        dd_results = run_dd_experiments(
            train_images, train_labels, test_images, test_labels,
            soft_labels, device=args.device, num_runs=args.num_runs
        )
        all_results.update(dd_results)
    
    # Save all results
    with open('/workspace/results/results_all_final.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print_results_table(all_results)


if __name__ == '__main__':
    main()
