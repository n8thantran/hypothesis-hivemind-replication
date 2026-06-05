"""
Final clean experiment script for reproducing Table 1 (tab:small_scale_c100)
from "Rethinking Dataset Distillation: Hard Truths About Soft Labels"

CIFAR-100, ConvNet-D3, IPC 10 and 50
Methods: DM, DC, TM, Random, K-centers
Settings: Hard Label (HL) and Soft Label (SL)

Evaluation hyperparameters (from paper's Table tab:stage3_hyper):
- HL: 300 epochs, SGD lr=0.01, StepLR@epoch151 (halve), batch=256, DSA, CE loss
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
import sys
from convnet import get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors


# ============================================================
# TEACHER TRAINING
# ============================================================
def train_teacher(train_images, train_labels, test_images, test_labels, epochs=300, device='cuda'):
    """Train a teacher ConvNet-D3 on full CIFAR-100 for soft label generation."""
    print("Training teacher model...")
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, drop_last=False)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            acc = evaluate_model(model, test_images, test_labels, device)
            print(f"  Teacher epoch {epoch+1}: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_state)
    print(f"  Best teacher accuracy: {best_acc:.2f}%")
    return model


def generate_soft_logits(model, images, device='cuda', batch_size=256):
    """Generate soft logits from teacher model."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


# ============================================================
# EVALUATION
# ============================================================
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
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_and_eval_hl(syn_images, syn_labels, test_images, test_labels, 
                       epochs=300, device='cuda', seed=0):
    """
    Train student with Hard Labels (HL setting).
    300 epochs, SGD lr=0.01, StepLR@151 (gamma=0.5), batch=256, DSA, CE loss
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    n = len(syn_images)
    batch_size = min(256, n)
    
    for epoch in range(epochs):
        model.train()
        # Shuffle
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            imgs = syn_images[idx].to(device)
            labels = syn_labels[idx].to(device)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return evaluate_model(model, test_images, test_labels, device)


def train_and_eval_sl(syn_images, syn_soft_logits, test_images, test_labels,
                       epochs=300, device='cuda', seed=0, T=20):
    """
    Train student with Soft Labels (SL setting).
    300 epochs, AdamW lr=1e-3, Cosine scheduler, batch=256, DSA, KL-Div(T=20)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n = len(syn_images)
    batch_size = min(256, n)
    
    # Pre-compute soft targets
    soft_targets = F.softmax(syn_soft_logits / T, dim=1)
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            imgs = syn_images[idx].to(device)
            targets = soft_targets[idx].to(device)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(imgs)
            log_probs = F.log_softmax(out / T, dim=1)
            loss = F.kl_div(log_probs, targets, reduction='batchmean') * (T * T)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return evaluate_model(model, test_images, test_labels, device)


def run_eval(syn_images, syn_labels, syn_soft_logits, test_images, test_labels,
             setting='both', n_runs=3, device='cuda', method_name=''):
    """Run evaluation with multiple seeds."""
    results = {}
    
    if setting in ['hl', 'both']:
        hl_accs = []
        for run in range(n_runs):
            acc = train_and_eval_hl(syn_images, syn_labels, test_images, test_labels,
                                     device=device, seed=run*100+42)
            hl_accs.append(acc)
            print(f"  {method_name} HL run {run+1}: {acc:.2f}%")
        results['hl_mean'] = np.mean(hl_accs)
        results['hl_std'] = np.std(hl_accs)
        results['hl_accs'] = hl_accs
        print(f"  {method_name} HL: {results['hl_mean']:.2f} ± {results['hl_std']:.2f}")
    
    if setting in ['sl', 'both']:
        sl_accs = []
        for run in range(n_runs):
            acc = train_and_eval_sl(syn_images, syn_soft_logits, test_images, test_labels,
                                     device=device, seed=run*100+42)
            sl_accs.append(acc)
            print(f"  {method_name} SL run {run+1}: {acc:.2f}%")
        results['sl_mean'] = np.mean(sl_accs)
        results['sl_std'] = np.std(sl_accs)
        results['sl_accs'] = sl_accs
        print(f"  {method_name} SL: {results['sl_mean']:.2f} ± {results['sl_std']:.2f}")
    
    return results


# ============================================================
# CORESET METHODS
# ============================================================
def select_random(train_images, train_labels, ipc, seed=0):
    """Random coreset selection."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_classes = 100
    indices = []
    for c in range(num_classes):
        class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_idx))[:ipc]
        indices.append(class_idx[perm])
    indices = torch.cat(indices)
    return train_images[indices], train_labels[indices], indices


def select_k_centers(train_images, train_labels, ipc, device='cuda'):
    """
    K-centers coreset selection using feature space.
    Uses a pre-trained ConvNet-D3 to extract features, then runs K-center greedy.
    Paper cites DeepCore for K-centers implementation.
    """
    print("  Computing K-centers selection...")
    
    # Train a quick model for feature extraction
    model = get_convnet_d3().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=76, gamma=0.5)
    
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    # Train for 150 epochs (enough for decent features)
    for epoch in range(150):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    # Extract features
    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(device)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    features = torch.cat(all_features, dim=0)
    
    # K-center greedy per class
    num_classes = 100
    selected_indices = []
    
    for c in range(num_classes):
        class_mask = (train_labels == c)
        class_idx = class_mask.nonzero(as_tuple=True)[0]
        class_feat = features[class_idx]
        
        # Normalize features
        class_feat = F.normalize(class_feat, dim=1)
        
        # K-center greedy: iteratively select point farthest from selected set
        n = len(class_feat)
        if n <= ipc:
            selected_indices.append(class_idx)
            continue
        
        # Start with the point closest to the mean (most representative)
        mean_feat = class_feat.mean(dim=0, keepdim=True)
        dists_to_mean = torch.cdist(class_feat, mean_feat).squeeze()
        first_idx = dists_to_mean.argmin().item()
        
        selected = [first_idx]
        # Min distance from each point to the selected set
        min_dists = torch.cdist(class_feat, class_feat[first_idx:first_idx+1]).squeeze()
        
        for _ in range(ipc - 1):
            # Select point with maximum minimum distance to selected set
            farthest = min_dists.argmax().item()
            selected.append(farthest)
            # Update min distances
            new_dists = torch.cdist(class_feat, class_feat[farthest:farthest+1]).squeeze()
            min_dists = torch.minimum(min_dists, new_dists)
        
        selected_indices.append(class_idx[torch.tensor(selected)])
    
    indices = torch.cat(selected_indices)
    return train_images[indices], train_labels[indices], indices


# ============================================================
# DISTILLATION METHODS
# ============================================================
def distill_dm(train_images, train_labels, ipc, n_iters=20000, device='cuda'):
    """
    Distribution Matching (DM) distillation.
    Matches mean features between synthetic and real data.
    """
    print(f"  Distilling with DM (IPC={ipc}, iters={n_iters})...")
    num_classes = 100
    
    # Initialize synthetic data
    syn_images = torch.randn(num_classes * ipc, 3, 32, 32, device=device, requires_grad=True)
    syn_labels = torch.arange(num_classes, device=device).repeat_interleave(ipc)
    
    # Initialize from class means
    with torch.no_grad():
        for c in range(num_classes):
            class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            # Random subset for initialization
            perm = torch.randperm(len(class_idx))[:ipc]
            syn_images[c*ipc:(c+1)*ipc] = train_images[class_idx[perm]].to(device)
    syn_images = syn_images.detach().requires_grad_(True)
    
    optimizer = torch.optim.SGD([syn_images], lr=1.0, momentum=0.5)
    
    for it in range(n_iters):
        # Random model for feature extraction
        model = get_convnet_d3().to(device)
        model.train()
        
        loss_total = torch.tensor(0.0, device=device)
        
        for c in range(num_classes):
            # Real data for this class
            class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(class_idx))[:256]
            real_batch = train_images[class_idx[perm]].to(device)
            real_batch = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate')
            
            # Synthetic data for this class
            syn_batch = syn_images[c*ipc:(c+1)*ipc]
            syn_batch_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate')
            
            # Match features
            with torch.no_grad():
                real_feat = model.embed(real_batch).mean(0)
            syn_feat = model.embed(syn_batch_aug).mean(0)
            
            loss_total += torch.sum((real_feat - syn_feat) ** 2)
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        if (it + 1) % 2000 == 0:
            print(f"    DM iter {it+1}/{n_iters}, loss: {loss_total.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_dc(train_images, train_labels, ipc, n_iters=5000, device='cuda'):
    """
    Dataset Condensation (DC) via gradient matching.
    """
    print(f"  Distilling with DC (IPC={ipc}, iters={n_iters})...")
    num_classes = 100
    
    # Initialize synthetic data from real images
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
        
        # Train model on synthetic data for a few steps, then match gradients
        loss_total = torch.tensor(0.0, device=device)
        
        for c in range(num_classes):
            # Real gradients
            class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(class_idx))[:256]
            real_batch = train_images[class_idx[perm]].to(device)
            real_labels = train_labels[class_idx[perm]].to(device)
            real_batch = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate')
            
            real_out = model(real_batch)
            real_loss = F.cross_entropy(real_out, real_labels)
            real_grads = torch.autograd.grad(real_loss, model.parameters(), create_graph=False)
            
            # Synthetic gradients
            syn_batch = syn_images[c*ipc:(c+1)*ipc]
            syn_lab = syn_labels[c*ipc:(c+1)*ipc]
            syn_batch_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate')
            
            syn_out = model(syn_batch_aug)
            syn_loss = F.cross_entropy(syn_out, syn_lab)
            syn_grads = torch.autograd.grad(syn_loss, model.parameters(), create_graph=True)
            
            # Match gradients (cosine distance)
            for rg, sg in zip(real_grads, syn_grads):
                rg_flat = rg.detach().flatten()
                sg_flat = sg.flatten()
                cos_sim = F.cosine_similarity(rg_flat.unsqueeze(0), sg_flat.unsqueeze(0))
                loss_total += (1 - cos_sim)
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        if (it + 1) % 1000 == 0:
            print(f"    DC iter {it+1}/{n_iters}, loss: {loss_total.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_tm(train_images, train_labels, ipc, n_iters=5000, 
               n_experts=10, expert_epochs=50, device='cuda'):
    """
    Trajectory Matching (TM) distillation.
    Matches training trajectories of models trained on synthetic vs real data.
    """
    print(f"  Distilling with TM (IPC={ipc}, iters={n_iters})...")
    num_classes = 100
    
    # Step 1: Train expert trajectories on real data
    print("    Training expert trajectories...")
    expert_trajectories = []
    
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    for exp_id in range(n_experts):
        torch.manual_seed(exp_id * 1000)
        model = get_convnet_d3().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        trajectory = [copy.deepcopy(model.state_dict())]
        
        for epoch in range(expert_epochs):
            model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
                optimizer.zero_grad()
                out = model(imgs)
                loss = F.cross_entropy(out, labels)
                loss.backward()
                optimizer.step()
            trajectory.append(copy.deepcopy(model.state_dict()))
        
        expert_trajectories.append(trajectory)
        if (exp_id + 1) % 5 == 0:
            print(f"      Expert {exp_id+1}/{n_experts} done")
    
    # Step 2: Match trajectories
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
    
    M = 2  # Number of student steps
    student_lr = 0.01
    
    for it in range(n_iters):
        # Sample random expert and starting point
        exp_id = np.random.randint(n_experts)
        trajectory = expert_trajectories[exp_id]
        
        # Sample starting epoch (not too late)
        max_start = max(1, len(trajectory) - M - 1)
        start_epoch = np.random.randint(0, max_start)
        
        # Load expert starting params
        start_params = trajectory[start_epoch]
        target_params = trajectory[min(start_epoch + M, len(trajectory) - 1)]
        
        # Create student model from expert starting point
        student = get_convnet_d3().to(device)
        student.load_state_dict(start_params)
        student.train()
        
        student_opt = torch.optim.SGD(student.parameters(), lr=student_lr, momentum=0.9)
        
        # Train student on synthetic data for M steps
        for step in range(M):
            perm = torch.randperm(len(syn_images))
            batch_size = min(256, len(syn_images))
            idx = perm[:batch_size]
            
            imgs = syn_images[idx]
            labels = syn_labels[idx]
            imgs_aug = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            student_opt.zero_grad()
            out = student(imgs_aug)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            student_opt.step()
        
        # Match: minimize distance between student params and target expert params
        match_loss = torch.tensor(0.0, device=device)
        target_model = get_convnet_d3().to(device)
        target_model.load_state_dict(target_params)
        
        for sp, tp in zip(student.parameters(), target_model.parameters()):
            match_loss += F.mse_loss(sp, tp.detach(), reduction='sum')
        
        # Normalize by parameter count
        n_params = sum(p.numel() for p in student.parameters())
        match_loss = match_loss / n_params
        
        syn_optimizer.zero_grad()
        match_loss.backward()
        syn_optimizer.step()
        
        if (it + 1) % 1000 == 0:
            print(f"    TM iter {it+1}/{n_iters}, loss: {match_loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('results', exist_ok=True)
    
    # Parse command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['random', 'k_centers', 'dm', 'dc', 'tm'])
    parser.add_argument('--ipcs', nargs='+', type=int, default=[10, 50])
    parser.add_argument('--settings', nargs='+', default=['hl', 'sl'])
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--dm_iters', type=int, default=20000)
    parser.add_argument('--dc_iters', type=int, default=5000)
    parser.add_argument('--tm_iters', type=int, default=5000)
    parser.add_argument('--tm_experts', type=int, default=10)
    parser.add_argument('--teacher_epochs', type=int, default=300)
    parser.add_argument('--skip_distill', action='store_true', help='Skip distillation, load from files')
    args = parser.parse_args()
    
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Train or load teacher for soft labels
    teacher_path = 'teacher_model_final.pt'
    logits_path = 'teacher_soft_logits_final.pt'
    
    if os.path.exists(teacher_path) and os.path.exists(logits_path):
        print("Loading existing teacher model and soft logits...")
        teacher = get_convnet_d3().to(device)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        all_soft_logits = torch.load(logits_path, map_location='cpu')
        teacher_acc = evaluate_model(teacher, test_images, test_labels, device)
        print(f"  Teacher accuracy: {teacher_acc:.2f}%")
    else:
        teacher = train_teacher(train_images, train_labels, test_images, test_labels,
                               epochs=args.teacher_epochs, device=device)
        torch.save(teacher.state_dict(), teacher_path)
        all_soft_logits = generate_soft_logits(teacher, train_images, device)
        torch.save(all_soft_logits, logits_path)
    
    all_results = {}
    
    for ipc in args.ipcs:
        for method in args.methods:
            key = f"{method}_ipc{ipc}"
            print(f"\n{'='*60}")
            print(f"Method: {method}, IPC: {ipc}")
            print(f"{'='*60}")
            
            # Get synthetic/selected data
            if method == 'random':
                syn_images, syn_labels, indices = select_random(train_images, train_labels, ipc, seed=42)
                syn_soft_logits = all_soft_logits[indices]
                
            elif method == 'k_centers':
                syn_images, syn_labels, indices = select_k_centers(train_images, train_labels, ipc, device)
                syn_soft_logits = all_soft_logits[indices]
                
            elif method == 'dm':
                distilled_path = f'distilled_dm_ipc{ipc}_final.pt'
                if args.skip_distill and os.path.exists(distilled_path):
                    data = torch.load(distilled_path, map_location='cpu')
                    syn_images, syn_labels = data['images'], data['labels']
                else:
                    syn_images, syn_labels = distill_dm(train_images, train_labels, ipc, 
                                                         n_iters=args.dm_iters, device=device)
                    torch.save({'images': syn_images, 'labels': syn_labels}, distilled_path)
                # Generate soft labels for distilled data using teacher
                syn_soft_logits = generate_soft_logits(teacher, syn_images, device)
                
            elif method == 'dc':
                distilled_path = f'distilled_dc_ipc{ipc}_final.pt'
                if args.skip_distill and os.path.exists(distilled_path):
                    data = torch.load(distilled_path, map_location='cpu')
                    syn_images, syn_labels = data['images'], data['labels']
                else:
                    syn_images, syn_labels = distill_dc(train_images, train_labels, ipc,
                                                         n_iters=args.dc_iters, device=device)
                    torch.save({'images': syn_images, 'labels': syn_labels}, distilled_path)
                syn_soft_logits = generate_soft_logits(teacher, syn_images, device)
                
            elif method == 'tm':
                distilled_path = f'distilled_tm_ipc{ipc}_final.pt'
                if args.skip_distill and os.path.exists(distilled_path):
                    data = torch.load(distilled_path, map_location='cpu')
                    syn_images, syn_labels = data['images'], data['labels']
                else:
                    syn_images, syn_labels = distill_tm(train_images, train_labels, ipc,
                                                         n_iters=args.tm_iters,
                                                         n_experts=args.tm_experts,
                                                         device=device)
                    torch.save({'images': syn_images, 'labels': syn_labels}, distilled_path)
                syn_soft_logits = generate_soft_logits(teacher, syn_images, device)
            
            print(f"  Data shape: {syn_images.shape}")
            
            # Evaluate
            for setting in args.settings:
                result_key = f"{key}_{setting}"
                print(f"\n  Evaluating {setting.upper()}...")
                
                if setting == 'hl':
                    accs = []
                    for run in range(args.n_runs):
                        acc = train_and_eval_hl(syn_images, syn_labels, test_images, test_labels,
                                                 device=device, seed=run*100+42)
                        accs.append(acc)
                        print(f"    Run {run+1}: {acc:.2f}%")
                    all_results[result_key] = {
                        'mean': np.mean(accs),
                        'std': np.std(accs),
                        'accs': accs
                    }
                    print(f"  {method} IPC{ipc} HL: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
                    
                elif setting == 'sl':
                    accs = []
                    for run in range(args.n_runs):
                        acc = train_and_eval_sl(syn_images, syn_soft_logits, test_images, test_labels,
                                                 device=device, seed=run*100+42)
                        accs.append(acc)
                        print(f"    Run {run+1}: {acc:.2f}%")
                    all_results[result_key] = {
                        'mean': np.mean(accs),
                        'std': np.std(accs),
                        'accs': accs
                    }
                    print(f"  {method} IPC{ipc} SL: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
            
            # Save intermediate results
            with open('results/results_final_table.json', 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Print final table
    print_table(all_results)
    
    # Save final results
    with open('results/results_final_table.json', 'w') as f:
        json.dump(all_results, f, indent=2)


def print_table(results):
    """Print results in paper table format."""
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("="*80)
    
    # Paper target values
    paper = {
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
    
    header = f"{'Method':<12} {'IPC':>4} | {'HL (ours)':>12} {'HL (paper)':>12} | {'SL (ours)':>12} {'SL (paper)':>12}"
    print(header)
    print("-" * len(header))
    
    for method in ['dm', 'dc', 'tm', 'random', 'k_centers']:
        for ipc in [10, 50]:
            hl_key = f"{method}_ipc{ipc}_hl"
            sl_key = f"{method}_ipc{ipc}_sl"
            
            hl_str = f"{results[hl_key]['mean']:.2f}±{results[hl_key]['std']:.2f}" if hl_key in results else "N/A"
            sl_str = f"{results[sl_key]['mean']:.2f}±{results[sl_key]['std']:.2f}" if sl_key in results else "N/A"
            
            hl_paper = f"{paper.get(hl_key, 'N/A')}"
            sl_paper = f"{paper.get(sl_key, 'N/A')}"
            
            name = method.replace('_', '-')
            print(f"{name:<12} {ipc:>4} | {hl_str:>12} {hl_paper:>12} | {sl_str:>12} {sl_paper:>12}")


if __name__ == '__main__':
    main()
