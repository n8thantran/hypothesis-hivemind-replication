"""
Complete pipeline for reproducing Table 1 from the paper.
CIFAR-100, ConvNet-D3, IPC 10 and 50.
Methods: Random, K-centers, DM, DC, TM
Settings: Hard Label (HL) and Soft Label (SL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import copy
from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_cifar100_tensors

DEVICE = 'cuda'
NUM_CLASSES = 100
DSA_STRATEGY = 'color_crop_cutout_flip_scale_rotate'


def get_model():
    return ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32))


# ============================================================
# EVALUATION: Train student on distilled/selected data
# ============================================================

def train_student(images, labels, test_images, test_labels,
                  label_type='hard', soft_labels=None,
                  epochs=300, batch_size=256, seed=0, verbose=False):
    """
    Train student model following paper's exact hyperparameters.
    
    HL: SGD, lr=0.01, StepLR@151, CE loss, DSA
    SL: AdamW, lr=1e-3, Cosine, KL-Div T=20, DSA
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_model().to(DEVICE)
    n = len(images)
    
    if label_type == 'hard':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    temperature = 20.0
    bs = min(batch_size, n)
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            imgs = DiffAugment(images[idx].to(DEVICE), strategy=DSA_STRATEGY)
            
            optimizer.zero_grad()
            out = model(imgs)
            
            if label_type == 'hard':
                loss = F.cross_entropy(out, labels[idx].to(DEVICE))
            else:
                target = F.softmax(soft_labels[idx].to(DEVICE) / temperature, dim=1)
                log_pred = F.log_softmax(out / temperature, dim=1)
                loss = F.kl_div(log_pred, target, reduction='batchmean') * (temperature ** 2)
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), 512):
            batch = test_images[i:i+512].to(DEVICE)
            pred = model(batch).argmax(1)
            correct += (pred == test_labels[i:i+512].to(DEVICE)).sum().item()
            total += len(batch)
    
    return 100.0 * correct / total


def evaluate_method(images, labels, test_images, test_labels,
                    soft_labels_all, label_type='hard', soft_labels=None,
                    num_runs=3, epochs=300, verbose=True):
    """Run multiple trials and return mean ± std."""
    accs = []
    for run in range(num_runs):
        acc = train_student(
            images, labels, test_images, test_labels,
            label_type=label_type, soft_labels=soft_labels,
            epochs=epochs, seed=run, verbose=False
        )
        accs.append(acc)
        if verbose:
            print(f"  Run {run+1}: {acc:.2f}%")
    
    mean = np.mean(accs)
    std = np.std(accs)
    if verbose:
        print(f"  => {mean:.2f} ± {std:.2f}%")
    return mean, std


# ============================================================
# CORESET SELECTION
# ============================================================

def random_select(train_labels, ipc, seed=42):
    """Random selection of IPC samples per class."""
    np.random.seed(seed)
    indices = []
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0].numpy()
        sel = np.random.choice(cls_idx, ipc, replace=False)
        indices.extend(sel.tolist())
    return indices


def kcenters_select(train_images, train_labels, ipc, seed=42):
    """
    K-means clustering in feature space, select nearest-to-centroid.
    Uses a pretrained ConvNet to extract features.
    """
    from sklearn.cluster import KMeans
    
    # Extract features using a randomly initialized model (or pretrained)
    # The paper uses K-centers from DeepCore which uses feature space
    model = get_model().to(DEVICE)
    
    # Load teacher if available for better features
    teacher_path = '/workspace/teacher_model.pt'
    if os.path.exists(teacher_path):
        model.load_state_dict(torch.load(teacher_path, weights_only=True))
        print("  Using teacher model for K-centers features")
    
    model.eval()
    
    # Extract features
    features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(DEVICE)
            # Get penultimate layer features
            feat = model.embed(batch)
            features.append(feat.cpu())
    features = torch.cat(features, dim=0).numpy()
    
    indices = []
    for c in range(NUM_CLASSES):
        cls_mask = (train_labels == c).numpy()
        cls_idx = np.where(cls_mask)[0]
        cls_feat = features[cls_idx]
        
        if len(cls_idx) <= ipc:
            indices.extend(cls_idx.tolist())
            continue
        
        # K-means clustering
        kmeans = KMeans(n_clusters=ipc, random_state=seed, n_init=10)
        kmeans.fit(cls_feat)
        
        # Select nearest to each centroid
        for k in range(ipc):
            center = kmeans.cluster_centers_[k]
            dists = np.linalg.norm(cls_feat - center, axis=1)
            nearest = cls_idx[np.argmin(dists)]
            indices.append(nearest)
    
    return indices


# ============================================================
# DISTRIBUTION MATCHING (DM)
# ============================================================

def distill_dm(train_images, train_labels, ipc, num_iters=20000, lr=0.01, seed=0):
    """
    Distribution Matching: match mean embeddings of real and synthetic data.
    Following Zhao & Bilen (2023) with 20000 iterations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize synthetic data from random real images
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(cls_idx))[:ipc]
        syn_images.append(train_images[cls_idx[perm]].clone())
        syn_labels.append(torch.full((ipc,), c, dtype=torch.long))
    
    syn_images = torch.cat(syn_images, dim=0).to(DEVICE).requires_grad_(True)
    syn_labels = torch.cat(syn_labels, dim=0).to(DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr, momentum=0.5)
    
    # Organize real data by class
    real_by_class = {}
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        real_by_class[c] = train_images[cls_idx]
    
    print(f"  DM distillation: {num_iters} iterations, IPC={ipc}")
    
    for it in range(num_iters):
        # Sample a new random network each iteration
        model = get_model().to(DEVICE)
        model.eval()
        
        loss = torch.tensor(0.0, device=DEVICE)
        
        # For each class, match mean embeddings
        for c in range(NUM_CLASSES):
            # Real data: sample a batch
            real_cls = real_by_class[c]
            real_idx = torch.randperm(len(real_cls))[:256]
            real_batch = real_cls[real_idx].to(DEVICE)
            
            # Synthetic data for this class
            syn_cls = syn_images[syn_labels == c]
            
            # Apply DSA augmentation
            real_aug = DiffAugment(real_batch, strategy=DSA_STRATEGY)
            syn_aug = DiffAugment(syn_cls, strategy=DSA_STRATEGY)
            
            # Get embeddings
            with torch.no_grad():
                real_feat = model.embed(real_aug)
            syn_feat = model.embed(syn_aug)
            
            # Match mean embeddings (MMD with mean)
            loss += torch.mean((real_feat.mean(0) - syn_feat.mean(0)) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 2000 == 0:
            print(f"    Iter {it+1}/{num_iters}, Loss: {loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# GRADIENT MATCHING (DC)
# ============================================================

def distill_dc(train_images, train_labels, ipc, outer_loops=1, inner_loops=1,
               num_iters=1000, lr_img=0.1, seed=0):
    """
    Dataset Condensation via Gradient Matching (DC).
    Following Zhao et al. (2021).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize synthetic data
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(cls_idx))[:ipc]
        syn_images.append(train_images[cls_idx[perm]].clone())
        syn_labels.append(torch.full((ipc,), c, dtype=torch.long))
    
    syn_images = torch.cat(syn_images, dim=0).to(DEVICE).requires_grad_(True)
    syn_labels = torch.cat(syn_labels, dim=0).to(DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    # Organize real data by class
    real_by_class = {}
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        real_by_class[c] = train_images[cls_idx]
    
    print(f"  DC distillation: {num_iters} outer iterations, IPC={ipc}")
    
    for it in range(num_iters):
        model = get_model().to(DEVICE)
        model.train()
        
        # Train model on synthetic data for a few steps, then match gradients
        net_params = list(model.parameters())
        optimizer_net = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        for ol in range(outer_loops):
            loss = torch.tensor(0.0, device=DEVICE)
            
            for c in range(NUM_CLASSES):
                # Real gradient
                real_cls = real_by_class[c]
                real_idx = torch.randperm(len(real_cls))[:256]
                real_batch = DiffAugment(real_cls[real_idx].to(DEVICE), strategy=DSA_STRATEGY)
                real_lab = torch.full((len(real_batch),), c, dtype=torch.long, device=DEVICE)
                
                real_out = model(real_batch)
                real_loss = F.cross_entropy(real_out, real_lab)
                real_grad = torch.autograd.grad(real_loss, net_params, create_graph=False)
                
                # Synthetic gradient
                syn_cls = syn_images[syn_labels == c]
                syn_aug = DiffAugment(syn_cls, strategy=DSA_STRATEGY)
                syn_lab = torch.full((len(syn_cls),), c, dtype=torch.long, device=DEVICE)
                
                syn_out = model(syn_aug)
                syn_loss = F.cross_entropy(syn_out, syn_lab)
                syn_grad = torch.autograd.grad(syn_loss, net_params, create_graph=True)
                
                # Gradient matching loss (cosine distance)
                for rg, sg in zip(real_grad, syn_grad):
                    rg_flat = rg.flatten()
                    sg_flat = sg.flatten()
                    cos_sim = F.cosine_similarity(rg_flat.unsqueeze(0), sg_flat.unsqueeze(0))
                    loss += (1 - cos_sim)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update network on synthetic data
            if ol < outer_loops - 1:
                syn_aug_net = DiffAugment(syn_images.detach(), strategy=DSA_STRATEGY)
                out_net = model(syn_aug_net)
                loss_net = F.cross_entropy(out_net, syn_labels)
                optimizer_net.zero_grad()
                loss_net.backward()
                optimizer_net.step()
        
        if (it + 1) % 200 == 0:
            print(f"    Iter {it+1}/{num_iters}, Loss: {loss.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# TRAJECTORY MATCHING (TM)
# ============================================================

def train_expert(train_images, train_labels, epochs=50, seed=0):
    """Train an expert model and save trajectory."""
    torch.manual_seed(seed)
    model = get_model().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    trajectory = [copy.deepcopy(model.state_dict())]
    n = len(train_images)
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            imgs = DiffAugment(train_images[idx].to(DEVICE), strategy=DSA_STRATEGY)
            labels = train_labels[idx].to(DEVICE)
            
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        trajectory.append(copy.deepcopy(model.state_dict()))
    
    return trajectory


def distill_tm(train_images, train_labels, ipc, num_experts=3, expert_epochs=50,
               num_iters=5000, lr_img=0.01, match_steps=50, seed=0):
    """
    Trajectory Matching (TM).
    Following Cazenavette et al. (2022).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Train expert trajectories
    print(f"  Training {num_experts} expert trajectories ({expert_epochs} epochs each)...")
    experts = []
    for e in range(num_experts):
        print(f"    Expert {e+1}/{num_experts}")
        traj = train_expert(train_images, train_labels, epochs=expert_epochs, seed=e)
        experts.append(traj)
    
    # Initialize synthetic data
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(cls_idx))[:ipc]
        syn_images.append(train_images[cls_idx[perm]].clone())
        syn_labels.append(torch.full((ipc,), c, dtype=torch.long))
    
    syn_images = torch.cat(syn_images, dim=0).to(DEVICE).requires_grad_(True)
    syn_labels = torch.cat(syn_labels, dim=0).to(DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    print(f"  TM distillation: {num_iters} iterations, IPC={ipc}")
    
    for it in range(num_iters):
        # Sample random expert and starting point
        expert_idx = np.random.randint(num_experts)
        traj = experts[expert_idx]
        max_start = len(traj) - match_steps - 1
        if max_start < 1:
            max_start = 1
        start_epoch = np.random.randint(0, max_start)
        
        # Load starting parameters
        start_params = traj[start_epoch]
        target_params = traj[min(start_epoch + match_steps, len(traj) - 1)]
        
        # Create student model from starting point
        student = get_model().to(DEVICE)
        student.load_state_dict(start_params)
        student.train()
        
        # Train student on synthetic data for N steps
        student_optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
        
        N = match_steps  # number of student training steps
        for step in range(N):
            # Sample a batch from synthetic data
            perm = torch.randperm(len(syn_images))[:256]
            batch_imgs = DiffAugment(syn_images[perm], strategy=DSA_STRATEGY)
            batch_labels = syn_labels[perm]
            
            out = student(batch_imgs)
            loss = F.cross_entropy(out, batch_labels)
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
        
        # Compute trajectory matching loss
        student_params = {k: v for k, v in student.named_parameters()}
        target_flat = torch.cat([target_params[k].to(DEVICE).flatten() for k in target_params])
        
        # Get student params in same order
        student_flat = torch.cat([student_params[k].flatten() for k in target_params if k in student_params])
        start_flat = torch.cat([start_params[k].to(DEVICE).flatten() for k in target_params])
        
        # Normalized parameter matching loss
        target_direction = target_flat - start_flat
        student_direction = student_flat - start_flat
        
        # L2 loss on normalized directions
        target_norm = target_direction / (target_direction.norm() + 1e-6)
        student_norm = student_direction / (student_direction.norm() + 1e-6)
        
        tm_loss = torch.mean((target_norm - student_norm) ** 2)
        
        optimizer.zero_grad()
        tm_loss.backward()
        optimizer.step()
        
        if (it + 1) % 500 == 0:
            print(f"    Iter {it+1}/{num_iters}, TM Loss: {tm_loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# TEACHER MODEL
# ============================================================

def train_teacher(train_images, train_labels, test_images, test_labels, epochs=200):
    """Train a teacher model for soft label generation."""
    model = get_model().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n = len(train_images)
    best_acc = 0
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            imgs = DiffAugment(train_images[idx].to(DEVICE), strategy=DSA_STRATEGY)
            labels = train_labels[idx].to(DEVICE)
            
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for i in range(0, len(test_images), 512):
                    batch = test_images[i:i+512].to(DEVICE)
                    pred = model(batch).argmax(1)
                    correct += (pred == test_labels[i:i+512].to(DEVICE)).sum().item()
                    total += len(batch)
            acc = 100.0 * correct / total
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), '/workspace/teacher_model.pt')
            print(f"  Epoch {epoch+1}/{epochs}, Test Acc: {acc:.2f}% (best: {best_acc:.2f}%)")
            model.train()
    
    return best_acc


def generate_soft_labels(train_images):
    """Generate soft labels using teacher model."""
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load('/workspace/teacher_model.pt', weights_only=True))
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(DEVICE)
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    return torch.cat(all_logits, dim=0)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='all',
                        choices=['teacher', 'distill', 'eval', 'all', 'quick'])
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'random', 'kcenter', 'dm', 'dc', 'tm'])
    parser.add_argument('--ipc', type=int, default=10)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--dm_iters', type=int, default=20000)
    parser.add_argument('--dc_iters', type=int, default=1000)
    parser.add_argument('--tm_iters', type=int, default=5000)
    args = parser.parse_args()
    
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Phase 1: Teacher
    if args.phase in ['teacher', 'all']:
        if not os.path.exists('/workspace/teacher_model.pt'):
            print("\n=== Training Teacher ===")
            acc = train_teacher(train_images, train_labels, test_images, test_labels)
            print(f"Teacher accuracy: {acc:.2f}%")
        
        if not os.path.exists('/workspace/soft_labels.pt'):
            print("\n=== Generating Soft Labels ===")
            sl = generate_soft_labels(train_images)
            torch.save(sl, '/workspace/soft_labels.pt')
            print(f"Soft labels shape: {sl.shape}")
    
    # Load soft labels
    soft_labels_all = torch.load('/workspace/soft_labels.pt', weights_only=True)
    
    results = {}
    
    # Phase 2: Distill + Eval
    methods_to_run = ['random', 'kcenter', 'dm', 'dc', 'tm'] if args.method == 'all' else [args.method]
    
    for method in methods_to_run:
        for ipc in ([args.ipc] if args.method != 'all' else [10, 50]):
            key = f"{method}_ipc{ipc}"
            print(f"\n{'='*60}")
            print(f"Method: {method}, IPC: {ipc}")
            print(f"{'='*60}")
            
            # Get data
            if method == 'random':
                indices = random_select(train_labels, ipc)
                images = train_images[indices]
                labels = train_labels[indices]
                sl = soft_labels_all[indices]
                
            elif method == 'kcenter':
                indices = kcenters_select(train_images, train_labels, ipc)
                images = train_images[indices]
                labels = train_labels[indices]
                sl = soft_labels_all[indices]
                
            elif method == 'dm':
                cache_path = f'/workspace/distilled_dm_ipc{ipc}_final.pt'
                if os.path.exists(cache_path):
                    data = torch.load(cache_path, weights_only=True)
                    images, labels = data['images'], data['labels']
                else:
                    images, labels = distill_dm(train_images, train_labels, ipc,
                                                num_iters=args.dm_iters)
                    torch.save({'images': images, 'labels': labels}, cache_path)
                # Generate soft labels for distilled images using teacher
                sl = generate_soft_labels_for_images(images)
                
            elif method == 'dc':
                cache_path = f'/workspace/distilled_dc_ipc{ipc}_final.pt'
                if os.path.exists(cache_path):
                    data = torch.load(cache_path, weights_only=True)
                    images, labels = data['images'], data['labels']
                else:
                    images, labels = distill_dc(train_images, train_labels, ipc,
                                                num_iters=args.dc_iters)
                    torch.save({'images': images, 'labels': labels}, cache_path)
                sl = generate_soft_labels_for_images(images)
                
            elif method == 'tm':
                cache_path = f'/workspace/distilled_tm_ipc{ipc}_final.pt'
                if os.path.exists(cache_path):
                    data = torch.load(cache_path, weights_only=True)
                    images, labels = data['images'], data['labels']
                else:
                    images, labels = distill_tm(train_images, train_labels, ipc,
                                                num_iters=args.tm_iters)
                    torch.save({'images': images, 'labels': labels}, cache_path)
                sl = generate_soft_labels_for_images(images)
            
            # Evaluate HL
            print(f"\n  Evaluating HL ({args.num_runs} runs)...")
            hl_mean, hl_std = evaluate_method(
                images, labels, test_images, test_labels,
                soft_labels_all, label_type='hard',
                num_runs=args.num_runs, epochs=300
            )
            
            # Evaluate SL
            print(f"\n  Evaluating SL ({args.num_runs} runs)...")
            sl_mean, sl_std = evaluate_method(
                images, labels, test_images, test_labels,
                soft_labels_all, label_type='soft', soft_labels=sl,
                num_runs=args.num_runs, epochs=300
            )
            
            results[key] = {
                'hl_mean': hl_mean, 'hl_std': hl_std,
                'sl_mean': sl_mean, 'sl_std': sl_std
            }
            
            print(f"\n  {method} IPC={ipc}: HL={hl_mean:.2f}±{hl_std:.2f}, SL={sl_mean:.2f}±{sl_std:.2f}")
            
            # Save intermediate results
            with open('/workspace/results/results_pipeline.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    # Print final table
    print_table(results)
    
    return results


def generate_soft_labels_for_images(images):
    """Generate soft labels for arbitrary images using teacher."""
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load('/workspace/teacher_model.pt', weights_only=True))
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), 512):
            batch = images[i:i+512].to(DEVICE)
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    return torch.cat(all_logits, dim=0)


def print_table(results):
    """Print results in a nice table format."""
    print("\n" + "="*80)
    print("RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("="*80)
    print(f"{'Method':<12} {'IPC':>4} {'HL (ours)':>14} {'HL (paper)':>14} {'SL (ours)':>14} {'SL (paper)':>14}")
    print("-"*80)
    
    paper_results = {
        'dm_ipc10': (29.23, 26.13), 'dm_ipc50': (42.32, 43.46),
        'dc_ipc10': (28.42, 23.54), 'dc_ipc50': (30.56, 33.46),
        'tm_ipc10': (38.18, 37.60), 'tm_ipc50': (46.32, 46.26),
        'random_ipc10': (18.64, 33.43), 'random_ipc50': (34.66, 45.39),
        'kcenter_ipc10': (25.04, 34.70), 'kcenter_ipc50': (38.64, 46.24),
    }
    
    for key in sorted(results.keys()):
        r = results[key]
        method, ipc_str = key.rsplit('_', 1)
        ipc = ipc_str.replace('ipc', '')
        
        paper = paper_results.get(key, (None, None))
        paper_hl = f"{paper[0]:.2f}" if paper[0] else "N/A"
        paper_sl = f"{paper[1]:.2f}" if paper[1] else "N/A"
        
        print(f"{method:<12} {ipc:>4} {r['hl_mean']:>6.2f}±{r['hl_std']:.2f} {paper_hl:>14} {r['sl_mean']:>6.2f}±{r['sl_std']:.2f} {paper_sl:>14}")
    
    print("="*80)
    
    # Save table to file
    with open('/workspace/results/table1.txt', 'w') as f:
        f.write("CIFAR-100, ConvNet-D3 Results\n")
        f.write(f"{'Method':<12} {'IPC':>4} {'HL (ours)':>14} {'HL (paper)':>14} {'SL (ours)':>14} {'SL (paper)':>14}\n")
        for key in sorted(results.keys()):
            r = results[key]
            method, ipc_str = key.rsplit('_', 1)
            ipc = ipc_str.replace('ipc', '')
            paper = paper_results.get(key, (None, None))
            paper_hl = f"{paper[0]:.2f}" if paper[0] else "N/A"
            paper_sl = f"{paper[1]:.2f}" if paper[1] else "N/A"
            f.write(f"{method:<12} {ipc:>4} {r['hl_mean']:>6.2f}±{r['hl_std']:.2f} {paper_hl:>14} {r['sl_mean']:>6.2f}±{r['sl_std']:.2f} {paper_sl:>14}\n")


if __name__ == '__main__':
    main()
