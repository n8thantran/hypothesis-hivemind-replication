#!/usr/bin/env python3
"""
Complete pipeline for replicating Table (CIFAR-100, ConvNet-D3) from
"Rethinking Dataset Distillation: Hard Truths About Soft Labels"

Steps:
1. Train teacher model (for soft labels)
2. Generate soft labels for full training set
3. Select coresets (Random, K-centers via K-means)
4. Distill datasets (DM, DC, TM)
5. Evaluate all methods in HL and SL settings
6. Generate results table
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import sys
from collections import defaultdict

from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 100
RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

DSA_STRATEGY = 'color_crop_cutout_flip_scale_rotate'


def model_fn():
    return ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32))


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def evaluate_model(model, test_images, test_labels, device=DEVICE, batch_size=512):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            imgs = test_images[i:i+batch_size].to(device)
            labs = test_labels[i:i+batch_size].to(device)
            out = model(imgs)
            correct += out.argmax(1).eq(labs).sum().item()
            total += labs.size(0)
    return 100.0 * correct / total


def train_and_eval(train_images, train_labels, test_images, test_labels,
                   label_type='hard', soft_labels=None,
                   epochs=300, batch_size=256, seed=0, verbose=True):
    """Train ConvNet-D3 and return test accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model_fn().to(DEVICE)
    n_train = len(train_images)
    
    if label_type == 'hard':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        temperature = 20.0
    
    eff_bs = min(batch_size, n_train)
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, eff_bs):
            idx = perm[i:i+eff_bs]
            imgs = train_images[idx].to(DEVICE)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            
            optimizer.zero_grad()
            out = model(imgs)
            
            if label_type == 'hard':
                loss = criterion(out, train_labels[idx].to(DEVICE))
            else:
                sl = soft_labels[idx].to(DEVICE)
                log_p = F.log_softmax(out / temperature, dim=1)
                tgt = F.softmax(sl / temperature, dim=1)
                loss = F.kl_div(log_p, tgt, reduction='batchmean') * (temperature ** 2)
            
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            acc = evaluate_model(model, test_images, test_labels)
            print(f"  Epoch {epoch+1}/{epochs}, Acc: {acc:.2f}%")
            model.train()
    
    return evaluate_model(model, test_images, test_labels)


def run_eval(train_images, train_labels, test_images, test_labels,
             label_type='hard', soft_labels=None, num_runs=3, verbose=True):
    """Run multiple trials."""
    accs = []
    for r in range(num_runs):
        if verbose:
            print(f"  Run {r+1}/{num_runs}")
        acc = train_and_eval(train_images, train_labels, test_images, test_labels,
                             label_type=label_type, soft_labels=soft_labels,
                             seed=r, verbose=verbose)
        accs.append(acc)
        if verbose:
            print(f"  Run {r+1} acc: {acc:.2f}%")
    return np.mean(accs), np.std(accs)


# ============================================================
# TEACHER TRAINING
# ============================================================

def train_teacher(train_images, train_labels, test_images, test_labels,
                  epochs=300, save_path='/workspace/teacher_best.pt'):
    """Train a teacher model on full CIFAR-100."""
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location='cpu', weights_only=False)
        print(f"Teacher already trained: {ckpt['accuracy']:.2f}%")
        return ckpt['state_dict'], ckpt['accuracy']
    
    torch.manual_seed(42)
    model = model_fn().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_sd = None
    
    n = len(train_images)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            imgs = train_images[idx].to(DEVICE)
            labs = train_labels[idx].to(DEVICE)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            
            optimizer.zero_grad()
            loss = criterion(model(imgs), labs)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            acc = evaluate_model(model, test_images, test_labels)
            if acc > best_acc:
                best_acc = acc
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Teacher epoch {epoch+1}/{epochs}, acc: {acc:.2f}%, best: {best_acc:.2f}%")
            model.train()
    
    torch.save({'state_dict': best_sd, 'accuracy': best_acc}, save_path)
    print(f"Teacher trained: {best_acc:.2f}%")
    return best_sd, best_acc


def generate_soft_labels(train_images, teacher_sd, save_path='/workspace/soft_labels_teacher.pt'):
    """Generate soft labels (logits) from teacher."""
    if os.path.exists(save_path):
        sl = torch.load(save_path, map_location='cpu', weights_only=False)
        print(f"Soft labels loaded: {sl.shape}")
        return sl
    
    model = model_fn().to(DEVICE)
    model.load_state_dict(teacher_sd)
    model.eval()
    
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(DEVICE)
            logits_list.append(model(batch).cpu())
    
    logits = torch.cat(logits_list, 0)
    torch.save(logits, save_path)
    print(f"Soft labels generated: {logits.shape}")
    return logits


# ============================================================
# CORESET SELECTION
# ============================================================

def random_select(labels, ipc, seed=0):
    np.random.seed(seed)
    ci = get_class_indices(labels, NUM_CLASSES)
    selected = []
    for c in range(NUM_CLASSES):
        chosen = np.random.choice(ci[c], size=ipc, replace=False)
        selected.extend(chosen.tolist())
    return sorted(selected)


def kmeans_select(images, labels, ipc, teacher_sd=None, seed=0):
    """K-centers via K-means: cluster each class in feature space, pick nearest to centroid."""
    np.random.seed(seed)
    
    # Extract features using teacher model
    model = model_fn().to(DEVICE)
    if teacher_sd is not None:
        model.load_state_dict(teacher_sd)
    model.eval()
    
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(DEVICE)
            all_feats.append(model.embed(batch).cpu())
    all_feats = torch.cat(all_feats, 0).numpy()
    
    ci = get_class_indices(labels, NUM_CLASSES)
    selected = []
    
    for c in range(NUM_CLASSES):
        indices = np.array(ci[c])
        feats = all_feats[indices]
        
        # Normalize features
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats_norm = feats / norms
        
        # K-means clustering
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=ipc, random_state=seed, n_init=10, max_iter=300)
        km.fit(feats_norm)
        
        # For each cluster, pick the sample nearest to centroid
        for k in range(ipc):
            cluster_mask = km.labels_ == k
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                # Fallback: random
                cluster_indices = np.array([np.random.randint(len(indices))])
            
            centroid = km.cluster_centers_[k]
            dists = np.sum((feats_norm[cluster_indices] - centroid) ** 2, axis=1)
            nearest = cluster_indices[np.argmin(dists)]
            selected.append(int(indices[nearest]))
    
    return sorted(selected)


# ============================================================
# DATASET DISTILLATION: DM
# ============================================================

def distill_dm(train_images, train_labels, ipc=10, iterations=20000, lr_img=1.0,
               batch_real=256, save_path=None, seed=0):
    """Distribution Matching."""
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, map_location='cpu', weights_only=False)
        print(f"DM IPC={ipc} loaded from {save_path}")
        return data['images'], data['labels']
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    ci = get_class_indices(train_labels, NUM_CLASSES)
    
    # Init from real images
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        perm = np.random.permutation(len(ci[c]))[:ipc]
        for p in perm:
            syn_images.append(train_images[ci[c][p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(DEVICE).requires_grad_(True)
    syn_labels_t = torch.tensor(syn_labels, dtype=torch.long, device=DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    # Pre-organize real data by class
    real_by_class = [train_images[ci[c]] for c in range(NUM_CLASSES)]
    min_class_size = min(len(r) for r in real_by_class)
    batch_per_class = min(batch_real, min_class_size)
    
    print(f"DM IPC={ipc}: {iterations} iterations...")
    t0 = time.time()
    
    for it in range(iterations):
        net = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(DEVICE)
        net.eval()
        
        # Batch all real samples
        real_samples = []
        for c in range(NUM_CLASSES):
            perm = torch.randperm(len(real_by_class[c]))[:batch_per_class]
            real_samples.append(real_by_class[c][perm])
        
        all_real = torch.cat(real_samples, 0).to(DEVICE)
        all_real_aug = DiffAugment(all_real, strategy=DSA_STRATEGY)
        
        with torch.no_grad():
            all_real_feat = net.embed(all_real_aug)
        
        # Real mean features per class
        real_means = []
        offset = 0
        for c in range(NUM_CLASSES):
            real_means.append(all_real_feat[offset:offset+batch_per_class].mean(0))
            offset += batch_per_class
        real_means = torch.stack(real_means)
        
        # Synthetic features
        all_syn_aug = DiffAugment(syn_images, strategy=DSA_STRATEGY)
        all_syn_feat = net.embed(all_syn_aug)
        
        syn_means = []
        for c in range(NUM_CLASSES):
            mask = syn_labels_t == c
            syn_means.append(all_syn_feat[mask].mean(0))
        syn_means = torch.stack(syn_means)
        
        loss = torch.mean((real_means - syn_means) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        del net, all_real, all_real_aug, all_real_feat, all_syn_aug, all_syn_feat
        
        if (it + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.6f}, Time: {elapsed:.0f}s")
    
    result_images = syn_images.detach().cpu()
    result_labels = torch.tensor(syn_labels, dtype=torch.long)
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels


# ============================================================
# DATASET DISTILLATION: DC
# ============================================================

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:
        gwr = gwr.reshape(shape[0], shape[1]*shape[2]*shape[3])
        gws = gws.reshape(shape[0], shape[1]*shape[2]*shape[3])
    elif len(shape) == 3:
        gwr = gwr.reshape(shape[0], shape[1]*shape[2])
        gws = gws.reshape(shape[0], shape[1]*shape[2])
    elif len(shape) == 2:
        pass
    elif len(shape) == 1:
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
    return torch.sum(1 - torch.sum(gwr*gws, dim=-1) / (torch.norm(gwr, dim=-1)*torch.norm(gws, dim=-1) + 1e-6))


def distill_dc(train_images, train_labels, ipc=10, outer_loops=100, inner_loops=1,
               lr_img=1.0, batch_real=256, save_path=None, seed=0):
    """Dataset Condensation via Gradient Matching."""
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, map_location='cpu', weights_only=False)
        print(f"DC IPC={ipc} loaded from {save_path}")
        return data['images'], data['labels']
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    ci = get_class_indices(train_labels, NUM_CLASSES)
    
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        perm = np.random.permutation(len(ci[c]))[:ipc]
        for p in perm:
            syn_images.append(train_images[ci[c][p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(DEVICE).requires_grad_(True)
    syn_labels_t = torch.tensor(syn_labels, dtype=torch.long, device=DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"DC IPC={ipc}: {outer_loops} outer × {inner_loops} inner loops...")
    t0 = time.time()
    
    for ol in range(outer_loops):
        net = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(DEVICE)
        net.train()
        net_opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        
        for il in range(inner_loops):
            loss = torch.tensor(0.0, device=DEVICE)
            
            for c in range(NUM_CLASSES):
                # Real gradients
                real_idx = ci[c]
                perm = np.random.permutation(len(real_idx))[:batch_real]
                real_batch = train_images[np.array(real_idx)[perm]].to(DEVICE)
                real_lab = torch.full((len(perm),), c, dtype=torch.long, device=DEVICE)
                
                real_aug = DiffAugment(real_batch, strategy=DSA_STRATEGY)
                out_real = net(real_aug)
                loss_real = criterion(out_real, real_lab)
                gw_real = torch.autograd.grad(loss_real, net.parameters(), create_graph=False)
                gw_real = [g.detach().clone() for g in gw_real]
                
                # Synthetic gradients
                syn_mask = syn_labels_t == c
                syn_batch = syn_images[syn_mask]
                syn_lab = torch.full((syn_batch.shape[0],), c, dtype=torch.long, device=DEVICE)
                
                syn_aug = DiffAugment(syn_batch, strategy=DSA_STRATEGY)
                out_syn = net(syn_aug)
                loss_syn = criterion(out_syn, syn_lab)
                gw_syn = torch.autograd.grad(loss_syn, net.parameters(), create_graph=True)
                
                for gwr, gws in zip(gw_real, gw_syn):
                    loss += distance_wb(gwr, gws)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update network on synthetic data
            if il < inner_loops - 1:
                syn_aug_net = DiffAugment(syn_images.detach(), strategy=DSA_STRATEGY)
                out_net = net(syn_aug_net)
                loss_net = criterion(out_net, syn_labels_t)
                net_opt.zero_grad()
                loss_net.backward()
                net_opt.step()
        
        del net
        
        if (ol + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Outer {ol+1}/{outer_loops}, Loss: {loss.item():.4f}, Time: {elapsed:.0f}s")
    
    result_images = syn_images.detach().cpu()
    result_labels = torch.tensor(syn_labels, dtype=torch.long)
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels


# ============================================================
# DATASET DISTILLATION: TM
# ============================================================

def train_experts(train_images, train_labels, num_experts=100, expert_epochs=50,
                  save_dir='/workspace/expert_trajectories', seed=0):
    """Train expert trajectories."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if already done
    existing = [f for f in os.listdir(save_dir) if f.startswith('expert_') and f.endswith('.pt')]
    if len(existing) >= num_experts:
        print(f"Expert trajectories already exist ({len(existing)} experts)")
        return save_dir
    
    n = len(train_images)
    criterion = nn.CrossEntropyLoss()
    
    for exp_idx in range(len(existing), num_experts):
        torch.manual_seed(seed + exp_idx * 1000)
        model = model_fn().to(DEVICE)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        trajectory = [model.state_dict()]
        model.train()
        
        for epoch in range(expert_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i+256]
                imgs = train_images[idx].to(DEVICE)
                labs = train_labels[idx].to(DEVICE)
                imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
                
                opt.zero_grad()
                criterion(model(imgs), labs).backward()
                opt.step()
            
            trajectory.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        
        torch.save(trajectory, os.path.join(save_dir, f'expert_{exp_idx}.pt'))
        print(f"  Expert {exp_idx+1}/{num_experts} done")
    
    return save_dir


def distill_tm(train_images, train_labels, ipc=10, expert_dir='/workspace/expert_trajectories',
               num_experts=100, iterations=5000, lr_img=1000.0, lr_lr=1e-5,
               syn_steps=30, expert_epochs=3, max_start_epoch=25,
               save_path=None, seed=0):
    """Trajectory Matching."""
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, map_location='cpu', weights_only=False)
        print(f"TM IPC={ipc} loaded from {save_path}")
        return data['images'], data['labels']
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load experts
    expert_files = sorted([f for f in os.listdir(expert_dir) if f.startswith('expert_')])
    n_exp = min(num_experts, len(expert_files))
    print(f"Loading {n_exp} expert trajectories...")
    
    trajectories = []
    for f in expert_files[:n_exp]:
        traj = torch.load(os.path.join(expert_dir, f), map_location='cpu', weights_only=False)
        trajectories.append(traj)
    
    ci = get_class_indices(train_labels, NUM_CLASSES)
    
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        perm = np.random.permutation(len(ci[c]))[:ipc]
        for p in perm:
            syn_images.append(train_images[ci[c][p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(DEVICE).requires_grad_(True)
    syn_labels_t = torch.tensor(syn_labels, dtype=torch.long, device=DEVICE)
    
    syn_lr = torch.tensor(0.01, device=DEVICE, requires_grad=True)
    
    opt_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    opt_lr = torch.optim.SGD([syn_lr], lr=lr_lr, momentum=0.5)
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"TM IPC={ipc}: {iterations} iterations...")
    t0 = time.time()
    
    for it in range(iterations):
        exp_idx = np.random.randint(n_exp)
        traj = trajectories[exp_idx]
        max_start = min(max_start_epoch, len(traj) - expert_epochs - 1)
        if max_start < 1:
            max_start = 1
        start_epoch = np.random.randint(0, max_start)
        
        start_params = traj[start_epoch]
        target_params = traj[start_epoch + expert_epochs]
        
        student = model_fn().to(DEVICE)
        student.load_state_dict({k: v.to(DEVICE) for k, v in start_params.items()})
        student.train()
        
        n_syn = len(syn_images)
        for step in range(syn_steps):
            perm = torch.randperm(n_syn, device=DEVICE)
            imgs = syn_images[perm]
            labs = syn_labels_t[perm]
            
            imgs_aug = DiffAugment(imgs, strategy=DSA_STRATEGY)
            out = student(imgs_aug)
            loss_s = criterion(out, labs)
            
            grads = torch.autograd.grad(loss_s, student.parameters(), create_graph=True)
            with torch.no_grad():
                for param, grad in zip(student.parameters(), grads):
                    param.sub_(syn_lr * grad)
        
        # Trajectory matching loss
        loss = torch.tensor(0.0, device=DEVICE)
        target_dict = {k: v.to(DEVICE) for k, v in target_params.items()}
        
        for (name, param), (_, target) in zip(student.named_parameters(), target_dict.items()):
            loss += F.mse_loss(param, target, reduction='sum')
        
        num_params = sum(p.numel() for p in student.parameters())
        loss = loss / num_params
        
        opt_img.zero_grad()
        opt_lr.zero_grad()
        loss.backward()
        opt_img.step()
        opt_lr.step()
        
        with torch.no_grad():
            syn_lr.clamp_(min=1e-6)
        
        del student
        
        if (it + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.8f}, lr: {syn_lr.item():.6f}, Time: {elapsed:.0f}s")
    
    result_images = syn_images.detach().cpu()
    result_labels = torch.tensor(syn_labels, dtype=torch.long)
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels


# ============================================================
# SOFT LABEL GENERATION FOR DISTILLED/CORESET DATA
# ============================================================

def get_soft_labels_for_subset(subset_images, teacher_sd):
    """Get teacher logits for a subset of images."""
    model = model_fn().to(DEVICE)
    model.load_state_dict(teacher_sd)
    model.eval()
    
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(subset_images), 256):
            batch = subset_images[i:i+256].to(DEVICE)
            logits_list.append(model(batch).cpu())
    
    return torch.cat(logits_list, 0)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'teacher', 'coresets', 'dm', 'dc', 'tm', 'eval', 'eval_only'])
    parser.add_argument('--ipc', type=int, default=0, help='0 = both 10 and 50')
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--dm_iters', type=int, default=20000)
    parser.add_argument('--dc_outer', type=int, default=100)
    parser.add_argument('--dc_inner', type=int, default=1)
    parser.add_argument('--tm_iters', type=int, default=5000)
    parser.add_argument('--num_experts', type=int, default=10)
    parser.add_argument('--expert_epochs', type=int, default=50)
    args = parser.parse_args()
    
    print("=" * 60)
    print("CIFAR-100 Dataset Distillation Replication Pipeline")
    print("=" * 60)
    
    # Load data
    print("\nLoading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    ipcs = [10, 50] if args.ipc == 0 else [args.ipc]
    
    results = {}
    
    # Step 1: Teacher
    if args.step in ['all', 'teacher']:
        print("\n" + "=" * 40)
        print("STEP 1: Train Teacher")
        print("=" * 40)
        teacher_sd, teacher_acc = train_teacher(train_images, train_labels, test_images, test_labels)
    else:
        ckpt = torch.load('/workspace/teacher_best.pt', map_location='cpu', weights_only=False)
        teacher_sd = ckpt['state_dict']
        teacher_acc = ckpt['accuracy']
        print(f"Teacher loaded: {teacher_acc:.2f}%")
    
    # Generate soft labels for full training set
    full_soft_labels = generate_soft_labels(train_images, teacher_sd)
    
    # Step 2: Coresets
    if args.step in ['all', 'coresets', 'eval', 'eval_only']:
        print("\n" + "=" * 40)
        print("STEP 2: Coreset Selection & Evaluation")
        print("=" * 40)
        
        for ipc in ipcs:
            # Random
            print(f"\n--- Random IPC={ipc} ---")
            rand_idx = random_select(train_labels, ipc, seed=0)
            rand_imgs = train_images[rand_idx]
            rand_labs = train_labels[rand_idx]
            rand_sl = full_soft_labels[rand_idx]
            
            print("  Evaluating HL...")
            hl_mean, hl_std = run_eval(rand_imgs, rand_labs, test_images, test_labels,
                                        label_type='hard', num_runs=args.num_runs)
            print("  Evaluating SL...")
            sl_mean, sl_std = run_eval(rand_imgs, rand_labs, test_images, test_labels,
                                        label_type='soft', soft_labels=rand_sl, num_runs=args.num_runs)
            
            results[f'random_ipc{ipc}'] = {
                'hl_mean': hl_mean, 'hl_std': hl_std,
                'sl_mean': sl_mean, 'sl_std': sl_std
            }
            print(f"  Random IPC={ipc}: HL={hl_mean:.2f}±{hl_std:.2f}, SL={sl_mean:.2f}±{sl_std:.2f}")
            
            # K-centers (K-means)
            print(f"\n--- K-centers IPC={ipc} ---")
            kc_idx = kmeans_select(train_images, train_labels, ipc, teacher_sd=teacher_sd, seed=0)
            kc_imgs = train_images[kc_idx]
            kc_labs = train_labels[kc_idx]
            kc_sl = full_soft_labels[kc_idx]
            
            print("  Evaluating HL...")
            hl_mean, hl_std = run_eval(kc_imgs, kc_labs, test_images, test_labels,
                                        label_type='hard', num_runs=args.num_runs)
            print("  Evaluating SL...")
            sl_mean, sl_std = run_eval(kc_imgs, kc_labs, test_images, test_labels,
                                        label_type='soft', soft_labels=kc_sl, num_runs=args.num_runs)
            
            results[f'kcenter_ipc{ipc}'] = {
                'hl_mean': hl_mean, 'hl_std': hl_std,
                'sl_mean': sl_mean, 'sl_std': sl_std
            }
            print(f"  K-centers IPC={ipc}: HL={hl_mean:.2f}±{hl_std:.2f}, SL={sl_mean:.2f}±{sl_std:.2f}")
    
    # Step 3: DM
    if args.step in ['all', 'dm']:
        print("\n" + "=" * 40)
        print("STEP 3: Distribution Matching (DM)")
        print("=" * 40)
        
        for ipc in ipcs:
            save_p = f'/workspace/distilled_dm_ipc{ipc}_final.pt'
            dm_imgs, dm_labs = distill_dm(train_images, train_labels, ipc=ipc,
                                           iterations=args.dm_iters, save_path=save_p)
    
    # Step 4: DC
    if args.step in ['all', 'dc']:
        print("\n" + "=" * 40)
        print("STEP 4: Gradient Matching (DC)")
        print("=" * 40)
        
        for ipc in ipcs:
            save_p = f'/workspace/distilled_dc_ipc{ipc}_final.pt'
            dc_imgs, dc_labs = distill_dc(train_images, train_labels, ipc=ipc,
                                           outer_loops=args.dc_outer, inner_loops=args.dc_inner,
                                           save_path=save_p)
    
    # Step 5: TM
    if args.step in ['all', 'tm']:
        print("\n" + "=" * 40)
        print("STEP 5: Trajectory Matching (TM)")
        print("=" * 40)
        
        # Train experts first
        train_experts(train_images, train_labels, num_experts=args.num_experts,
                      expert_epochs=args.expert_epochs)
        
        for ipc in ipcs:
            save_p = f'/workspace/distilled_tm_ipc{ipc}_final.pt'
            tm_imgs, tm_labs = distill_tm(train_images, train_labels, ipc=ipc,
                                           num_experts=args.num_experts,
                                           iterations=args.tm_iters,
                                           save_path=save_p)
    
    # Step 6: Evaluate DD methods
    if args.step in ['all', 'eval', 'eval_only']:
        print("\n" + "=" * 40)
        print("STEP 6: Evaluate DD Methods")
        print("=" * 40)
        
        for method in ['dm', 'dc', 'tm']:
            for ipc in ipcs:
                # Try to load distilled data
                save_p = f'/workspace/distilled_{method}_ipc{ipc}_final.pt'
                if not os.path.exists(save_p):
                    # Try older versions
                    save_p = f'/workspace/distilled_{method}_ipc{ipc}.pt'
                if not os.path.exists(save_p):
                    print(f"  {method.upper()} IPC={ipc}: no distilled data found, skipping")
                    continue
                
                data = torch.load(save_p, map_location='cpu', weights_only=False)
                if isinstance(data, dict):
                    dd_imgs = data['images']
                    dd_labs = data['labels']
                else:
                    dd_imgs, dd_labs = data
                
                print(f"\n--- {method.upper()} IPC={ipc} ---")
                
                # Generate soft labels for distilled data
                dd_sl = get_soft_labels_for_subset(dd_imgs, teacher_sd)
                
                print("  Evaluating HL...")
                hl_mean, hl_std = run_eval(dd_imgs, dd_labs, test_images, test_labels,
                                            label_type='hard', num_runs=args.num_runs)
                print("  Evaluating SL...")
                sl_mean, sl_std = run_eval(dd_imgs, dd_labs, test_images, test_labels,
                                            label_type='soft', soft_labels=dd_sl, num_runs=args.num_runs)
                
                results[f'{method}_ipc{ipc}'] = {
                    'hl_mean': hl_mean, 'hl_std': hl_std,
                    'sl_mean': sl_mean, 'sl_std': sl_std
                }
                print(f"  {method.upper()} IPC={ipc}: HL={hl_mean:.2f}±{hl_std:.2f}, SL={sl_mean:.2f}±{sl_std:.2f}")
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'results_final.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate table
    print("\n" + "=" * 60)
    print("RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("=" * 60)
    
    table_lines = []
    header = f"{'Method':<15} {'IPC':>4} {'HL (ours)':>12} {'HL (paper)':>12} {'SL (ours)':>12} {'SL (paper)':>12}"
    table_lines.append(header)
    table_lines.append("-" * len(header))
    
    paper_results = {
        'dm_ipc10': (29.23, 0.26, 26.13, 0.10),
        'dm_ipc50': (42.32, 0.37, 43.46, 0.18),
        'dc_ipc10': (28.42, 0.29, 23.54, 0.31),
        'dc_ipc50': (30.56, 0.56, 33.46, 0.38),
        'tm_ipc10': (38.18, 0.42, 37.60, 0.25),
        'tm_ipc50': (46.32, 0.26, 46.26, 0.30),
        'random_ipc10': (18.64, 0.25, 33.43, 0.18),
        'random_ipc50': (34.66, 0.41, 45.39, 0.23),
        'kcenter_ipc10': (25.04, 0.30, 34.70, 0.27),
        'kcenter_ipc50': (38.64, 0.43, 46.24, 0.12),
    }
    
    display_order = [
        ('DM', 'dm'), ('DC', 'dc'), ('TM', 'tm'),
        ('Random', 'random'), ('K-centers', 'kcenter')
    ]
    
    for name, key in display_order:
        for ipc in [10, 50]:
            rkey = f'{key}_ipc{ipc}'
            if rkey in results:
                r = results[rkey]
                ours_hl = f"{r['hl_mean']:.2f}±{r['hl_std']:.2f}"
                ours_sl = f"{r['sl_mean']:.2f}±{r['sl_std']:.2f}"
            else:
                ours_hl = "N/A"
                ours_sl = "N/A"
            
            if rkey in paper_results:
                p = paper_results[rkey]
                paper_hl = f"{p[0]:.2f}±{p[1]:.2f}"
                paper_sl = f"{p[2]:.2f}±{p[3]:.2f}"
            else:
                paper_hl = "N/A"
                paper_sl = "N/A"
            
            line = f"{name:<15} {ipc:>4} {ours_hl:>12} {paper_hl:>12} {ours_sl:>12} {paper_sl:>12}"
            table_lines.append(line)
    
    table_str = "\n".join(table_lines)
    print(table_str)
    
    with open(os.path.join(RESULTS_DIR, 'table_final.txt'), 'w') as f:
        f.write(table_str)
    
    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Done!")


if __name__ == '__main__':
    main()
