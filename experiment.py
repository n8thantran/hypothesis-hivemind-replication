"""
Clean, comprehensive experiment script for replicating Table small_scale_c100.
Paper: "Rethinking Dataset Distillation: Hard Truths About Soft Labels"

Target: CIFAR-100, ConvNet-D3, IPC 10 and IPC 50
Methods: DM, DC, TM (DD), Random, K-centers (coresets)
Settings: Hard Label (HL) and Soft Label (SL)

Hyperparameters from paper's Table stage3_hyper:
  HL: 300 epochs, CE loss, SGD lr=0.01, StepLR@151, batch=256, DSA
  SL: 300 epochs, KL-Div(T=20), AdamW lr=1e-3, Cosine, batch=256, DSA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
import time
import copy
import argparse
from torch.utils.data import DataLoader, TensorDataset
from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 100
IM_SIZE = (32, 32)
CHANNEL = 3
DSA_STRATEGY = 'color_crop_cutout_flip_scale_rotate'


# ============================================================
# Data Loading
# ============================================================
def load_cifar100():
    """Load CIFAR-100 train and test data as tensors."""
    import torchvision
    import torchvision.transforms as transforms
    
    # Load raw data
    trainset = torchvision.datasets.CIFAR100(root='/workspace/data', train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root='/workspace/data', train=False, download=True)
    
    # Convert to tensors [N, 3, 32, 32] in [0, 1]
    train_images = torch.tensor(trainset.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    train_labels = torch.tensor(trainset.targets, dtype=torch.long)
    test_images = torch.tensor(testset.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    test_labels = torch.tensor(testset.targets, dtype=torch.long)
    
    # Normalize with CIFAR-100 mean/std
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std
    
    return train_images, train_labels, test_images, test_labels


def get_class_indices(labels, num_classes=100):
    """Get indices for each class."""
    indices = {}
    for c in range(num_classes):
        indices[c] = (labels == c).nonzero(as_tuple=True)[0]
    return indices


# ============================================================
# Coreset Selection
# ============================================================
def select_random(train_images, train_labels, ipc, seed=42):
    """Random coreset selection."""
    rng = np.random.RandomState(seed)
    class_indices = get_class_indices(train_labels)
    selected_images = []
    selected_labels = []
    
    for c in range(NUM_CLASSES):
        idx = class_indices[c].numpy()
        chosen = rng.choice(idx, size=ipc, replace=False)
        selected_images.append(train_images[chosen])
        selected_labels.append(train_labels[chosen])
    
    return torch.cat(selected_images), torch.cat(selected_labels)


def select_k_centers(train_images, train_labels, ipc, seed=42):
    """
    K-centers coreset selection using feature embeddings.
    Following DeepCore: train a model, extract features, then use K-center greedy.
    K-center greedy: iteratively select the point that is farthest from the already selected set.
    This is the standard facility location / K-center algorithm.
    """
    print("K-centers: Training feature extractor...")
    torch.manual_seed(seed)
    
    # Train a simple model to get features
    model = get_convnet_d3().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # Quick training - 50 epochs with basic augmentation
    dataset = TensorDataset(train_images, train_labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    model.train()
    for epoch in range(30):
        for imgs, labs in loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            # Simple augmentation: random flip
            if torch.rand(1).item() > 0.5:
                imgs = imgs.flip(3)
            out = model(imgs)
            loss = F.cross_entropy(out, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Extract features
    print("K-centers: Extracting features...")
    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(DEVICE)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    all_features = torch.cat(all_features)  # [50000, feat_dim]
    
    # Normalize features
    all_features = F.normalize(all_features, dim=1)
    
    # K-center greedy per class
    print("K-centers: Running K-center greedy selection...")
    class_indices = get_class_indices(train_labels)
    selected_images = []
    selected_labels = []
    
    for c in range(NUM_CLASSES):
        idx = class_indices[c].numpy()
        feats = all_features[idx]  # [N_c, feat_dim]
        N_c = len(idx)
        
        # K-center greedy: select point farthest from current selected set
        # Start with the point closest to the mean (most representative)
        mean_feat = feats.mean(dim=0, keepdim=True)
        dists_to_mean = torch.cdist(feats, mean_feat).squeeze()
        first = dists_to_mean.argmin().item()
        
        selected = [first]
        # min_dists[i] = min distance from point i to any selected point
        min_dists = torch.cdist(feats, feats[first:first+1]).squeeze()
        
        for _ in range(ipc - 1):
            # Select the point with maximum min-distance to selected set
            # But mask already selected
            mask = torch.ones(N_c, dtype=torch.bool)
            for s in selected:
                mask[s] = False
            masked_dists = min_dists.clone()
            masked_dists[~mask] = -1
            new_idx = masked_dists.argmax().item()
            selected.append(new_idx)
            
            # Update min_dists
            new_dists = torch.cdist(feats, feats[new_idx:new_idx+1]).squeeze()
            min_dists = torch.min(min_dists, new_dists)
        
        chosen_global = idx[selected]
        selected_images.append(train_images[chosen_global])
        selected_labels.append(train_labels[chosen_global])
    
    return torch.cat(selected_images), torch.cat(selected_labels)


# ============================================================
# Dataset Distillation Methods
# ============================================================
def distill_dm(train_images, train_labels, ipc, num_iters=20000, lr=0.01, seed=42):
    """Distribution Matching (DM) distillation."""
    print(f"DM distillation: IPC={ipc}, iters={num_iters}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels)
    
    # Initialize synthetic data from random real images
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        idx = class_indices[c].numpy()
        chosen = np.random.choice(idx, size=ipc, replace=False)
        syn_images.append(train_images[chosen].clone())
        syn_labels.append(torch.full((ipc,), c, dtype=torch.long))
    
    syn_images = torch.cat(syn_images).to(DEVICE).requires_grad_(True)
    syn_labels = torch.cat(syn_labels).to(DEVICE)
    
    optimizer = optim.SGD([syn_images], lr=lr, momentum=0.5)
    
    for it in range(num_iters):
        # Sample a random model
        model = get_convnet_d3().to(DEVICE)
        model.eval()
        
        # Compute loss: match mean embeddings per class
        loss = torch.tensor(0.0, device=DEVICE)
        for c in range(NUM_CLASSES):
            # Real images for this class
            idx = class_indices[c]
            real_batch_idx = idx[torch.randperm(len(idx))[:ipc * 2]]
            real_batch = train_images[real_batch_idx].to(DEVICE)
            
            # Synthetic images for this class
            syn_c = syn_images[syn_labels == c]
            
            # Apply DSA
            seed_aug = int(torch.randint(0, 100000, (1,)).item())
            real_aug = DiffAugment(real_batch, strategy=DSA_STRATEGY, seed=seed_aug)
            syn_aug = DiffAugment(syn_c, strategy=DSA_STRATEGY, seed=seed_aug)
            
            with torch.no_grad():
                real_feat = model.embed(real_aug)
            syn_feat = model.embed(syn_aug)
            
            # MMD loss (mean matching)
            loss += torch.mean((real_feat.mean(0) - syn_feat.mean(0)) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 2000 == 0:
            print(f"  DM iter {it+1}/{num_iters}, loss={loss.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_dc(train_images, train_labels, ipc, num_iters=5000, lr=0.01, seed=42):
    """Dataset Condensation via Gradient Matching (DC)."""
    print(f"DC distillation: IPC={ipc}, iters={num_iters}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels)
    
    # Initialize synthetic data
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        idx = class_indices[c].numpy()
        chosen = np.random.choice(idx, size=ipc, replace=False)
        syn_images.append(train_images[chosen].clone())
        syn_labels.append(torch.full((ipc,), c, dtype=torch.long))
    
    syn_images = torch.cat(syn_images).to(DEVICE).requires_grad_(True)
    syn_labels = torch.cat(syn_labels).to(DEVICE)
    
    optimizer = optim.SGD([syn_images], lr=lr, momentum=0.5)
    
    for it in range(num_iters):
        model = get_convnet_d3().to(DEVICE)
        model.train()
        
        loss = torch.tensor(0.0, device=DEVICE)
        
        for c in range(NUM_CLASSES):
            # Real batch
            idx = class_indices[c]
            real_idx = idx[torch.randperm(len(idx))[:256]]
            real_batch = train_images[real_idx].to(DEVICE)
            real_lab = train_labels[real_idx].to(DEVICE)
            
            # Synthetic batch
            syn_c = syn_images[syn_labels == c]
            syn_lab_c = syn_labels[syn_labels == c]
            
            # DSA
            seed_aug = int(torch.randint(0, 100000, (1,)).item())
            real_aug = DiffAugment(real_batch, strategy=DSA_STRATEGY, seed=seed_aug)
            syn_aug = DiffAugment(syn_c, strategy=DSA_STRATEGY, seed=seed_aug)
            
            # Compute gradients
            out_real = model(real_aug)
            loss_real = F.cross_entropy(out_real, real_lab)
            grad_real = torch.autograd.grad(loss_real, model.parameters(), create_graph=False)
            
            out_syn = model(syn_aug)
            loss_syn = F.cross_entropy(out_syn, syn_lab_c)
            grad_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)
            
            # Match gradients (cosine distance)
            for gr, gs in zip(grad_real, grad_syn):
                gr = gr.detach()
                cos_sim = F.cosine_similarity(gr.flatten().unsqueeze(0), 
                                               gs.flatten().unsqueeze(0))
                loss += (1 - cos_sim)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 1000 == 0:
            print(f"  DC iter {it+1}/{num_iters}, loss={loss.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_tm(train_images, train_labels, ipc, num_iters=5000, lr=0.01, 
               expert_dir='/workspace/expert_trajectories', num_experts=5, seed=42):
    """Trajectory Matching (TM) distillation."""
    print(f"TM distillation: IPC={ipc}, iters={num_iters}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels)
    
    # First, generate expert trajectories if they don't exist
    if not os.path.exists(expert_dir):
        os.makedirs(expert_dir, exist_ok=True)
        print("  Generating expert trajectories...")
        generate_expert_trajectories(train_images, train_labels, expert_dir, 
                                      num_experts=num_experts)
    
    # Load expert trajectories
    expert_trajectories = []
    for f in sorted(os.listdir(expert_dir)):
        if f.endswith('.pt'):
            traj = torch.load(os.path.join(expert_dir, f), map_location='cpu')
            expert_trajectories.append(traj)
    
    if len(expert_trajectories) == 0:
        print("  No expert trajectories found, generating...")
        generate_expert_trajectories(train_images, train_labels, expert_dir, 
                                      num_experts=num_experts)
        for f in sorted(os.listdir(expert_dir)):
            if f.endswith('.pt'):
                traj = torch.load(os.path.join(expert_dir, f), map_location='cpu')
                expert_trajectories.append(traj)
    
    print(f"  Loaded {len(expert_trajectories)} expert trajectories")
    
    # Initialize synthetic data
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        idx = class_indices[c].numpy()
        chosen = np.random.choice(idx, size=ipc, replace=False)
        syn_images.append(train_images[chosen].clone())
        syn_labels.append(torch.full((ipc,), c, dtype=torch.long))
    
    syn_images = torch.cat(syn_images).to(DEVICE).requires_grad_(True)
    syn_labels = torch.cat(syn_labels).to(DEVICE)
    
    optimizer = optim.SGD([syn_images], lr=lr, momentum=0.5)
    
    # TM parameters
    expert_epochs = 3  # Number of expert steps to match
    syn_steps = 50     # Number of student steps on synthetic data
    lr_student = 0.01
    
    for it in range(num_iters):
        # Pick random expert trajectory and starting point
        expert_idx = np.random.randint(len(expert_trajectories))
        traj = expert_trajectories[expert_idx]
        max_start = len(traj) - expert_epochs - 1
        if max_start < 1:
            max_start = 1
        start_epoch = np.random.randint(0, max_start)
        
        # Get starting and target parameters
        start_params = traj[start_epoch]
        target_params = traj[start_epoch + expert_epochs]
        
        # Initialize student from expert starting point
        student = get_convnet_d3().to(DEVICE)
        with torch.no_grad():
            for p, sp in zip(student.parameters(), start_params):
                p.copy_(sp.to(DEVICE))
        
        student_opt = optim.SGD(student.parameters(), lr=lr_student, momentum=0.9)
        
        # Train student on synthetic data for syn_steps
        student.train()
        for step in range(syn_steps):
            # Random subset of synthetic data
            perm = torch.randperm(len(syn_images))[:256]
            batch_img = syn_images[perm]
            batch_lab = syn_labels[perm]
            
            seed_aug = int(torch.randint(0, 100000, (1,)).item())
            batch_aug = DiffAugment(batch_img, strategy=DSA_STRATEGY, seed=seed_aug)
            
            out = student(batch_aug)
            loss_ce = F.cross_entropy(out, batch_lab)
            student_opt.zero_grad()
            loss_ce.backward()
            student_opt.step()
        
        # Match student params to target expert params
        loss = torch.tensor(0.0, device=DEVICE)
        for p, tp in zip(student.parameters(), target_params):
            tp = tp.to(DEVICE)
            loss += F.mse_loss(p, tp, reduction='sum')
        
        # Normalize by number of params
        num_params = sum(p.numel() for p in student.parameters())
        loss = loss / num_params
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 500 == 0:
            print(f"  TM iter {it+1}/{num_iters}, loss={loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def generate_expert_trajectories(train_images, train_labels, save_dir, num_experts=5):
    """Generate expert training trajectories for TM."""
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = TensorDataset(train_images, train_labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    for exp_id in range(num_experts):
        print(f"  Training expert {exp_id+1}/{num_experts}...")
        torch.manual_seed(exp_id)
        model = get_convnet_d3().to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        trajectory = []
        # Save initial params
        trajectory.append([p.detach().cpu().clone() for p in model.parameters()])
        
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            for imgs, labs in loader:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
                out = model(imgs)
                loss = F.cross_entropy(out, labs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Save params at end of each epoch
            trajectory.append([p.detach().cpu().clone() for p in model.parameters()])
        
        torch.save(trajectory, os.path.join(save_dir, f'expert_{exp_id}.pt'))
        print(f"  Expert {exp_id+1} done, trajectory length: {len(trajectory)}")


# ============================================================
# Teacher Model & Soft Labels
# ============================================================
def train_teacher(train_images, train_labels, test_images, test_labels, epochs=200):
    """Train a teacher model for soft label generation."""
    print("Training teacher model...")
    model = get_convnet_d3().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset = TensorDataset(train_images, train_labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for imgs, labs in loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            # Basic augmentation
            if torch.rand(1).item() > 0.5:
                imgs = imgs.flip(3)
            out = model(imgs)
            loss = F.cross_entropy(out, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            acc = evaluate_accuracy(model, test_images, test_labels)
            print(f"  Teacher epoch {epoch+1}/{epochs}, acc={acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_state)
    print(f"  Best teacher accuracy: {best_acc:.2f}%")
    return model


def generate_soft_labels(model, images, temperature=20.0, batch_size=256):
    """Generate soft labels from teacher model."""
    model.eval()
    all_soft = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(DEVICE)
            logits = model(batch)
            soft = F.softmax(logits / temperature, dim=1)
            all_soft.append(soft.cpu())
    return torch.cat(all_soft)


# ============================================================
# Evaluation (Student Training)
# ============================================================
def evaluate_accuracy(model, test_images, test_labels, batch_size=256):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            imgs = test_images[i:i+batch_size].to(DEVICE)
            labs = test_labels[i:i+batch_size].to(DEVICE)
            out = model(imgs)
            pred = out.argmax(dim=1)
            correct += (pred == labs).sum().item()
            total += labs.size(0)
    return 100.0 * correct / total


def train_student_hl(syn_images, syn_labels, test_images, test_labels, 
                     epochs=300, lr=0.01, batch_size=256, seed=0):
    """
    Train student with Hard Labels.
    Paper: 300 epochs, SGD, lr=0.01, StepLR@151, batch=256, DSA, CE loss
    """
    torch.manual_seed(seed)
    model = get_convnet_d3().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    dataset = TensorDataset(syn_images, syn_labels)
    # If dataset is smaller than batch_size, use full dataset
    actual_bs = min(batch_size, len(syn_images))
    loader = DataLoader(dataset, batch_size=actual_bs, shuffle=True, num_workers=0, drop_last=False)
    
    for epoch in range(epochs):
        model.train()
        for imgs, labs in loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            out = model(imgs)
            loss = F.cross_entropy(out, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    acc = evaluate_accuracy(model, test_images, test_labels)
    return acc


def train_student_sl(syn_images, soft_labels, test_images, test_labels,
                     epochs=300, lr=1e-3, batch_size=256, temperature=20.0, seed=0):
    """
    Train student with Soft Labels.
    Paper: 300 epochs, AdamW, lr=1e-3, Cosine, batch=256, DSA, KL-Div(T=20)
    """
    torch.manual_seed(seed)
    model = get_convnet_d3().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset = TensorDataset(syn_images, soft_labels)
    actual_bs = min(batch_size, len(syn_images))
    loader = DataLoader(dataset, batch_size=actual_bs, shuffle=True, num_workers=0, drop_last=False)
    
    for epoch in range(epochs):
        model.train()
        for imgs, targets in loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            logits = model(imgs)
            
            # KL-Div loss with temperature
            log_probs = F.log_softmax(logits / temperature, dim=1)
            loss = F.kl_div(log_probs, targets, reduction='batchmean') * (temperature ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    acc = evaluate_accuracy(model, test_images, test_labels)
    return acc


def run_evaluation(syn_images, syn_labels, soft_labels, test_images, test_labels,
                   label, num_runs=3):
    """Run HL and SL evaluation with multiple seeds."""
    results = {}
    
    # HL evaluation
    hl_accs = []
    for run in range(num_runs):
        acc = train_student_hl(syn_images, syn_labels, test_images, test_labels, seed=run)
        hl_accs.append(acc)
        print(f"  {label} HL run {run+1}: {acc:.2f}%")
    
    results[f'{label}_hl'] = {
        'mean': np.mean(hl_accs),
        'std': np.std(hl_accs),
        'accs': hl_accs
    }
    
    # SL evaluation
    sl_accs = []
    for run in range(num_runs):
        acc = train_student_sl(syn_images, soft_labels, test_images, test_labels, seed=run)
        sl_accs.append(acc)
        print(f"  {label} SL run {run+1}: {acc:.2f}%")
    
    results[f'{label}_sl'] = {
        'mean': np.mean(sl_accs),
        'std': np.std(sl_accs),
        'accs': sl_accs
    }
    
    return results


# ============================================================
# Main Experiment Runner
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='random', 
                        choices=['random', 'k_centers', 'dm', 'dc', 'tm', 'all_coresets', 'all'])
    parser.add_argument('--ipc', type=int, default=10)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--setting', type=str, default='both', choices=['hl', 'sl', 'both'])
    parser.add_argument('--dm_iters', type=int, default=20000)
    parser.add_argument('--dc_iters', type=int, default=5000)
    parser.add_argument('--tm_iters', type=int, default=5000)
    args = parser.parse_args()
    
    print(f"Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = load_cifar100()
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Load or train teacher for soft labels
    teacher_path = '/workspace/teacher_model.pt'
    soft_labels_path = '/workspace/teacher_soft_labels.pt'
    
    if os.path.exists(soft_labels_path):
        print("Loading cached soft labels...")
        all_soft_labels = torch.load(soft_labels_path, map_location='cpu')
    else:
        if os.path.exists(teacher_path):
            print("Loading cached teacher model...")
            teacher = get_convnet_d3().to(DEVICE)
            teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
        else:
            teacher = train_teacher(train_images, train_labels, test_images, test_labels)
            torch.save(teacher.state_dict(), teacher_path)
        
        print("Generating soft labels for full training set...")
        all_soft_labels = generate_soft_labels(teacher, train_images, temperature=20.0)
        torch.save(all_soft_labels, soft_labels_path)
    
    print(f"  Soft labels shape: {all_soft_labels.shape}")
    
    all_results = {}
    
    methods_to_run = []
    if args.method == 'all_coresets':
        methods_to_run = ['random', 'k_centers']
    elif args.method == 'all':
        methods_to_run = ['random', 'k_centers', 'dm', 'dc', 'tm']
    else:
        methods_to_run = [args.method]
    
    for method in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Method: {method}, IPC: {args.ipc}")
        print(f"{'='*60}")
        
        if method == 'random':
            syn_images, syn_labels = select_random(train_images, train_labels, args.ipc)
        elif method == 'k_centers':
            syn_images, syn_labels = select_k_centers(train_images, train_labels, args.ipc)
        elif method == 'dm':
            cache_path = f'/workspace/distilled_dm_ipc{args.ipc}_v2.pt'
            if os.path.exists(cache_path):
                data = torch.load(cache_path, map_location='cpu')
                syn_images, syn_labels = data['images'], data['labels']
            else:
                syn_images, syn_labels = distill_dm(train_images, train_labels, args.ipc, 
                                                     num_iters=args.dm_iters)
                torch.save({'images': syn_images, 'labels': syn_labels}, cache_path)
        elif method == 'dc':
            cache_path = f'/workspace/distilled_dc_ipc{args.ipc}_v2.pt'
            if os.path.exists(cache_path):
                data = torch.load(cache_path, map_location='cpu')
                syn_images, syn_labels = data['images'], data['labels']
            else:
                syn_images, syn_labels = distill_dc(train_images, train_labels, args.ipc,
                                                     num_iters=args.dc_iters)
                torch.save({'images': syn_images, 'labels': syn_labels}, cache_path)
        elif method == 'tm':
            cache_path = f'/workspace/distilled_tm_ipc{args.ipc}_v2.pt'
            if os.path.exists(cache_path):
                data = torch.load(cache_path, map_location='cpu')
                syn_images, syn_labels = data['images'], data['labels']
            else:
                syn_images, syn_labels = distill_tm(train_images, train_labels, args.ipc,
                                                     num_iters=args.tm_iters)
                torch.save({'images': syn_images, 'labels': syn_labels}, cache_path)
        
        print(f"  Synthetic data: {syn_images.shape}")
        
        # Get soft labels for selected/synthetic images
        if method in ['random', 'k_centers']:
            # For coresets, get soft labels from teacher for the selected real images
            # Need to find which indices were selected
            # Easier: just generate soft labels for the selected images
            teacher = get_convnet_d3().to(DEVICE)
            if os.path.exists(teacher_path):
                teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
            soft_labels = generate_soft_labels(teacher, syn_images, temperature=20.0)
        else:
            # For DD methods, generate soft labels from teacher for synthetic images
            teacher = get_convnet_d3().to(DEVICE)
            if os.path.exists(teacher_path):
                teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
            soft_labels = generate_soft_labels(teacher, syn_images, temperature=20.0)
        
        # Run evaluation
        if args.setting == 'hl':
            hl_accs = []
            for run in range(args.num_runs):
                acc = train_student_hl(syn_images, syn_labels, test_images, test_labels, seed=run)
                hl_accs.append(acc)
                print(f"  {method} IPC{args.ipc} HL run {run+1}: {acc:.2f}%")
            all_results[f'{method}_ipc{args.ipc}_hl'] = {
                'mean': np.mean(hl_accs), 'std': np.std(hl_accs), 'accs': hl_accs
            }
        elif args.setting == 'sl':
            sl_accs = []
            for run in range(args.num_runs):
                acc = train_student_sl(syn_images, soft_labels, test_images, test_labels, seed=run)
                sl_accs.append(acc)
                print(f"  {method} IPC{args.ipc} SL run {run+1}: {acc:.2f}%")
            all_results[f'{method}_ipc{args.ipc}_sl'] = {
                'mean': np.mean(sl_accs), 'std': np.std(sl_accs), 'accs': sl_accs
            }
        else:
            results = run_evaluation(syn_images, syn_labels, soft_labels, 
                                     test_images, test_labels, 
                                     f'{method}_ipc{args.ipc}', args.num_runs)
            all_results.update(results)
    
    # Save results
    os.makedirs('/workspace/results', exist_ok=True)
    result_file = f'/workspace/results/exp_{args.method}_ipc{args.ipc}_{args.setting}.json'
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for key, val in all_results.items():
        print(f"  {key}: {val['mean']:.2f} ± {val['std']:.2f}")
    
    return all_results


if __name__ == '__main__':
    main()
