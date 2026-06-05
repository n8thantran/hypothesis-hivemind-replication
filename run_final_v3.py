"""
Final comprehensive experiment runner.
Trains teacher, generates soft labels, runs all DD methods, evaluates everything.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import sys

from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices
from train_eval import train_and_evaluate, evaluate

device = 'cuda'
NUM_CLASSES = 100
CHANNEL = 3
IM_SIZE = (32, 32)

# ============================================================
# Step 0: Load Data
# ============================================================
def load_data():
    print("=" * 60)
    print("STEP 0: Loading CIFAR-100 data")
    print("=" * 60)
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    return train_images, train_labels, test_images, test_labels


# ============================================================
# Step 1: Train Teacher (for soft labels)
# ============================================================
def train_teacher(train_images, train_labels, test_images, test_labels,
                  epochs=200, batch_size=256, save_path='teacher_v3.pt'):
    """Train a ConvNet-D3 teacher with heavy augmentation."""
    print("=" * 60)
    print(f"STEP 1: Training teacher for {epochs} epochs")
    print("=" * 60)
    
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, weights_only=False)
        if ckpt.get('accuracy', 0) >= 58.0:
            print(f"Teacher already trained: {ckpt['accuracy']:.2f}%")
            return ckpt['state_dict'], ckpt['accuracy']
    
    model = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    n_train = len(train_images)
    best_acc = 0
    best_state = None
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = train_images[idx].to(device)
            labs = train_labels[idx].to(device)
            # Use DSA augmentation same as DD evaluation  
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            acc = evaluate(model, test_images, test_labels, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches:.4f}, Acc: {acc:.2f}%, Best: {best_acc:.2f}%")
    
    torch.save({'state_dict': best_state, 'accuracy': best_acc}, save_path)
    print(f"Teacher trained: {best_acc:.2f}%")
    return best_state, best_acc


# ============================================================
# Step 2: Generate Soft Labels
# ============================================================
def generate_soft_labels(teacher_state, train_images, train_labels, save_path='soft_labels_v3.pt'):
    """Generate soft labels using teacher model."""
    print("=" * 60)
    print("STEP 2: Generating soft labels")
    print("=" * 60)
    
    if os.path.exists(save_path):
        sl = torch.load(save_path, weights_only=False)
        print(f"Soft labels loaded: {sl.shape}")
        return sl
    
    model = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
    model.load_state_dict(teacher_state)
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    soft_labels = torch.cat(all_logits, dim=0)
    torch.save(soft_labels, save_path)
    print(f"Soft labels shape: {soft_labels.shape}")
    return soft_labels


# ============================================================
# Step 3: Coreset Methods (Random, K-centers)
# ============================================================
def random_select(train_labels, ipc, seed=0):
    """Select random subset with IPC images per class."""
    np.random.seed(seed)
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    selected = []
    for c in range(NUM_CLASSES):
        perm = np.random.permutation(len(class_indices[c]))[:ipc]
        selected.extend([class_indices[c][p] for p in perm])
    return selected


def kcenters_select(train_images, train_labels, ipc, device='cuda', seed=0):
    """
    K-centers coreset selection using feature-space K-means clustering.
    For each class: cluster features into IPC clusters, select nearest-to-centroid.
    Uses a pretrained model for feature extraction.
    """
    print(f"  K-centers selection (IPC={ipc})...")
    np.random.seed(seed)
    
    # Use a pretrained teacher or a randomly initialized network for features
    # The paper references DeepCore - try using average of multiple random networks
    model = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
    model.eval()
    
    # If we have a teacher, use it for better features
    if os.path.exists('teacher_v3.pt'):
        ckpt = torch.load('teacher_v3.pt', weights_only=False)
        model.load_state_dict({k: v.to(device) for k, v in ckpt['state_dict'].items()})
        print("    Using teacher features for K-centers")
    
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    selected = []
    
    # Extract all features
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(device)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    all_features = torch.cat(all_features, dim=0)  # (N, feat_dim)
    
    for c in range(NUM_CLASSES):
        indices = np.array(class_indices[c])
        features = all_features[indices].numpy()  # (n_c, feat_dim)
        
        if len(indices) <= ipc:
            selected.extend(indices.tolist())
            continue
        
        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=ipc, random_state=seed, n_init=3, max_iter=100)
        kmeans.fit(features)
        centroids = kmeans.cluster_centers_  # (ipc, feat_dim)
        labels_km = kmeans.labels_
        
        # For each cluster, select the sample nearest to the centroid
        for k in range(ipc):
            cluster_mask = labels_km == k
            cluster_features = features[cluster_mask]
            cluster_indices = indices[cluster_mask]
            
            if len(cluster_features) == 0:
                # Fallback: random sample
                selected.append(np.random.choice(indices))
                continue
            
            dists = np.linalg.norm(cluster_features - centroids[k], axis=1)
            nearest = np.argmin(dists)
            selected.append(cluster_indices[nearest])
    
    return selected


# ============================================================  
# Step 4: Distribution Matching (DM)
# ============================================================
def distribution_matching_v3(train_images, train_labels, ipc=10, iterations=20000,
                              lr_img=1.0, batch_real=64, device='cuda', seed=0,
                              save_path=None):
    """Optimized DM synthesis."""
    print(f"  DM synthesis (IPC={ipc}, {iterations} iterations)...")
    
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        print(f"  Loaded existing DM data from {save_path}")
        return data['images'], data['labels']
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    
    # Initialize from real images
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images.append(train_images[indices[p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    # Pre-organize real images by class
    real_by_class = []
    for c in range(NUM_CLASSES):
        real_by_class.append(train_images[class_indices[c]])
    
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    
    for it in range(iterations):
        # New random network each iteration
        net = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        net.eval()
        
        # Sample real data (batched)
        real_samples = []
        for c in range(NUM_CLASSES):
            n_c = len(real_by_class[c])
            perm = torch.randperm(n_c)[:batch_real]
            real_samples.append(real_by_class[c][perm])
        
        all_real = torch.cat(real_samples, dim=0).to(device)
        all_real_aug = DiffAugment(all_real, strategy=dsa_strategy)
        
        with torch.no_grad():
            all_real_feat = net.embed(all_real_aug)
        
        # Compute per-class mean features for real
        real_means = []
        offset = 0
        for c in range(NUM_CLASSES):
            n = min(batch_real, len(real_by_class[c]))
            real_means.append(all_real_feat[offset:offset+n].mean(0))
            offset += n
        real_means = torch.stack(real_means)
        
        # Synthetic forward
        all_syn_aug = DiffAugment(syn_images, strategy=dsa_strategy)
        all_syn_feat = net.embed(all_syn_aug)
        
        syn_means = []
        for c in range(NUM_CLASSES):
            mask = syn_labels == c
            syn_means.append(all_syn_feat[mask].mean(0))
        syn_means = torch.stack(syn_means)
        
        # Loss: MSE between means
        loss = torch.mean((real_means - syn_means) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        del net, all_real, all_real_aug, all_real_feat, all_syn_aug, all_syn_feat
        
        if (it + 1) % 2000 == 0:
            print(f"    Iter {it+1}/{iterations}, Loss: {loss.item():.6f}")
    
    result_images = syn_images.detach().cpu()
    result_labels = syn_labels.cpu()
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels


# ============================================================
# Step 5: Gradient Matching (DC)
# ============================================================
def gradient_matching_v3(train_images, train_labels, ipc=10, 
                         outer_loops=100, inner_loops=1, lr_img=1.0,
                         batch_real=256, device='cuda', seed=0,
                         save_path=None):
    """Optimized DC synthesis - matches standard DC setup."""
    print(f"  DC synthesis (IPC={ipc}, {outer_loops}x{inner_loops} loops)...")
    
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        print(f"  Loaded existing DC data from {save_path}")
        return data['images'], data['labels']
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images.append(train_images[indices[p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    
    real_by_class = []
    for c in range(NUM_CLASSES):
        real_by_class.append(train_images[class_indices[c]])
    
    for ol in range(outer_loops):
        # New random network
        net = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        net.train()
        optimizer_net = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        
        for il in range(inner_loops):
            loss = torch.tensor(0.0, device=device)
            
            for c in range(NUM_CLASSES):
                # Real gradient
                n_c = len(real_by_class[c])
                perm = torch.randperm(n_c)[:batch_real]
                real_batch = real_by_class[c][perm].to(device)
                real_labels_batch = torch.full((len(perm),), c, dtype=torch.long, device=device)
                
                real_aug = DiffAugment(real_batch, strategy=dsa_strategy)
                out_real = net(real_aug)
                loss_real = criterion(out_real, real_labels_batch)
                gw_real = torch.autograd.grad(loss_real, net.parameters(), create_graph=False)
                gw_real = [g.detach().clone() for g in gw_real]
                
                # Synthetic gradient
                syn_mask = syn_labels == c
                syn_batch = syn_images[syn_mask]
                syn_labels_batch = torch.full((syn_batch.shape[0],), c, dtype=torch.long, device=device)
                
                syn_aug = DiffAugment(syn_batch, strategy=dsa_strategy)
                out_syn = net(syn_aug)
                loss_syn = criterion(out_syn, syn_labels_batch)
                gw_syn = torch.autograd.grad(loss_syn, net.parameters(), create_graph=True)
                
                # Distance (cosine-based from DC paper)
                for ig in range(len(gw_real)):
                    gwr = gw_real[ig]
                    gws = gw_syn[ig]
                    shape = gwr.shape
                    if len(shape) == 4:
                        gwr = gwr.reshape(shape[0], -1)
                        gws = gws.reshape(shape[0], -1)
                    elif len(shape) == 1:
                        gwr = gwr.reshape(1, -1)
                        gws = gws.reshape(1, -1)
                    elif len(shape) == 2:
                        pass
                    else:
                        gwr = gwr.reshape(shape[0], -1)
                        gws = gws.reshape(shape[0], -1)
                    
                    loss += torch.sum(1 - torch.sum(gwr * gws, dim=-1) / 
                                     (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 1e-6))
            
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            
            # Train network on current synthetic data
            if il < inner_loops - 1:
                syn_aug_net = DiffAugment(syn_images.detach(), strategy=dsa_strategy)
                out_net = net(syn_aug_net)
                loss_net = criterion(out_net, syn_labels)
                optimizer_net.zero_grad()
                loss_net.backward()
                optimizer_net.step()
        
        del net
        
        if (ol + 1) % 20 == 0:
            print(f"    Outer {ol+1}/{outer_loops}, Loss: {loss.item():.4f}")
    
    result_images = syn_images.detach().cpu()
    result_labels = syn_labels.cpu()
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels


# ============================================================
# Step 6: Trajectory Matching (TM)
# ============================================================
def train_experts_v3(train_images, train_labels, num_experts=3, expert_epochs=50,
                     save_dir='/workspace/expert_trajectories_v3', device='cuda', seed=0):
    """Train expert trajectories for TM."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if already done
    existing = [f for f in os.listdir(save_dir) if f.startswith('expert_')]
    if len(existing) >= num_experts:
        print(f"  {len(existing)} experts already exist in {save_dir}")
        return save_dir
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_train = len(train_images)
    batch_size = 256
    criterion = nn.CrossEntropyLoss()
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    
    for exp_idx in range(num_experts):
        save_path = os.path.join(save_dir, f'expert_{exp_idx}.pt')
        if os.path.exists(save_path):
            print(f"  Expert {exp_idx} already exists")
            continue
            
        print(f"  Training expert {exp_idx+1}/{num_experts}...")
        torch.manual_seed(seed + exp_idx * 1000)
        
        model = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        trajectory = [{k: v.cpu().clone() for k, v in model.state_dict().items()}]
        
        model.train()
        for epoch in range(expert_epochs):
            perm = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i+batch_size]
                imgs = train_images[idx].to(device)
                labs = train_labels[idx].to(device)
                imgs = DiffAugment(imgs, strategy=dsa_strategy)
                
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labs)
                loss.backward()
                optimizer.step()
            
            trajectory.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            
            if (epoch + 1) % 10 == 0:
                acc = evaluate(model, test_images, test_labels, device)
                print(f"    Expert {exp_idx}, epoch {epoch+1}/{expert_epochs}, acc={acc:.2f}%")
        
        torch.save(trajectory, save_path)
    
    return save_dir


def trajectory_matching_v3(train_images, train_labels, ipc=10,
                           expert_dir='/workspace/expert_trajectories_v3',
                           iterations=5000, lr_img=1000.0, lr_lr=1e-5,
                           syn_steps=20, expert_epochs=3, max_start_epoch=25,
                           device='cuda', seed=0, save_path=None):
    """TM synthesis."""
    print(f"  TM synthesis (IPC={ipc}, {iterations} iterations)...")
    
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        print(f"  Loaded existing TM data from {save_path}")
        return data['images'], data['labels']
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load experts
    expert_files = sorted([f for f in os.listdir(expert_dir) if f.startswith('expert_')])
    num_experts = len(expert_files)
    print(f"    Loading {num_experts} expert trajectories...")
    
    expert_trajectories = []
    for f in expert_files:
        traj = torch.load(os.path.join(expert_dir, f), map_location='cpu', weights_only=False)
        expert_trajectories.append(traj)
    
    # Initialize from real images
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images.append(train_images[indices[p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    syn_lr = torch.tensor(0.01, device=device, requires_grad=True)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=lr_lr, momentum=0.5)
    
    criterion = nn.CrossEntropyLoss()
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    
    for it in range(iterations):
        exp_idx = np.random.randint(num_experts)
        traj = expert_trajectories[exp_idx]
        max_start = min(max_start_epoch, len(traj) - expert_epochs - 1)
        if max_start < 1:
            max_start = 1
        start_epoch = np.random.randint(0, max_start)
        
        start_params = traj[start_epoch]
        target_params = traj[start_epoch + expert_epochs]
        
        # Init student from start checkpoint
        student = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        student.load_state_dict({k: v.to(device) for k, v in start_params.items()})
        student.train()
        
        # Train student on synthetic data
        n_syn = len(syn_images)
        for step in range(syn_steps):
            perm = torch.randperm(n_syn, device=device)
            batch_imgs = syn_images[perm]
            batch_labels = syn_labels[perm]
            batch_imgs_aug = DiffAugment(batch_imgs, strategy=dsa_strategy)
            
            out = student(batch_imgs_aug)
            loss_s = criterion(out, batch_labels)
            
            grads = torch.autograd.grad(loss_s, student.parameters(), create_graph=True)
            with torch.no_grad():
                for param, grad in zip(student.parameters(), grads):
                    param.sub_(syn_lr * grad)
        
        # Trajectory matching loss
        loss = torch.tensor(0.0, device=device)
        target_dict = {k: v.to(device) for k, v in target_params.items()}
        
        # Compute parameter-wise matching loss with normalization
        for (name, param), (_, target) in zip(student.named_parameters(), target_dict.items()):
            target_flat = target.reshape(-1)
            param_flat = param.reshape(-1)
            # Normalized parameter matching (from MTT paper)
            target_norm = target_flat / (torch.norm(target_flat) + 1e-6)
            param_norm = param_flat / (torch.norm(param_flat) + 1e-6)
            loss += torch.sum((param_norm - target_norm) ** 2)
        
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        loss.backward()
        optimizer_img.step()
        optimizer_lr.step()
        
        with torch.no_grad():
            syn_lr.clamp_(min=1e-6)
        
        del student
        
        if (it + 1) % 1000 == 0:
            print(f"    Iter {it+1}/{iterations}, Loss: {loss.item():.6f}, lr: {syn_lr.item():.6f}")
    
    result_images = syn_images.detach().cpu()
    result_labels = syn_labels.cpu()
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels


# ============================================================
# Step 7: Generate Soft Labels for Distilled/Coreset Data
# ============================================================
def get_soft_labels_for_subset(images, labels, teacher_state):
    """Get soft labels for a subset using teacher."""
    model = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
    model.load_state_dict({k: v.to(device) for k, v in teacher_state.items()})
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), 512):
            batch = images[i:i+512].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    return torch.cat(all_logits, dim=0)


# ============================================================
# Step 8: Evaluate
# ============================================================
def evaluate_config(images, labels, test_images, test_labels, 
                    label_type, soft_labels=None, num_runs=3):
    """Evaluate a config with multiple runs."""
    model_fn = lambda: ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE)
    
    accs = []
    for run in range(num_runs):
        acc = train_and_evaluate(
            images, labels, test_images, test_labels,
            model_fn, num_classes=NUM_CLASSES, device=device,
            label_type=label_type, soft_labels=soft_labels,
            epochs=300, batch_size=256, seed=run, verbose=False
        )
        accs.append(acc)
        print(f"    Run {run+1}: {acc:.2f}%")
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    return mean_acc, std_acc


# ============================================================
# Main Pipeline
# ============================================================
if __name__ == '__main__':
    start_time = time.time()
    
    # Parse args - allow skipping steps
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default='all', 
                       help='Which step to run: teacher, distill, evaluate, all')
    parser.add_argument('--method', type=str, default='all',
                       help='Which method: random, kcenters, dm, dc, tm, all')
    parser.add_argument('--ipc', type=int, default=0, help='IPC (0=both)')
    args = parser.parse_args()
    
    # Load data
    train_images, train_labels, test_images, test_labels = load_data()
    
    results = {}
    
    # === STEP 1: Train Teacher ===
    if args.step in ['all', 'teacher']:
        teacher_state, teacher_acc = train_teacher(
            train_images, train_labels, test_images, test_labels,
            epochs=200, save_path='teacher_v3.pt'
        )
    else:
        ckpt = torch.load('teacher_v3.pt', weights_only=False)
        teacher_state = ckpt['state_dict']
        teacher_acc = ckpt['accuracy']
    
    # === STEP 2: Generate full soft labels ===
    full_soft_labels = generate_soft_labels(teacher_state, train_images, train_labels, 
                                            save_path='soft_labels_v3.pt')
    
    # === STEP 3+4: Distill and Evaluate ===
    ipcs = [10, 50] if args.ipc == 0 else [args.ipc]
    methods = ['random', 'kcenters', 'dm', 'dc', 'tm'] if args.method == 'all' else [args.method]
    
    for ipc in ipcs:
        for method in methods:
            print("\n" + "=" * 60)
            print(f"Method: {method}, IPC: {ipc}")
            print("=" * 60)
            
            key = f"{method}_ipc{ipc}"
            
            # Get/create distilled data
            if method == 'random':
                selected = random_select(train_labels, ipc, seed=42)
                images = train_images[selected]
                labels = train_labels[selected]
                sl = full_soft_labels[selected]
                
            elif method == 'kcenters':
                selected = kcenters_select(train_images, train_labels, ipc, device, seed=42)
                images = train_images[selected]
                labels = train_labels[selected]
                sl = full_soft_labels[selected]
                
            elif method == 'dm':
                iters = 20000 if ipc == 10 else 10000
                images, labels = distribution_matching_v3(
                    train_images, train_labels, ipc=ipc,
                    iterations=iters, lr_img=1.0, batch_real=64,
                    device=device, seed=0,
                    save_path=f'distilled_dm_ipc{ipc}_v3.pt'
                )
                sl = get_soft_labels_for_subset(images, labels, teacher_state)
                
            elif method == 'dc':
                outer = 200 if ipc == 10 else 100
                images, labels = gradient_matching_v3(
                    train_images, train_labels, ipc=ipc,
                    outer_loops=outer, inner_loops=1, lr_img=1.0,
                    batch_real=256, device=device, seed=0,
                    save_path=f'distilled_dc_ipc{ipc}_v3.pt'
                )
                sl = get_soft_labels_for_subset(images, labels, teacher_state)
                
            elif method == 'tm':
                # First ensure experts exist
                train_experts_v3(train_images, train_labels, 
                                num_experts=3, expert_epochs=50,
                                save_dir='/workspace/expert_trajectories_v3',
                                device=device)
                
                images, labels = trajectory_matching_v3(
                    train_images, train_labels, ipc=ipc,
                    expert_dir='/workspace/expert_trajectories_v3',
                    iterations=5000, lr_img=1000.0, syn_steps=20,
                    expert_epochs=3, max_start_epoch=40,
                    device=device, seed=0,
                    save_path=f'distilled_tm_ipc{ipc}_v3.pt'
                )
                sl = get_soft_labels_for_subset(images, labels, teacher_state)
            
            # Evaluate HL
            print(f"  Evaluating HL...")
            hl_mean, hl_std = evaluate_config(images, labels, test_images, test_labels,
                                               'hard', num_runs=3)
            print(f"  HL: {hl_mean:.2f} ± {hl_std:.2f}")
            
            # Evaluate SL
            print(f"  Evaluating SL...")
            sl_mean, sl_std = evaluate_config(images, labels, test_images, test_labels,
                                               'soft', soft_labels=sl, num_runs=3)
            print(f"  SL: {sl_mean:.2f} ± {sl_std:.2f}")
            
            results[key] = {
                'hl_mean': hl_mean, 'hl_std': hl_std,
                'sl_mean': sl_mean, 'sl_std': sl_std
            }
            
            # Save intermediate results
            os.makedirs('results', exist_ok=True)
            with open('results/results_v3.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    # Print final table
    elapsed = time.time() - start_time
    print(f"\n\nTotal time: {elapsed/60:.1f} min")
    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("=" * 80)
    print(f"{'Method':<12} {'IPC':>4}  {'HL (ours)':>14}  {'HL (paper)':>14}  {'SL (ours)':>14}  {'SL (paper)':>14}")
    print("-" * 80)
    
    paper = {
        'random_ipc10': (18.64, 33.43), 'random_ipc50': (34.66, 45.39),
        'kcenters_ipc10': (25.04, 34.70), 'kcenters_ipc50': (38.64, 46.24),
        'dm_ipc10': (29.23, 26.13), 'dm_ipc50': (42.32, 43.46),
        'dc_ipc10': (28.42, 23.54), 'dc_ipc50': (30.56, 33.46),
        'tm_ipc10': (38.18, 37.60), 'tm_ipc50': (46.32, 46.26),
    }
    
    for key in sorted(results.keys()):
        r = results[key]
        method, ipc_str = key.rsplit('_', 1)
        ipc = ipc_str.replace('ipc', '')
        p_hl, p_sl = paper.get(key, (0, 0))
        print(f"{method:<12} {ipc:>4}  {r['hl_mean']:>5.2f}±{r['hl_std']:.2f}  {p_hl:>5.2f}        {r['sl_mean']:>5.2f}±{r['sl_std']:.2f}  {p_sl:>5.2f}")
    
    # Save final results
    with open('results/results_v3.json', 'w') as f:
        json.dump(results, f, indent=2)
