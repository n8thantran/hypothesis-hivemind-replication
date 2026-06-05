#!/usr/bin/env python3
"""
Comprehensive experiment runner for replicating Table 1 of the paper:
"Rethinking Dataset Distillation: Hard Truths About Soft Labels"

CIFAR-100, ConvNet-D3, IPC 10 and 50
Methods: DM, DC, TM, Random, K-centers
Settings: Hard Label (HL) and Soft Label (SL)

Key hyperparameters from paper (Table stage3_hyper):
- HL: SGD, lr=1e-2, StepLR@151, CE loss, 300 epochs, DSA, batch_size=256
- SL: AdamW, lr=1e-3, Cosine, KL-Div T=20, 300 epochs, DSA, batch_size=256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import sys
import argparse
from collections import defaultdict

# ============================================================
# ConvNet-D3
# ============================================================
class ConvNet(nn.Module):
    def __init__(self, num_classes=100, channel=3, im_size=(32, 32),
                 net_width=128, net_depth=3, net_norm='instancenorm',
                 net_act='relu', net_pooling='avgpool'):
        super().__init__()
        layers = []
        in_ch = channel
        for i in range(net_depth):
            layers.append(nn.Conv2d(in_ch, net_width, 3, padding=1))
            if net_norm == 'instancenorm':
                layers.append(nn.GroupNorm(net_width, net_width, affine=True))
            elif net_norm == 'batchnorm':
                layers.append(nn.BatchNorm2d(net_width))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.AvgPool2d(2))
            in_ch = net_width
        self.features = nn.Sequential(*layers)
        feat_size = im_size[0] // (2 ** net_depth)
        self.classifier = nn.Linear(net_width * feat_size * feat_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def embed(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


# ============================================================
# DSA Augmentation (simplified but effective)
# ============================================================
def DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate', seed=-1, param=None):
    if seed == -1:
        param_aug = ParamDiffAug()
    else:
        param_aug = ParamDiffAug()
        param_aug.Siamese = True
        param_aug.latestseed = seed

    if strategy == 'None' or strategy == '':
        return x

    if strategy:
        for p in strategy.split('_'):
            for f in AUGMENT_FNS[p]:
                x = f(x, param_aug)
    return x


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S'
        self.Siamese = False
        self.latestseed = -1


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def rand_brightness(x, param):
    set_seed_DiffAug(param)
    ratio = 1.0 + (torch.rand(x.shape[0], 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x * ratio

def rand_saturation(x, param):
    set_seed_DiffAug(param)
    x_mean = x.mean(dim=1, keepdim=True)
    ratio = 2.0 * torch.rand(x.shape[0], 1, 1, 1, dtype=x.dtype, device=x.device)
    return (x - x_mean) * ratio + x_mean

def rand_contrast(x, param):
    set_seed_DiffAug(param)
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    ratio = 0.5 + 1.5 * torch.rand(x.shape[0], 1, 1, 1, dtype=x.dtype, device=x.device)
    return (x - x_mean) * ratio + x_mean

def rand_crop(x, param):
    set_seed_DiffAug(param)
    ratio = 0.125
    shift_x = int(x.shape[2] * ratio + 0.5)
    shift_y = int(x.shape[3] * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, size=[x.shape[0], 1, 1], device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, size=[x.shape[0], 1, 1], device=x.device)
    grid_b, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.shape[0], dtype=torch.long, device=x.device),
        torch.arange(x.shape[2], dtype=torch.long, device=x.device),
        torch.arange(x.shape[3], dtype=torch.long, device=x.device),
        indexing='ij')
    grid_x = torch.clamp(grid_x + tx + 1, 0, x.shape[2] + 1)
    grid_y = torch.clamp(grid_y + ty + 1, 0, x.shape[3] + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_b, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_cutout(x, param):
    set_seed_DiffAug(param)
    ratio = 0.5
    cutout_size = int(x.shape[2] * ratio + 0.5), int(x.shape[3] * ratio + 0.5)
    offset_x = torch.randint(0, x.shape[2] + (1 - cutout_size[0] % 2), size=[x.shape[0], 1, 1], device=x.device)
    offset_y = torch.randint(0, x.shape[3] + (1 - cutout_size[1] % 2), size=[x.shape[0], 1, 1], device=x.device)
    grid_b, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.shape[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij')
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.shape[2] - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.shape[3] - 1)
    mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], dtype=x.dtype, device=x.device)
    mask[grid_b, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)

def rand_flip(x, param):
    set_seed_DiffAug(param)
    prob = 0.5
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    return torch.where(randf < prob, x.flip(3), x)

def rand_scale(x, param):
    set_seed_DiffAug(param)
    ratio = 1.2
    sx = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * (ratio - 1.0/ratio) + 1.0/ratio
    sy = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = torch.zeros(x.shape[0], 2, 3, device=x.device)
    theta[:, 0, 0] = sx.squeeze()
    theta[:, 1, 1] = sy.squeeze()
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)

def rand_rotate(x, param):
    set_seed_DiffAug(param)
    ratio = 15.0
    theta = (torch.rand(x.shape[0], device=x.device) - 0.5) * 2 * ratio / 180 * 3.14159
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    aff = torch.zeros(x.shape[0], 2, 3, device=x.device)
    aff[:, 0, 0] = cos_t
    aff[:, 0, 1] = -sin_t
    aff[:, 1, 0] = sin_t
    aff[:, 1, 1] = cos_t
    grid = F.affine_grid(aff, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


# ============================================================
# Data Loading
# ============================================================
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

def get_cifar100_tensors(data_path='/workspace/data/hf_cache'):
    from datasets import load_dataset
    from PIL import Image
    import torchvision.transforms as transforms

    ds = load_dataset('uoft-cs/cifar100', cache_dir=data_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    def process_split(split):
        images, labels = [], []
        for ex in split:
            img = ex['img']
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            images.append(transform(img))
            labels.append(ex['fine_label'])
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)

    train_images, train_labels = process_split(ds['train'])
    test_images, test_labels = process_split(ds['test'])
    return train_images, train_labels, test_images, test_labels


def get_class_indices(labels, num_classes=100):
    ci = defaultdict(list)
    for i in range(len(labels)):
        ci[int(labels[i])].append(i)
    return ci


# ============================================================
# Teacher Training
# ============================================================
def train_teacher(train_images, train_labels, test_images, test_labels,
                  device='cuda', epochs=300, save_path='teacher_best.pt'):
    """Train a ConvNet-D3 teacher on full CIFAR-100."""
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location='cpu')
        print(f"Loaded teacher from {save_path}, accuracy: {ckpt.get('accuracy', 'unknown')}")
        return ckpt

    model = ConvNet(num_classes=100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)

    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labs)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            acc = evaluate_model(model, test_images, test_labels, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Teacher epoch {epoch+1}/{epochs}, acc: {acc:.2f}%, best: {best_acc:.2f}%")

    ckpt = {'state_dict': best_state, 'accuracy': best_acc}
    torch.save(ckpt, save_path)
    print(f"Teacher saved to {save_path}, accuracy: {best_acc:.2f}%")
    return ckpt


def evaluate_model(model, test_images, test_labels, device='cuda', batch_size=512):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            imgs = test_images[i:i+batch_size].to(device)
            labs = test_labels[i:i+batch_size].to(device)
            pred = model(imgs).argmax(1)
            correct += pred.eq(labs).sum().item()
            total += labs.size(0)
    return 100.0 * correct / total


# ============================================================
# Soft Label Generation
# ============================================================
def generate_soft_labels_from_teacher(train_images, teacher_state_dict, device='cuda'):
    """Generate soft labels (logits) from a trained teacher."""
    model = ConvNet(num_classes=100).to(device)
    model.load_state_dict(teacher_state_dict)
    model.eval()

    all_logits = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


# ============================================================
# Coreset Selection
# ============================================================
def random_select(labels, ipc, num_classes=100, seed=0):
    np.random.seed(seed)
    ci = get_class_indices(labels, num_classes)
    selected = []
    for c in range(num_classes):
        chosen = np.random.choice(ci[c], size=ipc, replace=False)
        selected.extend(chosen.tolist())
    return sorted(selected)


def k_centers_select(images, labels, ipc, num_classes=100, seed=0,
                     feature_model_state=None, device='cuda'):
    """
    K-Centers coreset selection using pretrained features.
    Uses K-means clustering: cluster each class into IPC clusters,
    then select the sample nearest to each centroid.
    This gives representative samples (not diverse outliers).
    """
    np.random.seed(seed)
    ci = get_class_indices(labels, num_classes)

    # Extract features using pretrained teacher
    model = ConvNet(num_classes=100).to(device)
    if feature_model_state is not None:
        model.load_state_dict(feature_model_state)
    model.eval()

    all_features = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    features_all = torch.cat(all_features, dim=0).numpy()

    selected = []
    for c in range(num_classes):
        indices = np.array(ci[c])
        features = features_all[indices]

        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features_norm = features / norms

        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=ipc, random_state=seed, n_init=3, max_iter=100)
        kmeans.fit(features_norm)
        centroids = kmeans.cluster_centers_

        # Select nearest sample to each centroid
        for k in range(ipc):
            cluster_mask = kmeans.labels_ == k
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                # Fallback: pick random
                selected.append(int(indices[np.random.randint(len(indices))]))
                continue
            cluster_feats = features_norm[cluster_indices]
            dists = np.sum((cluster_feats - centroids[k:k+1]) ** 2, axis=1)
            nearest = cluster_indices[np.argmin(dists)]
            selected.append(int(indices[nearest]))

    return sorted(selected)


# ============================================================
# Distribution Matching (DM)
# ============================================================
def distill_dm(train_images, train_labels, ipc, num_classes=100, device='cuda',
               iterations=20000, lr_img=1.0, batch_real=256):
    """
    Distribution Matching: minimize MMD between real and synthetic feature distributions.
    Following DC-Bench / DM paper settings.
    """
    print(f"DM distillation: IPC={ipc}, iterations={iterations}")
    ci = get_class_indices(train_labels, num_classes)

    # Initialize synthetic data from real data
    syn_images = []
    syn_labels = []
    for c in range(num_classes):
        indices = ci[c]
        chosen = np.random.choice(indices, size=ipc, replace=False)
        syn_images.append(train_images[chosen].clone())
        syn_labels.extend([c] * ipc)

    syn_images = torch.cat(syn_images, dim=0).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)

    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)

    for it in range(iterations):
        # Sample a new random model each iteration
        model = ConvNet(num_classes=100).to(device)
        model.eval()

        # Compute loss per class
        loss = torch.tensor(0.0, device=device)
        for c in range(num_classes):
            # Real samples for this class
            real_indices = ci[c]
            batch_idx = np.random.choice(real_indices, size=min(batch_real, len(real_indices)), replace=False)
            real_batch = train_images[batch_idx].to(device)

            # Synthetic samples for this class
            syn_mask = syn_labels == c
            syn_batch = syn_images[syn_mask]

            # Apply DSA
            real_aug = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate')
            syn_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate')

            with torch.no_grad():
                real_feat = model.embed(real_aug)
            syn_feat = model.embed(syn_aug)

            # MMD loss
            loss += torch.mean((real_feat.mean(0) - syn_feat.mean(0)) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % 1000 == 0:
            print(f"  DM iter {it+1}/{iterations}, loss: {loss.item():.6f}")

    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# Dataset Condensation (DC) - Gradient Matching
# ============================================================
def distill_dc(train_images, train_labels, ipc, num_classes=100, device='cuda',
               outer_loops=200, inner_loops=50, lr_img=1.0):
    """
    DC: Match gradients between real and synthetic data.
    """
    print(f"DC distillation: IPC={ipc}, outer={outer_loops}, inner={inner_loops}")
    ci = get_class_indices(train_labels, num_classes)

    # Initialize from real data
    syn_images = []
    syn_labels = []
    for c in range(num_classes):
        indices = ci[c]
        chosen = np.random.choice(indices, size=ipc, replace=False)
        syn_images.append(train_images[chosen].clone())
        syn_labels.extend([c] * ipc)

    syn_images = torch.cat(syn_images, dim=0).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)

    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)

    for outer in range(outer_loops):
        model = ConvNet(num_classes=100).to(device)
        model.train()
        model_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        for inner in range(inner_loops):
            # Gradient matching loss
            loss = torch.tensor(0.0, device=device)

            for c in range(num_classes):
                # Real gradient
                real_indices = ci[c]
                batch_idx = np.random.choice(real_indices, size=min(256, len(real_indices)), replace=False)
                real_batch = train_images[batch_idx].to(device)
                real_labels_batch = train_labels[batch_idx].to(device)

                real_aug = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate')
                real_loss = F.cross_entropy(model(real_aug), real_labels_batch)
                real_grad = torch.autograd.grad(real_loss, model.parameters(), create_graph=False)

                # Synthetic gradient
                syn_mask = syn_labels == c
                syn_batch = syn_images[syn_mask]
                syn_labels_batch = syn_labels[syn_mask]

                syn_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate')
                syn_loss = F.cross_entropy(model(syn_aug), syn_labels_batch)
                syn_grad = torch.autograd.grad(syn_loss, model.parameters(), create_graph=True)

                # Gradient matching (cosine distance)
                for rg, sg in zip(real_grad, syn_grad):
                    rg = rg.detach()
                    cos_sim = F.cosine_similarity(rg.flatten().unsqueeze(0), sg.flatten().unsqueeze(0))
                    loss += (1 - cos_sim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update model on synthetic data
            model_optimizer.zero_grad()
            syn_all_aug = DiffAugment(syn_images.detach(), strategy='color_crop_cutout_flip_scale_rotate')
            model_loss = F.cross_entropy(model(syn_all_aug), syn_labels)
            model_loss.backward()
            model_optimizer.step()

        if (outer + 1) % 20 == 0:
            print(f"  DC outer {outer+1}/{outer_loops}")

    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# Trajectory Matching (TM)
# ============================================================
def train_expert_trajectories(train_images, train_labels, num_experts=10,
                              expert_epochs=50, device='cuda', save_dir='expert_trajs'):
    """Train expert trajectories for TM."""
    os.makedirs(save_dir, exist_ok=True)

    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)

    for exp_idx in range(num_experts):
        save_path = os.path.join(save_dir, f'expert_{exp_idx}.pt')
        if os.path.exists(save_path):
            print(f"Expert {exp_idx} already exists, skipping")
            continue

        print(f"Training expert {exp_idx+1}/{num_experts}...")
        model = ConvNet(num_classes=100).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        trajectory = [model.state_dict()]

        for epoch in range(expert_epochs):
            model.train()
            for imgs, labs in loader:
                imgs, labs = imgs.to(device), labs.to(device)
                imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
                optimizer.zero_grad()
                F.cross_entropy(model(imgs), labs).backward()
                optimizer.step()
            trajectory.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        torch.save(trajectory, save_path)
        print(f"  Expert {exp_idx} saved ({len(trajectory)} checkpoints)")


def distill_tm(train_images, train_labels, ipc, num_classes=100, device='cuda',
               iterations=5000, lr_img=1000.0, expert_dir='expert_trajs',
               expert_epochs=50, num_experts=10, match_window=10):
    """
    Trajectory Matching: match synthetic training trajectory to expert trajectory.
    """
    print(f"TM distillation: IPC={ipc}, iterations={iterations}")

    # Train experts if needed
    train_expert_trajectories(train_images, train_labels, num_experts=num_experts,
                              expert_epochs=expert_epochs, device=device, save_dir=expert_dir)

    # Load expert trajectories
    expert_trajs = []
    for i in range(num_experts):
        traj = torch.load(os.path.join(expert_dir, f'expert_{i}.pt'), map_location='cpu')
        expert_trajs.append(traj)
    print(f"Loaded {len(expert_trajs)} expert trajectories")

    ci = get_class_indices(train_labels, num_classes)

    # Initialize synthetic data from real data
    syn_images = []
    syn_labels = []
    for c in range(num_classes):
        indices = ci[c]
        chosen = np.random.choice(indices, size=ipc, replace=False)
        syn_images.append(train_images[chosen].clone())
        syn_labels.extend([c] * ipc)

    syn_images = torch.cat(syn_images, dim=0).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)

    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    max_start_epoch = expert_epochs - match_window

    for it in range(iterations):
        # Sample random expert and starting epoch
        exp_idx = np.random.randint(len(expert_trajs))
        start_epoch = np.random.randint(0, max(1, max_start_epoch))
        end_epoch = min(start_epoch + match_window, expert_epochs)

        # Load starting parameters
        start_params = expert_trajs[exp_idx][start_epoch]
        target_params = expert_trajs[exp_idx][end_epoch]

        # Create student model with starting parameters
        student = ConvNet(num_classes=100).to(device)
        student.load_state_dict({k: v.to(device) for k, v in start_params.items()})
        student.train()

        # Train student on synthetic data for a few steps
        student_opt = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)

        n_steps = match_window  # One step per epoch to match
        for step in range(n_steps):
            syn_aug = DiffAugment(syn_images, strategy='color_crop_cutout_flip_scale_rotate')
            student_opt.zero_grad()
            loss = F.cross_entropy(student(syn_aug), syn_labels)
            loss.backward()
            student_opt.step()

        # Compute trajectory matching loss
        tm_loss = torch.tensor(0.0, device=device)
        target_sd = {k: v.to(device) for k, v in target_params.items()}
        for (name, param) in student.named_parameters():
            if name in target_sd:
                target = target_sd[name]
                start = start_params[name].to(device)
                # Normalized parameter matching
                direction = target - start
                norm = direction.norm() + 1e-8
                student_direction = param - start
                tm_loss += ((student_direction / norm - direction / norm) ** 2).sum()

        optimizer.zero_grad()
        tm_loss.backward()
        optimizer.step()
        scheduler.step()

        if (it + 1) % 500 == 0:
            print(f"  TM iter {it+1}/{iterations}, loss: {tm_loss.item():.6f}")

    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# Training & Evaluation
# ============================================================
def train_and_evaluate(train_images, train_labels, test_images, test_labels,
                       label_type='hard', soft_labels=None,
                       epochs=300, batch_size=256, seed=0, device='cuda', verbose=True):
    """Train ConvNet-D3 and evaluate."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ConvNet(num_classes=100).to(device)

    if label_type == 'hard':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        temperature = 20.0

    n_train = len(train_images)
    eff_bs = min(batch_size, n_train)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)

        for i in range(0, n_train, eff_bs):
            idx = perm[i:i+eff_bs]
            imgs = train_images[idx].to(device)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')

            optimizer.zero_grad()
            outputs = model(imgs)

            if label_type == 'hard':
                labs = train_labels[idx].to(device)
                loss = F.cross_entropy(outputs, labs)
            else:
                sl = soft_labels[idx].to(device)
                log_probs = F.log_softmax(outputs / temperature, dim=1)
                targets = F.softmax(sl / temperature, dim=1)
                loss = F.kl_div(log_probs, targets, reduction='batchmean') * (temperature ** 2)

            loss.backward()
            optimizer.step()

        scheduler.step()

        if verbose and (epoch + 1) % 100 == 0:
            acc = evaluate_model(model, test_images, test_labels, device)
            print(f"    Epoch {epoch+1}/{epochs}, acc: {acc:.2f}%")

    return evaluate_model(model, test_images, test_labels, device)


def run_eval(train_images, train_labels, test_images, test_labels,
             label_type='hard', soft_labels=None, num_runs=3, device='cuda', verbose=True):
    """Run multiple evaluation trials."""
    accs = []
    for run in range(num_runs):
        if verbose:
            print(f"  Run {run+1}/{num_runs}")
        acc = train_and_evaluate(train_images, train_labels, test_images, test_labels,
                                 label_type=label_type, soft_labels=soft_labels,
                                 seed=run, device=device, verbose=verbose)
        accs.append(acc)
        if verbose:
            print(f"  Run {run+1} accuracy: {acc:.2f}%")
    return np.mean(accs), np.std(accs)


# ============================================================
# Main Pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='all',
                        choices=['teacher', 'softlabels', 'coreset', 'dm', 'dc', 'tm', 'eval_all', 'all'])
    parser.add_argument('--ipc', type=int, default=10)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dm_iters', type=int, default=20000)
    parser.add_argument('--dc_outer', type=int, default=200)
    parser.add_argument('--dc_inner', type=int, default=50)
    parser.add_argument('--tm_iters', type=int, default=5000)
    parser.add_argument('--tm_experts', type=int, default=10)
    args = parser.parse_args()

    device = args.device
    results = {}

    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")

    # Phase 1: Teacher
    if args.phase in ['teacher', 'softlabels', 'all']:
        teacher_ckpt = train_teacher(train_images, train_labels, test_images, test_labels,
                                     device=device, save_path='teacher_best.pt')
        teacher_state = teacher_ckpt['state_dict']

    # Phase 2: Soft labels
    if args.phase in ['softlabels', 'all']:
        sl_path = 'soft_labels_teacher.pt'
        if os.path.exists(sl_path):
            all_soft_labels = torch.load(sl_path, map_location='cpu')
            print(f"Loaded soft labels from {sl_path}")
        else:
            all_soft_labels = generate_soft_labels_from_teacher(train_images, teacher_state, device)
            torch.save(all_soft_labels, sl_path)
            print(f"Saved soft labels to {sl_path}")

    # Phase 3: Coreset evaluation
    if args.phase in ['coreset', 'eval_all', 'all']:
        teacher_ckpt = torch.load('teacher_best.pt', map_location='cpu')
        teacher_state = teacher_ckpt['state_dict']
        all_soft_labels = torch.load('soft_labels_teacher.pt', map_location='cpu') if os.path.exists('soft_labels_teacher.pt') else None

        for ipc in [10, 50]:
            print(f"\n{'='*60}")
            print(f"Evaluating coresets IPC={ipc}")
            print(f"{'='*60}")

            # Random
            for seed_offset in range(3):  # 3 different random selections
                pass  # We'll use the same selection for all runs

            sel = random_select(train_labels, ipc, seed=42)
            sub_imgs = train_images[sel]
            sub_labs = train_labels[sel]

            print(f"\nRandom IPC={ipc} HL:")
            mean, std = run_eval(sub_imgs, sub_labs, test_images, test_labels,
                                 'hard', num_runs=args.num_runs, device=device)
            results[f'Random_IPC{ipc}_HL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

            if all_soft_labels is not None:
                sub_sl = all_soft_labels[sel]
                print(f"\nRandom IPC={ipc} SL:")
                mean, std = run_eval(sub_imgs, sub_labs, test_images, test_labels,
                                     'soft', sub_sl, num_runs=args.num_runs, device=device)
                results[f'Random_IPC{ipc}_SL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

            # K-centers
            sel_kc = k_centers_select(train_images, train_labels, ipc, seed=42,
                                      feature_model_state=teacher_state, device=device)
            sub_imgs_kc = train_images[sel_kc]
            sub_labs_kc = train_labels[sel_kc]

            print(f"\nK-centers IPC={ipc} HL:")
            mean, std = run_eval(sub_imgs_kc, sub_labs_kc, test_images, test_labels,
                                 'hard', num_runs=args.num_runs, device=device)
            results[f'Kcenters_IPC{ipc}_HL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

            if all_soft_labels is not None:
                sub_sl_kc = all_soft_labels[sel_kc]
                print(f"\nK-centers IPC={ipc} SL:")
                mean, std = run_eval(sub_imgs_kc, sub_labs_kc, test_images, test_labels,
                                     'soft', sub_sl_kc, num_runs=args.num_runs, device=device)
                results[f'Kcenters_IPC{ipc}_SL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

            # Save intermediate results
            with open('results/results_comprehensive.json', 'w') as f:
                json.dump(results, f, indent=2)

    # Phase 4: DM distillation
    if args.phase in ['dm', 'all']:
        for ipc in [10, 50]:
            dm_path = f'distilled_dm_ipc{ipc}_final.pt'
            if os.path.exists(dm_path):
                print(f"DM IPC={ipc} already distilled, skipping")
                continue
            iters = args.dm_iters if ipc == 10 else min(args.dm_iters, 10000)
            syn_imgs, syn_labs = distill_dm(train_images, train_labels, ipc,
                                            device=device, iterations=iters)
            torch.save({'images': syn_imgs, 'labels': syn_labs}, dm_path)
            print(f"Saved DM IPC={ipc} to {dm_path}")

    # Phase 5: DC distillation
    if args.phase in ['dc', 'all']:
        for ipc in [10, 50]:
            dc_path = f'distilled_dc_ipc{ipc}_final.pt'
            if os.path.exists(dc_path):
                print(f"DC IPC={ipc} already distilled, skipping")
                continue
            outer = args.dc_outer if ipc == 10 else min(args.dc_outer, 100)
            syn_imgs, syn_labs = distill_dc(train_images, train_labels, ipc,
                                            device=device, outer_loops=outer,
                                            inner_loops=args.dc_inner)
            torch.save({'images': syn_imgs, 'labels': syn_labs}, dc_path)
            print(f"Saved DC IPC={ipc} to {dc_path}")

    # Phase 6: TM distillation
    if args.phase in ['tm', 'all']:
        for ipc in [10, 50]:
            tm_path = f'distilled_tm_ipc{ipc}_final.pt'
            if os.path.exists(tm_path):
                print(f"TM IPC={ipc} already distilled, skipping")
                continue
            syn_imgs, syn_labs = distill_tm(train_images, train_labels, ipc,
                                            device=device, iterations=args.tm_iters,
                                            num_experts=args.tm_experts)
            torch.save({'images': syn_imgs, 'labels': syn_labs}, tm_path)
            print(f"Saved TM IPC={ipc} to {tm_path}")

    # Phase 7: Evaluate DD methods
    if args.phase in ['eval_all', 'all']:
        all_soft_labels = torch.load('soft_labels_teacher.pt', map_location='cpu') if os.path.exists('soft_labels_teacher.pt') else None
        teacher_ckpt = torch.load('teacher_best.pt', map_location='cpu')
        teacher_state = teacher_ckpt['state_dict']

        for method in ['dm', 'dc', 'tm']:
            for ipc in [10, 50]:
                # Try final version first, then fall back
                dd_path = f'distilled_{method}_ipc{ipc}_final.pt'
                if not os.path.exists(dd_path):
                    dd_path = f'distilled_{method}_ipc{ipc}.pt'
                if not os.path.exists(dd_path):
                    print(f"No distilled set for {method} IPC={ipc}, skipping")
                    continue

                data = torch.load(dd_path, map_location='cpu')
                syn_imgs = data['images']
                syn_labs = data['labels']

                print(f"\n{'='*60}")
                print(f"{method.upper()} IPC={ipc}")
                print(f"{'='*60}")

                # HL evaluation
                print(f"\n{method.upper()} IPC={ipc} HL:")
                mean, std = run_eval(syn_imgs, syn_labs, test_images, test_labels,
                                     'hard', num_runs=args.num_runs, device=device)
                results[f'{method.upper()}_IPC{ipc}_HL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

                # SL evaluation - generate soft labels for synthetic data
                if all_soft_labels is not None:
                    # For DD methods, generate soft labels from teacher on synthetic images
                    syn_sl = generate_soft_labels_from_teacher(syn_imgs, teacher_state, device)
                    print(f"\n{method.upper()} IPC={ipc} SL:")
                    mean, std = run_eval(syn_imgs, syn_labs, test_images, test_labels,
                                         'soft', syn_sl, num_runs=args.num_runs, device=device)
                    results[f'{method.upper()}_IPC{ipc}_SL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

                # Save intermediate
                with open('results/results_comprehensive.json', 'w') as f:
                    json.dump(results, f, indent=2)

    # Print final table
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    print(f"{'Method':<12} {'IPC':>4} {'HL':>12} {'SL':>12}")
    print("-"*44)
    for method in ['DM', 'DC', 'TM', 'Random', 'Kcenters']:
        for ipc in [10, 50]:
            hl_key = f'{method}_IPC{ipc}_HL'
            sl_key = f'{method}_IPC{ipc}_SL'
            hl = results.get(hl_key, {})
            sl = results.get(sl_key, {})
            hl_str = f"{hl.get('mean', '-'):.2f}±{hl.get('std', 0):.2f}" if hl else "-"
            sl_str = f"{sl.get('mean', '-'):.2f}±{sl.get('std', 0):.2f}" if sl else "-"
            print(f"{method:<12} {ipc:>4} {hl_str:>12} {sl_str:>12}")

    # Save final results
    os.makedirs('results', exist_ok=True)
    with open('results/results_comprehensive.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    main()
