#!/usr/bin/env python3
"""
Complete experiment pipeline for replicating Table 1:
"Rethinking Dataset Distillation: Hard Truths About Soft Labels"

CIFAR-100, ConvNet-D3, IPC 10 and 50
Methods: DM, DC, TM, Random, K-centers
Settings: Hard Label (HL) and Soft Label (SL)

Paper hyperparameters:
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
from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment


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
# Evaluation
# ============================================================
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


def train_and_evaluate(train_images, train_labels, test_images, test_labels,
                       label_type='hard', soft_labels=None,
                       epochs=300, batch_size=256, seed=0, device='cuda', verbose=True):
    """Train ConvNet-D3 on distilled/coreset data and evaluate on test set."""
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
                                 seed=run*42, device=device, verbose=verbose)
        accs.append(acc)
        if verbose:
            print(f"  Run {run+1} accuracy: {acc:.2f}%")
    return np.mean(accs), np.std(accs)


# ============================================================
# Soft Label Generation
# ============================================================
def generate_soft_labels(train_images, teacher_state_dict, device='cuda'):
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
    K-Centers coreset selection using K-means clustering in feature space.
    Uses pretrained teacher features for better representation.
    For each class: cluster into IPC clusters, select nearest-to-centroid.
    """
    from sklearn.cluster import KMeans
    
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
        kmeans = KMeans(n_clusters=ipc, random_state=seed, n_init=3, max_iter=100)
        kmeans.fit(features_norm)
        centroids = kmeans.cluster_centers_

        # Select nearest sample to each centroid
        for k in range(ipc):
            cluster_mask = kmeans.labels_ == k
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
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
    """Distribution Matching distillation."""
    print(f"DM distillation: IPC={ipc}, iterations={iterations}")
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

    for it in range(iterations):
        model = ConvNet(num_classes=100).to(device)
        model.eval()

        loss = torch.tensor(0.0, device=device)
        for c in range(num_classes):
            real_indices = ci[c]
            batch_idx = np.random.choice(real_indices, size=min(batch_real, len(real_indices)), replace=False)
            real_batch = train_images[batch_idx].to(device)

            syn_mask = syn_labels == c
            syn_batch = syn_images[syn_mask]

            real_aug = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate')
            syn_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate')

            with torch.no_grad():
                real_feat = model.embed(real_aug)
            syn_feat = model.embed(syn_aug)

            loss += torch.mean((real_feat.mean(0) - syn_feat.mean(0)) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % 2000 == 0:
            print(f"  DM iter {it+1}/{iterations}, loss: {loss.item():.6f}")

    return syn_images.detach().cpu(), syn_labels.cpu()


# ============================================================
# Dataset Condensation (DC) - Gradient Matching
# ============================================================
def distill_dc(train_images, train_labels, ipc, num_classes=100, device='cuda',
               outer_loops=200, inner_loops=50, lr_img=1.0):
    """DC: Match gradients between real and synthetic data."""
    print(f"DC distillation: IPC={ipc}, outer={outer_loops}, inner={inner_loops}")
    ci = get_class_indices(train_labels, num_classes)

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
            loss = torch.tensor(0.0, device=device)

            for c in range(num_classes):
                real_indices = ci[c]
                batch_idx = np.random.choice(real_indices, size=min(256, len(real_indices)), replace=False)
                real_batch = train_images[batch_idx].to(device)
                real_labels_batch = train_labels[batch_idx].to(device)

                real_aug = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate')
                real_loss = F.cross_entropy(model(real_aug), real_labels_batch)
                real_grad = torch.autograd.grad(real_loss, model.parameters(), create_graph=False)

                syn_mask = syn_labels == c
                syn_batch = syn_images[syn_mask]
                syn_labels_batch = syn_labels[syn_mask]

                syn_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate')
                syn_loss = F.cross_entropy(model(syn_aug), syn_labels_batch)
                syn_grad = torch.autograd.grad(syn_loss, model.parameters(), create_graph=True)

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

        trajectory = [{k: v.cpu().clone() for k, v in model.state_dict().items()}]

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
    """Trajectory Matching distillation."""
    print(f"TM distillation: IPC={ipc}, iterations={iterations}")

    train_expert_trajectories(train_images, train_labels, num_experts=num_experts,
                              expert_epochs=expert_epochs, device=device, save_dir=expert_dir)

    expert_trajs = []
    for i in range(num_experts):
        traj = torch.load(os.path.join(expert_dir, f'expert_{i}.pt'), map_location='cpu')
        expert_trajs.append(traj)
    print(f"Loaded {len(expert_trajs)} expert trajectories")

    ci = get_class_indices(train_labels, num_classes)

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
        exp_idx = np.random.randint(len(expert_trajs))
        start_epoch = np.random.randint(0, max(1, max_start_epoch))
        end_epoch = min(start_epoch + match_window, expert_epochs)

        start_params = expert_trajs[exp_idx][start_epoch]
        target_params = expert_trajs[exp_idx][end_epoch]

        student = ConvNet(num_classes=100).to(device)
        student.load_state_dict({k: v.to(device) for k, v in start_params.items()})
        student.train()

        student_opt = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)

        n_steps = match_window
        for step in range(n_steps):
            syn_aug = DiffAugment(syn_images, strategy='color_crop_cutout_flip_scale_rotate')
            student_opt.zero_grad()
            loss = F.cross_entropy(student(syn_aug), syn_labels)
            loss.backward()
            student_opt.step()

        tm_loss = torch.tensor(0.0, device=device)
        target_sd = {k: v.to(device) for k, v in target_params.items()}
        for (name, param) in student.named_parameters():
            if name in target_sd:
                target = target_sd[name]
                start = start_params[name].to(device)
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
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='coreset',
                        choices=['coreset', 'dm', 'dc', 'tm', 'eval_dd', 'all'])
    parser.add_argument('--ipc', type=int, nargs='+', default=[10, 50])
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dm_iters', type=int, default=20000)
    parser.add_argument('--dc_outer', type=int, default=200)
    parser.add_argument('--dc_inner', type=int, default=50)
    parser.add_argument('--tm_iters', type=int, default=5000)
    args = parser.parse_args()

    device = args.device
    results = {}

    # Load existing results
    results_path = 'results/results_comprehensive.json'
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")

    # Load teacher
    teacher_ckpt = torch.load('teacher_best.pt', map_location='cpu')
    teacher_state = teacher_ckpt['state_dict']
    print(f"Teacher accuracy: {teacher_ckpt['accuracy']}")

    # Generate soft labels if needed
    sl_path = 'soft_labels_teacher.pt'
    if not os.path.exists(sl_path):
        print("Generating soft labels from teacher...")
        all_soft_labels = generate_soft_labels(train_images, teacher_state, device)
        torch.save(all_soft_labels, sl_path)
    else:
        all_soft_labels = torch.load(sl_path, map_location='cpu')
    print(f"Soft labels shape: {all_soft_labels.shape}")

    # ---- Coreset evaluation ----
    if args.phase in ['coreset', 'all']:
        for ipc in args.ipc:
            print(f"\n{'='*60}")
            print(f"CORESET EVALUATION IPC={ipc}")
            print(f"{'='*60}")

            # Random
            sel = random_select(train_labels, ipc, seed=42)
            sub_imgs = train_images[sel]
            sub_labs = train_labels[sel]
            sub_sl = all_soft_labels[sel]

            print(f"\nRandom IPC={ipc} HL:")
            mean, std = run_eval(sub_imgs, sub_labs, test_images, test_labels,
                                 'hard', num_runs=args.num_runs, device=device)
            results[f'Random_IPC{ipc}_HL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

            print(f"\nRandom IPC={ipc} SL:")
            mean, std = run_eval(sub_imgs, sub_labs, test_images, test_labels,
                                 'soft', sub_sl, num_runs=args.num_runs, device=device)
            results[f'Random_IPC{ipc}_SL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

            # K-centers
            sel_kc = k_centers_select(train_images, train_labels, ipc, seed=42,
                                      feature_model_state=teacher_state, device=device)
            sub_imgs_kc = train_images[sel_kc]
            sub_labs_kc = train_labels[sel_kc]
            sub_sl_kc = all_soft_labels[sel_kc]

            print(f"\nK-centers IPC={ipc} HL:")
            mean, std = run_eval(sub_imgs_kc, sub_labs_kc, test_images, test_labels,
                                 'hard', num_runs=args.num_runs, device=device)
            results[f'Kcenters_IPC{ipc}_HL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

            print(f"\nK-centers IPC={ipc} SL:")
            mean, std = run_eval(sub_imgs_kc, sub_labs_kc, test_images, test_labels,
                                 'soft', sub_sl_kc, num_runs=args.num_runs, device=device)
            results[f'Kcenters_IPC{ipc}_SL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

            # Save intermediate
            os.makedirs('results', exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

    # ---- DM distillation ----
    if args.phase in ['dm', 'all']:
        for ipc in args.ipc:
            dm_path = f'distilled_dm_ipc{ipc}_final.pt'
            if not os.path.exists(dm_path):
                iters = args.dm_iters if ipc == 10 else min(args.dm_iters, 10000)
                syn_imgs, syn_labs = distill_dm(train_images, train_labels, ipc,
                                                device=device, iterations=iters)
                torch.save({'images': syn_imgs, 'labels': syn_labs}, dm_path)
            else:
                print(f"DM IPC={ipc} already exists at {dm_path}")

    # ---- DC distillation ----
    if args.phase in ['dc', 'all']:
        for ipc in args.ipc:
            dc_path = f'distilled_dc_ipc{ipc}_final.pt'
            if not os.path.exists(dc_path):
                outer = args.dc_outer if ipc == 10 else min(args.dc_outer, 100)
                syn_imgs, syn_labs = distill_dc(train_images, train_labels, ipc,
                                                device=device, outer_loops=outer,
                                                inner_loops=args.dc_inner)
                torch.save({'images': syn_imgs, 'labels': syn_labs}, dc_path)
            else:
                print(f"DC IPC={ipc} already exists at {dc_path}")

    # ---- TM distillation ----
    if args.phase in ['tm', 'all']:
        for ipc in args.ipc:
            tm_path = f'distilled_tm_ipc{ipc}_final.pt'
            if not os.path.exists(tm_path):
                syn_imgs, syn_labs = distill_tm(train_images, train_labels, ipc,
                                                device=device, iterations=args.tm_iters)
                torch.save({'images': syn_imgs, 'labels': syn_labs}, tm_path)
            else:
                print(f"TM IPC={ipc} already exists at {tm_path}")

    # ---- Evaluate DD methods ----
    if args.phase in ['eval_dd', 'all']:
        for method in ['dm', 'dc', 'tm']:
            for ipc in args.ipc:
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

                # HL
                print(f"\n{method.upper()} IPC={ipc} HL:")
                mean, std = run_eval(syn_imgs, syn_labs, test_images, test_labels,
                                     'hard', num_runs=args.num_runs, device=device)
                results[f'{method.upper()}_IPC{ipc}_HL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

                # SL - generate soft labels for synthetic images from teacher
                syn_sl = generate_soft_labels(syn_imgs, teacher_state, device)
                print(f"\n{method.upper()} IPC={ipc} SL:")
                mean, std = run_eval(syn_imgs, syn_labs, test_images, test_labels,
                                     'soft', syn_sl, num_runs=args.num_runs, device=device)
                results[f'{method.upper()}_IPC{ipc}_SL'] = {'mean': round(mean, 2), 'std': round(std, 2)}

                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)

    # Print table
    print_results_table(results)

    # Save
    os.makedirs('results', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)


def print_results_table(results):
    print("\n" + "="*80)
    print("RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("="*80)
    print(f"{'Method':<12} {'IPC':>4} {'HL':>14} {'SL':>14}")
    print("-"*48)
    for method in ['DM', 'DC', 'TM', 'Random', 'Kcenters']:
        for ipc in [10, 50]:
            hl_key = f'{method}_IPC{ipc}_HL'
            sl_key = f'{method}_IPC{ipc}_SL'
            hl = results.get(hl_key, {})
            sl = results.get(sl_key, {})
            hl_str = f"{hl['mean']:.2f}±{hl['std']:.2f}" if hl else "-"
            sl_str = f"{sl['mean']:.2f}±{sl['std']:.2f}" if sl else "-"
            print(f"{method:<12} {ipc:>4} {hl_str:>14} {sl_str:>14}")


if __name__ == '__main__':
    main()
