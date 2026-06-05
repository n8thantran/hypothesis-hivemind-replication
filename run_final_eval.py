#!/usr/bin/env python3
"""
Final comprehensive evaluation script.
Evaluates all methods (Random, K-centers, DM, DC, TM) in HL and SL settings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import sys

from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices

DEVICE = 'cuda'
NUM_CLASSES = 100
DSA_STRATEGY = 'color_crop_cutout_flip_scale_rotate'
RESULTS_DIR = '/workspace/results'


def model_fn():
    return ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32))


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
    """Train ConvNet-D3 with paper-specified hyperparameters."""
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
            print(f"    Epoch {epoch+1}/{epochs}, Acc: {acc:.2f}%")
            model.train()
    
    return evaluate_model(model, test_images, test_labels)


def random_select(labels, ipc, seed=42):
    """Select IPC random samples per class."""
    np.random.seed(seed)
    indices = []
    for c in range(NUM_CLASSES):
        cls_idx = (labels == c).nonzero(as_tuple=True)[0].numpy()
        sel = np.random.choice(cls_idx, size=ipc, replace=False)
        indices.extend(sel.tolist())
    return indices


def kmeans_select(images, labels, ipc, seed=42):
    """Select IPC samples per class using K-means in feature space."""
    from sklearn.cluster import KMeans
    
    # Extract features using a pretrained model
    model = model_fn().to(DEVICE)
    model.eval()
    
    all_features = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(DEVICE)
            # Get features before final layer
            feat = model.features(batch)
            feat = feat.view(feat.size(0), -1)
            all_features.append(feat.cpu())
    features = torch.cat(all_features, dim=0).numpy()
    
    indices = []
    for c in range(NUM_CLASSES):
        cls_idx = (labels == c).nonzero(as_tuple=True)[0].numpy()
        cls_feat = features[cls_idx]
        
        if len(cls_idx) <= ipc:
            indices.extend(cls_idx.tolist())
            continue
        
        kmeans = KMeans(n_clusters=ipc, random_state=seed, n_init=3, max_iter=100)
        kmeans.fit(cls_feat)
        centers = kmeans.cluster_centers_
        
        # For each center, find nearest real sample
        for k in range(ipc):
            dists = np.sum((cls_feat - centers[k:k+1]) ** 2, axis=1)
            nearest = np.argmin(dists)
            indices.append(cls_idx[nearest])
    
    return indices


def generate_dd_soft_labels(images, labels, teacher_sd):
    """Generate soft labels for distilled images using teacher."""
    model = model_fn().to(DEVICE)
    model.load_state_dict(teacher_sd)
    model.eval()
    
    logits = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(DEVICE)
            out = model(batch)
            logits.append(out.cpu())
    return torch.cat(logits, dim=0)


def main():
    print("=" * 70)
    print("FINAL EVALUATION - Dataset Distillation Paper Replication")
    print("=" * 70)
    
    # Load data
    print("\nLoading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Load teacher and soft labels
    teacher_ckpt = torch.load('/workspace/teacher_best.pt', map_location='cpu', weights_only=False)
    teacher_sd = teacher_ckpt['state_dict']
    print(f"Teacher accuracy: {teacher_ckpt['accuracy']:.2f}%")
    
    # Use teacher-generated soft labels for full training set
    sl_path = '/workspace/soft_labels_teacher.pt'
    if os.path.exists(sl_path):
        full_soft_labels = torch.load(sl_path, map_location='cpu', weights_only=True)
    else:
        full_soft_labels = torch.load('/workspace/soft_labels.pt', map_location='cpu', weights_only=True)
    print(f"Soft labels shape: {full_soft_labels.shape}")
    
    results = {}
    NUM_RUNS = 3
    
    # ================================================================
    # CORESET METHODS
    # ================================================================
    for ipc in [10, 50]:
        # RANDOM
        print(f"\n{'='*60}")
        print(f"Evaluating Random IPC={ipc}")
        print(f"{'='*60}")
        
        sel_idx = random_select(train_labels, ipc)
        sel_images = train_images[sel_idx]
        sel_labels = train_labels[sel_idx]
        sel_soft = full_soft_labels[sel_idx]
        
        # HL
        print("  HL evaluation:")
        accs = []
        for r in range(NUM_RUNS):
            acc = train_and_eval(sel_images, sel_labels, test_images, test_labels,
                                label_type='hard', seed=r, verbose=(r==0))
            accs.append(acc)
            print(f"    Run {r+1}: {acc:.2f}%")
        results[f'random_ipc{ipc}_hl'] = {'mean': np.mean(accs), 'std': np.std(accs)}
        print(f"  HL: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
        
        # SL
        print("  SL evaluation:")
        accs = []
        for r in range(NUM_RUNS):
            acc = train_and_eval(sel_images, sel_labels, test_images, test_labels,
                                label_type='soft', soft_labels=sel_soft, seed=r, verbose=(r==0))
            accs.append(acc)
            print(f"    Run {r+1}: {acc:.2f}%")
        results[f'random_ipc{ipc}_sl'] = {'mean': np.mean(accs), 'std': np.std(accs)}
        print(f"  SL: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
        
        # K-CENTERS (K-means)
        print(f"\n{'='*60}")
        print(f"Evaluating K-centers IPC={ipc}")
        print(f"{'='*60}")
        
        sel_idx = kmeans_select(train_images, train_labels, ipc)
        sel_images = train_images[sel_idx]
        sel_labels = train_labels[sel_idx]
        sel_soft = full_soft_labels[sel_idx]
        
        # HL
        print("  HL evaluation:")
        accs = []
        for r in range(NUM_RUNS):
            acc = train_and_eval(sel_images, sel_labels, test_images, test_labels,
                                label_type='hard', seed=r, verbose=(r==0))
            accs.append(acc)
            print(f"    Run {r+1}: {acc:.2f}%")
        results[f'kcenter_ipc{ipc}_hl'] = {'mean': np.mean(accs), 'std': np.std(accs)}
        print(f"  HL: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
        
        # SL
        print("  SL evaluation:")
        accs = []
        for r in range(NUM_RUNS):
            acc = train_and_eval(sel_images, sel_labels, test_images, test_labels,
                                label_type='soft', soft_labels=sel_soft, seed=r, verbose=(r==0))
            accs.append(acc)
            print(f"    Run {r+1}: {acc:.2f}%")
        results[f'kcenter_ipc{ipc}_sl'] = {'mean': np.mean(accs), 'std': np.std(accs)}
        print(f"  SL: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
    
    # Save intermediate results
    with open(f'{RESULTS_DIR}/results_coreset_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nCoreset results saved.")
    
    # ================================================================
    # DD METHODS
    # ================================================================
    dd_methods = {
        'DM': {'ipc10': 'distilled_dm_ipc10.pt', 'ipc50': 'distilled_dm_ipc50.pt'},
        'DC': {'ipc10': 'distilled_dc_ipc10.pt', 'ipc50': 'distilled_dc_ipc50.pt'},
        'TM': {'ipc10': 'distilled_tm_ipc10.pt', 'ipc50': 'distilled_tm_ipc50.pt'},
    }
    
    for method_name, files in dd_methods.items():
        for ipc_key in ['ipc10', 'ipc50']:
            ipc = int(ipc_key.replace('ipc', ''))
            fpath = f'/workspace/{files[ipc_key]}'
            
            if not os.path.exists(fpath):
                print(f"\n{method_name} {ipc_key}: FILE NOT FOUND, skipping")
                continue
            
            print(f"\n{'='*60}")
            print(f"Evaluating {method_name} IPC={ipc}")
            print(f"{'='*60}")
            
            data = torch.load(fpath, map_location='cpu', weights_only=True)
            dd_images = data['images']
            dd_labels = data['labels']
            print(f"  Loaded: {dd_images.shape}, labels: {dd_labels.shape}")
            
            # Generate soft labels for DD images from teacher
            dd_soft = generate_dd_soft_labels(dd_images, dd_labels, teacher_sd)
            
            # HL
            print("  HL evaluation:")
            accs = []
            for r in range(NUM_RUNS):
                acc = train_and_eval(dd_images, dd_labels, test_images, test_labels,
                                    label_type='hard', seed=r, verbose=(r==0))
                accs.append(acc)
                print(f"    Run {r+1}: {acc:.2f}%")
            results[f'{method_name.lower()}_ipc{ipc}_hl'] = {'mean': np.mean(accs), 'std': np.std(accs)}
            print(f"  HL: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
            
            # SL
            print("  SL evaluation:")
            accs = []
            for r in range(NUM_RUNS):
                acc = train_and_eval(dd_images, dd_labels, test_images, test_labels,
                                    label_type='soft', soft_labels=dd_soft, seed=r, verbose=(r==0))
                accs.append(acc)
                print(f"    Run {r+1}: {acc:.2f}%")
            results[f'{method_name.lower()}_ipc{ipc}_sl'] = {'mean': np.mean(accs), 'std': np.std(accs)}
            print(f"  SL: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
    
    # Save all results
    with open(f'{RESULTS_DIR}/results_all_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # ================================================================
    # GENERATE TABLE
    # ================================================================
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS TABLE - CIFAR-100, ConvNet-D3")
    print("=" * 80)
    
    # Paper's reference values
    paper = {
        'dm_ipc10_hl': 29.23, 'dm_ipc10_sl': 26.13,
        'dm_ipc50_hl': 42.32, 'dm_ipc50_sl': 43.46,
        'dc_ipc10_hl': 28.42, 'dc_ipc10_sl': 23.54,
        'dc_ipc50_hl': 30.56, 'dc_ipc50_sl': 33.46,
        'tm_ipc10_hl': 38.18, 'tm_ipc10_sl': 37.60,
        'tm_ipc50_hl': 46.32, 'tm_ipc50_sl': 46.26,
        'random_ipc10_hl': 18.64, 'random_ipc10_sl': 33.43,
        'random_ipc50_hl': 34.66, 'random_ipc50_sl': 45.39,
        'kcenter_ipc10_hl': 25.04, 'kcenter_ipc10_sl': 34.70,
        'kcenter_ipc50_hl': 38.64, 'kcenter_ipc50_sl': 46.24,
    }
    
    header = f"{'Method':<12} {'IPC':>4} | {'HL (ours)':>12} {'HL (paper)':>12} | {'SL (ours)':>12} {'SL (paper)':>12}"
    print(header)
    print("-" * len(header))
    
    table_lines = [header, "-" * len(header)]
    
    for method in ['DM', 'DC', 'TM', 'Random', 'K-centers']:
        for ipc in [10, 50]:
            mkey = method.lower().replace('-', '') if method != 'K-centers' else 'kcenter'
            hl_key = f'{mkey}_ipc{ipc}_hl'
            sl_key = f'{mkey}_ipc{ipc}_sl'
            
            hl_ours = results.get(hl_key, {})
            sl_ours = results.get(sl_key, {})
            
            hl_str = f"{hl_ours.get('mean', 0):.2f}±{hl_ours.get('std', 0):.2f}" if hl_ours else "N/A"
            sl_str = f"{sl_ours.get('mean', 0):.2f}±{sl_ours.get('std', 0):.2f}" if sl_ours else "N/A"
            
            hl_paper = paper.get(hl_key, 0)
            sl_paper = paper.get(sl_key, 0)
            
            line = f"{method:<12} {ipc:>4} | {hl_str:>12} {hl_paper:>12.2f} | {sl_str:>12} {sl_paper:>12.2f}"
            print(line)
            table_lines.append(line)
    
    # Save table
    with open(f'{RESULTS_DIR}/final_table.txt', 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"\nAll results saved to {RESULTS_DIR}/")
    return results


if __name__ == '__main__':
    main()
