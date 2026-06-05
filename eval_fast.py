#!/usr/bin/env python3
"""
Fast evaluation script with cached data loading.
Matches paper's exact hyperparameters.
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
from collections import defaultdict

from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_cached_data():
    """Load cached CIFAR-100 tensors."""
    cache_path = '/workspace/cifar100_tensors.pt'
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location='cpu')
        return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']
    else:
        from data_utils import get_cifar100_tensors
        return get_cifar100_tensors()


def test_accuracy(model, test_images, test_labels, batch_size=512):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            x = test_images[i:i+batch_size].to(DEVICE)
            y = test_labels[i:i+batch_size].to(DEVICE)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return 100.0 * correct / total


def evaluate_hl(images, labels, test_images, test_labels, num_runs=3, epochs=300, seed=0):
    """Hard Label: 300 epochs, SGD lr=0.01, StepLR@151 gamma=0.5, batch=256, DSA, CE"""
    accs = []
    for run in range(num_runs):
        torch.manual_seed(seed + run * 100)
        np.random.seed(seed + run * 100)
        
        model = get_convnet_d3().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
        
        n = len(images)
        batch_size = min(256, n)
        
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                x = images[idx].to(DEVICE)
                y = labels[idx].to(DEVICE)
                x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate')
                
                optimizer.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
        
        acc = test_accuracy(model, test_images, test_labels)
        accs.append(acc)
        print(f"  [HL Run {run+1}] {acc:.2f}%")
    
    return np.mean(accs), np.std(accs), accs


def evaluate_sl(images, soft_labels, test_images, test_labels, num_runs=3, epochs=300, seed=0, temperature=20):
    """Soft Label: 300 epochs, AdamW lr=1e-3, Cosine, batch=256, DSA, KL-Div(T=20)"""
    accs = []
    for run in range(num_runs):
        torch.manual_seed(seed + run * 100)
        np.random.seed(seed + run * 100)
        
        model = get_convnet_d3().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        n = len(images)
        batch_size = min(256, n)
        T = temperature
        
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                x = images[idx].to(DEVICE)
                sl = soft_labels[idx].to(DEVICE)
                x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate')
                
                optimizer.zero_grad()
                out = model(x)
                
                log_student = F.log_softmax(out / T, dim=1)
                teacher_prob = F.softmax(sl / T, dim=1)
                loss = F.kl_div(log_student, teacher_prob, reduction='batchmean') * (T * T)
                
                loss.backward()
                optimizer.step()
            
            scheduler.step()
        
        acc = test_accuracy(model, test_images, test_labels)
        accs.append(acc)
        print(f"  [SL Run {run+1}] {acc:.2f}%")
    
    return np.mean(accs), np.std(accs), accs


def get_class_indices(labels, num_classes=100):
    class_indices = defaultdict(list)
    for i in range(len(labels)):
        class_indices[int(labels[i])].append(i)
    return class_indices


def select_random(train_labels, ipc, seed=42):
    np.random.seed(seed)
    class_idx = get_class_indices(train_labels)
    indices = []
    for c in range(100):
        chosen = np.random.choice(class_idx[c], size=ipc, replace=False)
        indices.extend(chosen.tolist())
    return sorted(indices)


def select_kcenter(train_images, train_labels, ipc, seed=42):
    """K-centers with pretrained features."""
    # Load pretrained teacher
    teacher_path = '/workspace/teacher_for_kcenter.pt'
    model = get_convnet_d3().to(DEVICE)
    model.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
    model.eval()
    
    # Extract features
    print("  Extracting features...")
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(DEVICE)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    features = torch.cat(all_features, dim=0).numpy()
    print(f"  Features shape: {features.shape}")
    
    # K-centers per class
    np.random.seed(seed)
    class_idx = get_class_indices(train_labels)
    indices = []
    
    for c in range(100):
        cls_indices = np.array(class_idx[c])
        feats = features[cls_indices]
        
        # Normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.maximum(norms, 1e-8)
        
        # Farthest-first traversal
        chosen = [np.random.randint(len(cls_indices))]
        min_dists = np.full(len(cls_indices), np.inf)
        
        for _ in range(ipc - 1):
            last = chosen[-1]
            dists = np.sum((feats - feats[last:last+1]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, dists)
            for c_idx in chosen:
                min_dists[c_idx] = -1
            next_idx = np.argmax(min_dists)
            chosen.append(next_idx)
        
        indices.extend([int(cls_indices[c_idx]) for c_idx in chosen])
    
    return sorted(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--ipc', type=int, required=True)
    parser.add_argument('--mode', type=str, default='both', choices=['hl', 'sl', 'both'])
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()
    
    print(f"Loading data...")
    train_images, train_labels, test_images, test_labels = load_cached_data()
    all_soft_labels = torch.load('/workspace/soft_labels_clean.pt', map_location='cpu')
    
    print(f"Method: {args.method}, IPC: {args.ipc}")
    
    if args.method == 'random':
        indices = select_random(train_labels, args.ipc)
        images = train_images[indices]
        labels = train_labels[indices]
        soft_labs = all_soft_labels[indices]
        
    elif args.method == 'kcenter':
        indices = select_kcenter(train_images, train_labels, args.ipc)
        images = train_images[indices]
        labels = train_labels[indices]
        soft_labs = all_soft_labels[indices]
        
    elif args.method in ['dm', 'dc', 'tm']:
        distilled_path = f'/workspace/distilled_{args.method}_ipc{args.ipc}.pt'
        if not os.path.exists(distilled_path):
            print(f"ERROR: {distilled_path} not found. Run distillation first.")
            sys.exit(1)
        
        data = torch.load(distilled_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            images = data.get('images', data.get('data'))
            labels = data.get('labels', data.get('targets'))
        elif isinstance(data, torch.Tensor):
            images = data
            labels = torch.arange(100).repeat_interleave(args.ipc)
        
        # Generate soft labels for distilled data
        sl_path = f'/workspace/soft_labels_{args.method}_ipc{args.ipc}.pt'
        if os.path.exists(sl_path):
            soft_labs = torch.load(sl_path, map_location='cpu')
            if isinstance(soft_labs, dict):
                soft_labs = soft_labs.get('soft_labels', soft_labs.get('logits'))
        else:
            # Generate from teacher
            teacher_path = '/workspace/teacher_for_kcenter.pt'
            model = get_convnet_d3().to(DEVICE)
            model.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
            model.eval()
            with torch.no_grad():
                sl_list = []
                for i in range(0, len(images), 256):
                    batch = images[i:i+256].to(DEVICE)
                    logits = model(batch)
                    sl_list.append(logits.cpu())
                soft_labs = torch.cat(sl_list, dim=0)
            torch.save(soft_labs, sl_path)
    
    print(f"Dataset: {images.shape}, Labels: {labels.shape}")
    
    result = {'method': args.method, 'ipc': args.ipc}
    
    if args.mode in ['hl', 'both']:
        print(f"\n--- HL Evaluation ---")
        t0 = time.time()
        hl_mean, hl_std, hl_accs = evaluate_hl(images, labels, test_images, test_labels, 
                                                 num_runs=args.num_runs)
        print(f"HL: {hl_mean:.2f} ± {hl_std:.2f} (took {time.time()-t0:.0f}s)")
        result['hl_mean'] = hl_mean
        result['hl_std'] = hl_std
        result['hl_accs'] = hl_accs
    
    if args.mode in ['sl', 'both']:
        print(f"\n--- SL Evaluation ---")
        t0 = time.time()
        sl_mean, sl_std, sl_accs = evaluate_sl(images, soft_labs, test_images, test_labels,
                                                 num_runs=args.num_runs)
        print(f"SL: {sl_mean:.2f} ± {sl_std:.2f} (took {time.time()-t0:.0f}s)")
        result['sl_mean'] = sl_mean
        result['sl_std'] = sl_std
        result['sl_accs'] = sl_accs
    
    # Save
    os.makedirs('/workspace/results', exist_ok=True)
    result_file = f'/workspace/results/{args.method}_ipc{args.ipc}_{args.mode}.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {result_file}")
    
    return result


if __name__ == '__main__':
    main()
