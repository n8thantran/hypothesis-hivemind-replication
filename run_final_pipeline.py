"""
Final evaluation pipeline for Table 1 (tab:small_scale_c100) replication.
CIFAR-100, ConvNet-D3, all methods × IPCs × {HL, SL}

Paper target (CIFAR-100, ConvNet-D3):
| Method    | IPC | HL          | SL          |
|-----------|-----|-------------|-------------|
| DM        | 10  | 29.23±0.26  | 26.13±0.10  |
| DM        | 50  | 42.32±0.37  | 43.46±0.18  |
| DC        | 10  | 28.42±0.29  | 23.54±0.31  |
| DC        | 50  | 30.56±0.56  | 33.46±0.38  |
| TM        | 10  | 38.18±0.42  | 37.60±0.25  |
| TM        | 50  | 46.32±0.26  | 46.26±0.30  |
| Random    | 10  | 18.64±0.25  | 33.43±0.18  |
| Random    | 50  | 34.66±0.41  | 45.39±0.23  |
| K-centers | 10  | 25.04±0.30  | 34.70±0.27  |
| K-centers | 50  | 38.64±0.43  | 46.24±0.12  |
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import sys
import os

from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, random_select, k_centers_select, get_class_indices

DEVICE = 'cuda'


def train_and_eval(train_images, train_labels, test_images, test_labels,
                   label_type='hard', soft_labels=None, seed=0, epochs=300,
                   batch_size=256, verbose=True):
    """
    Train ConvNet-D3 on given data and evaluate.
    
    HL: SGD, lr=0.01, momentum=0.9, wd=5e-4, StepLR@151(gamma=0.1), CE, DSA
    SL: AdamW, lr=1e-3, wd=0.01, CosineAnnealing, KL-Div(T=20), DSA
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3().to(DEVICE)
    
    if label_type == 'hard':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        T = 20.0
    
    n = len(train_images)
    bs = min(batch_size, n)
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            imgs = train_images[idx].to(DEVICE)
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(imgs)
            
            if label_type == 'hard':
                loss = F.cross_entropy(out, train_labels[idx].to(DEVICE))
            else:
                sl = soft_labels[idx].to(DEVICE)
                log_p = F.log_softmax(out / T, dim=1)
                tgt = F.softmax(sl / T, dim=1)
                loss = F.kl_div(log_p, tgt, reduction='batchmean') * (T ** 2)
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            acc = eval_model(model, test_images, test_labels)
            print(f"  Epoch {epoch+1}/{epochs}, Acc: {acc:.2f}%")
    
    acc = eval_model(model, test_images, test_labels)
    return acc


def eval_model(model, test_images, test_labels, batch_size=512):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            x = test_images[i:i+batch_size].to(DEVICE)
            y = test_labels[i:i+batch_size].to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return 100.0 * correct / total


def load_distilled(method, ipc):
    """Load distilled dataset."""
    fname = f'distilled_{method}_ipc{ipc}.pt'
    d = torch.load(fname, map_location='cpu')
    return d['images'], d['labels']


def load_soft_labels_for_distilled(method, ipc):
    """Load soft labels for distilled dataset."""
    fname = f'soft_labels_{method}_ipc{ipc}_v2.pt'
    return torch.load(fname, map_location='cpu')


def get_coreset_soft_labels(indices, full_soft_labels):
    """Get soft labels for coreset indices from full training set soft labels."""
    return full_soft_labels[indices]


def main():
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    print("=" * 60)
    print("FINAL EVALUATION PIPELINE")
    print(f"Runs per experiment: {num_runs}")
    print("=" * 60)
    
    # Load data
    print("\nLoading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Load teacher soft labels for full training set
    full_soft_labels = torch.load('soft_labels.pt', map_location='cpu')
    print(f"Full soft labels: {full_soft_labels.shape}")
    
    # Load teacher for K-centers feature extraction
    teacher_state = torch.load('teacher.pt', map_location='cpu')
    teacher_model = get_convnet_d3().to(DEVICE)
    teacher_model.load_state_dict(teacher_state['model_state_dict'])
    teacher_model.eval()
    print(f"Teacher accuracy: {teacher_state['accuracy']}%")
    
    results = {}
    
    # ============================================================
    # DD Methods: DM, DC, TM
    # ============================================================
    for method in ['dm', 'dc', 'tm']:
        for ipc in [10, 50]:
            key = f"{method}_ipc{ipc}"
            print(f"\n{'='*60}")
            print(f"Evaluating {method.upper()} IPC={ipc}")
            print(f"{'='*60}")
            
            images, labels = load_distilled(method, ipc)
            soft_labels = load_soft_labels_for_distilled(method, ipc)
            
            print(f"  Data: {images.shape}, Labels: {labels.shape}, SL: {soft_labels.shape}")
            
            hl_accs = []
            sl_accs = []
            
            for run in range(num_runs):
                print(f"\n  --- HL Run {run+1}/{num_runs} ---")
                acc = train_and_eval(images, labels, test_images, test_labels,
                                     label_type='hard', seed=run)
                hl_accs.append(acc)
                print(f"  HL accuracy: {acc:.2f}%")
                
                print(f"\n  --- SL Run {run+1}/{num_runs} ---")
                acc = train_and_eval(images, labels, test_images, test_labels,
                                     label_type='soft', soft_labels=soft_labels, seed=run)
                sl_accs.append(acc)
                print(f"  SL accuracy: {acc:.2f}%")
            
            results[key] = {
                'method': method.upper(),
                'ipc': ipc,
                'hl_mean': np.mean(hl_accs),
                'hl_std': np.std(hl_accs),
                'hl_accs': hl_accs,
                'sl_mean': np.mean(sl_accs),
                'sl_std': np.std(sl_accs),
                'sl_accs': sl_accs,
            }
            
            # Save intermediate results
            with open('results/pipeline_results.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    # ============================================================
    # Coreset Methods: Random, K-centers
    # ============================================================
    for ipc in [10, 50]:
        # Random
        key = f"random_ipc{ipc}"
        print(f"\n{'='*60}")
        print(f"Evaluating Random IPC={ipc}")
        print(f"{'='*60}")
        
        hl_accs = []
        sl_accs = []
        
        for run in range(num_runs):
            indices = random_select(train_labels, ipc=ipc, seed=run)
            images = train_images[indices]
            labels = train_labels[indices]
            soft_labels = get_coreset_soft_labels(indices, full_soft_labels)
            
            print(f"\n  --- HL Run {run+1}/{num_runs} ---")
            acc = train_and_eval(images, labels, test_images, test_labels,
                                 label_type='hard', seed=run)
            hl_accs.append(acc)
            print(f"  HL accuracy: {acc:.2f}%")
            
            print(f"\n  --- SL Run {run+1}/{num_runs} ---")
            acc = train_and_eval(images, labels, test_images, test_labels,
                                 label_type='soft', soft_labels=soft_labels, seed=run)
            sl_accs.append(acc)
            print(f"  SL accuracy: {acc:.2f}%")
        
        results[key] = {
            'method': 'Random',
            'ipc': ipc,
            'hl_mean': np.mean(hl_accs),
            'hl_std': np.std(hl_accs),
            'hl_accs': hl_accs,
            'sl_mean': np.mean(sl_accs),
            'sl_std': np.std(sl_accs),
            'sl_accs': sl_accs,
        }
        
        with open('results/pipeline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # K-centers (using teacher features)
        key = f"kcenter_ipc{ipc}"
        print(f"\n{'='*60}")
        print(f"Evaluating K-centers IPC={ipc}")
        print(f"{'='*60}")
        
        hl_accs = []
        sl_accs = []
        
        for run in range(num_runs):
            indices = k_centers_select(train_images, train_labels, ipc=ipc,
                                       use_features=True, feature_model=teacher_model,
                                       seed=run, device=DEVICE)
            images = train_images[indices]
            labels = train_labels[indices]
            soft_labels = get_coreset_soft_labels(indices, full_soft_labels)
            
            print(f"\n  --- HL Run {run+1}/{num_runs} ---")
            acc = train_and_eval(images, labels, test_images, test_labels,
                                 label_type='hard', seed=run)
            hl_accs.append(acc)
            print(f"  HL accuracy: {acc:.2f}%")
            
            print(f"\n  --- SL Run {run+1}/{num_runs} ---")
            acc = train_and_eval(images, labels, test_images, test_labels,
                                 label_type='soft', soft_labels=soft_labels, seed=run)
            sl_accs.append(acc)
            print(f"  SL accuracy: {acc:.2f}%")
        
        results[key] = {
            'method': 'K-centers',
            'ipc': ipc,
            'hl_mean': np.mean(hl_accs),
            'hl_std': np.std(hl_accs),
            'hl_accs': hl_accs,
            'sl_mean': np.mean(sl_accs),
            'sl_std': np.std(sl_accs),
            'sl_accs': sl_accs,
        }
        
        with open('results/pipeline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # ============================================================
    # Print final table
    # ============================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE (tab:small_scale_c100)")
    print("=" * 80)
    print(f"{'Method':<12} {'IPC':>4} {'HL (ours)':>12} {'HL (paper)':>12} {'SL (ours)':>12} {'SL (paper)':>12}")
    print("-" * 80)
    
    paper = {
        'dm_ipc10': (29.23, 26.13), 'dm_ipc50': (42.32, 43.46),
        'dc_ipc10': (28.42, 23.54), 'dc_ipc50': (30.56, 33.46),
        'tm_ipc10': (38.18, 37.60), 'tm_ipc50': (46.32, 46.26),
        'random_ipc10': (18.64, 33.43), 'random_ipc50': (34.66, 45.39),
        'kcenter_ipc10': (25.04, 34.70), 'kcenter_ipc50': (38.64, 46.24),
    }
    
    for key in ['dm_ipc10', 'dm_ipc50', 'dc_ipc10', 'dc_ipc50', 'tm_ipc10', 'tm_ipc50',
                'random_ipc10', 'random_ipc50', 'kcenter_ipc10', 'kcenter_ipc50']:
        if key in results:
            r = results[key]
            p = paper[key]
            hl_str = f"{r['hl_mean']:.2f}±{r['hl_std']:.2f}"
            sl_str = f"{r['sl_mean']:.2f}±{r['sl_std']:.2f}"
            print(f"{r['method']:<12} {r['ipc']:>4} {hl_str:>12} {p[0]:>12.2f} {sl_str:>12} {p[1]:>12.2f}")
    
    print("=" * 80)
    
    # Save final results
    with open('results/pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/pipeline_results.json")


if __name__ == '__main__':
    main()
