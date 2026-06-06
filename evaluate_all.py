"""
Unified evaluation script for all methods in Table 1 of the paper.
Loads DCBench distilled datasets, evaluates with HL and SL settings.

Paper: "Rethinking Dataset Distillation: Hard Truths About Soft Labels"
Target: Table 1 (tab:small_scale_c100) - CIFAR-100, ConvNet-D3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
import sys
import time
import argparse

from convnet import get_convnet_d3
from dsa import DiffAugment


# ============================================================
# Data Loading
# ============================================================

def load_dcbench_data(method, ipc):
    """Load distilled dataset from DCBench."""
    base = 'dcbench_data/data/condensed'
    
    if method in ['DC', 'DM', 'DSA']:
        path = f'{base}/{method}/CIFAR100/res_{method}_CIFAR100_ConvNet_{ipc}ipc.pt'
        d = torch.load(path, map_location='cpu', weights_only=False)
        images = d['data'][0][0]
        labels = d['data'][0][1]
    elif method == 'TM':
        images = torch.load(f'{base}/TM/CIFAR100/IPC{ipc}/images_best.pt', map_location='cpu', weights_only=False)
        labels = torch.load(f'{base}/TM/CIFAR100/IPC{ipc}/labels_best.pt', map_location='cpu', weights_only=False)
    elif method == 'Random':
        images = torch.load(f'{base}/random/CIFAR100/CIFAR100_IPC{ipc}_normalize_images.pt', map_location='cpu', weights_only=False)
        labels = torch.load(f'{base}/random/CIFAR100/CIFAR100_IPC{ipc}_normalize_labels.pt', map_location='cpu', weights_only=False)
    elif method == 'K-centers':
        images = torch.load(f'{base}/kmeans-emb/CIFAR100/CIFAR100_IPC{ipc}_images.pt', map_location='cpu', weights_only=False)
        labels = torch.load(f'{base}/kmeans-emb/CIFAR100/CIFAR100_IPC{ipc}_labels.pt', map_location='cpu', weights_only=False)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    images = images.float()
    labels = labels.long()
    
    return images, labels


def load_test_data():
    """Load CIFAR-100 test set, normalized."""
    import torchvision
    import torchvision.transforms as transforms
    
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    
    return testloader


def generate_soft_labels(images, labels, teacher_path='teacher.pt', device='cuda'):
    """Generate soft labels for distilled images using teacher model."""
    teacher = get_convnet_d3(num_classes=100).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher.eval()
    
    all_logits = []
    with torch.no_grad():
        # Process in batches to avoid OOM
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            logits = teacher(batch)
            all_logits.append(logits.cpu())
    
    soft_labels = torch.cat(all_logits, dim=0)  # Raw logits
    return soft_labels


# ============================================================
# Training Functions
# ============================================================

def evaluate(model, testloader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def train_hl(images, labels, testloader, device='cuda', seed=0):
    """
    Train with Hard Labels (HL) setting from paper.
    - 300 epochs, SGD, lr=0.01, momentum=0.9, weight_decay=5e-4
    - StepLR@epoch151 (gamma=0.1), batch=256, DSA augmentation, CE loss
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3(num_classes=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    images_gpu = images.to(device)
    labels_gpu = labels.to(device)
    n = len(images)
    batch_size = min(256, n)
    
    for epoch in range(300):
        model.train()
        # Shuffle
        perm = torch.randperm(n)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x = images_gpu[idx]
            y = labels_gpu[idx]
            
            # DSA augmentation
            x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
    
    acc = evaluate(model, testloader, device)
    return acc


def train_sl(images, labels, soft_labels, testloader, device='cuda', seed=0, temperature=20.0):
    """
    Train with Soft Labels (SL) setting from paper.
    - 300 epochs, AdamW, lr=1e-3, weight_decay=0.01, Cosine scheduler
    - batch=256, DSA augmentation, KL-Div(T=20), NO warmup
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3(num_classes=100).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    
    images_gpu = images.to(device)
    labels_gpu = labels.to(device)  # Hard labels (for reference)
    soft_labels_gpu = soft_labels.to(device)  # Teacher logits
    
    n = len(images)
    batch_size = min(256, n)
    T = temperature
    
    for epoch in range(300):
        model.train()
        perm = torch.randperm(n)
        
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x = images_gpu[idx]
            sl = soft_labels_gpu[idx]
            
            # DSA augmentation
            x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(x)
            
            # KL divergence with temperature scaling
            # KL(teacher || student) = sum(p * log(p/q))
            # Using PyTorch convention: KLDivLoss expects log-probs as input, probs as target
            log_student = F.log_softmax(out / T, dim=1)
            teacher_probs = F.softmax(sl / T, dim=1)
            loss = F.kl_div(log_student, teacher_probs, reduction='batchmean') * (T * T)
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    acc = evaluate(model, testloader, device)
    return acc


# ============================================================
# Main Experiment Runner
# ============================================================

def run_experiment(method, ipc, label_type, num_runs=3, device='cuda'):
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Method: {method}, IPC: {ipc}, Labels: {label_type}")
    print(f"{'='*60}")
    
    # Load data
    images, labels = load_dcbench_data(method, ipc)
    print(f"Loaded {method} IPC{ipc}: images={images.shape}, labels={labels.shape}")
    
    # Load test data
    testloader = load_test_data()
    
    # Generate soft labels if needed
    soft_labels = None
    if label_type == 'SL':
        soft_labels = generate_soft_labels(images, labels, device=device)
        print(f"Generated soft labels: {soft_labels.shape}")
    
    # Run multiple seeds
    accs = []
    for run in range(num_runs):
        seed = run * 42
        t0 = time.time()
        
        if label_type == 'HL':
            acc = train_hl(images, labels, testloader, device=device, seed=seed)
        else:
            acc = train_sl(images, labels, soft_labels, testloader, device=device, seed=seed)
        
        elapsed = time.time() - t0
        accs.append(acc)
        print(f"  Run {run+1}/{num_runs}: {acc:.2f}% ({elapsed:.1f}s)")
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"  Result: {mean_acc:.2f} ± {std_acc:.2f}%")
    
    return {
        'method': method,
        'ipc': ipc,
        'label_type': label_type,
        'accs': accs,
        'mean': float(mean_acc),
        'std': float(std_acc),
    }


def run_all_experiments(num_runs=3, device='cuda'):
    """Run all experiments for Table 1."""
    methods = ['DC', 'DM', 'TM', 'Random', 'K-centers']
    ipcs = [10, 50]
    label_types = ['HL', 'SL']
    
    results = []
    
    for method in methods:
        for ipc in ipcs:
            for lt in label_types:
                result = run_experiment(method, ipc, lt, num_runs=num_runs, device=device)
                results.append(result)
                
                # Save incrementally
                os.makedirs('results', exist_ok=True)
                with open('results/table1_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
    
    return results


def print_table(results):
    """Print results in table format matching paper."""
    print("\n" + "="*70)
    print("Table 1: CIFAR-100, ConvNet-D3 Evaluation")
    print("="*70)
    print(f"{'Method':<12} {'IPC':>4} {'HL (Ours)':>14} {'HL (Paper)':>14} {'SL (Ours)':>14} {'SL (Paper)':>14}")
    print("-"*70)
    
    # Paper values for comparison
    paper = {
        ('DC', 10): (28.42, 23.54),
        ('DC', 50): (30.56, 33.46),
        ('DM', 10): (29.23, 26.13),
        ('DM', 50): (42.32, 43.46),
        ('TM', 10): (38.18, 37.60),
        ('TM', 50): (46.32, 46.26),
        ('Random', 10): (18.64, 33.43),
        ('Random', 50): (34.66, 45.39),
        ('K-centers', 10): (25.04, 34.70),
        ('K-centers', 50): (38.64, 46.24),
    }
    
    # Organize results
    result_dict = {}
    for r in results:
        key = (r['method'], r['ipc'], r['label_type'])
        result_dict[key] = r
    
    for method in ['DC', 'DM', 'TM', 'Random', 'K-centers']:
        for ipc in [10, 50]:
            hl_key = (method, ipc, 'HL')
            sl_key = (method, ipc, 'SL')
            
            hl_str = f"{result_dict[hl_key]['mean']:.2f}±{result_dict[hl_key]['std']:.2f}" if hl_key in result_dict else "---"
            sl_str = f"{result_dict[sl_key]['mean']:.2f}±{result_dict[sl_key]['std']:.2f}" if sl_key in result_dict else "---"
            
            paper_hl, paper_sl = paper.get((method, ipc), (0, 0))
            
            print(f"{method:<12} {ipc:>4} {hl_str:>14} {paper_hl:>10.2f}    {sl_str:>14} {paper_sl:>10.2f}")
    
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default=None, help='Single method to run')
    parser.add_argument('--ipc', type=int, default=None, help='Single IPC to run')
    parser.add_argument('--label_type', type=str, default=None, choices=['HL', 'SL'])
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs per config')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if args.method and args.ipc and args.label_type:
        # Run single experiment
        result = run_experiment(args.method, args.ipc, args.label_type, 
                              num_runs=args.num_runs, device=args.device)
        os.makedirs('results', exist_ok=True)
        fname = f"results/{args.method}_{args.ipc}_{args.label_type}.json"
        with open(fname, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        # Run all experiments
        results = run_all_experiments(num_runs=args.num_runs, device=args.device)
        print_table(results)
