"""
Complete experiment pipeline for replicating Table: tab:small_scale_c100
from "Rethinking Dataset Distillation: Hard Truths About Soft Labels"

Evaluates DD methods (DM, DC, TM) and coreset methods (Random, K-centers)
in both Hard Label (HL) and Soft Label (SL) settings on CIFAR-100 with ConvNet-D3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import sys
from convnet import ConvNet
from dsa import DiffAugment

# ============================================================
# Configuration
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 100
NUM_SEEDS = 3  # Paper uses multiple seeds; we use 3 for speed

# HL hyperparameters (from paper Table: tab:stage3_hyper)
HL_EPOCHS = 300
HL_LR = 0.01
HL_MOMENTUM = 0.9
HL_WD = 5e-4
HL_BATCH = 256
HL_STEP_EPOCH = 151
HL_GAMMA = 0.1

# SL hyperparameters (from paper Table: tab:stage3_hyper)
SL_EPOCHS = 300
SL_LR = 1e-3
SL_WD = 0.01
SL_BATCH = 256
SL_TEMP = 20.0

DSA_STRATEGY = 'color_crop_cutout_flip_scale_rotate'


def load_cifar100():
    """Load CIFAR-100 tensors."""
    data = torch.load('cifar100_tensors.pt', map_location='cpu')
    return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']


def evaluate_test(model, test_images, test_labels, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for j in range(0, len(test_images), 512):
            imgs = test_images[j:j+512].to(device)
            labels = test_labels[j:j+512].to(device)
            pred = model(imgs).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_hl(images, labels, test_images, test_labels, seed=0):
    """Train with Hard Labels using paper's exact hyperparameters."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=HL_LR, momentum=HL_MOMENTUM, weight_decay=HL_WD)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=HL_STEP_EPOCH, gamma=HL_GAMMA)
    
    n = len(images)
    for epoch in range(HL_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, HL_BATCH):
            idx = perm[i:i+HL_BATCH]
            imgs = images[idx].to(DEVICE)
            labs = labels[idx].to(DEVICE)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labs)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return evaluate_test(model, test_images, test_labels, DEVICE)


def train_sl(images, soft_labels, test_images, test_labels, seed=0):
    """Train with Soft Labels using paper's exact hyperparameters."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SL_LR, weight_decay=SL_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SL_EPOCHS)
    
    n = len(images)
    T = SL_TEMP
    for epoch in range(SL_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, SL_BATCH):
            idx = perm[i:i+SL_BATCH]
            imgs = images[idx].to(DEVICE)
            soft = soft_labels[idx].to(DEVICE)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            
            optimizer.zero_grad()
            out = model(imgs)
            log_p = F.log_softmax(out / T, dim=1)
            target_p = F.softmax(soft / T, dim=1)
            loss = F.kl_div(log_p, target_p, reduction='batchmean') * (T ** 2)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return evaluate_test(model, test_images, test_labels, DEVICE)


def get_random_subset(train_images, train_labels, ipc, seed=0):
    """Get random subset with ipc images per class."""
    np.random.seed(seed)
    indices = []
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0].numpy()
        sel = np.random.choice(cls_idx, ipc, replace=False)
        indices.extend(sel.tolist())
    return torch.tensor(indices)


def get_kcenters_subset(train_images, train_labels, ipc, seed=0):
    """Get K-centers subset using feature-space greedy selection."""
    torch.manual_seed(seed)
    
    # Use a pretrained ConvNet to extract features
    model = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(DEVICE)
    model.eval()
    
    # Extract features
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(DEVICE)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    all_features = torch.cat(all_features, dim=0)
    
    indices = []
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        cls_feat = all_features[cls_idx]
        
        # Greedy K-centers
        selected = []
        # Start with the sample closest to the class mean
        mean_feat = cls_feat.mean(0)
        dists = torch.cdist(cls_feat.unsqueeze(0), mean_feat.unsqueeze(0).unsqueeze(0)).squeeze()
        first = dists.argmin().item()
        selected.append(first)
        
        # Min distance to selected set
        min_dists = torch.cdist(cls_feat.unsqueeze(0), cls_feat[selected].unsqueeze(0)).squeeze(-1)
        if min_dists.dim() == 0:
            min_dists = min_dists.unsqueeze(0)
        
        for _ in range(ipc - 1):
            # Pick the point with maximum min-distance to selected set
            if len(selected) == 1:
                dists_to_selected = torch.cdist(cls_feat.unsqueeze(0), cls_feat[selected].unsqueeze(0)).squeeze()
            else:
                dists_to_selected = torch.cdist(cls_feat.unsqueeze(0), cls_feat[selected].unsqueeze(0)).squeeze(0)
                dists_to_selected = dists_to_selected.min(dim=1)[0]
            
            # Mask already selected
            for s in selected:
                dists_to_selected[s] = -1
            
            next_idx = dists_to_selected.argmax().item()
            selected.append(next_idx)
        
        # Map back to global indices
        for s in selected:
            indices.append(cls_idx[s].item())
    
    return torch.tensor(indices)


def run_experiment(name, images, labels, soft_labels, test_images, test_labels, num_seeds=NUM_SEEDS):
    """Run HL and SL experiments for a given dataset."""
    hl_accs = []
    sl_accs = []
    
    for seed in range(num_seeds):
        print(f"  {name} - HL seed {seed}...", end=' ', flush=True)
        t0 = time.time()
        hl_acc = train_hl(images, labels, test_images, test_labels, seed=seed)
        hl_accs.append(hl_acc)
        print(f"{hl_acc:.2f}% ({time.time()-t0:.0f}s)")
        
        print(f"  {name} - SL seed {seed}...", end=' ', flush=True)
        t0 = time.time()
        sl_acc = train_sl(images, soft_labels, test_images, test_labels, seed=seed)
        sl_accs.append(sl_acc)
        print(f"{sl_acc:.2f}% ({time.time()-t0:.0f}s)")
    
    hl_mean = np.mean(hl_accs)
    hl_std = np.std(hl_accs)
    sl_mean = np.mean(sl_accs)
    sl_std = np.std(sl_accs)
    
    return {
        'hl_mean': hl_mean, 'hl_std': hl_std, 'hl_accs': hl_accs,
        'sl_mean': sl_mean, 'sl_std': sl_std, 'sl_accs': sl_accs,
    }


def main():
    print("=" * 60)
    print("Replicating Table: tab:small_scale_c100")
    print("CIFAR-100, ConvNet-D3, HL vs SL evaluation")
    print("=" * 60)
    
    # Load data
    print("\nLoading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = load_cifar100()
    full_soft_labels = torch.load('soft_labels.pt', map_location='cpu')
    
    results = {}
    
    # ============================================================
    # DD Methods
    # ============================================================
    for method in ['dm', 'dc', 'tm']:
        for ipc in [10, 50]:
            key = f"{method}_ipc{ipc}"
            print(f"\n--- {method.upper()} IPC={ipc} ---")
            
            # Load distilled data
            dd_data = torch.load(f'distilled_{method}_ipc{ipc}.pt', map_location='cpu')
            dd_images = dd_data['images']
            dd_labels = dd_data['labels']
            dd_soft = torch.load(f'soft_labels_{method}_ipc{ipc}_correct.pt', map_location='cpu')
            
            print(f"  Loaded: {dd_images.shape[0]} images")
            results[key] = run_experiment(key, dd_images, dd_labels, dd_soft, 
                                          test_images, test_labels)
    
    # ============================================================
    # Coreset Methods
    # ============================================================
    for ipc in [10, 50]:
        # Random
        key = f"random_ipc{ipc}"
        print(f"\n--- Random IPC={ipc} ---")
        indices = get_random_subset(train_images, train_labels, ipc, seed=0)
        sub_images = train_images[indices]
        sub_labels = train_labels[indices]
        sub_soft = full_soft_labels[indices]
        results[key] = run_experiment(key, sub_images, sub_labels, sub_soft,
                                      test_images, test_labels)
        
        # K-centers
        key = f"kcenters_ipc{ipc}"
        print(f"\n--- K-centers IPC={ipc} ---")
        indices = get_kcenters_subset(train_images, train_labels, ipc, seed=0)
        sub_images = train_images[indices]
        sub_labels = train_labels[indices]
        sub_soft = full_soft_labels[indices]
        results[key] = run_experiment(key, sub_images, sub_labels, sub_soft,
                                      test_images, test_labels)
    
    # ============================================================
    # Print Results Table
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS TABLE: tab:small_scale_c100 (CIFAR-100, ConvNet-D3)")
    print("=" * 70)
    print(f"{'Method':<12} {'IPC':>4} {'HL (ours)':>14} {'SL (ours)':>14} {'HL (paper)':>14} {'SL (paper)':>14}")
    print("-" * 70)
    
    paper_results = {
        'dm_ipc10':      {'hl': '29.23±0.26', 'sl': '26.13±0.10'},
        'dm_ipc50':      {'hl': '42.32±0.37', 'sl': '43.46±0.18'},
        'dc_ipc10':      {'hl': '28.42±0.29', 'sl': '23.54±0.31'},
        'dc_ipc50':      {'hl': '30.56±0.56', 'sl': '33.46±0.38'},
        'tm_ipc10':      {'hl': '38.18±0.42', 'sl': '37.60±0.25'},
        'tm_ipc50':      {'hl': '46.32±0.26', 'sl': '46.26±0.30'},
        'random_ipc10':  {'hl': '18.64±0.25', 'sl': '33.43±0.18'},
        'random_ipc50':  {'hl': '34.66±0.41', 'sl': '45.39±0.23'},
        'kcenters_ipc10':{'hl': '25.04±0.30', 'sl': '34.70±0.27'},
        'kcenters_ipc50':{'hl': '38.64±0.43', 'sl': '46.24±0.12'},
    }
    
    display_order = [
        ('DM', 10, 'dm_ipc10'), ('DM', 50, 'dm_ipc50'),
        ('DC', 10, 'dc_ipc10'), ('DC', 50, 'dc_ipc50'),
        ('TM', 10, 'tm_ipc10'), ('TM', 50, 'tm_ipc50'),
        ('Random', 10, 'random_ipc10'), ('Random', 50, 'random_ipc50'),
        ('K-centers', 10, 'kcenters_ipc10'), ('K-centers', 50, 'kcenters_ipc50'),
    ]
    
    for method_name, ipc, key in display_order:
        r = results[key]
        ours_hl = f"{r['hl_mean']:.2f}±{r['hl_std']:.2f}"
        ours_sl = f"{r['sl_mean']:.2f}±{r['sl_std']:.2f}"
        paper_hl = paper_results[key]['hl']
        paper_sl = paper_results[key]['sl']
        print(f"{method_name:<12} {ipc:>4} {ours_hl:>14} {ours_sl:>14} {paper_hl:>14} {paper_sl:>14}")
    
    # ============================================================
    # Analysis of Key Claims
    # ============================================================
    print("\n" + "=" * 70)
    print("KEY CLAIMS ANALYSIS")
    print("=" * 70)
    
    # Claim 1: In HL, DD methods >> coresets
    tm10_hl = results['tm_ipc10']['hl_mean']
    rand10_hl = results['random_ipc10']['hl_mean']
    kc10_hl = results['kcenters_ipc10']['hl_mean']
    print(f"\n1. HL gap (IPC10): TM={tm10_hl:.1f}% vs Random={rand10_hl:.1f}% (gap={tm10_hl-rand10_hl:.1f}%)")
    print(f"   Paper: TM=38.2% vs Random=18.6% (gap=19.6%)")
    
    # Claim 2: In SL, gap narrows
    tm10_sl = results['tm_ipc10']['sl_mean']
    rand10_sl = results['random_ipc10']['sl_mean']
    kc10_sl = results['kcenters_ipc10']['sl_mean']
    print(f"\n2. SL gap (IPC10): TM={tm10_sl:.1f}% vs Random={rand10_sl:.1f}% (gap={tm10_sl-rand10_sl:.1f}%)")
    print(f"   Paper: TM=37.6% vs Random=33.4% (gap=4.2%)")
    
    # Claim 3: TM is best DD method
    for ipc in [10, 50]:
        tm_hl = results[f'tm_ipc{ipc}']['hl_mean']
        dm_hl = results[f'dm_ipc{ipc}']['hl_mean']
        dc_hl = results[f'dc_ipc{ipc}']['hl_mean']
        print(f"\n3. Best DD (IPC{ipc} HL): TM={tm_hl:.1f}%, DM={dm_hl:.1f}%, DC={dc_hl:.1f}%")
    
    # Claim 4: K-centers > Random in HL
    for ipc in [10, 50]:
        kc_hl = results[f'kcenters_ipc{ipc}']['hl_mean']
        rand_hl = results[f'random_ipc{ipc}']['hl_mean']
        kc_sl = results[f'kcenters_ipc{ipc}']['sl_mean']
        rand_sl = results[f'random_ipc{ipc}']['sl_mean']
        print(f"\n4. Coreset gap (IPC{ipc}): HL: K-centers={kc_hl:.1f}% vs Random={rand_hl:.1f}% (gap={kc_hl-rand_hl:.1f}%)")
        print(f"   SL: K-centers={kc_sl:.1f}% vs Random={rand_sl:.1f}% (gap={kc_sl-rand_sl:.1f}%)")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Convert numpy types for JSON serialization
    json_results = {}
    for k, v in results.items():
        json_results[k] = {
            'hl_mean': float(v['hl_mean']),
            'hl_std': float(v['hl_std']),
            'hl_accs': [float(x) for x in v['hl_accs']],
            'sl_mean': float(v['sl_mean']),
            'sl_std': float(v['sl_std']),
            'sl_accs': [float(x) for x in v['sl_accs']],
        }
    
    with open('results/experiment_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save formatted table
    with open('results/table_small_scale_c100.txt', 'w') as f:
        f.write("Table: tab:small_scale_c100 (CIFAR-100, ConvNet-D3)\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Method':<12} {'IPC':>4} {'HL (ours)':>14} {'SL (ours)':>14} {'HL (paper)':>14} {'SL (paper)':>14}\n")
        f.write("-" * 70 + "\n")
        for method_name, ipc, key in display_order:
            r = results[key]
            ours_hl = f"{r['hl_mean']:.2f}±{r['hl_std']:.2f}"
            ours_sl = f"{r['sl_mean']:.2f}±{r['sl_std']:.2f}"
            paper_hl = paper_results[key]['hl']
            paper_sl = paper_results[key]['sl']
            f.write(f"{method_name:<12} {ipc:>4} {ours_hl:>14} {ours_sl:>14} {paper_hl:>14} {paper_sl:>14}\n")
    
    print(f"\nResults saved to results/experiment_results.json")
    print(f"Table saved to results/table_small_scale_c100.txt")


if __name__ == '__main__':
    main()
