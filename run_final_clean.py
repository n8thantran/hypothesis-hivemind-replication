"""
Final comprehensive experiment runner.
Runs ALL experiments for Table 1 (small_scale_c100):
- 5 methods: Random, K-centers, DM, DC, TM
- 2 IPC values: 10, 50
- 2 label types: HL (hard label), SL (soft label)
= 20 configurations, 3 trials each = 60 evaluations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import sys

# Import our modules
from convnet import ConvNet, get_convnet_d3
from data_utils import get_cifar100_tensors, get_class_indices, random_select
from train_eval import train_and_evaluate, evaluate
from dsa import DiffAugment

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_RUNS = 3
NUM_CLASSES = 100


def k_centers_select_with_teacher(train_images, train_labels, ipc, teacher_path, device='cuda', seed=0):
    """K-centers selection using pretrained teacher features."""
    np.random.seed(seed)
    
    # Load teacher model
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    state = torch.load(teacher_path, map_location=device)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    
    # Extract features
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(device)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    features_np = torch.cat(all_features, dim=0).numpy()
    
    # Per-class greedy farthest-first traversal
    class_indices = {}
    for i in range(len(train_labels)):
        c = int(train_labels[i])
        if c not in class_indices:
            class_indices[c] = []
        class_indices[c].append(i)
    
    selected = []
    for c in range(NUM_CLASSES):
        indices = np.array(class_indices[c])
        features = features_np[indices]
        
        # Start with random point
        first = np.random.randint(len(indices))
        chosen = [first]
        dists = np.full(len(indices), np.inf)
        
        for _ in range(ipc - 1):
            last = chosen[-1]
            new_dists = np.sum((features - features[last:last+1]) ** 2, axis=1)
            dists = np.minimum(dists, new_dists)
            next_idx = np.argmax(dists)
            chosen.append(next_idx)
        
        selected.extend([int(indices[c_idx]) for c_idx in chosen])
    
    return sorted(selected)


def generate_soft_labels_for_subset(sub_images, teacher_path, device='cuda'):
    """Generate soft labels for a subset using the teacher model."""
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    state = torch.load(teacher_path, map_location=device)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(sub_images), 256):
            batch = sub_images[i:i+256].to(device)
            logits = model(batch)
            logits_list.append(logits.cpu())
    return torch.cat(logits_list, dim=0)


def run_eval(train_imgs, train_labels, test_imgs, test_labels, label_type, 
             soft_labels=None, num_runs=NUM_RUNS):
    """Run evaluation with multiple trials."""
    model_fn = lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))
    accs = []
    for run in range(num_runs):
        acc = train_and_evaluate(
            train_imgs, train_labels, test_imgs, test_labels,
            model_fn, num_classes=100, device=DEVICE,
            label_type=label_type, soft_labels=soft_labels,
            epochs=300, batch_size=256, seed=run, verbose=False
        )
        accs.append(acc)
        print(f"    Run {run+1}: {acc:.2f}%")
    return accs


def main():
    print("="*60)
    print("FINAL EXPERIMENT RUNNER")
    print("="*60)
    
    # Load data
    print("\n[1] Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Move test data to GPU for faster eval
    test_images_gpu = test_images.to(DEVICE)
    test_labels_gpu = test_labels.to(DEVICE)
    
    teacher_path = '/workspace/teacher_final.pt'
    full_soft_labels_path = '/workspace/soft_labels_final.pt'
    
    # Load full soft labels
    print("\n[2] Loading soft labels...")
    full_soft_labels = torch.load(full_soft_labels_path, map_location='cpu')
    print(f"  Soft labels shape: {full_soft_labels.shape}")
    
    results = {}
    
    # =====================================================
    # CORESET METHODS: Random, K-centers
    # =====================================================
    for ipc in [10, 50]:
        print(f"\n{'='*60}")
        print(f"IPC = {ipc}")
        print(f"{'='*60}")
        
        # --- RANDOM ---
        for seed_base in [0]:  # Single seed for selection, 3 runs for eval
            print(f"\n  [Random IPC={ipc}]")
            sel_idx = random_select(train_labels, ipc=ipc, seed=42)
            sub_imgs = train_images[sel_idx]
            sub_labels = train_labels[sel_idx]
            sub_soft = full_soft_labels[sel_idx]
            
            print(f"    HL:")
            hl_accs = run_eval(sub_imgs, sub_labels, test_images, test_labels, 'hard')
            
            print(f"    SL:")
            sl_accs = run_eval(sub_imgs, sub_labels, test_images, test_labels, 'soft', sub_soft)
            
            results[f'random_ipc{ipc}_hl'] = {'mean': np.mean(hl_accs), 'std': np.std(hl_accs), 'accs': hl_accs}
            results[f'random_ipc{ipc}_sl'] = {'mean': np.mean(sl_accs), 'std': np.std(sl_accs), 'accs': sl_accs}
        
        # --- K-CENTERS ---
        print(f"\n  [K-centers IPC={ipc}]")
        sel_idx = k_centers_select_with_teacher(train_images, train_labels, ipc, teacher_path, seed=42)
        sub_imgs = train_images[sel_idx]
        sub_labels = train_labels[sel_idx]
        sub_soft = full_soft_labels[sel_idx]
        
        # Verify class balance
        unique, counts = np.unique(sub_labels.numpy(), return_counts=True)
        assert len(unique) == 100, f"Expected 100 classes, got {len(unique)}"
        assert all(c == ipc for c in counts), f"Expected {ipc} per class, got range {min(counts)}-{max(counts)}"
        print(f"    Class balance OK: {len(unique)} classes, {ipc} per class")
        
        print(f"    HL:")
        hl_accs = run_eval(sub_imgs, sub_labels, test_images, test_labels, 'hard')
        
        print(f"    SL:")
        sl_accs = run_eval(sub_imgs, sub_labels, test_images, test_labels, 'soft', sub_soft)
        
        results[f'kcenter_ipc{ipc}_hl'] = {'mean': np.mean(hl_accs), 'std': np.std(hl_accs), 'accs': hl_accs}
        results[f'kcenter_ipc{ipc}_sl'] = {'mean': np.mean(sl_accs), 'std': np.std(sl_accs), 'accs': sl_accs}
    
    # =====================================================
    # DD METHODS: DM, DC, TM
    # =====================================================
    dd_methods = ['dm', 'dc', 'tm']
    
    for method in dd_methods:
        for ipc in [10, 50]:
            print(f"\n  [{method.upper()} IPC={ipc}]")
            
            # Load distilled data
            img_path = f'/workspace/distilled_{method}_ipc{ipc}.pt'
            
            if not os.path.exists(img_path):
                print(f"    WARNING: {img_path} not found, skipping")
                continue
            
            data = torch.load(img_path, map_location='cpu')
            if isinstance(data, dict):
                dd_imgs = data['images']
                dd_labels = data['labels']
            else:
                dd_imgs = data
                # Create labels: ipc images per class, 100 classes
                dd_labels = torch.arange(NUM_CLASSES).repeat_interleave(ipc)
            
            print(f"    Loaded: {dd_imgs.shape}, labels: {dd_labels.shape}")
            
            # Generate fresh soft labels for DD images
            dd_soft = generate_soft_labels_for_subset(dd_imgs, teacher_path)
            
            print(f"    HL:")
            hl_accs = run_eval(dd_imgs, dd_labels, test_images, test_labels, 'hard')
            
            print(f"    SL:")
            sl_accs = run_eval(dd_imgs, dd_labels, test_images, test_labels, 'soft', dd_soft)
            
            results[f'{method}_ipc{ipc}_hl'] = {'mean': np.mean(hl_accs), 'std': np.std(hl_accs), 'accs': hl_accs}
            results[f'{method}_ipc{ipc}_sl'] = {'mean': np.mean(sl_accs), 'std': np.std(sl_accs), 'accs': sl_accs}
    
    # Save results
    os.makedirs('/workspace/results', exist_ok=True)
    with open('/workspace/results/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n\n" + "="*80)
    print("RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("="*80)
    print(f"{'Method':<12} {'IPC':>4} {'HL (Ours)':>16} {'HL (Paper)':>14} {'SL (Ours)':>16} {'SL (Paper)':>14}")
    print("-"*80)
    
    paper = {
        'random_ipc10': (18.64, 33.43),
        'random_ipc50': (34.66, 45.39),
        'kcenter_ipc10': (25.04, 34.70),
        'kcenter_ipc50': (38.64, 46.24),
        'dm_ipc10': (29.23, 26.13),
        'dm_ipc50': (42.32, 43.46),
        'dc_ipc10': (28.42, 23.54),
        'dc_ipc50': (30.56, 33.46),
        'tm_ipc10': (38.18, 37.60),
        'tm_ipc50': (46.32, 46.26),
    }
    
    for method_name, method_key in [('Random', 'random'), ('K-centers', 'kcenter'), 
                                      ('DM', 'dm'), ('DC', 'dc'), ('TM', 'tm')]:
        for ipc in [10, 50]:
            key = f'{method_key}_ipc{ipc}'
            hl_key = f'{key}_hl'
            sl_key = f'{key}_sl'
            
            hl_str = "N/A"
            sl_str = "N/A"
            
            if hl_key in results:
                hl_str = f"{results[hl_key]['mean']:.2f}±{results[hl_key]['std']:.2f}"
            if sl_key in results:
                sl_str = f"{results[sl_key]['mean']:.2f}±{results[sl_key]['std']:.2f}"
            
            paper_hl, paper_sl = paper.get(key, ('?', '?'))
            
            print(f"{method_name:<12} {ipc:>4} {hl_str:>16} {paper_hl:>14} {sl_str:>16} {paper_sl:>14}")
    
    print("="*80)
    print(f"\nResults saved to /workspace/results/final_results.json")
    
    return results


if __name__ == '__main__':
    results = main()
