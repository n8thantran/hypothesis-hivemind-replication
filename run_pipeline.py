"""
Complete pipeline for reproducing Table 1 (small-scale CIFAR-100 results)
from "Rethinking Dataset Distillation: Hard Truths About Soft Labels"

This script:
1. Generates coreset baselines (Random, K-centers) for IPC 10 and 50
2. Runs DM distillation for IPC 10 and 50
3. Runs DC distillation for IPC 10 and 50
4. Trains expert trajectories and runs TM for IPC 10 and 50
5. Evaluates all methods in both HL and SL settings
6. Produces the final results table
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from convnet import ConvNet
from data_utils import get_cifar100_tensors, random_select, get_class_indices
from train_eval import train_and_evaluate, run_experiment
from distill_dm import distribution_matching
from distill_dc import gradient_matching
from distill_tm import train_expert_trajectories, trajectory_matching

DEVICE = 'cuda'
NUM_CLASSES = 100
NUM_RUNS = 3  # Paper uses 5, we use 3 for speed
RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def model_fn():
    return ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32))

def load_data():
    """Load CIFAR-100 data."""
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    return train_images, train_labels, test_images, test_labels

def load_teacher_logits():
    """Load pre-computed teacher logits."""
    data = torch.load('/workspace/teacher_logits_v2.pt', map_location='cpu')
    return data['logits'], data['labels']

def get_soft_labels_for_subset(indices, teacher_logits):
    """Get teacher logits for a subset of training data."""
    return teacher_logits[indices]

def evaluate_method(name, train_imgs, train_lbls, test_imgs, test_lbls,
                    teacher_logits=None, indices=None, num_runs=NUM_RUNS):
    """Evaluate a method in both HL and SL settings."""
    results = {}
    
    # HL evaluation
    print(f"\n  Evaluating {name} - HL setting...")
    hl_accs = []
    for run in range(num_runs):
        acc = train_and_evaluate(
            train_imgs, train_lbls, test_imgs, test_lbls,
            model_fn, label_type='hard', epochs=300, batch_size=256,
            seed=run, verbose=False
        )
        hl_accs.append(acc)
        print(f"    Run {run+1}: {acc:.2f}%")
    results['HL'] = {'mean': np.mean(hl_accs), 'std': np.std(hl_accs), 'runs': hl_accs}
    print(f"  {name} HL: {results['HL']['mean']:.2f} ± {results['HL']['std']:.2f}%")
    
    # SL evaluation
    if teacher_logits is not None:
        print(f"\n  Evaluating {name} - SL setting...")
        if indices is not None:
            soft_labels = teacher_logits[indices]
        else:
            soft_labels = teacher_logits
        
        sl_accs = []
        for run in range(num_runs):
            acc = train_and_evaluate(
                train_imgs, train_lbls, test_imgs, test_lbls,
                model_fn, label_type='soft', soft_labels=soft_labels,
                epochs=300, batch_size=256, seed=run, verbose=False
            )
            sl_accs.append(acc)
            print(f"    Run {run+1}: {acc:.2f}%")
        results['SL'] = {'mean': np.mean(sl_accs), 'std': np.std(sl_accs), 'runs': sl_accs}
        print(f"  {name} SL: {results['SL']['mean']:.2f} ± {results['SL']['std']:.2f}%")
    
    return results

def k_centers_select(train_images, train_labels, ipc, num_classes=100, seed=0):
    """K-centers coreset selection using K-means in feature space."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Use a random ConvNet to extract features
    model = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(DEVICE)
    model.eval()
    
    # Extract features
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(DEVICE)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    all_features = torch.cat(all_features, dim=0)
    
    # K-centers per class
    class_indices = get_class_indices(train_labels, num_classes)
    selected = []
    
    for c in range(num_classes):
        indices = class_indices[c]
        feats = all_features[indices].numpy()
        
        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=ipc, random_state=seed, n_init=3, max_iter=100)
        kmeans.fit(feats)
        
        # Select closest sample to each center
        centers = kmeans.cluster_centers_
        for center in centers:
            dists = np.sum((feats - center) ** 2, axis=1)
            closest = np.argmin(dists)
            selected.append(indices[closest])
    
    return np.array(selected)

def generate_dm_soft_labels(syn_images, syn_labels, teacher_model_path, device='cuda'):
    """Generate soft labels for distilled images using teacher model."""
    model = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(device)
    model.load_state_dict(torch.load(teacher_model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        logits = model(syn_images.to(device))
    return logits.cpu()

def run_all(skip_distillation=False, skip_tm=False):
    """Run the complete pipeline."""
    train_images, train_labels, test_images, test_labels = load_data()
    teacher_logits, _ = load_teacher_logits()
    
    all_results = {}
    
    for ipc in [10, 50]:
        print(f"\n{'='*60}")
        print(f"IPC = {ipc}")
        print(f"{'='*60}")
        
        # ===== RANDOM BASELINE =====
        print(f"\n--- Random (IPC={ipc}) ---")
        indices = random_select(train_labels, ipc=ipc, seed=42)
        sub_images = train_images[indices]
        sub_labels = train_labels[indices]
        sub_soft = teacher_logits[indices]
        
        results = evaluate_method(
            f"Random_IPC{ipc}", sub_images, sub_labels, test_images, test_labels,
            teacher_logits=sub_soft, indices=None
        )
        all_results[f"Random_IPC{ipc}"] = results
        save_results(all_results)
        
        # ===== K-CENTERS BASELINE =====
        print(f"\n--- K-centers (IPC={ipc}) ---")
        kc_indices = k_centers_select(train_images, train_labels, ipc=ipc)
        kc_images = train_images[kc_indices]
        kc_labels = train_labels[kc_indices]
        kc_soft = teacher_logits[kc_indices]
        
        results = evaluate_method(
            f"Kcenter_IPC{ipc}", kc_images, kc_labels, test_images, test_labels,
            teacher_logits=kc_soft, indices=None
        )
        all_results[f"Kcenter_IPC{ipc}"] = results
        save_results(all_results)
        
        if skip_distillation:
            continue
        
        # ===== DM =====
        print(f"\n--- DM (IPC={ipc}) ---")
        dm_file = f'/workspace/distilled_dm_ipc{ipc}.pt'
        if os.path.exists(dm_file):
            print(f"  Loading cached DM from {dm_file}")
            dm_data = torch.load(dm_file, map_location='cpu')
            dm_images = dm_data['images']
            dm_labels = dm_data['labels']
        else:
            dm_iters = 20000 if ipc == 10 else 20000
            dm_images, dm_labels = distribution_matching(
                train_images, train_labels, ipc=ipc,
                iterations=dm_iters, lr_img=1.0, batch_real=256
            )
            torch.save({'images': dm_images, 'labels': dm_labels}, dm_file)
        
        # Generate soft labels for DM images
        dm_soft = generate_dm_soft_labels(dm_images, dm_labels, '/workspace/teacher_best_v2.pt')
        
        results = evaluate_method(
            f"DM_IPC{ipc}", dm_images, dm_labels, test_images, test_labels,
            teacher_logits=dm_soft, indices=None
        )
        all_results[f"DM_IPC{ipc}"] = results
        save_results(all_results)
        
        # ===== DC =====
        print(f"\n--- DC (IPC={ipc}) ---")
        dc_file = f'/workspace/distilled_dc_ipc{ipc}.pt'
        if os.path.exists(dc_file):
            print(f"  Loading cached DC from {dc_file}")
            dc_data = torch.load(dc_file, map_location='cpu')
            dc_images = dc_data['images']
            dc_labels = dc_data['labels']
        else:
            # DC uses outer_loops * inner_loops total iterations
            # Paper: 1000 iterations for DC
            dc_images, dc_labels = gradient_matching(
                train_images, train_labels, ipc=ipc,
                outer_loops=10, inner_loops=100, lr_img=1.0, batch_real=256
            )
            torch.save({'images': dc_images, 'labels': dc_labels}, dc_file)
        
        # Generate soft labels for DC images
        dc_soft = generate_dm_soft_labels(dc_images, dc_labels, '/workspace/teacher_best_v2.pt')
        
        results = evaluate_method(
            f"DC_IPC{ipc}", dc_images, dc_labels, test_images, test_labels,
            teacher_logits=dc_soft, indices=None
        )
        all_results[f"DC_IPC{ipc}"] = results
        save_results(all_results)
        
        if skip_tm:
            continue
            
        # ===== TM =====
        print(f"\n--- TM (IPC={ipc}) ---")
        tm_file = f'/workspace/distilled_tm_ipc{ipc}.pt'
        expert_dir = '/workspace/expert_trajectories'
        
        # Train experts if needed
        if not os.path.exists(expert_dir) or len(os.listdir(expert_dir)) < 10:
            print("  Training expert trajectories...")
            train_expert_trajectories(
                train_images, train_labels,
                num_experts=20, expert_epochs=50,
                save_dir=expert_dir
            )
        
        if os.path.exists(tm_file):
            print(f"  Loading cached TM from {tm_file}")
            tm_data = torch.load(tm_file, map_location='cpu')
            tm_images = tm_data['images']
            tm_labels = tm_data['labels']
        else:
            tm_images, tm_labels = trajectory_matching(
                train_images, train_labels, ipc=ipc,
                expert_dir=expert_dir,
                num_experts=20, iterations=5000,
                lr_img=1000.0, lr_lr=1e-5,
                syn_steps=30, expert_epochs=3, max_start_epoch=25
            )
            torch.save({'images': tm_images, 'labels': tm_labels}, tm_file)
        
        # Generate soft labels for TM images
        tm_soft = generate_dm_soft_labels(tm_images, tm_labels, '/workspace/teacher_best_v2.pt')
        
        results = evaluate_method(
            f"TM_IPC{ipc}", tm_images, tm_labels, test_images, test_labels,
            teacher_logits=tm_soft, indices=None
        )
        all_results[f"TM_IPC{ipc}"] = results
        save_results(all_results)
    
    # Print final table
    print_table(all_results)
    return all_results

def save_results(results):
    """Save results to JSON."""
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable = {}
    for k, v in results.items():
        serializable[k] = {}
        for setting, data in v.items():
            serializable[k][setting] = {
                'mean': convert(data['mean']),
                'std': convert(data['std']),
                'runs': [convert(x) for x in data['runs']]
            }
    
    with open(os.path.join(RESULTS_DIR, 'final_table_results.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

def print_table(results):
    """Print results in a nice table format."""
    print(f"\n{'='*80}")
    print("CIFAR-100, ConvNet-D3 Results")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'IPC':<5} {'HL (Ours)':<20} {'SL (Ours)':<20}")
    print(f"{'-'*60}")
    
    # Paper target values for comparison
    paper = {
        'Random_IPC10': {'HL': 18.64, 'SL': 33.43},
        'Random_IPC50': {'HL': 34.66, 'SL': 45.39},
        'Kcenter_IPC10': {'HL': 25.04, 'SL': 34.70},
        'Kcenter_IPC50': {'HL': 38.64, 'SL': 46.24},
        'DM_IPC10': {'HL': 29.23, 'SL': 26.13},
        'DM_IPC50': {'HL': 42.32, 'SL': 43.46},
        'DC_IPC10': {'HL': 28.42, 'SL': 23.54},
        'DC_IPC50': {'HL': 30.56, 'SL': 33.46},
        'TM_IPC10': {'HL': 38.18, 'SL': 37.60},
        'TM_IPC50': {'HL': 46.32, 'SL': 46.26},
    }
    
    for method in ['Random', 'Kcenter', 'DM', 'DC', 'TM']:
        for ipc in [10, 50]:
            key = f"{method}_IPC{ipc}"
            if key in results:
                hl = results[key].get('HL', {})
                sl = results[key].get('SL', {})
                hl_str = f"{hl.get('mean', 0):.2f}±{hl.get('std', 0):.2f}" if hl else "N/A"
                sl_str = f"{sl.get('mean', 0):.2f}±{sl.get('std', 0):.2f}" if sl else "N/A"
                
                # Paper values
                p = paper.get(key, {})
                p_hl = f"({p.get('HL', '?')})" if p else ""
                p_sl = f"({p.get('SL', '?')})" if p else ""
                
                print(f"{method:<15} {ipc:<5} {hl_str:<20} {sl_str:<20}")
    
    print(f"\n{'='*80}")
    print("Paper target values in parentheses:")
    for method in ['Random', 'Kcenter', 'DM', 'DC', 'TM']:
        for ipc in [10, 50]:
            key = f"{method}_IPC{ipc}"
            p = paper.get(key, {})
            print(f"  {key}: HL={p.get('HL', '?')}, SL={p.get('SL', '?')}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-distillation', action='store_true', help='Skip DD methods, only run coresets')
    parser.add_argument('--skip-tm', action='store_true', help='Skip TM (slowest method)')
    parser.add_argument('--coresets-only', action='store_true', help='Only run coreset baselines')
    args = parser.parse_args()
    
    if args.coresets_only:
        args.skip_distillation = True
    
    run_all(skip_distillation=args.skip_distillation, skip_tm=args.skip_tm)
