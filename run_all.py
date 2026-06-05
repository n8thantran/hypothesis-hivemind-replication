"""
Main experiment runner for replicating Table small_scale_c100 from the paper.
Compares DD methods (DM, DC, TM) vs coreset methods (Random, K-centers)
under Hard Label (HL) and Soft Label (SL) settings on CIFAR-100 with ConvNet-D3.

Optimized for single-GPU execution with reasonable time budget.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import argparse
import gc

from convnet import ConvNet
from data_utils import (get_cifar100_tensors, random_select, k_centers_select,
                        get_class_indices)
from train_eval import train_and_evaluate, run_experiment
from dsa import DiffAugment

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = '/workspace/results'
DISTILLED_DIR = '/workspace/distilled_data'
EXPERT_DIR = '/workspace/expert_trajectories'
SOFT_LABELS_PATH = '/workspace/soft_labels.pt'


def get_model_fn():
    return lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))


def generate_soft_labels(train_images, train_labels, num_classes=100, device='cuda',
                         num_models=1, epochs=100, seed=42):
    """Generate soft labels using teacher ensemble."""
    print(f"Generating soft labels ({num_models} teacher(s), {epochs} epochs)...")
    
    all_logits = []
    n = len(train_images)
    
    for m_idx in range(num_models):
        print(f"  Training teacher {m_idx+1}/{num_models}...")
        torch.manual_seed(seed + m_idx)
        model = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//2, gamma=0.1)
        
        model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i+256]
                x = train_images[idx].to(device)
                y = train_labels[idx].to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            if (epoch + 1) % 50 == 0:
                # Quick eval
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for i in range(0, n, 512):
                        x = train_images[i:i+512].to(device)
                        y = train_labels[i:i+512].to(device)
                        out = model(x)
                        correct += (out.argmax(1) == y).sum().item()
                        total += y.size(0)
                print(f"    Epoch {epoch+1}: train acc = {100*correct/total:.1f}%")
                model.train()
        
        # Get logits on all training data
        model.eval()
        logits_list = []
        with torch.no_grad():
            for i in range(0, n, 512):
                x = train_images[i:i+512].to(device)
                logits = model(x)
                logits_list.append(logits.cpu())
        all_logits.append(torch.cat(logits_list, dim=0))
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    # Average logits across teachers
    soft_labels = torch.stack(all_logits).mean(dim=0)
    return soft_labels


def ensure_soft_labels(train_images, train_labels):
    """Generate or load soft labels."""
    if os.path.exists(SOFT_LABELS_PATH):
        print("Loading cached soft labels...")
        return torch.load(SOFT_LABELS_PATH, weights_only=False)
    
    soft_labels = generate_soft_labels(train_images, train_labels, 
                                        num_models=1, epochs=100)
    torch.save(soft_labels, SOFT_LABELS_PATH)
    print(f"Soft labels saved to {SOFT_LABELS_PATH}")
    return soft_labels


def generate_soft_labels_for_synthetic(syn_images, train_images, train_labels, device='cuda'):
    """Generate soft labels for synthetic images using a teacher trained on full data."""
    print("  Generating soft labels for synthetic data...")
    torch.manual_seed(42)
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    n = len(train_images)
    model.train()
    for epoch in range(100):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            x = train_images[idx].to(device)
            y = train_labels[idx].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    model.eval()
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(syn_images), 256):
            x = syn_images[i:i+256].to(device)
            logits = model(x)
            logits_list.append(logits.cpu())
    
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return torch.cat(logits_list, dim=0)


# ============================================================
# Distillation methods (inlined for efficiency)
# ============================================================

def distill_dm(train_images, train_labels, ipc=10, iterations=3000, device='cuda'):
    """Distribution Matching."""
    from distill_dm import distribution_matching
    return distribution_matching(train_images, train_labels, ipc=ipc, device=device,
                                  iterations=iterations, lr_img=1.0, batch_real=256)


def distill_dc(train_images, train_labels, ipc=10, outer_loops=5, inner_loops=10, device='cuda'):
    """Dataset Condensation (Gradient Matching)."""
    from distill_dc import gradient_matching
    return gradient_matching(train_images, train_labels, ipc=ipc, device=device,
                              outer_loops=outer_loops, inner_loops=inner_loops,
                              lr_img=1.0, batch_real=256)


def distill_tm(train_images, train_labels, ipc=10, device='cuda'):
    """Trajectory Matching."""
    from distill_tm import train_expert_trajectories, trajectory_matching
    
    # Train experts if needed
    if not os.path.exists(EXPERT_DIR) or len(os.listdir(EXPERT_DIR)) < 5:
        print("Training expert trajectories...")
        train_expert_trajectories(
            train_images, train_labels, device=device,
            num_experts=5, expert_epochs=20,
            save_dir=EXPERT_DIR
        )
    
    return trajectory_matching(
        train_images, train_labels, ipc=ipc, device=device,
        expert_dir=EXPERT_DIR, num_experts=5,
        iterations=1000, lr_img=1000.0, syn_steps=20,
        expert_epochs=2, max_start_epoch=15
    )


def ensure_distilled_data(method, ipc, train_images, train_labels, device='cuda'):
    """Generate or load distilled data."""
    os.makedirs(DISTILLED_DIR, exist_ok=True)
    path = os.path.join(DISTILLED_DIR, f'{method}_ipc{ipc}.pt')
    
    if os.path.exists(path):
        print(f"Loading cached {method} IPC={ipc}...")
        data = torch.load(path, weights_only=False)
        return data['images'], data['labels']
    
    print(f"\nSynthesizing {method} IPC={ipc}...")
    start = time.time()
    
    if method == 'dm':
        syn_images, syn_labels = distill_dm(train_images, train_labels, ipc=ipc, device=device)
    elif method == 'dc':
        syn_images, syn_labels = distill_dc(train_images, train_labels, ipc=ipc, device=device)
    elif method == 'tm':
        syn_images, syn_labels = distill_tm(train_images, train_labels, ipc=ipc, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed = time.time() - start
    print(f"  {method} IPC={ipc} distillation took {elapsed:.0f}s")
    
    torch.save({'images': syn_images, 'labels': syn_labels}, path)
    return syn_images, syn_labels


def run_single_experiment(method, ipc, label_type, train_images, train_labels,
                          test_images, test_labels, soft_labels, num_runs=3):
    """Run a single experiment configuration."""
    key = f"{method}_ipc{ipc}_{label_type}"
    print(f"\n{'='*60}")
    print(f"Running: {key}")
    print(f"{'='*60}")
    
    dd_methods = ['dm', 'dc', 'tm']
    coreset_methods = ['random', 'k_centers']
    
    results = []
    
    for run in range(num_runs):
        print(f"\n  --- Run {run+1}/{num_runs} ---")
        
        if method in coreset_methods:
            if method == 'random':
                selected = random_select(train_labels, ipc=ipc, seed=run)
            elif method == 'k_centers':
                selected = k_centers_select(train_images, train_labels, ipc=ipc, seed=run)
            
            sub_images = train_images[selected]
            sub_labels = train_labels[selected]
            sub_soft = soft_labels[selected] if soft_labels is not None else None
            
        elif method in dd_methods:
            syn_images, syn_labels = ensure_distilled_data(method, ipc, train_images, train_labels)
            sub_images = syn_images
            sub_labels = syn_labels
            
            if label_type == 'soft':
                # Generate soft labels for synthetic data
                sl_path = os.path.join(DISTILLED_DIR, f'{method}_ipc{ipc}_soft_labels.pt')
                if os.path.exists(sl_path):
                    sub_soft = torch.load(sl_path, weights_only=False)
                else:
                    sub_soft = generate_soft_labels_for_synthetic(
                        syn_images, train_images, train_labels
                    )
                    torch.save(sub_soft, sl_path)
            else:
                sub_soft = None
        
        acc = train_and_evaluate(
            sub_images, sub_labels, test_images, test_labels,
            model_fn=get_model_fn(), device=DEVICE,
            label_type=label_type, soft_labels=sub_soft,
            epochs=300, batch_size=256, seed=run, verbose=False
        )
        results.append(acc)
        print(f"  {key} run {run+1}: {acc:.2f}%")
    
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    print(f"\n>>> {key}: {mean_acc:.2f} ± {std_acc:.2f}%")
    
    return {
        'mean': float(mean_acc),
        'std': float(std_acc),
        'runs': [float(r) for r in results],
        'method': method,
        'ipc': ipc,
        'label_type': label_type
    }


def print_results_table(results):
    """Print results in a nice table format matching paper's Table small_scale_c100."""
    print("\n" + "="*80)
    print("RESULTS TABLE: CIFAR-100, ConvNet-D3")
    print("="*80)
    
    methods_order = ['dm', 'dc', 'tm', 'random', 'k_centers']
    method_names = {'dm': 'DM', 'dc': 'DC', 'tm': 'TM', 'random': 'Random', 'k_centers': 'K-centers'}
    
    for label_type in ['hard', 'soft']:
        print(f"\n--- {label_type.upper()} Label Setting ---")
        print(f"{'Method':<15} {'IPC 10':<20} {'IPC 50':<20}")
        print("-" * 55)
        
        for method in methods_order:
            row = f"{method_names.get(method, method):<15}"
            for ipc in [10, 50]:
                key = f"{method}_ipc{ipc}_{label_type}"
                if key in results:
                    r = results[key]
                    row += f"{r['mean']:.2f} ± {r['std']:.2f}    "
                else:
                    row += f"{'N/A':<20}"
            print(row)
    
    # Also print paper's reference values
    print("\n--- Paper Reference Values (HL) ---")
    paper_hl = {
        'DM': {'10': '29.23±0.26', '50': '42.32±0.37'},
        'DC': {'10': '28.42±0.29', '50': '30.56±0.56'},
        'TM': {'10': '38.18±0.42', '50': '46.32±0.26'},
        'Random': {'10': '18.64±0.25', '50': '34.66±0.41'},
        'K-centers': {'10': '25.04±0.30', '50': '38.64±0.43'},
    }
    print(f"{'Method':<15} {'IPC 10':<20} {'IPC 50':<20}")
    print("-" * 55)
    for method in ['DM', 'DC', 'TM', 'Random', 'K-centers']:
        print(f"{method:<15} {paper_hl[method]['10']:<20} {paper_hl[method]['50']:<20}")
    
    print("\n--- Paper Reference Values (SL) ---")
    paper_sl = {
        'DM': {'10': '26.13±0.10', '50': '43.46±0.18'},
        'DC': {'10': '23.54±0.31', '50': '33.46±0.38'},
        'TM': {'10': '37.60±0.25', '50': '46.26±0.30'},
        'Random': {'10': '33.43±0.18', '50': '45.39±0.23'},
        'K-centers': {'10': '34.70±0.27', '50': '46.24±0.12'},
    }
    print(f"{'Method':<15} {'IPC 10':<20} {'IPC 50':<20}")
    print("-" * 55)
    for method in ['DM', 'DC', 'TM', 'Random', 'K-centers']:
        print(f"{method:<15} {paper_sl[method]['10']:<20} {paper_sl[method]['50']:<20}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', 
                        default=['random', 'k_centers', 'dm', 'dc', 'tm'],
                        help='Methods to evaluate')
    parser.add_argument('--ipcs', nargs='+', type=int, default=[10, 50],
                        help='IPC values')
    parser.add_argument('--label_types', nargs='+', default=['hard', 'soft'],
                        help='Label types')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of evaluation runs')
    args = parser.parse_args()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    total_start = time.time()
    
    # Load data
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Generate soft labels
    soft_labels = None
    if 'soft' in args.label_types:
        soft_labels = ensure_soft_labels(train_images, train_labels)
    
    # Run all experiments
    all_results = {}
    
    # Load any existing results
    results_path = os.path.join(RESULTS_DIR, 'results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results")
    
    for method in args.methods:
        for ipc in args.ipcs:
            for label_type in args.label_types:
                key = f"{method}_ipc{ipc}_{label_type}"
                
                # Skip if already computed
                if key in all_results:
                    print(f"\nSkipping {key} (already computed)")
                    continue
                
                try:
                    result = run_single_experiment(
                        method, ipc, label_type,
                        train_images, train_labels,
                        test_images, test_labels,
                        soft_labels, num_runs=args.num_runs
                    )
                    all_results[key] = result
                    
                    # Save intermediate results
                    with open(results_path, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    
                except Exception as e:
                    print(f"ERROR in {key}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Clean up GPU memory
                gc.collect()
                torch.cuda.empty_cache()
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.0f}s = {total_elapsed/60:.1f}min")
    
    # Print final results
    print_results_table(all_results)
    
    # Save final results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
