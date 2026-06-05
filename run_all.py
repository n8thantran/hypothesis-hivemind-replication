"""
Complete pipeline for replicating Table 1 (small_scale_c100) from the paper.
Runs all methods (DM, DC, TM, Random, K-centers) at IPC 10 and 50,
evaluates under both HL and SL settings.
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

from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, random_select, kcenter_select, get_class_indices
from train_eval import train_and_evaluate, run_experiment

# ============================================================
# Configuration
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 100
CHANNEL = 3
IM_SIZE = (32, 32)
NUM_EVAL_RUNS = 3  # Paper uses 3 runs for mean±std

def model_fn():
    return ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE)


# ============================================================
# Teacher for soft labels
# ============================================================
def train_teacher(train_images, train_labels, test_images, test_labels, 
                  epochs=200, device=DEVICE):
    """Train a ConvNet-D3 teacher model on full CIFAR-100."""
    teacher_path = '/workspace/teacher_model.pt'
    
    if os.path.exists(teacher_path):
        print("Loading existing teacher model...")
        teacher = model_fn().to(device)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
        teacher.eval()
        # Quick accuracy check
        from train_eval import evaluate
        acc = evaluate(teacher, test_images, test_labels, device)
        print(f"Teacher accuracy: {acc:.2f}%")
        return teacher
    
    print(f"Training teacher model for {epochs} epochs...")
    teacher = model_fn().to(device)
    optimizer = torch.optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    n_train = len(train_images)
    batch_size = 256
    
    teacher.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch_imgs = train_images[idx].to(device)
            batch_labels = train_labels[idx].to(device)
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = teacher(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            from train_eval import evaluate
            acc = evaluate(teacher, test_images, test_labels, device)
            print(f"  Epoch {epoch+1}/{epochs}, Test Acc: {acc:.2f}%")
    
    torch.save(teacher.state_dict(), teacher_path)
    teacher.eval()
    return teacher


def generate_soft_labels(teacher, train_images, device=DEVICE, batch_size=512):
    """Generate soft labels (raw logits) for all training images."""
    sl_path = '/workspace/soft_labels.pt'
    
    if os.path.exists(sl_path):
        print("Loading existing soft labels...")
        return torch.load(sl_path, map_location='cpu', weights_only=True)
    
    print("Generating soft labels...")
    teacher.eval()
    all_logits = []
    
    with torch.no_grad():
        for i in range(0, len(train_images), batch_size):
            batch = train_images[i:i+batch_size].to(device)
            logits = teacher(batch)
            all_logits.append(logits.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    torch.save(all_logits, sl_path)
    print(f"Soft labels shape: {all_logits.shape}")
    return all_logits


# ============================================================
# Distillation Methods
# ============================================================
def run_dm(train_images, train_labels, ipc, device=DEVICE, iterations=5000):
    """Run Distribution Matching distillation."""
    cache_path = f'/workspace/distilled/dm_ipc{ipc}.pt'
    if os.path.exists(cache_path):
        print(f"Loading cached DM IPC={ipc}...")
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        return data['images'], data['labels']
    
    from distill_dm import distribution_matching
    syn_images, syn_labels = distribution_matching(
        train_images, train_labels, 
        num_classes=NUM_CLASSES, ipc=ipc,
        channel=CHANNEL, im_size=IM_SIZE, device=device,
        iterations=iterations, lr_img=1.0, batch_real=256,
        seed=0
    )
    
    os.makedirs('/workspace/distilled', exist_ok=True)
    torch.save({'images': syn_images, 'labels': syn_labels}, cache_path)
    return syn_images, syn_labels


def run_dc(train_images, train_labels, ipc, device=DEVICE, outer_loops=10, inner_loops=50):
    """Run Dataset Condensation (gradient matching)."""
    cache_path = f'/workspace/distilled/dc_ipc{ipc}.pt'
    if os.path.exists(cache_path):
        print(f"Loading cached DC IPC={ipc}...")
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        return data['images'], data['labels']
    
    from distill_dc import gradient_matching
    syn_images, syn_labels = gradient_matching(
        train_images, train_labels,
        num_classes=NUM_CLASSES, ipc=ipc,
        channel=CHANNEL, im_size=IM_SIZE, device=device,
        outer_loops=outer_loops, inner_loops=inner_loops,
        lr_img=1.0, batch_real=256, dis_metric='ours',
        seed=0
    )
    
    os.makedirs('/workspace/distilled', exist_ok=True)
    torch.save({'images': syn_images, 'labels': syn_labels}, cache_path)
    return syn_images, syn_labels


def run_tm(train_images, train_labels, ipc, device=DEVICE, 
           iterations=2000, num_experts=10, expert_epochs=50):
    """Run Trajectory Matching distillation."""
    cache_path = f'/workspace/distilled/tm_ipc{ipc}.pt'
    if os.path.exists(cache_path):
        print(f"Loading cached TM IPC={ipc}...")
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        return data['images'], data['labels']
    
    expert_dir = '/workspace/expert_trajectories'
    
    # Train expert trajectories if needed
    if not os.path.exists(expert_dir) or len(os.listdir(expert_dir)) < num_experts:
        from distill_tm import train_expert_trajectories
        print(f"Training {num_experts} expert trajectories...")
        train_expert_trajectories(
            train_images, train_labels,
            num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE,
            device=device, num_experts=num_experts, expert_epochs=expert_epochs,
            lr=0.01, batch_size=256, save_dir=expert_dir, seed=0
        )
    
    from distill_tm import trajectory_matching
    syn_images, syn_labels = trajectory_matching(
        train_images, train_labels,
        num_classes=NUM_CLASSES, ipc=ipc,
        channel=CHANNEL, im_size=IM_SIZE, device=device,
        expert_dir=expert_dir, num_experts=num_experts,
        iterations=iterations, lr_img=1000.0, lr_lr=1e-5,
        syn_steps=30, expert_epochs=3, max_start_epoch=25,
        seed=0
    )
    
    os.makedirs('/workspace/distilled', exist_ok=True)
    torch.save({'images': syn_images, 'labels': syn_labels}, cache_path)
    return syn_images, syn_labels


# ============================================================
# Evaluation
# ============================================================
def evaluate_method(name, images, labels, test_images, test_labels,
                    soft_labels_all=None, train_labels_all=None,
                    selected_indices=None, device=DEVICE):
    """
    Evaluate a method under both HL and SL settings.
    
    For coreset methods, selected_indices maps to original training set indices
    (needed to look up soft labels).
    For DD methods, soft_labels_all is None and we generate soft labels from teacher.
    """
    results = {}
    
    # HL evaluation
    print(f"\n  [{name}] HL evaluation ({NUM_EVAL_RUNS} runs)...")
    hl_mean, hl_std = run_experiment(
        images, labels, test_images, test_labels,
        model_fn, NUM_CLASSES, device,
        label_type='hard', soft_labels=None,
        epochs=300, batch_size=256,
        num_runs=NUM_EVAL_RUNS, verbose=True
    )
    results['hl_mean'] = hl_mean
    results['hl_std'] = hl_std
    print(f"  [{name}] HL: {hl_mean:.2f} ± {hl_std:.2f}%")
    
    # SL evaluation
    if soft_labels_all is not None and selected_indices is not None:
        # Coreset: use pre-computed soft labels for selected indices
        sl = soft_labels_all[selected_indices]
    elif soft_labels_all is not None:
        # DD method: generate soft labels for synthetic images using teacher
        # We need to get teacher predictions on synthetic images
        sl = generate_soft_labels_for_images(images, device)
    else:
        print(f"  [{name}] Skipping SL (no soft labels available)")
        return results
    
    print(f"\n  [{name}] SL evaluation ({NUM_EVAL_RUNS} runs)...")
    sl_mean, sl_std = run_experiment(
        images, labels, test_images, test_labels,
        model_fn, NUM_CLASSES, device,
        label_type='soft', soft_labels=sl,
        epochs=300, batch_size=256,
        num_runs=NUM_EVAL_RUNS, verbose=True
    )
    results['sl_mean'] = sl_mean
    results['sl_std'] = sl_std
    print(f"  [{name}] SL: {sl_mean:.2f} ± {sl_std:.2f}%")
    
    return results


def generate_soft_labels_for_images(images, device=DEVICE):
    """Generate soft labels for arbitrary images using the teacher model."""
    teacher_path = '/workspace/teacher_model.pt'
    teacher = model_fn().to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), 512):
            batch = images[i:i+512].to(device)
            logits = teacher(batch)
            all_logits.append(logits.cpu())
    
    return torch.cat(all_logits, dim=0)


# ============================================================
# Main Pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['random', 'kcenter', 'dm', 'dc', 'tm'],
                        help='Methods to run')
    parser.add_argument('--ipcs', nargs='+', type=int, default=[10, 50],
                        help='IPC values')
    parser.add_argument('--dm_iters', type=int, default=5000, help='DM iterations')
    parser.add_argument('--dc_outer', type=int, default=10, help='DC outer loops')
    parser.add_argument('--dc_inner', type=int, default=50, help='DC inner loops')
    parser.add_argument('--tm_iters', type=int, default=2000, help='TM iterations')
    parser.add_argument('--tm_experts', type=int, default=10, help='Number of TM experts')
    parser.add_argument('--eval_runs', type=int, default=3, help='Number of eval runs')
    args = parser.parse_args()
    
    global NUM_EVAL_RUNS
    NUM_EVAL_RUNS = args.eval_runs
    
    print("=" * 60)
    print("DATASET DISTILLATION REPLICATION PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\nLoading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Train teacher and generate soft labels
    teacher = train_teacher(train_images, train_labels, test_images, test_labels)
    soft_labels_all = generate_soft_labels(teacher, train_images)
    del teacher
    torch.cuda.empty_cache()
    
    all_results = {}
    
    for ipc in args.ipcs:
        print(f"\n{'='*60}")
        print(f"IPC = {ipc}")
        print(f"{'='*60}")
        
        # ---- Coreset methods ----
        if 'random' in args.methods:
            print(f"\n--- Random IPC={ipc} ---")
            indices = random_select(train_labels, ipc=ipc, seed=42)
            images = train_images[indices]
            labels = train_labels[indices]
            results = evaluate_method(
                f'Random_IPC{ipc}', images, labels, test_images, test_labels,
                soft_labels_all=soft_labels_all, selected_indices=indices
            )
            all_results[f'Random_IPC{ipc}'] = results
            save_results(all_results)
        
        if 'kcenter' in args.methods:
            print(f"\n--- K-centers IPC={ipc} ---")
            indices = kcenter_select(train_images, train_labels, ipc=ipc, device=DEVICE)
            images = train_images[indices]
            labels = train_labels[indices]
            results = evaluate_method(
                f'Kcenter_IPC{ipc}', images, labels, test_images, test_labels,
                soft_labels_all=soft_labels_all, selected_indices=indices
            )
            all_results[f'Kcenter_IPC{ipc}'] = results
            save_results(all_results)
        
        # ---- DD methods ----
        if 'dm' in args.methods:
            print(f"\n--- DM IPC={ipc} ---")
            syn_images, syn_labels = run_dm(train_images, train_labels, ipc, 
                                             iterations=args.dm_iters)
            # For DD methods, generate soft labels from teacher on synthetic images
            syn_sl = generate_soft_labels_for_images(syn_images)
            results = evaluate_method(
                f'DM_IPC{ipc}', syn_images, syn_labels, test_images, test_labels,
                soft_labels_all=syn_sl, selected_indices=torch.arange(len(syn_images))
            )
            all_results[f'DM_IPC{ipc}'] = results
            save_results(all_results)
            del syn_images, syn_labels, syn_sl
            torch.cuda.empty_cache()
        
        if 'dc' in args.methods:
            print(f"\n--- DC IPC={ipc} ---")
            syn_images, syn_labels = run_dc(train_images, train_labels, ipc,
                                             outer_loops=args.dc_outer,
                                             inner_loops=args.dc_inner)
            syn_sl = generate_soft_labels_for_images(syn_images)
            results = evaluate_method(
                f'DC_IPC{ipc}', syn_images, syn_labels, test_images, test_labels,
                soft_labels_all=syn_sl, selected_indices=torch.arange(len(syn_images))
            )
            all_results[f'DC_IPC{ipc}'] = results
            save_results(all_results)
            del syn_images, syn_labels, syn_sl
            torch.cuda.empty_cache()
        
        if 'tm' in args.methods:
            print(f"\n--- TM IPC={ipc} ---")
            syn_images, syn_labels = run_tm(train_images, train_labels, ipc,
                                             iterations=args.tm_iters,
                                             num_experts=args.tm_experts)
            syn_sl = generate_soft_labels_for_images(syn_images)
            results = evaluate_method(
                f'TM_IPC{ipc}', syn_images, syn_labels, test_images, test_labels,
                soft_labels_all=syn_sl, selected_indices=torch.arange(len(syn_images))
            )
            all_results[f'TM_IPC{ipc}'] = results
            save_results(all_results)
            del syn_images, syn_labels, syn_sl
            torch.cuda.empty_cache()
    
    # Print final table
    print_results_table(all_results)
    save_results(all_results)
    
    print("\nDone! Results saved to /workspace/results/")


def save_results(results):
    """Save results to JSON."""
    os.makedirs('/workspace/results', exist_ok=True)
    with open('/workspace/results/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)


def print_results_table(results):
    """Print results in a formatted table matching the paper."""
    print("\n" + "=" * 80)
    print("RESULTS TABLE (Table 1: small_scale_c100)")
    print("=" * 80)
    print(f"{'Method':<15} {'IPC':>5} {'HL (Ours)':>15} {'HL (Paper)':>15} {'SL (Ours)':>15} {'SL (Paper)':>15}")
    print("-" * 80)
    
    # Paper reference values
    paper = {
        'DM_IPC10':      {'hl': '29.23±0.26', 'sl': '26.13±0.10'},
        'DM_IPC50':      {'hl': '42.32±0.37', 'sl': '43.46±0.18'},
        'DC_IPC10':      {'hl': '28.42±0.29', 'sl': '23.54±0.31'},
        'DC_IPC50':      {'hl': '30.56±0.56', 'sl': '33.46±0.38'},
        'TM_IPC10':      {'hl': '38.18±0.42', 'sl': '37.60±0.25'},
        'TM_IPC50':      {'hl': '46.32±0.26', 'sl': '46.26±0.30'},
        'Random_IPC10':  {'hl': '18.64±0.25', 'sl': '33.43±0.18'},
        'Random_IPC50':  {'hl': '34.66±0.41', 'sl': '45.39±0.23'},
        'Kcenter_IPC10': {'hl': '25.04±0.30', 'sl': '34.70±0.27'},
        'Kcenter_IPC50': {'hl': '38.64±0.43', 'sl': '46.24±0.12'},
    }
    
    for method_name in ['DM_IPC10', 'DM_IPC50', 'DC_IPC10', 'DC_IPC50', 
                         'TM_IPC10', 'TM_IPC50', 'Random_IPC10', 'Random_IPC50',
                         'Kcenter_IPC10', 'Kcenter_IPC50']:
        parts = method_name.rsplit('_IPC', 1)
        method = parts[0]
        ipc = parts[1]
        
        if method_name in results:
            r = results[method_name]
            hl_str = f"{r.get('hl_mean', 0):.2f}±{r.get('hl_std', 0):.2f}"
            sl_str = f"{r.get('sl_mean', 0):.2f}±{r.get('sl_std', 0):.2f}" if 'sl_mean' in r else 'N/A'
        else:
            hl_str = 'N/A'
            sl_str = 'N/A'
        
        p = paper.get(method_name, {'hl': 'N/A', 'sl': 'N/A'})
        print(f"{method:<15} {ipc:>5} {hl_str:>15} {p['hl']:>15} {sl_str:>15} {p['sl']:>15}")
    
    print("=" * 80)
    
    # Save table to file
    with open('/workspace/results/table1.txt', 'w') as f:
        f.write("RESULTS TABLE (Table 1: small_scale_c100)\n")
        f.write(f"{'Method':<15} {'IPC':>5} {'HL (Ours)':>15} {'HL (Paper)':>15} {'SL (Ours)':>15} {'SL (Paper)':>15}\n")
        f.write("-" * 80 + "\n")
        for method_name in ['DM_IPC10', 'DM_IPC50', 'DC_IPC10', 'DC_IPC50', 
                             'TM_IPC10', 'TM_IPC50', 'Random_IPC10', 'Random_IPC50',
                             'Kcenter_IPC10', 'Kcenter_IPC50']:
            parts = method_name.rsplit('_IPC', 1)
            method = parts[0]
            ipc = parts[1]
            
            if method_name in results:
                r = results[method_name]
                hl_str = f"{r.get('hl_mean', 0):.2f}±{r.get('hl_std', 0):.2f}"
                sl_str = f"{r.get('sl_mean', 0):.2f}±{r.get('sl_std', 0):.2f}" if 'sl_mean' in r else 'N/A'
            else:
                hl_str = 'N/A'
                sl_str = 'N/A'
            
            p = paper.get(method_name, {'hl': 'N/A', 'sl': 'N/A'})
            f.write(f"{method:<15} {ipc:>5} {hl_str:>15} {p['hl']:>15} {sl_str:>15} {p['sl']:>15}\n")


if __name__ == '__main__':
    main()
