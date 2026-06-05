"""
Main experiment runner for replicating Table small_scale_c100 from the paper.
Compares DD methods (DM, DC, TM) vs coreset methods (Random, K-centers)
under Hard Label (HL) and Soft Label (SL) settings on CIFAR-100 with ConvNet-D3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import argparse

from convnet import ConvNet, get_convnet_d3
from data_utils import (get_cifar100_tensors, random_select, k_centers_select,
                        get_class_indices, generate_soft_labels)
from train_eval import train_and_evaluate, run_experiment
from distill_dm import distribution_matching
from distill_dc import gradient_matching
from distill_tm import train_expert_trajectories, trajectory_matching


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = '/workspace/results'
DISTILLED_DIR = '/workspace/distilled_data'
EXPERT_DIR = '/workspace/expert_trajectories'
SOFT_LABELS_PATH = '/workspace/soft_labels.pt'


def get_model_fn():
    return lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))


def ensure_soft_labels(train_images, train_labels):
    """Generate or load soft labels from teacher models."""
    if os.path.exists(SOFT_LABELS_PATH):
        print("Loading cached soft labels...")
        return torch.load(SOFT_LABELS_PATH, weights_only=False)
    
    print("Generating soft labels from teacher models...")
    soft_labels = generate_soft_labels(
        train_images, train_labels, 
        model_fn=get_model_fn(),
        num_classes=100, device=DEVICE,
        num_models=3, epochs=200, seed=42
    )
    torch.save(soft_labels, SOFT_LABELS_PATH)
    print(f"Soft labels saved to {SOFT_LABELS_PATH}")
    return soft_labels


def ensure_distilled_data(method, ipc, train_images, train_labels):
    """Generate or load distilled data."""
    os.makedirs(DISTILLED_DIR, exist_ok=True)
    path = os.path.join(DISTILLED_DIR, f'{method}_ipc{ipc}.pt')
    
    if os.path.exists(path):
        print(f"Loading cached {method} IPC={ipc}...")
        data = torch.load(path, weights_only=False)
        return data['images'], data['labels']
    
    print(f"Synthesizing {method} IPC={ipc}...")
    
    if method == 'dm':
        syn_images, syn_labels = distribution_matching(
            train_images, train_labels, ipc=ipc, device=DEVICE,
            iterations=20000, lr_img=1.0, batch_real=256
        )
    elif method == 'dc':
        syn_images, syn_labels = gradient_matching(
            train_images, train_labels, ipc=ipc, device=DEVICE,
            outer_loops=10, inner_loops=50, lr_img=1.0, batch_real=256
        )
    elif method == 'tm':
        # First ensure expert trajectories exist
        if not os.path.exists(EXPERT_DIR) or len(os.listdir(EXPERT_DIR)) < 20:
            print("Training expert trajectories for TM...")
            train_expert_trajectories(
                train_images, train_labels, device=DEVICE,
                num_experts=20, expert_epochs=50,
                save_dir=EXPERT_DIR
            )
        
        syn_images, syn_labels = trajectory_matching(
            train_images, train_labels, ipc=ipc, device=DEVICE,
            expert_dir=EXPERT_DIR, num_experts=20,
            iterations=5000, lr_img=1000.0, syn_steps=30,
            expert_epochs=3, max_start_epoch=25
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    torch.save({'images': syn_images, 'labels': syn_labels}, path)
    print(f"Saved {method} IPC={ipc} to {path}")
    return syn_images, syn_labels


def run_coreset_experiment(method, ipc, train_images, train_labels, 
                           test_images, test_labels, soft_labels,
                           label_type='hard', num_runs=3):
    """Run coreset experiment (Random or K-centers)."""
    results = []
    
    for run in range(num_runs):
        # Select coreset
        if method == 'random':
            selected = random_select(train_labels, ipc=ipc, seed=run)
        elif method == 'k_centers':
            selected = k_centers_select(train_images, train_labels, ipc=ipc, seed=run)
        else:
            raise ValueError(f"Unknown coreset method: {method}")
        
        sub_images = train_images[selected]
        sub_labels = train_labels[selected]
        
        # Get soft labels for selected subset
        sub_soft_labels = soft_labels[selected] if soft_labels is not None else None
        
        acc = train_and_evaluate(
            sub_images, sub_labels, test_images, test_labels,
            model_fn=get_model_fn(), device=DEVICE,
            label_type=label_type, soft_labels=sub_soft_labels,
            epochs=300, batch_size=256, seed=run, verbose=True
        )
        results.append(acc)
        print(f"  {method} IPC={ipc} {label_type} run {run+1}: {acc:.2f}%")
    
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    return mean_acc, std_acc


def run_dd_experiment(method, ipc, train_images, train_labels,
                      test_images, test_labels, soft_labels,
                      label_type='hard', num_runs=3):
    """Run DD experiment (DM, DC, TM)."""
    # Get distilled data
    syn_images, syn_labels = ensure_distilled_data(method, ipc, train_images, train_labels)
    
    # For SL setting, we need soft labels for the synthetic data
    # For DD methods, we generate soft labels by running teacher on synthetic images
    if label_type == 'soft' and soft_labels is not None:
        # Generate soft labels for synthetic data using teacher
        print(f"Generating soft labels for {method} synthetic data...")
        syn_soft_labels = generate_soft_labels_for_synthetic(
            syn_images, train_images, train_labels
        )
    else:
        syn_soft_labels = None
    
    results = []
    for run in range(num_runs):
        acc = train_and_evaluate(
            syn_images, syn_labels, test_images, test_labels,
            model_fn=get_model_fn(), device=DEVICE,
            label_type=label_type, soft_labels=syn_soft_labels,
            epochs=300, batch_size=256, seed=run, verbose=True
        )
        results.append(acc)
        print(f"  {method} IPC={ipc} {label_type} run {run+1}: {acc:.2f}%")
    
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    return mean_acc, std_acc


def generate_soft_labels_for_synthetic(syn_images, train_images, train_labels):
    """Generate soft labels for synthetic images using teacher models trained on full data."""
    # Train teacher on full data and get predictions on synthetic images
    model_fn = get_model_fn()
    
    all_logits = []
    for m_idx in range(3):
        print(f"  Training teacher {m_idx+1}/3 for synthetic soft labels...")
        torch.manual_seed(42 + m_idx)
        model = model_fn().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        
        dataset = torch.utils.data.TensorDataset(train_images, train_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
        
        model.train()
        for epoch in range(200):
            for batch_imgs, batch_labels in loader:
                batch_imgs = batch_imgs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_imgs)
                loss = F.cross_entropy(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        # Get predictions on synthetic images
        model.eval()
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(syn_images), 256):
                batch = syn_images[i:i+256].to(DEVICE)
                logits = model(batch)
                logits_list.append(logits.cpu())
        all_logits.append(torch.cat(logits_list, dim=0))
    
    return torch.stack(all_logits).mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['random', 'k_centers', 'dm', 'dc', 'tm'],
                        help='Methods to evaluate')
    parser.add_argument('--ipcs', nargs='+', type=int, default=[10, 50],
                        help='IPC values')
    parser.add_argument('--label_types', nargs='+', default=['hard', 'soft'],
                        help='Label types')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of evaluation runs')
    parser.add_argument('--skip_distill', action='store_true',
                        help='Skip distillation, only evaluate cached data')
    args = parser.parse_args()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Generate soft labels for SL setting
    soft_labels = None
    if 'soft' in args.label_types:
        soft_labels = ensure_soft_labels(train_images, train_labels)
    
    # Run experiments
    all_results = {}
    
    dd_methods = ['dm', 'dc', 'tm']
    coreset_methods = ['random', 'k_centers']
    
    for method in args.methods:
        for ipc in args.ipcs:
            for label_type in args.label_types:
                key = f"{method}_ipc{ipc}_{label_type}"
                print(f"\n{'='*60}")
                print(f"Running: {key}")
                print(f"{'='*60}")
                
                try:
                    if method in coreset_methods:
                        mean_acc, std_acc = run_coreset_experiment(
                            method, ipc, train_images, train_labels,
                            test_images, test_labels, soft_labels,
                            label_type=label_type, num_runs=args.num_runs
                        )
                    elif method in dd_methods:
                        mean_acc, std_acc = run_dd_experiment(
                            method, ipc, train_images, train_labels,
                            test_images, test_labels, soft_labels,
                            label_type=label_type, num_runs=args.num_runs
                        )
                    else:
                        print(f"Unknown method: {method}")
                        continue
                    
                    all_results[key] = {
                        'mean': float(mean_acc),
                        'std': float(std_acc),
                        'method': method,
                        'ipc': ipc,
                        'label_type': label_type
                    }
                    print(f"\n>>> {key}: {mean_acc:.2f} ± {std_acc:.2f}%")
                    
                    # Save intermediate results
                    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
                        json.dump(all_results, f, indent=2)
                        
                except Exception as e:
                    print(f"ERROR in {key}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Print final results table
    print_results_table(all_results)
    
    # Save final results
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


def print_results_table(results):
    """Print results in a nice table format."""
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
    
    print("="*80)


if __name__ == '__main__':
    main()
