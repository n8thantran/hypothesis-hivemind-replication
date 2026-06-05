"""
Final comprehensive experiment runner for all 20 experiments.
Runs: 5 methods × 2 IPC × 2 label types × 3 trials each.
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
from data_utils import get_cifar100_tensors, random_select, k_centers_select
from train_eval import train_and_evaluate

DEVICE = 'cuda'
NUM_CLASSES = 100
NUM_RUNS = 3
EPOCHS = 300


def get_model_fn():
    return lambda: ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32))


def get_soft_labels_for_real_data(indices, full_soft_labels):
    """Get soft labels for real data subset by indexing into full soft labels."""
    return full_soft_labels[indices]


def get_soft_labels_for_dd_data(method, ipc):
    """Get teacher-assigned soft labels for DD distilled data."""
    # Use v2 soft labels (from better teacher) if available
    path_v2 = f'/workspace/soft_labels_{method}_ipc{ipc}_v2.pt'
    path_v1 = f'/workspace/soft_labels_{method}_ipc{ipc}.pt'
    
    if os.path.exists(path_v2):
        return torch.load(path_v2, map_location='cpu')
    elif os.path.exists(path_v1):
        return torch.load(path_v1, map_location='cpu')
    else:
        return None


def generate_dd_soft_labels(images, teacher, device='cuda'):
    """Generate soft labels for DD images using teacher model."""
    teacher.eval()
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            logits = teacher(batch)
            logits_list.append(logits.cpu())
    return torch.cat(logits_list, dim=0)


def run_single_experiment(train_images, train_labels, test_images, test_labels,
                          label_type, soft_labels=None, tag="", num_runs=NUM_RUNS):
    """Run a single experiment with multiple trials."""
    model_fn = get_model_fn()
    accs = []
    for run in range(num_runs):
        print(f"  {tag} - Run {run+1}/{num_runs}...", end=" ", flush=True)
        t0 = time.time()
        acc = train_and_evaluate(
            train_images, train_labels, test_images, test_labels,
            model_fn, NUM_CLASSES, DEVICE, label_type, soft_labels,
            EPOCHS, batch_size=256, seed=run*17+42, verbose=False
        )
        dt = time.time() - t0
        print(f"{acc:.2f}% ({dt:.0f}s)")
        accs.append(acc)
    
    mean = np.mean(accs)
    std = np.std(accs)
    print(f"  => {tag}: {mean:.2f} ± {std:.2f}%")
    return {"mean": round(mean, 2), "std": round(std, 2), "runs": [round(a, 2) for a in accs]}


def main():
    results = {}
    
    # Load test data
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Load teacher and full soft labels  
    print("Loading teacher model and soft labels...")
    teacher_data = torch.load('/workspace/teacher_final.pt', map_location='cpu')
    teacher = ConvNet(num_classes=100, channel=3, im_size=(32, 32))
    teacher.load_state_dict(teacher_data['state_dict'])
    teacher = teacher.to(DEVICE)
    teacher.eval()
    
    # Full soft labels for real data
    if os.path.exists('/workspace/soft_labels_final.pt'):
        full_soft_labels = torch.load('/workspace/soft_labels_final.pt', map_location='cpu')
    else:
        print("Generating full soft labels...")
        full_soft_labels = generate_dd_soft_labels(train_images, teacher, DEVICE)
        torch.save(full_soft_labels, '/workspace/soft_labels_final.pt')
    
    # Check which experiments to run (allow selective running)
    methods_to_run = sys.argv[1:] if len(sys.argv) > 1 else ['random', 'kcenter', 'dm', 'dc', 'tm']
    
    for ipc in [10, 50]:
        print(f"\n{'='*60}")
        print(f"IPC = {ipc}")
        print(f"{'='*60}")
        
        # ===== RANDOM =====
        if 'random' in methods_to_run:
            print(f"\n--- Random IPC={ipc} ---")
            selected = random_select(train_labels, ipc=ipc, seed=42)
            sub_images = train_images[selected]
            sub_labels = train_labels[selected]
            sub_soft = full_soft_labels[selected]
            
            # HL
            tag = f"Random_IPC{ipc}_HL"
            results[tag] = run_single_experiment(
                sub_images, sub_labels, test_images, test_labels,
                'hard', tag=tag
            )
            
            # SL
            tag = f"Random_IPC{ipc}_SL"
            results[tag] = run_single_experiment(
                sub_images, sub_labels, test_images, test_labels,
                'soft', sub_soft, tag=tag
            )
        
        # ===== K-CENTERS =====
        if 'kcenter' in methods_to_run:
            print(f"\n--- K-Centers IPC={ipc} ---")
            selected = k_centers_select(
                train_images, train_labels, ipc=ipc,
                use_features=True, feature_model=teacher, device=DEVICE, seed=42
            )
            sub_images = train_images[selected]
            sub_labels = train_labels[selected]
            sub_soft = full_soft_labels[selected]
            
            # HL
            tag = f"KCenter_IPC{ipc}_HL"
            results[tag] = run_single_experiment(
                sub_images, sub_labels, test_images, test_labels,
                'hard', tag=tag
            )
            
            # SL
            tag = f"KCenter_IPC{ipc}_SL"
            results[tag] = run_single_experiment(
                sub_images, sub_labels, test_images, test_labels,
                'soft', sub_soft, tag=tag
            )
        
        # ===== DM =====
        if 'dm' in methods_to_run:
            print(f"\n--- DM IPC={ipc} ---")
            dd_data = torch.load(f'/workspace/distilled_dm_ipc{ipc}.pt', map_location='cpu')
            dd_images = dd_data['images']
            dd_labels = dd_data['labels']
            
            # Generate fresh soft labels from teacher
            dd_soft = generate_dd_soft_labels(dd_images, teacher, DEVICE)
            
            tag = f"DM_IPC{ipc}_HL"
            results[tag] = run_single_experiment(
                dd_images, dd_labels, test_images, test_labels,
                'hard', tag=tag
            )
            
            tag = f"DM_IPC{ipc}_SL"
            results[tag] = run_single_experiment(
                dd_images, dd_labels, test_images, test_labels,
                'soft', dd_soft, tag=tag
            )
        
        # ===== DC =====
        if 'dc' in methods_to_run:
            print(f"\n--- DC IPC={ipc} ---")
            dd_data = torch.load(f'/workspace/distilled_dc_ipc{ipc}.pt', map_location='cpu')
            dd_images = dd_data['images']
            dd_labels = dd_data['labels']
            
            dd_soft = generate_dd_soft_labels(dd_images, teacher, DEVICE)
            
            tag = f"DC_IPC{ipc}_HL"
            results[tag] = run_single_experiment(
                dd_images, dd_labels, test_images, test_labels,
                'hard', tag=tag
            )
            
            tag = f"DC_IPC{ipc}_SL"
            results[tag] = run_single_experiment(
                dd_images, dd_labels, test_images, test_labels,
                'soft', dd_soft, tag=tag
            )
        
        # ===== TM =====
        if 'tm' in methods_to_run:
            print(f"\n--- TM IPC={ipc} ---")
            dd_data = torch.load(f'/workspace/distilled_tm_ipc{ipc}.pt', map_location='cpu')
            dd_images = dd_data['images']
            dd_labels = dd_data['labels']
            
            dd_soft = generate_dd_soft_labels(dd_images, teacher, DEVICE)
            
            tag = f"TM_IPC{ipc}_HL"
            results[tag] = run_single_experiment(
                dd_images, dd_labels, test_images, test_labels,
                'hard', tag=tag
            )
            
            tag = f"TM_IPC{ipc}_SL"
            results[tag] = run_single_experiment(
                dd_images, dd_labels, test_images, test_labels,
                'soft', dd_soft, tag=tag
            )
    
    # Save results
    os.makedirs('/workspace/results', exist_ok=True)
    with open('/workspace/results/all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print table
    print_table(results)
    
    return results


def print_table(results):
    """Print results in the format of paper Table small_scale_c100."""
    print("\n" + "="*80)
    print("RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("="*80)
    
    # Paper reference values
    paper = {
        'Random_IPC10_HL': '18.64±0.25', 'Random_IPC10_SL': '33.43±0.18',
        'Random_IPC50_HL': '34.66±0.41', 'Random_IPC50_SL': '45.39±0.23',
        'KCenter_IPC10_HL': '25.04±0.30', 'KCenter_IPC10_SL': '34.70±0.27',
        'KCenter_IPC50_HL': '38.64±0.43', 'KCenter_IPC50_SL': '46.24±0.12',
        'DM_IPC10_HL': '29.23±0.26', 'DM_IPC10_SL': '26.13±0.10',
        'DM_IPC50_HL': '42.32±0.37', 'DM_IPC50_SL': '43.46±0.18',
        'DC_IPC10_HL': '28.42±0.29', 'DC_IPC10_SL': '23.54±0.31',
        'DC_IPC50_HL': '30.56±0.56', 'DC_IPC50_SL': '33.46±0.38',
        'TM_IPC10_HL': '38.18±0.42', 'TM_IPC10_SL': '37.60±0.25',
        'TM_IPC50_HL': '46.32±0.26', 'TM_IPC50_SL': '46.26±0.30',
    }
    
    methods = ['Random', 'KCenter', 'DM', 'DC', 'TM']
    method_names = {'Random': 'Random', 'KCenter': 'K-Centers', 'DM': 'DM', 'DC': 'DC', 'TM': 'TM'}
    
    header = f"{'Method':<12} | {'IPC':>4} | {'HL (Ours)':>14} | {'HL (Paper)':>14} | {'SL (Ours)':>14} | {'SL (Paper)':>14}"
    print(header)
    print("-" * len(header))
    
    for method in methods:
        for ipc in [10, 50]:
            hl_key = f"{method}_IPC{ipc}_HL"
            sl_key = f"{method}_IPC{ipc}_SL"
            
            hl_ours = f"{results[hl_key]['mean']:.2f}±{results[hl_key]['std']:.2f}" if hl_key in results else "N/A"
            sl_ours = f"{results[sl_key]['mean']:.2f}±{results[sl_key]['std']:.2f}" if sl_key in results else "N/A"
            hl_paper = paper.get(hl_key, 'N/A')
            sl_paper = paper.get(sl_key, 'N/A')
            
            print(f"{method_names[method]:<12} | {ipc:>4} | {hl_ours:>14} | {hl_paper:>14} | {sl_ours:>14} | {sl_paper:>14}")
    
    # Also save as text file
    with open('/workspace/results/final_table.txt', 'w') as f:
        f.write("CIFAR-100, ConvNet-D3 Results\n")
        f.write("="*80 + "\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for method in methods:
            for ipc in [10, 50]:
                hl_key = f"{method}_IPC{ipc}_HL"
                sl_key = f"{method}_IPC{ipc}_SL"
                hl_ours = f"{results[hl_key]['mean']:.2f}±{results[hl_key]['std']:.2f}" if hl_key in results else "N/A"
                sl_ours = f"{results[sl_key]['mean']:.2f}±{results[sl_key]['std']:.2f}" if sl_key in results else "N/A"
                hl_paper = paper.get(hl_key, 'N/A')
                sl_paper = paper.get(sl_key, 'N/A')
                f.write(f"{method_names[method]:<12} | {ipc:>4} | {hl_ours:>14} | {hl_paper:>14} | {sl_ours:>14} | {sl_paper:>14}\n")


if __name__ == '__main__':
    main()
