"""
Run coreset experiments (Random + K-centers) with correct paper hyperparameters.
This is the fastest way to validate the evaluation pipeline.
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import copy
import sys

from convnet import get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices
from pipeline import (train_teacher, generate_soft_labels_from_teacher,
                      evaluate_model, train_and_eval_hl, train_and_eval_sl,
                      select_random, select_k_centers)

DEVICE = 'cuda'


def main():
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Train or load teacher
    teacher_path = '/workspace/teacher_model.pt'
    soft_labels_path = '/workspace/teacher_soft_logits.pt'
    
    if os.path.exists(teacher_path) and os.path.exists(soft_labels_path):
        print("Loading cached teacher...")
        teacher = get_convnet_d3().to(DEVICE)
        teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
        soft_logits = torch.load(soft_labels_path, map_location='cpu')
        acc = evaluate_model(teacher, test_images, test_labels, DEVICE)
        print(f"Teacher accuracy: {acc:.2f}%")
    else:
        print("Training teacher model...")
        teacher = train_teacher(train_images, train_labels, test_images, test_labels,
                               epochs=300, device=DEVICE)
        torch.save(teacher.state_dict(), teacher_path)
        soft_logits = generate_soft_labels_from_teacher(teacher, train_images, DEVICE)
        torch.save(soft_logits, soft_labels_path)
    
    results = {}
    
    # Parse command line for specific experiment
    if len(sys.argv) > 1:
        experiments = [sys.argv[1]]  # e.g., "random_ipc10_hl"
    else:
        experiments = []
        for method in ['random', 'k_centers']:
            for ipc in [10, 50]:
                for setting in ['hl', 'sl']:
                    experiments.append(f"{method}_ipc{ipc}_{setting}")
    
    for exp_name in experiments:
        parts = exp_name.split('_')
        if parts[0] == 'k':
            method = 'k_centers'
            ipc = int(parts[2].replace('ipc', ''))
            setting = parts[3]
        else:
            method = parts[0]
            ipc = int(parts[1].replace('ipc', ''))
            setting = parts[2]
        
        print(f"\n{'='*60}")
        print(f"Running: {method} IPC={ipc} {setting.upper()}")
        print(f"{'='*60}")
        
        accs = []
        num_runs = 3
        
        for run in range(num_runs):
            seed = run * 42 + 1
            
            if method == 'random':
                indices = select_random(train_labels, ipc, seed=seed)
            elif method == 'k_centers':
                indices = select_k_centers(train_images, train_labels, ipc, seed=seed,
                                          model=teacher, device=DEVICE)
            
            syn_images = train_images[indices]
            syn_labels = train_labels[indices]
            
            if setting == 'hl':
                acc = train_and_eval_hl(syn_images, syn_labels, test_images, test_labels,
                                       device=DEVICE, seed=seed)
            elif setting == 'sl':
                syn_soft = soft_logits[indices]
                acc = train_and_eval_sl(syn_images, syn_soft, test_images, test_labels,
                                       device=DEVICE, seed=seed)
            
            accs.append(acc)
            print(f"  Run {run+1}: {acc:.2f}%")
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        
        results[exp_name] = {
            'method': method, 'ipc': ipc, 'setting': setting,
            'mean': round(mean_acc, 2), 'std': round(std_acc, 2),
            'accs': [round(a, 2) for a in accs]
        }
        print(f"  Result: {mean_acc:.2f} ± {std_acc:.2f}")
        
        # Save after each experiment
        os.makedirs('/workspace/results', exist_ok=True)
        with open('/workspace/results/results_coresets_v2.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print comparison
    paper = {
        'random_ipc10_hl': 18.64, 'random_ipc10_sl': 33.43,
        'random_ipc50_hl': 34.66, 'random_ipc50_sl': 45.39,
        'k_centers_ipc10_hl': 25.04, 'k_centers_ipc10_sl': 34.70,
        'k_centers_ipc50_hl': 38.64, 'k_centers_ipc50_sl': 46.24,
    }
    
    print(f"\n{'='*70}")
    print(f"{'Experiment':<25} {'Ours':>12} {'Paper':>12} {'Diff':>8}")
    print(f"{'-'*70}")
    for key in sorted(results.keys()):
        ours = results[key]['mean']
        pap = paper.get(key, 'N/A')
        diff = f"{ours - pap:+.2f}" if isinstance(pap, float) else "N/A"
        print(f"{key:<25} {ours:>10.2f}% {pap:>10}  {diff:>8}")


if __name__ == '__main__':
    main()
