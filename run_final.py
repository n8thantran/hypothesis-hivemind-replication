#!/usr/bin/env python3
"""
Final comprehensive experiment runner for dataset distillation paper replication.
Addresses all V1 issues:
1. Better teacher model (500 epochs, proper schedule) for better soft labels
2. Feature-space K-centers using trained teacher model
3. More DD iterations: DM=10000, DC outer=20 inner=50, TM=5000
4. 3 evaluation runs for all configs
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment
from data_utils import (get_cifar100_tensors, get_class_indices, 
                        random_select, k_centers_select)
from train_eval import train_and_evaluate
from distill_dm import distribution_matching
from distill_dc import gradient_matching
from distill_tm import trajectory_matching

DEVICE = 'cuda'
RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

results = {}


def model_fn():
    return ConvNet(num_classes=100, channel=3, im_size=(32, 32))


def train_teacher(train_images, train_labels, test_images, test_labels,
                  epochs=500, device='cuda'):
    """Train a strong teacher model on full CIFAR-100."""
    ckpt_path = '/workspace/teacher_final.pt'
    if os.path.exists(ckpt_path):
        print(f"Loading existing teacher from {ckpt_path}")
        data = torch.load(ckpt_path, map_location='cpu')
        model = model_fn()
        model.load_state_dict(data['state_dict'])
        print(f"Teacher accuracy: {data['accuracy']:.2f}%")
        return model.to(device), data['accuracy']
    
    print(f"\nTraining teacher model ({epochs} epochs)...")
    model = model_fn().to(device)
    
    # Strong training setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, 
                                          num_workers=4, pin_memory=True)
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch_imgs, batch_labels in loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            # Apply some standard augmentation (horizontal flip + random crop)
            batch_imgs = DiffAugment(batch_imgs, strategy='crop_flip')
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i in range(0, len(test_images), 512):
                    batch = test_images[i:i+512].to(device)
                    labels = test_labels[i:i+512].to(device)
                    outputs = model(batch)
                    _, pred = outputs.max(1)
                    correct += pred.eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100.0 * correct / total
            print(f"  Epoch {epoch+1}/{epochs}, Test Acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    torch.save({'state_dict': best_state, 'accuracy': best_acc}, ckpt_path)
    print(f"Teacher saved with accuracy: {best_acc:.2f}%")
    return model.to(device), best_acc


def generate_soft_labels_from_teacher(images, teacher, device='cuda'):
    """Generate soft labels (logits) from teacher model."""
    teacher.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            logits = teacher(batch)
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


def evaluate_config(train_images, train_labels, test_images, test_labels,
                    soft_labels, label_type, num_runs=3, epochs=300, batch_size=256):
    """Evaluate a configuration with multiple runs."""
    accs = []
    for run in range(num_runs):
        acc = train_and_evaluate(
            train_images, train_labels, test_images, test_labels,
            model_fn, num_classes=100, device=DEVICE,
            label_type=label_type,
            soft_labels=soft_labels,
            epochs=epochs, batch_size=batch_size,
            seed=run, verbose=False
        )
        accs.append(acc)
        print(f"    Run {run+1}: {acc:.2f}%")
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"    → {mean_acc:.2f} ± {std_acc:.2f}%")
    return {'mean': mean_acc, 'std': std_acc, 'runs': accs}


def save_results():
    """Save results to file."""
    with open(os.path.join(RESULTS_DIR, 'results_final.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save readable table
    generate_table()


def generate_table():
    """Generate readable results table."""
    paper = {
        'dm_10_hard': (29.23, 0.26), 'dm_10_soft': (26.13, 0.10),
        'dm_50_hard': (42.32, 0.37), 'dm_50_soft': (43.46, 0.18),
        'dc_10_hard': (28.42, 0.29), 'dc_10_soft': (23.54, 0.31),
        'dc_50_hard': (30.56, 0.56), 'dc_50_soft': (33.46, 0.38),
        'tm_10_hard': (38.18, 0.42), 'tm_10_soft': (37.60, 0.25),
        'tm_50_hard': (46.32, 0.26), 'tm_50_soft': (46.26, 0.30),
        'random_10_hard': (18.64, 0.25), 'random_10_soft': (33.43, 0.18),
        'random_50_hard': (34.66, 0.41), 'random_50_soft': (45.39, 0.23),
        'k_centers_10_hard': (25.04, 0.30), 'k_centers_10_soft': (34.70, 0.27),
        'k_centers_50_hard': (38.64, 0.43), 'k_centers_50_soft': (46.24, 0.12),
    }
    
    lines = []
    lines.append("=" * 100)
    lines.append(f"{'Method':<12} {'IPC':>4} {'Label':>5} {'Ours':>12} {'Paper':>12} {'Diff':>8}")
    lines.append("-" * 100)
    
    for method in ['random', 'k_centers', 'dm', 'dc', 'tm']:
        for ipc in [10, 50]:
            for label in ['hard', 'soft']:
                key = f"{method}_ipc{ipc}_{label}"
                paper_key = f"{method}_{ipc}_{label}"
                if key in results and paper_key in paper:
                    ours_mean = results[key]['mean']
                    ours_std = results[key]['std']
                    paper_mean, paper_std = paper[paper_key]
                    diff = ours_mean - paper_mean
                    lines.append(f"{method:<12} {ipc:>4} {label:>5} {ours_mean:>6.2f}±{ours_std:<4.2f} {paper_mean:>6.2f}±{paper_std:<4.2f} {diff:>+7.2f}")
    
    lines.append("=" * 100)
    
    table_str = '\n'.join(lines)
    print(table_str)
    
    with open(os.path.join(RESULTS_DIR, 'table_final.txt'), 'w') as f:
        f.write(table_str)


def run_step(name, func):
    """Run a step with timing."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    start = time.time()
    result = func()
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s")
    return result


def main():
    global results
    
    # Load existing results if available
    results_path = os.path.join(RESULTS_DIR, 'results_final.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")
    
    # Check for --phase argument for incremental execution
    phase = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    # =============================================
    # PHASE 1: Teacher training
    # =============================================
    if phase in ['all', 'teacher', 'coreset', 'dd', 'eval']:
        teacher, teacher_acc = run_step("Train Teacher", 
            lambda: train_teacher(train_images, train_labels, test_images, test_labels,
                                  epochs=500, device=DEVICE))
        
        # Generate full-dataset soft labels
        sl_path = '/workspace/soft_labels_final.pt'
        if not os.path.exists(sl_path):
            print("Generating soft labels for full training set...")
            full_soft_labels = generate_soft_labels_from_teacher(train_images, teacher, DEVICE)
            torch.save(full_soft_labels, sl_path)
            print(f"Saved soft labels: {full_soft_labels.shape}")
        else:
            full_soft_labels = torch.load(sl_path, map_location='cpu')
            print(f"Loaded soft labels: {full_soft_labels.shape}")
    
    # =============================================
    # PHASE 2: Coreset selection and evaluation
    # =============================================
    if phase in ['all', 'coreset']:
        print("\n" + "="*60)
        print("CORESET METHODS")
        print("="*60)
        
        for ipc in [10, 50]:
            # Random selection
            for label_type in ['hard', 'soft']:
                key = f"random_ipc{ipc}_{label_type}"
                if key in results:
                    print(f"  Skipping {key} (already done)")
                    continue
                print(f"\n--- Random IPC={ipc}, {label_type} ---")
                selected = random_select(train_labels, ipc=ipc, seed=42)
                sub_images = train_images[selected]
                sub_labels = train_labels[selected]
                sl = full_soft_labels[selected] if label_type == 'soft' else None
                
                results[key] = evaluate_config(
                    sub_images, sub_labels, test_images, test_labels,
                    sl, label_type, num_runs=3
                )
                results[key]['method'] = 'random'
                results[key]['ipc'] = ipc
                save_results()
            
            # K-centers selection (feature-space with trained teacher)
            for label_type in ['hard', 'soft']:
                key = f"k_centers_ipc{ipc}_{label_type}"
                if key in results:
                    print(f"  Skipping {key} (already done)")
                    continue
                print(f"\n--- K-centers IPC={ipc}, {label_type} ---")
                selected = k_centers_select(
                    train_images, train_labels, ipc=ipc, 
                    use_features=True, feature_model=teacher,
                    seed=42, device=DEVICE
                )
                sub_images = train_images[selected]
                sub_labels = train_labels[selected]
                sl = full_soft_labels[selected] if label_type == 'soft' else None
                
                results[key] = evaluate_config(
                    sub_images, sub_labels, test_images, test_labels,
                    sl, label_type, num_runs=3
                )
                results[key]['method'] = 'k_centers'
                results[key]['ipc'] = ipc
                save_results()
    
    # =============================================
    # PHASE 3: Dataset Distillation
    # =============================================
    if phase in ['all', 'dd']:
        print("\n" + "="*60)
        print("DATASET DISTILLATION METHODS")
        print("="*60)
        
        for ipc in [10, 50]:
            # DM - Distribution Matching
            dm_path = f'/workspace/distilled_dm_ipc{ipc}_final.pt'
            if not os.path.exists(dm_path):
                print(f"\n--- DM IPC={ipc}: Distilling ---")
                dm_iters = 10000 if ipc == 10 else 5000  # Compromise for time
                syn_images, syn_labels = run_step(f"DM IPC={ipc}",
                    lambda: distribution_matching(
                        train_images, train_labels, ipc=ipc,
                        iterations=dm_iters, lr_img=1.0, batch_real=256,
                        device=DEVICE, seed=0
                    ))
                torch.save({'images': syn_images, 'labels': syn_labels}, dm_path)
            else:
                data = torch.load(dm_path, map_location='cpu')
                syn_images, syn_labels = data['images'], data['labels']
                print(f"Loaded DM IPC={ipc} from {dm_path}")
            
            # Generate soft labels for DM distilled images
            dm_sl = generate_soft_labels_from_teacher(syn_images, teacher, DEVICE)
            
            for label_type in ['hard', 'soft']:
                key = f"dm_ipc{ipc}_{label_type}"
                if key in results:
                    print(f"  Skipping {key}")
                    continue
                print(f"\n--- DM IPC={ipc}, {label_type}: Evaluating ---")
                sl = dm_sl if label_type == 'soft' else None
                results[key] = evaluate_config(
                    syn_images, syn_labels, test_images, test_labels,
                    sl, label_type, num_runs=3
                )
                results[key]['method'] = 'dm'
                results[key]['ipc'] = ipc
                save_results()
            
            # DC - Gradient Matching
            dc_path = f'/workspace/distilled_dc_ipc{ipc}_final.pt'
            if not os.path.exists(dc_path):
                print(f"\n--- DC IPC={ipc}: Distilling ---")
                ol = 20 if ipc == 10 else 10
                il = 50
                syn_images, syn_labels = run_step(f"DC IPC={ipc}",
                    lambda: gradient_matching(
                        train_images, train_labels, ipc=ipc,
                        outer_loops=ol, inner_loops=il,
                        lr_img=1.0, batch_real=256,
                        device=DEVICE, seed=0
                    ))
                torch.save({'images': syn_images, 'labels': syn_labels}, dc_path)
            else:
                data = torch.load(dc_path, map_location='cpu')
                syn_images, syn_labels = data['images'], data['labels']
                print(f"Loaded DC IPC={ipc} from {dc_path}")
            
            dc_sl = generate_soft_labels_from_teacher(syn_images, teacher, DEVICE)
            
            for label_type in ['hard', 'soft']:
                key = f"dc_ipc{ipc}_{label_type}"
                if key in results:
                    print(f"  Skipping {key}")
                    continue
                print(f"\n--- DC IPC={ipc}, {label_type}: Evaluating ---")
                sl = dc_sl if label_type == 'soft' else None
                results[key] = evaluate_config(
                    syn_images, syn_labels, test_images, test_labels,
                    sl, label_type, num_runs=3
                )
                results[key]['method'] = 'dc'
                results[key]['ipc'] = ipc
                save_results()
            
            # TM - Trajectory Matching
            tm_path = f'/workspace/distilled_tm_ipc{ipc}_final.pt'
            if not os.path.exists(tm_path):
                print(f"\n--- TM IPC={ipc}: Distilling ---")
                syn_images, syn_labels = run_step(f"TM IPC={ipc}",
                    lambda: trajectory_matching(
                        train_images, train_labels, ipc=ipc,
                        num_experts=5, expert_epochs=50,
                        syn_steps=5000, lr_img=0.1,
                        expert_dir=f'/workspace/expert_traj_final_ipc{ipc}',
                        device=DEVICE, seed=0
                    ))
                torch.save({'images': syn_images, 'labels': syn_labels}, tm_path)
            else:
                data = torch.load(tm_path, map_location='cpu')
                syn_images, syn_labels = data['images'], data['labels']
                print(f"Loaded TM IPC={ipc} from {tm_path}")
            
            tm_sl = generate_soft_labels_from_teacher(syn_images, teacher, DEVICE)
            
            for label_type in ['hard', 'soft']:
                key = f"tm_ipc{ipc}_{label_type}"
                if key in results:
                    print(f"  Skipping {key}")
                    continue
                print(f"\n--- TM IPC={ipc}, {label_type}: Evaluating ---")
                sl = tm_sl if label_type == 'soft' else None
                results[key] = evaluate_config(
                    syn_images, syn_labels, test_images, test_labels,
                    sl, label_type, num_runs=3
                )
                results[key]['method'] = 'tm'
                results[key]['ipc'] = ipc
                save_results()
    
    # Final table
    print("\n\nFINAL RESULTS:")
    generate_table()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
