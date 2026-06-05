"""
V2 Comprehensive Runner - Improved experiments to match paper results.

Key improvements over V1:
1. Better teacher model (300 epochs, cosine schedule, DSA augmentation)
2. Feature-space K-centers using trained model
3. Proper iteration counts for DD methods
4. 3 evaluation runs for all configs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import sys
import gc

from convnet import ConvNet, get_convnet_d3
from data_utils import get_cifar100_tensors, get_class_indices, random_select, k_centers_select
from train_eval import train_and_evaluate
from dsa import DiffAugment


def train_teacher_v2(train_images, train_labels, test_images, test_labels,
                     device='cuda', epochs=300, save_path='/workspace/teacher_v2.pt'):
    """Train a strong teacher model for soft label generation."""
    if os.path.exists(save_path):
        print(f"Loading existing teacher from {save_path}")
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        model = get_convnet_d3().to(device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Teacher test accuracy: {checkpoint.get('test_acc', 'unknown')}%")
        return model
    
    print(f"Training V2 teacher model ({epochs} epochs, cosine schedule, DSA aug)...")
    model = get_convnet_d3().to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, 
                                          num_workers=0, pin_memory=True)
    
    best_acc = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_imgs, batch_labels in loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            
            # Apply augmentation during teacher training
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i in range(0, len(test_images), 512):
                    batch = test_images[i:i+512].to(device)
                    labels = test_labels[i:i+512].to(device)
                    outputs = model(batch)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100.0 * correct / total
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{epochs}, Test Acc: {acc:.2f}%, Time: {elapsed:.0f}s")
            if acc > best_acc:
                best_acc = acc
    
    # Save teacher
    torch.save({'state_dict': model.state_dict(), 'test_acc': best_acc}, save_path)
    print(f"Teacher saved. Best test accuracy: {best_acc:.2f}%")
    return model


def generate_soft_labels_v2(train_images, teacher_model, device='cuda',
                            save_path='/workspace/soft_labels_v2.pt'):
    """Generate soft labels using trained teacher model."""
    if os.path.exists(save_path):
        print(f"Loading existing soft labels from {save_path}")
        return torch.load(save_path, map_location='cpu', weights_only=False)
    
    print("Generating soft labels from teacher...")
    teacher_model.eval()
    all_logits = []
    
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(device)
            logits = teacher_model(batch)
            all_logits.append(logits.cpu())
    
    soft_labels = torch.cat(all_logits, dim=0)
    torch.save(soft_labels, save_path)
    print(f"Soft labels saved: {soft_labels.shape}")
    return soft_labels


def run_eval(train_images, train_labels, test_images, test_labels,
             model_fn, label_type, soft_labels=None, num_runs=3, device='cuda'):
    """Run evaluation with multiple seeds."""
    accs = []
    for run in range(num_runs):
        acc = train_and_evaluate(
            train_images, train_labels, test_images, test_labels,
            model_fn, num_classes=100, device=device,
            label_type=label_type, soft_labels=soft_labels,
            epochs=300, batch_size=256, seed=run, verbose=False
        )
        accs.append(acc)
        print(f"    Run {run+1}: {acc:.2f}%")
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"    → {mean_acc:.2f} ± {std_acc:.2f}%")
    return {'mean': float(mean_acc), 'std': float(std_acc), 'runs': [float(a) for a in accs]}


def save_results(results, path='/workspace/results/results_v2.json'):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def print_results_table(results):
    """Print formatted results table."""
    paper = {
        'dm_ipc10_hard': 29.23, 'dm_ipc10_soft': 26.13,
        'dm_ipc50_hard': 42.32, 'dm_ipc50_soft': 43.46,
        'dc_ipc10_hard': 28.42, 'dc_ipc10_soft': 23.54,
        'dc_ipc50_hard': 30.56, 'dc_ipc50_soft': 33.46,
        'tm_ipc10_hard': 38.18, 'tm_ipc10_soft': 37.60,
        'tm_ipc50_hard': 46.32, 'tm_ipc50_soft': 46.26,
        'random_ipc10_hard': 18.64, 'random_ipc10_soft': 33.43,
        'random_ipc50_hard': 34.66, 'random_ipc50_soft': 45.39,
        'k_centers_ipc10_hard': 25.04, 'k_centers_ipc10_soft': 34.70,
        'k_centers_ipc50_hard': 38.64, 'k_centers_ipc50_soft': 46.24,
    }
    
    header = f"{'Method':<12} {'IPC':>4} {'Label':>6} {'Ours':>12} {'Paper':>8} {'Diff':>8}"
    sep = "-" * 55
    lines = [header, sep]
    
    for method in ['dm', 'dc', 'tm', 'random', 'k_centers']:
        for ipc in [10, 50]:
            for label in ['hard', 'soft']:
                key = f'{method}_ipc{ipc}_{label}'
                if key in results:
                    ours = results[key]['mean']
                    std = results[key]['std']
                    pval = paper.get(key, 0)
                    diff = ours - pval
                    line = f"{method:<12} {ipc:>4} {label:>6} {ours:>7.2f}±{std:.2f} {pval:>8.2f} {diff:>+8.2f}"
                    lines.append(line)
    
    for line in lines:
        print(line)
    
    # Save table to file
    table_path = '/workspace/results/table_v2.txt'
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load existing results if any
    results_path = '/workspace/results/results_v2.json'
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")
    else:
        results = {}
    
    # Load data
    print("=" * 60)
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    model_fn = lambda: get_convnet_d3()
    
    # ============================================================
    # Step 1: Train teacher and generate soft labels
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: Training teacher model")
    teacher = train_teacher_v2(train_images, train_labels, test_images, test_labels, device)
    
    print("\nSTEP 2: Generating soft labels")
    soft_labels_full = generate_soft_labels_v2(train_images, teacher, device)
    
    # ============================================================
    # Step 2: Coreset methods (Random, K-centers)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Coreset methods")
    
    for method_name in ['random', 'k_centers']:
        for ipc in [10, 50]:
            hl_key = f'{method_name}_ipc{ipc}_hard'
            sl_key = f'{method_name}_ipc{ipc}_soft'
            
            if hl_key in results and sl_key in results:
                print(f"  Skipping {method_name} IPC={ipc} (already done)")
                continue
            
            print(f"\n--- {method_name} IPC={ipc} ---")
            
            if method_name == 'random':
                selected = random_select(train_labels, ipc=ipc, seed=42)
            else:
                # Feature-space K-centers using trained teacher
                selected = k_centers_select(
                    train_images, train_labels, ipc=ipc, 
                    use_features=True, feature_model=teacher, device=device, seed=42
                )
            
            sub_images = train_images[selected]
            sub_labels = train_labels[selected]
            sub_soft = soft_labels_full[selected]
            
            if hl_key not in results:
                print(f"  HL evaluation:")
                res = run_eval(sub_images, sub_labels, test_images, test_labels,
                              model_fn, 'hard', num_runs=3, device=device)
                results[hl_key] = res
            
            if sl_key not in results:
                print(f"  SL evaluation:")
                res = run_eval(sub_images, sub_labels, test_images, test_labels,
                              model_fn, 'soft', soft_labels=sub_soft, num_runs=3, device=device)
                results[sl_key] = res
            
            save_results(results)
    
    # ============================================================
    # Step 3: DM distillation (5000 iterations - feasible compromise)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: Distribution Matching (DM)")
    
    from distill_dm import distribution_matching
    
    for ipc in [10, 50]:
        hl_key = f'dm_ipc{ipc}_hard'
        sl_key = f'dm_ipc{ipc}_soft'
        
        if hl_key in results and sl_key in results:
            print(f"  Skipping DM IPC={ipc} (already done)")
            continue
        
        dm_path = f'/workspace/distilled_dm_v2_ipc{ipc}.pt'
        dm_sl_path = f'/workspace/soft_labels_dm_v2_ipc{ipc}.pt'
        
        if os.path.exists(dm_path):
            print(f"Loading existing DM IPC={ipc} from {dm_path}")
            data = torch.load(dm_path, map_location='cpu', weights_only=False)
            syn_images, syn_labels = data['images'], data['labels']
        else:
            print(f"\n--- DM IPC={ipc} (5000 iterations) ---")
            t0 = time.time()
            syn_images, syn_labels = distribution_matching(
                train_images, train_labels, ipc=ipc,
                iterations=5000, lr_img=1.0, batch_real=256,
                device=device, seed=0
            )
            print(f"  DM IPC={ipc} done in {time.time()-t0:.0f}s")
            torch.save({'images': syn_images, 'labels': syn_labels}, dm_path)
        
        # Generate soft labels for distilled data
        if os.path.exists(dm_sl_path):
            syn_soft = torch.load(dm_sl_path, map_location='cpu', weights_only=False)
        else:
            teacher.eval()
            with torch.no_grad():
                syn_soft = teacher(syn_images.to(device)).cpu()
            torch.save(syn_soft, dm_sl_path)
        
        # Evaluate
        if hl_key not in results:
            print(f"  DM IPC={ipc} HL evaluation:")
            res = run_eval(syn_images, syn_labels, test_images, test_labels,
                          model_fn, 'hard', num_runs=3, device=device)
            results[hl_key] = res
        
        if sl_key not in results:
            print(f"  DM IPC={ipc} SL evaluation:")
            res = run_eval(syn_images, syn_labels, test_images, test_labels,
                          model_fn, 'soft', soft_labels=syn_soft, num_runs=3, device=device)
            results[sl_key] = res
        
        save_results(results)
        gc.collect()
        torch.cuda.empty_cache()
    
    # ============================================================
    # Step 4: DC distillation (10 outer × 50 inner = 500 loops)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: Dataset Condensation (DC)")
    
    from distill_dc import gradient_matching
    
    for ipc in [10, 50]:
        hl_key = f'dc_ipc{ipc}_hard'
        sl_key = f'dc_ipc{ipc}_soft'
        
        if hl_key in results and sl_key in results:
            print(f"  Skipping DC IPC={ipc} (already done)")
            continue
        
        dc_path = f'/workspace/distilled_dc_v2_ipc{ipc}.pt'
        dc_sl_path = f'/workspace/soft_labels_dc_v2_ipc{ipc}.pt'
        
        if os.path.exists(dc_path):
            print(f"Loading existing DC IPC={ipc} from {dc_path}")
            data = torch.load(dc_path, map_location='cpu', weights_only=False)
            syn_images, syn_labels = data['images'], data['labels']
        else:
            print(f"\n--- DC IPC={ipc} (10 outer × 50 inner) ---")
            t0 = time.time()
            syn_images, syn_labels = gradient_matching(
                train_images, train_labels, ipc=ipc,
                outer_loops=10, inner_loops=50, lr_img=1.0,
                batch_real=256, device=device, seed=0
            )
            print(f"  DC IPC={ipc} done in {time.time()-t0:.0f}s")
            torch.save({'images': syn_images, 'labels': syn_labels}, dc_path)
        
        # Generate soft labels
        if os.path.exists(dc_sl_path):
            syn_soft = torch.load(dc_sl_path, map_location='cpu', weights_only=False)
        else:
            teacher.eval()
            with torch.no_grad():
                syn_soft = teacher(syn_images.to(device)).cpu()
            torch.save(syn_soft, dc_sl_path)
        
        # Evaluate
        if hl_key not in results:
            print(f"  DC IPC={ipc} HL evaluation:")
            res = run_eval(syn_images, syn_labels, test_images, test_labels,
                          model_fn, 'hard', num_runs=3, device=device)
            results[hl_key] = res
        
        if sl_key not in results:
            print(f"  DC IPC={ipc} SL evaluation:")
            res = run_eval(syn_images, syn_labels, test_images, test_labels,
                          model_fn, 'soft', soft_labels=syn_soft, num_runs=3, device=device)
            results[sl_key] = res
        
        save_results(results)
        gc.collect()
        torch.cuda.empty_cache()
    
    # ============================================================
    # Step 5: TM distillation
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 6: Trajectory Matching (TM)")
    
    from distill_tm import train_expert_trajectories, trajectory_matching
    
    # Train experts if needed
    expert_dir = '/workspace/expert_trajectories_v2'
    num_experts = 10
    expert_epochs = 50
    
    if not os.path.exists(expert_dir) or len([f for f in os.listdir(expert_dir) if f.startswith('expert_')]) < num_experts:
        print(f"Training {num_experts} expert trajectories ({expert_epochs} epochs each)...")
        t0 = time.time()
        train_expert_trajectories(
            train_images, train_labels,
            num_experts=num_experts, expert_epochs=expert_epochs,
            save_dir=expert_dir, device=device, seed=0
        )
        print(f"  Expert training done in {time.time()-t0:.0f}s")
    else:
        n_existing = len([f for f in os.listdir(expert_dir) if f.startswith('expert_')])
        print(f"Using existing {n_existing} expert trajectories")
    
    for ipc in [10, 50]:
        hl_key = f'tm_ipc{ipc}_hard'
        sl_key = f'tm_ipc{ipc}_soft'
        
        if hl_key in results and sl_key in results:
            print(f"  Skipping TM IPC={ipc} (already done)")
            continue
        
        tm_path = f'/workspace/distilled_tm_v2_ipc{ipc}.pt'
        tm_sl_path = f'/workspace/soft_labels_tm_v2_ipc{ipc}.pt'
        
        if os.path.exists(tm_path):
            print(f"Loading existing TM IPC={ipc} from {tm_path}")
            data = torch.load(tm_path, map_location='cpu', weights_only=False)
            syn_images, syn_labels = data['images'], data['labels']
        else:
            print(f"\n--- TM IPC={ipc} (5000 iterations) ---")
            t0 = time.time()
            syn_images, syn_labels = trajectory_matching(
                train_images, train_labels, ipc=ipc,
                expert_dir=expert_dir, num_experts=num_experts,
                iterations=5000, lr_img=1000.0, lr_lr=1e-5,
                syn_steps=30, expert_epochs=3, max_start_epoch=25,
                device=device, seed=0
            )
            print(f"  TM IPC={ipc} done in {time.time()-t0:.0f}s")
            torch.save({'images': syn_images, 'labels': syn_labels}, tm_path)
        
        # Generate soft labels
        if os.path.exists(tm_sl_path):
            syn_soft = torch.load(tm_sl_path, map_location='cpu', weights_only=False)
        else:
            teacher.eval()
            with torch.no_grad():
                syn_soft = teacher(syn_images.to(device)).cpu()
            torch.save(syn_soft, tm_sl_path)
        
        # Evaluate
        if hl_key not in results:
            print(f"  TM IPC={ipc} HL evaluation:")
            res = run_eval(syn_images, syn_labels, test_images, test_labels,
                          model_fn, 'hard', num_runs=3, device=device)
            results[hl_key] = res
        
        if sl_key not in results:
            print(f"  TM IPC={ipc} SL evaluation:")
            res = run_eval(syn_images, syn_labels, test_images, test_labels,
                          model_fn, 'soft', soft_labels=syn_soft, num_runs=3, device=device)
            results[sl_key] = res
        
        save_results(results)
        gc.collect()
        torch.cuda.empty_cache()
    
    # ============================================================
    # Final: Print results table
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print_results_table(results)
    save_results(results)
    print("\nDone! Results saved to /workspace/results/results_v2.json")


if __name__ == '__main__':
    main()
