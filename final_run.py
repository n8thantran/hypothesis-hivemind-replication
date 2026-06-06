"""
Final clean pipeline to reproduce Table: tab:small_scale_c100
CIFAR-100, ConvNet-D3, IPC 10 and 50

Methods: DM, DC, TM (dataset distillation) + Random, K-centers (coresets)
Settings: HL (hard labels) and SL (soft labels from teacher)

Paper hyperparameters (Table tab:stage3_hyper):
- HL: 300 epochs, SGD, lr=0.01, momentum=0.9, wd=5e-4, StepLR@151 (gamma=0.1), batch=256, DSA, CE loss
- SL: 300 epochs, AdamW, lr=1e-3, wd=0.01, Cosine scheduler, batch=256, DSA, KL-Div(T=20), no warmup
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
from dsa import DiffAugment


def evaluate_model(model, test_images, test_labels, device='cuda', batch_size=512):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            imgs = test_images[i:i+batch_size].to(device)
            labels = test_labels[i:i+batch_size].to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    model.train()
    return 100.0 * correct / total


def train_and_eval(train_images, train_labels, test_images, test_labels,
                   label_type='hard', soft_labels=None, seed=0, device='cuda',
                   epochs=300, verbose=True):
    """
    Train ConvNet-D3 on given data and evaluate.
    
    For HL: train_labels are integer class labels, CE loss
    For SL: soft_labels are raw teacher logits (not probabilities), KL-Div loss
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    
    batch_size = 256
    n_train = len(train_images)
    
    if label_type == 'hard':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        T = 20.0
    
    # Move data to GPU once if possible
    train_images_gpu = train_images.to(device)
    train_labels_gpu = train_labels.to(device) if train_labels is not None else None
    if soft_labels is not None:
        soft_labels_gpu = soft_labels.to(device)
    
    eff_bs = min(batch_size, n_train)
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train, device='cpu')
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, n_train, eff_bs):
            idx = perm[start:start+eff_bs]
            batch_imgs = train_images_gpu[idx]
            
            # Apply DSA augmentation
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            
            if label_type == 'hard':
                loss = F.cross_entropy(outputs, train_labels_gpu[idx])
            else:
                # KL-Div with temperature T=20
                batch_soft = soft_labels_gpu[idx]
                log_student = F.log_softmax(outputs / T, dim=1)
                target_teacher = F.softmax(batch_soft / T, dim=1)
                loss = F.kl_div(log_student, target_teacher, reduction='batchmean') * (T * T)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            acc = evaluate_model(model, test_images, test_labels, device)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/max(n_batches,1):.4f}, Acc: {acc:.2f}%")
    
    acc = evaluate_model(model, test_images, test_labels, device)
    return acc


def train_teacher(train_images, train_labels, test_images, test_labels, 
                  epochs=300, device='cuda'):
    """Train a teacher model on full CIFAR-100 with DSA augmentation."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
    
    batch_size = 256
    n_train = len(train_images)
    
    # Move to GPU
    train_images_gpu = train_images.to(device)
    train_labels_gpu = train_labels.to(device)
    
    best_acc = 0
    best_state = None
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start+batch_size]
            imgs = train_images_gpu[idx]
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, train_labels_gpu[idx])
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            acc = evaluate_model(model, test_images, test_labels, device)
            print(f"  Teacher epoch {epoch+1}/{epochs}, Acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    print(f"Best teacher accuracy: {best_acc:.2f}%")
    return best_state, best_acc


def generate_soft_labels(teacher_state, images, device='cuda', batch_size=256):
    """Generate teacher logits for given images."""
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    model.load_state_dict(teacher_state)
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    return torch.cat(all_logits, dim=0)


def coreset_random(train_labels, ipc, seed=0):
    """Random coreset selection: IPC samples per class."""
    np.random.seed(seed)
    indices = []
    for c in range(100):
        class_idx = (train_labels == c).nonzero(as_tuple=True)[0].numpy()
        selected = np.random.choice(class_idx, size=ipc, replace=False)
        indices.extend(selected.tolist())
    return indices


def coreset_kcenters(train_images, train_labels, ipc, seed=0):
    """K-centers coreset selection using feature-space distance."""
    np.random.seed(seed)
    indices = []
    
    for c in range(100):
        class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        class_imgs = train_images[class_idx]
        
        # Flatten to feature vectors
        feats = class_imgs.view(len(class_idx), -1).numpy()
        
        # Greedy K-centers: iteratively select furthest point
        n = len(feats)
        selected = [np.random.randint(n)]  # start with random
        
        # Distance from each point to nearest selected point  
        min_dists = np.full(n, np.inf)
        
        for _ in range(ipc - 1):
            last = selected[-1]
            dists = np.sum((feats - feats[last:last+1]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected] = -1  # already selected
            next_idx = np.argmax(min_dists)
            selected.append(next_idx)
        
        indices.extend(class_idx[selected].tolist())
    
    return indices


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['all'], 
                       help='Methods to evaluate: all, random, kcenter, dm, dc, tm')
    parser.add_argument('--ipcs', nargs='+', type=int, default=[10, 50])
    parser.add_argument('--settings', nargs='+', default=['hl', 'sl'])
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--retrain-teacher', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    
    if 'all' in args.methods:
        args.methods = ['random', 'kcenter', 'dm', 'dc', 'tm']
    
    # Load CIFAR-100
    print("Loading CIFAR-100...")
    data = torch.load('cifar100_tensors.pt', map_location='cpu')
    train_images = data['train_images']  # (50000, 3, 32, 32)
    train_labels = data['train_labels']  # (50000,)
    test_images = data['test_images']    # (10000, 3, 32, 32)
    test_labels = data['test_labels']    # (10000,)
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Get or train teacher
    teacher_path = 'teacher_final.pt'
    if args.retrain_teacher or not os.path.exists(teacher_path):
        print("\n=== Training Teacher ===")
        teacher_state, teacher_acc = train_teacher(
            train_images, train_labels, test_images, test_labels, 
            epochs=300, device=device
        )
        torch.save({'state_dict': teacher_state, 'accuracy': teacher_acc}, teacher_path)
    else:
        print(f"\nLoading teacher from {teacher_path}...")
        ckpt = torch.load(teacher_path, map_location='cpu')
        teacher_state = ckpt['state_dict']
        teacher_acc = ckpt['accuracy']
        print(f"Teacher accuracy: {teacher_acc:.2f}%")
    
    # Generate soft labels for full training set
    full_sl_path = 'soft_labels_full_final.pt'
    if not os.path.exists(full_sl_path):
        print("Generating soft labels for full training set...")
        full_logits = generate_soft_labels(teacher_state, train_images, device)
        torch.save(full_logits, full_sl_path)
    else:
        full_logits = torch.load(full_sl_path, map_location='cpu')
    print(f"Full soft labels: {full_logits.shape}")
    
    results = {}
    
    for ipc in args.ipcs:
        print(f"\n{'='*60}")
        print(f"IPC = {ipc}")
        print(f"{'='*60}")
        
        for method in args.methods:
            print(f"\n--- Method: {method.upper()} ---")
            
            # Get training data
            if method == 'random':
                for setting in args.settings:
                    key = f"Random_IPC{ipc}_{setting.upper()}"
                    accs = []
                    for run in range(args.runs):
                        print(f"  {key} run {run+1}/{args.runs}")
                        indices = coreset_random(train_labels, ipc, seed=run)
                        sub_images = train_images[indices]
                        sub_labels = train_labels[indices]
                        
                        if setting == 'hl':
                            acc = train_and_eval(sub_images, sub_labels, 
                                               test_images, test_labels,
                                               label_type='hard', seed=run, 
                                               device=device, epochs=args.epochs,
                                               verbose=False)
                        else:
                            sub_soft = full_logits[indices]
                            acc = train_and_eval(sub_images, sub_labels,
                                               test_images, test_labels,
                                               label_type='soft', soft_labels=sub_soft,
                                               seed=run, device=device, epochs=args.epochs,
                                               verbose=False)
                        accs.append(acc)
                        print(f"    acc = {acc:.2f}%")
                    
                    results[key] = {'mean': np.mean(accs), 'std': np.std(accs), 'runs': accs}
                    print(f"  {key}: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
                    
            elif method == 'kcenter':
                # K-centers: same selection for all runs
                indices = coreset_kcenters(train_images, train_labels, ipc, seed=0)
                sub_images = train_images[indices]
                sub_labels = train_labels[indices]
                sub_soft = full_logits[indices]
                
                for setting in args.settings:
                    key = f"Kcenter_IPC{ipc}_{setting.upper()}"
                    accs = []
                    for run in range(args.runs):
                        print(f"  {key} run {run+1}/{args.runs}")
                        if setting == 'hl':
                            acc = train_and_eval(sub_images, sub_labels,
                                               test_images, test_labels,
                                               label_type='hard', seed=run,
                                               device=device, epochs=args.epochs,
                                               verbose=False)
                        else:
                            acc = train_and_eval(sub_images, sub_labels,
                                               test_images, test_labels,
                                               label_type='soft', soft_labels=sub_soft,
                                               seed=run, device=device, epochs=args.epochs,
                                               verbose=False)
                        accs.append(acc)
                        print(f"    acc = {acc:.2f}%")
                    
                    results[key] = {'mean': np.mean(accs), 'std': np.std(accs), 'runs': accs}
                    print(f"  {key}: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
                    
            elif method in ['dm', 'dc', 'tm']:
                # Load distilled set
                dd_path = f'distilled_{method}_ipc{ipc}.pt'
                if not os.path.exists(dd_path):
                    print(f"  WARNING: {dd_path} not found, skipping")
                    continue
                    
                dd = torch.load(dd_path, map_location='cpu')
                if isinstance(dd, dict):
                    dd_images = dd['images']
                    dd_labels = dd['labels']
                else:
                    dd_images, dd_labels = dd[0], dd[1]
                    
                print(f"  Distilled set: {dd_images.shape}, labels: {dd_labels.shape}")
                
                # Generate soft labels for distilled images using teacher
                dd_soft = generate_soft_labels(teacher_state, dd_images, device)
                
                for setting in args.settings:
                    key = f"{method.upper()}_IPC{ipc}_{setting.upper()}"
                    accs = []
                    for run in range(args.runs):
                        print(f"  {key} run {run+1}/{args.runs}")
                        if setting == 'hl':
                            acc = train_and_eval(dd_images, dd_labels,
                                               test_images, test_labels,
                                               label_type='hard', seed=run,
                                               device=device, epochs=args.epochs,
                                               verbose=False)
                        else:
                            acc = train_and_eval(dd_images, dd_labels,
                                               test_images, test_labels,
                                               label_type='soft', soft_labels=dd_soft,
                                               seed=run, device=device, epochs=args.epochs,
                                               verbose=False)
                        accs.append(acc)
                        print(f"    acc = {acc:.2f}%")
                    
                    results[key] = {'mean': np.mean(accs), 'std': np.std(accs), 'runs': accs}
                    print(f"  {key}: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/final_table.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print formatted table
    print_table(results)
    
    return results


def print_table(results):
    """Print results in the format of Table tab:small_scale_c100."""
    print("\n" + "="*70)
    print("Table: Small-scale CIFAR-100 (ConvNet-D3)")
    print("="*70)
    print(f"{'Method':<12} {'IPC':<6} {'HL':>16} {'SL':>16}")
    print("-"*50)
    
    for method in ['DM', 'DC', 'TM', 'Random', 'Kcenter']:
        for ipc in [10, 50]:
            hl_key = f"{method}_IPC{ipc}_HL"
            sl_key = f"{method}_IPC{ipc}_SL"
            
            hl_str = "--"
            sl_str = "--"
            
            if hl_key in results:
                hl_str = f"{results[hl_key]['mean']:.2f}±{results[hl_key]['std']:.2f}"
            if sl_key in results:
                sl_str = f"{results[sl_key]['mean']:.2f}±{results[sl_key]['std']:.2f}"
            
            name = method if method != 'Kcenter' else 'K-centers'
            print(f"{name:<12} {ipc:<6} {hl_str:>16} {sl_str:>16}")
    
    print("="*70)
    
    # Print paper reference values
    print("\nPaper reference values (for comparison):")
    print(f"{'Method':<12} {'IPC':<6} {'HL':>16} {'SL':>16}")
    print("-"*50)
    paper = {
        ('DM', 10): ('29.23±0.26', '26.13±0.10'),
        ('DM', 50): ('42.32±0.37', '43.46±0.18'),
        ('DC', 10): ('28.42±0.29', '23.54±0.31'),
        ('DC', 50): ('30.56±0.56', '33.46±0.38'),
        ('TM', 10): ('38.18±0.42', '37.60±0.25'),
        ('TM', 50): ('46.32±0.26', '46.26±0.30'),
        ('Random', 10): ('18.64±0.25', '33.43±0.18'),
        ('Random', 50): ('34.66±0.41', '45.39±0.23'),
        ('K-centers', 10): ('25.04±0.30', '34.70±0.27'),
        ('K-centers', 50): ('38.64±0.43', '46.24±0.12'),
    }
    for (m, ipc), (hl, sl) in paper.items():
        print(f"{m:<12} {ipc:<6} {hl:>16} {sl:>16}")


if __name__ == '__main__':
    main()
