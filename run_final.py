"""
Final comprehensive experiment runner for paper replication.
Runs all 20 configs (5 methods × 2 IPC × 2 label types) with 3 trials each.
Time-optimized with GPU-resident data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import sys
from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_cifar100_tensors

DEVICE = 'cuda'
NUM_CLASSES = 100


def train_and_eval_fast(train_imgs, train_labels, test_imgs_gpu, test_labels_gpu,
                        label_type='hard', soft_labels=None,
                        epochs=300, batch_size=256, seed=0):
    """Fast training with GPU-resident test data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(DEVICE)
    
    # Move train data to GPU
    train_imgs_gpu = train_imgs.to(DEVICE)
    train_labels_gpu = train_labels.to(DEVICE)
    if soft_labels is not None:
        soft_labels_gpu = soft_labels.to(DEVICE)
    
    n_train = len(train_imgs_gpu)
    
    if label_type == 'hard':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        temperature = 20.0
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = train_imgs_gpu[idx]
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            out = model(imgs)
            
            if label_type == 'hard':
                loss = criterion(out, train_labels_gpu[idx])
            else:
                log_probs = F.log_softmax(out / temperature, dim=1)
                targets = F.softmax(soft_labels_gpu[idx] / temperature, dim=1)
                loss = F.kl_div(log_probs, targets, reduction='batchmean') * (temperature ** 2)
            
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(test_imgs_gpu), 512):
            out = model(test_imgs_gpu[i:i+512])
            correct += out.argmax(1).eq(test_labels_gpu[i:i+512]).sum().item()
    return 100.0 * correct / len(test_labels_gpu)


def run_trials(train_imgs, train_labels, test_imgs_gpu, test_labels_gpu,
               label_type, soft_labels=None, num_trials=3, epochs=300):
    """Run multiple trials and return mean±std."""
    accs = []
    for trial in range(num_trials):
        acc = train_and_eval_fast(train_imgs, train_labels, test_imgs_gpu, test_labels_gpu,
                                   label_type=label_type, soft_labels=soft_labels,
                                   epochs=epochs, seed=trial)
        accs.append(acc)
        print(f"    Trial {trial+1}: {acc:.2f}%")
    mean = np.mean(accs)
    std = np.std(accs)
    print(f"    => {mean:.2f} ± {std:.2f}%")
    return mean, std, accs


def random_select(labels, ipc, seed=42):
    """Select ipc images per class randomly."""
    rng = np.random.RandomState(seed)
    indices = []
    for c in range(NUM_CLASSES):
        cls_idx = (labels == c).nonzero(as_tuple=True)[0].numpy()
        sel = rng.choice(cls_idx, size=ipc, replace=False)
        indices.extend(sel.tolist())
    return indices


def k_centers_select(images, labels, ipc, teacher_model=None, seed=42):
    """K-centers coreset selection in feature space."""
    # Get features from teacher
    if teacher_model is not None:
        features_list = []
        teacher_model.eval()
        with torch.no_grad():
            for i in range(0, len(images), 256):
                batch = images[i:i+256].to(DEVICE)
                # Get features from the embedding layer
                feat = teacher_model.embed(batch)
                features_list.append(feat.cpu())
        features = torch.cat(features_list, dim=0)
    else:
        features = images.view(len(images), -1)
    
    indices = []
    for c in range(NUM_CLASSES):
        cls_mask = (labels == c)
        cls_idx = cls_mask.nonzero(as_tuple=True)[0]
        cls_feat = features[cls_idx]
        
        # Greedy k-centers
        n = len(cls_feat)
        selected = []
        # Start with random point
        rng = np.random.RandomState(seed + c)
        first = rng.randint(n)
        selected.append(first)
        
        # Compute distances to first center
        dists = torch.cdist(cls_feat, cls_feat[first:first+1]).squeeze(1)
        
        for _ in range(1, ipc):
            # Select point with max min-distance
            farthest = dists.argmax().item()
            selected.append(farthest)
            new_dists = torch.cdist(cls_feat, cls_feat[farthest:farthest+1]).squeeze(1)
            dists = torch.min(dists, new_dists)
        
        indices.extend(cls_idx[selected].tolist())
    
    return indices


def distill_dm(train_images, train_labels, ipc, n_iter=10000, lr_img=1.0):
    """Distribution Matching distillation."""
    print(f"  DM distillation: ipc={ipc}, iter={n_iter}")
    
    # Initialize synthetic data from class means
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        cls_imgs = train_images[cls_idx]
        perm = torch.randperm(len(cls_imgs))[:ipc]
        syn_images.append(cls_imgs[perm].clone())
        syn_labels.extend([c] * ipc)
    
    syn_images = torch.cat(syn_images, dim=0).to(DEVICE).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, device=DEVICE)
    train_images_gpu = train_images.to(DEVICE)
    train_labels_gpu = train_labels.to(DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    for it in range(n_iter):
        # New random model each iteration
        model = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(DEVICE)
        model.train()
        
        loss_total = 0
        for c in range(NUM_CLASSES):
            # Real features for class c
            real_idx = (train_labels_gpu == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(real_idx))[:256]
            real_batch = train_images_gpu[real_idx[perm]]
            real_batch = DiffAugment(real_batch, strategy='color_crop_cutout_flip_scale_rotate')
            
            # Synthetic features for class c
            syn_idx = (syn_labels == c).nonzero(as_tuple=True)[0]
            syn_batch = syn_images[syn_idx]
            syn_batch_aug = DiffAugment(syn_batch, strategy='color_crop_cutout_flip_scale_rotate')
            
            with torch.no_grad():
                real_feat = model.embed(real_batch)
            syn_feat = model.embed(syn_batch_aug)
            
            # Match means
            loss_total += ((real_feat.mean(0) - syn_feat.mean(0)) ** 2).sum()
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        if (it + 1) % 2000 == 0:
            print(f"    Iter {it+1}/{n_iter}, Loss: {loss_total.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_dc(train_images, train_labels, ipc, outer_loops=50, inner_loops=50, lr_img=1.0):
    """Dataset Condensation via gradient matching."""
    print(f"  DC distillation: ipc={ipc}, outer={outer_loops}, inner={inner_loops}")
    
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(cls_idx))[:ipc]
        syn_images.append(train_images[cls_idx[perm]].clone())
        syn_labels.extend([c] * ipc)
    
    syn_images = torch.cat(syn_images, dim=0).to(DEVICE).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, device=DEVICE)
    train_images_gpu = train_images.to(DEVICE)
    train_labels_gpu = train_labels.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    for outer in range(outer_loops):
        model = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(DEVICE)
        model.train()
        
        for inner in range(inner_loops):
            # Gradient on real data
            real_idx = torch.randperm(len(train_images_gpu))[:256]
            real_batch = DiffAugment(train_images_gpu[real_idx], strategy='color_crop_cutout_flip_scale_rotate')
            real_labels = train_labels_gpu[real_idx]
            
            loss_real = criterion(model(real_batch), real_labels)
            grad_real = torch.autograd.grad(loss_real, model.parameters(), create_graph=False)
            
            # Gradient on synthetic data
            syn_aug = DiffAugment(syn_images, strategy='color_crop_cutout_flip_scale_rotate')
            loss_syn = criterion(model(syn_aug), syn_labels)
            grad_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)
            
            # Match gradients
            loss_match = sum(((gs - gr.detach()) ** 2).sum() / (gr.detach() ** 2).sum().clamp(min=1e-8)
                           for gs, gr in zip(grad_syn, grad_real))
            
            optimizer.zero_grad()
            loss_match.backward()
            optimizer.step()
            
            # Update model with synthetic gradient (no graph)
            with torch.no_grad():
                for p, g in zip(model.parameters(), grad_syn):
                    p.sub_(0.01 * g.detach())
        
        if (outer + 1) % 10 == 0:
            print(f"    Outer {outer+1}/{outer_loops}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def distill_tm(train_images, train_labels, ipc, n_iter=3000, lr_img=1000.0, n_experts=3, expert_epochs=30):
    """Trajectory Matching distillation."""
    print(f"  TM distillation: ipc={ipc}, iter={n_iter}, experts={n_experts}")
    
    train_images_gpu = train_images.to(DEVICE)
    train_labels_gpu = train_labels.to(DEVICE)
    
    # Train expert trajectories
    print("    Training expert trajectories...")
    expert_trajectories = []
    for exp_id in range(n_experts):
        model = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        trajectory = [{ k: v.cpu().clone() for k, v in model.state_dict().items() }]
        
        for ep in range(expert_epochs):
            model.train()
            perm = torch.randperm(len(train_images_gpu), device=DEVICE)
            for i in range(0, len(train_images_gpu), 256):
                idx = perm[i:i+256]
                imgs = DiffAugment(train_images_gpu[idx], strategy='color_crop_cutout_flip_scale_rotate')
                loss = criterion(model(imgs), train_labels_gpu[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            trajectory.append({ k: v.cpu().clone() for k, v in model.state_dict().items() })
        
        expert_trajectories.append(trajectory)
        print(f"    Expert {exp_id+1}/{n_experts} trained ({expert_epochs} epochs)")
    
    # Initialize synthetic data
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(cls_idx))[:ipc]
        syn_images.append(train_images[cls_idx[perm]].clone())
        syn_labels.extend([c] * ipc)
    
    syn_images = torch.cat(syn_images, dim=0).to(DEVICE).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, device=DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    
    max_start_epoch = expert_epochs - 2  # Leave room for target
    
    for it in range(n_iter):
        # Pick random expert and starting point
        exp_idx = np.random.randint(n_experts)
        start_epoch = np.random.randint(0, max(1, max_start_epoch))
        
        # Load starting params
        start_params = expert_trajectories[exp_idx][start_epoch]
        target_params = expert_trajectories[exp_idx][min(start_epoch + 2, expert_epochs)]
        
        # Create student model at start point
        student = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(DEVICE)
        student.load_state_dict({k: v.to(DEVICE) for k, v in start_params.items()})
        student.train()
        
        # Train student on synthetic data for a few steps
        student_opt = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
        
        for _ in range(10):  # inner steps
            syn_aug = DiffAugment(syn_images, strategy='color_crop_cutout_flip_scale_rotate')
            loss_s = criterion(student(syn_aug), syn_labels)
            student_opt.zero_grad()
            loss_s.backward()
            student_opt.step()
        
        # Compare student params to target trajectory
        loss_match = 0
        target_flat = torch.cat([v.to(DEVICE).reshape(-1) for v in target_params.values()])
        student_flat = torch.cat([p.reshape(-1) for p in student.parameters()])
        start_flat = torch.cat([v.to(DEVICE).reshape(-1) for v in start_params.values()])
        
        target_direction = target_flat - start_flat
        student_direction = student_flat - start_flat
        
        loss_match = 1 - F.cosine_similarity(student_direction.unsqueeze(0), 
                                               target_direction.unsqueeze(0).detach())
        
        optimizer.zero_grad()
        loss_match.backward()
        optimizer.step()
        
        if (it + 1) % 1000 == 0:
            print(f"    Iter {it+1}/{n_iter}, Loss: {loss_match.item():.4f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def main():
    """Run all experiments."""
    # Check for resume
    results_file = '/workspace/results/results_final.json'
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print(f"Resuming with {len(results)} existing results")
    else:
        results = {}
    
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Move test data to GPU permanently
    test_imgs_gpu = test_images.to(DEVICE)
    test_labels_gpu = test_labels.to(DEVICE)
    
    # Load teacher for K-centers feature extraction
    teacher_data = torch.load('/workspace/teacher_final.pt', map_location='cpu')
    teacher = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32))
    teacher.load_state_dict(teacher_data['state_dict'])
    teacher = teacher.to(DEVICE)
    teacher.eval()
    print(f"Teacher accuracy: {teacher_data['accuracy']:.2f}%")
    
    # Load soft labels
    full_soft_labels = torch.load('/workspace/soft_labels_final.pt', map_location='cpu')
    
    configs = []
    for ipc in [10, 50]:
        for method in ['random', 'k_centers', 'dm', 'dc', 'tm']:
            for label_type in ['hard', 'soft']:
                key = f"{method}_ipc{ipc}_{label_type}"
                if key not in results:
                    configs.append((method, ipc, label_type, key))
    
    print(f"\n{len(configs)} experiments remaining out of 20 total")
    
    # Cache distilled datasets
    distilled_cache = {}
    
    for cfg_idx, (method, ipc, label_type, key) in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"[{cfg_idx+1}/{len(configs)}] {key}")
        print(f"{'='*60}")
        
        t0 = time.time()
        
        # Get or create the dataset
        cache_key = f"{method}_ipc{ipc}"
        
        if cache_key in distilled_cache:
            sub_images, sub_labels = distilled_cache[cache_key]
        elif method == 'random':
            selected = random_select(train_labels, ipc, seed=42)
            sub_images = train_images[selected]
            sub_labels = train_labels[selected]
            distilled_cache[cache_key] = (sub_images, sub_labels)
        elif method == 'k_centers':
            selected = k_centers_select(train_images, train_labels, ipc, teacher_model=teacher, seed=42)
            sub_images = train_images[selected]
            sub_labels = train_labels[selected]
            distilled_cache[cache_key] = (sub_images, sub_labels)
        elif method == 'dm':
            # Check if previously distilled
            dm_path = f'/workspace/distilled_v2_dm_ipc{ipc}.pt'
            if os.path.exists(dm_path):
                data = torch.load(dm_path, map_location='cpu')
                sub_images, sub_labels = data['images'], data['labels']
            else:
                n_iter = 15000 if ipc == 10 else 10000
                sub_images, sub_labels = distill_dm(train_images, train_labels, ipc, n_iter=n_iter)
                torch.save({'images': sub_images, 'labels': sub_labels}, dm_path)
            distilled_cache[cache_key] = (sub_images, sub_labels)
        elif method == 'dc':
            dc_path = f'/workspace/distilled_v2_dc_ipc{ipc}.pt'
            if os.path.exists(dc_path):
                data = torch.load(dc_path, map_location='cpu')
                sub_images, sub_labels = data['images'], data['labels']
            else:
                outer = 40 if ipc == 10 else 30
                inner = 40 if ipc == 10 else 30
                sub_images, sub_labels = distill_dc(train_images, train_labels, ipc, 
                                                     outer_loops=outer, inner_loops=inner)
                torch.save({'images': sub_images, 'labels': sub_labels}, dc_path)
            distilled_cache[cache_key] = (sub_images, sub_labels)
        elif method == 'tm':
            tm_path = f'/workspace/distilled_v2_tm_ipc{ipc}.pt'
            if os.path.exists(tm_path):
                data = torch.load(tm_path, map_location='cpu')
                sub_images, sub_labels = data['images'], data['labels']
            else:
                sub_images, sub_labels = distill_tm(train_images, train_labels, ipc,
                                                     n_iter=3000, n_experts=5, expert_epochs=40)
                torch.save({'images': sub_images, 'labels': sub_labels}, tm_path)
            distilled_cache[cache_key] = (sub_images, sub_labels)
        
        # Get soft labels for this subset
        if method in ['random', 'k_centers']:
            # These are real images - use pre-computed soft labels
            selected = [i for i in range(len(train_images)) if any(
                torch.equal(train_images[i], sub_images[j]) for j in range(min(3, len(sub_images)))
            )]
            # Actually, re-select to get indices
            if method == 'random':
                selected = random_select(train_labels, ipc, seed=42)
            else:
                selected = k_centers_select(train_images, train_labels, ipc, teacher_model=teacher, seed=42)
            sub_sl = full_soft_labels[selected]
        else:
            # For distilled images, generate soft labels from teacher
            teacher.eval()
            with torch.no_grad():
                sub_sl_list = []
                for i in range(0, len(sub_images), 256):
                    batch = sub_images[i:i+256].to(DEVICE)
                    logits = teacher(batch)
                    sub_sl_list.append(logits.cpu())
                sub_sl = torch.cat(sub_sl_list, dim=0)
        
        # Run trials
        sl_arg = sub_sl if label_type == 'soft' else None
        mean, std, accs = run_trials(sub_images, sub_labels, test_imgs_gpu, test_labels_gpu,
                                      label_type=label_type, soft_labels=sl_arg, num_trials=3)
        
        elapsed = time.time() - t0
        results[key] = {
            'method': method, 'ipc': ipc, 'label_type': label_type,
            'mean': mean, 'std': std, 'accs': accs, 'time': elapsed
        }
        
        # Save after each experiment
        os.makedirs('/workspace/results', exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved. Time: {elapsed:.0f}s")
    
    # Print final table
    print("\n\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    
    paper_values = {
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
    
    print(f"{'Method':<12} {'IPC':>4} {'Label':>6} {'Ours':>12} {'Paper':>8} {'Diff':>8}")
    print("-" * 55)
    
    for ipc in [10, 50]:
        for method in ['random', 'k_centers', 'dc', 'dm', 'tm']:
            for label_type in ['hard', 'soft']:
                key = f"{method}_ipc{ipc}_{label_type}"
                if key in results:
                    r = results[key]
                    paper_val = paper_values.get(key, 0)
                    lt = 'HL' if label_type == 'hard' else 'SL'
                    diff = r['mean'] - paper_val
                    print(f"{method:<12} {ipc:>4} {lt:>6} {r['mean']:>6.2f}±{r['std']:.2f} {paper_val:>8.2f} {diff:>+8.2f}")
        print()
    
    # Save table to file
    with open('/workspace/results/table_final.txt', 'w') as f:
        f.write(f"{'Method':<12} {'IPC':>4} {'Label':>6} {'Ours':>12} {'Paper':>8} {'Diff':>8}\n")
        f.write("-" * 55 + "\n")
        for ipc in [10, 50]:
            for method in ['random', 'k_centers', 'dc', 'dm', 'tm']:
                for label_type in ['hard', 'soft']:
                    key = f"{method}_ipc{ipc}_{label_type}"
                    if key in results:
                        r = results[key]
                        paper_val = paper_values.get(key, 0)
                        lt = 'HL' if label_type == 'hard' else 'SL'
                        diff = r['mean'] - paper_val
                        f.write(f"{method:<12} {ipc:>4} {lt:>6} {r['mean']:>6.2f}±{r['std']:.2f} {paper_val:>8.2f} {diff:>+8.2f}\n")
            f.write("\n")
    
    print("\nResults saved to /workspace/results/results_final.json")
    print("Table saved to /workspace/results/table_final.txt")


if __name__ == '__main__':
    main()
