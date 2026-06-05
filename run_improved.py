"""
Improved experiment runner with better distillation parameters and multiple runs.
Focuses on getting closer to paper results.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from convnet import ConvNet
from data_utils import get_cifar100_tensors, get_class_indices, random_select, k_centers_select
from train_eval import train_and_evaluate
from dsa import DiffAugment

# Load data once
print("Loading CIFAR-100...")
train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
model_fn = lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))

# Results storage
results_path = 'results/results_v2.json'
os.makedirs('results', exist_ok=True)
if os.path.exists(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
else:
    all_results = {}

def save_results():
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

NUM_EVAL_RUNS = 3

def evaluate_config(sub_images, sub_labels, label_type, sub_soft=None, num_runs=NUM_EVAL_RUNS):
    """Evaluate a single config and return (mean, std, runs)."""
    accs = []
    for run in range(num_runs):
        acc = train_and_evaluate(
            sub_images, sub_labels, test_images, test_labels,
            model_fn, num_classes=100, device=device,
            label_type=label_type, soft_labels=sub_soft,
            epochs=300, batch_size=256, seed=run, verbose=False
        )
        accs.append(acc)
        print(f"    Run {run+1}: {acc:.2f}%")
    return np.mean(accs), np.std(accs), accs

def train_teacher(epochs=300, seed=42):
    """Train a strong teacher model on full CIFAR-100 with DSA augmentation."""
    cache_path = 'teacher_model_v2.pt'
    if os.path.exists(cache_path):
        model = model_fn().to(device)
        model.load_state_dict(torch.load(cache_path, weights_only=True))
        model.eval()
        return model
    
    print("Training teacher model (300 epochs with DSA)...")
    torch.manual_seed(seed)
    model = model_fn().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    n_train = len(train_images)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 256):
            idx = perm[i:i+256]
            batch_imgs = train_images[idx].to(device)
            batch_labels = train_labels[idx].to(device)
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            from train_eval import evaluate
            acc = evaluate(model, test_images, test_labels, device)
            print(f"  Teacher epoch {epoch+1}: {acc:.2f}%")
    
    torch.save(model.state_dict(), cache_path)
    model.eval()
    return model

def generate_soft_labels_for_data(images, teacher_model):
    """Generate soft labels using teacher model."""
    teacher_model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            logits = teacher_model(batch)
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)

def run_and_save(method, ipc, label_type, sub_images, sub_labels, sub_soft=None, num_runs=NUM_EVAL_RUNS):
    """Run eval and save results."""
    key = f"{method}_ipc{ipc}_{label_type}"
    if key in all_results:
        print(f"  SKIP {key} (already done): {all_results[key]['mean']:.2f}%")
        return all_results[key]['mean']
    
    print(f"  Evaluating {key}...")
    start = time.time()
    mean, std, runs = evaluate_config(sub_images, sub_labels, label_type, sub_soft, num_runs)
    elapsed = time.time() - start
    
    all_results[key] = {
        'mean': float(mean), 'std': float(std), 'runs': [float(r) for r in runs],
        'method': method, 'ipc': ipc, 'label_type': label_type, 'time': elapsed
    }
    save_results()
    print(f"  {key}: {mean:.2f} ± {std:.2f}% ({elapsed:.0f}s)")
    return mean


# =====================================================
# STEP 0: Train teacher model
# =====================================================
print("\n" + "="*60)
print("STEP 0: Training Teacher Model")
print("="*60)
teacher = train_teacher(epochs=300, seed=42)

# Generate soft labels for full training set
soft_labels_path = 'soft_labels_v2.pt'
if os.path.exists(soft_labels_path):
    soft_labels_all = torch.load(soft_labels_path, weights_only=True)
else:
    print("Generating soft labels for full training set...")
    soft_labels_all = generate_soft_labels_for_data(train_images, teacher)
    torch.save(soft_labels_all, soft_labels_path)
print(f"Soft labels: {soft_labels_all.shape}")

# =====================================================
# STEP 1: Coreset methods (Random, K-centers)
# =====================================================
print("\n" + "="*60)
print("STEP 1: Coreset Methods")
print("="*60)

for ipc in [10, 50]:
    # Random
    print(f"\nRandom IPC={ipc}:")
    selected = random_select(train_labels, ipc=ipc, seed=0)
    sub_images = train_images[selected]
    sub_labels = train_labels[selected]
    sub_soft = soft_labels_all[selected]
    
    run_and_save('random', ipc, 'hard', sub_images, sub_labels)
    run_and_save('random', ipc, 'soft', sub_images, sub_labels, sub_soft)
    
    # K-centers
    print(f"\nK-centers IPC={ipc}:")
    selected = k_centers_select(train_images, train_labels, ipc=ipc, seed=0)
    sub_images = train_images[selected]
    sub_labels = train_labels[selected]
    sub_soft = soft_labels_all[selected]
    
    run_and_save('k_centers', ipc, 'hard', sub_images, sub_labels)
    run_and_save('k_centers', ipc, 'soft', sub_images, sub_labels, sub_soft)

torch.cuda.empty_cache()
gc.collect()

# =====================================================
# STEP 2: DM Distillation (improved)
# =====================================================
print("\n" + "="*60)
print("STEP 2: Distribution Matching (DM) - Improved")
print("="*60)

from distill_dm import distribution_matching

for ipc in [10, 50]:
    cache_path = f'distilled_dm_ipc{ipc}_v2.pt'
    if os.path.exists(cache_path):
        print(f"DM IPC={ipc}: loaded from cache")
        data = torch.load(cache_path, weights_only=True)
    else:
        print(f"DM IPC={ipc}: distilling (5000 iterations)...")
        start = time.time()
        syn_img, syn_lbl = distribution_matching(
            train_images, train_labels,
            num_classes=100, ipc=ipc, device=device,
            iterations=5000, lr_img=1.0, batch_real=256
        )
        data = {'images': syn_img, 'labels': syn_lbl}
        torch.save(data, cache_path)
        print(f"  Distilled in {time.time()-start:.0f}s")
    
    # Generate soft labels for synthetic data
    soft_cache = f'soft_labels_dm_ipc{ipc}_v2.pt'
    if os.path.exists(soft_cache):
        syn_soft = torch.load(soft_cache, weights_only=True)
    else:
        print(f"  Generating soft labels for DM IPC={ipc}...")
        syn_soft = generate_soft_labels_for_data(data['images'], teacher)
        torch.save(syn_soft, soft_cache)
    
    run_and_save('dm', ipc, 'hard', data['images'], data['labels'])
    run_and_save('dm', ipc, 'soft', data['images'], data['labels'], syn_soft)
    
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================
# STEP 3: DC Distillation (improved)
# =====================================================
print("\n" + "="*60)
print("STEP 3: Dataset Condensation (DC) - Improved")
print("="*60)

from distill_dc import gradient_matching

for ipc in [10, 50]:
    cache_path = f'distilled_dc_ipc{ipc}_v2.pt'
    if os.path.exists(cache_path):
        print(f"DC IPC={ipc}: loaded from cache")
        data = torch.load(cache_path, weights_only=True)
    else:
        print(f"DC IPC={ipc}: distilling (50 outer × 50 inner)...")
        start = time.time()
        syn_img, syn_lbl = gradient_matching(
            train_images, train_labels,
            num_classes=100, ipc=ipc, device=device,
            outer_loops=50, inner_loops=50, lr_img=1.0, batch_real=256
        )
        data = {'images': syn_img, 'labels': syn_lbl}
        torch.save(data, cache_path)
        print(f"  Distilled in {time.time()-start:.0f}s")
    
    # Generate soft labels
    soft_cache = f'soft_labels_dc_ipc{ipc}_v2.pt'
    if os.path.exists(soft_cache):
        syn_soft = torch.load(soft_cache, weights_only=True)
    else:
        print(f"  Generating soft labels for DC IPC={ipc}...")
        syn_soft = generate_soft_labels_for_data(data['images'], teacher)
        torch.save(syn_soft, soft_cache)
    
    run_and_save('dc', ipc, 'hard', data['images'], data['labels'])
    run_and_save('dc', ipc, 'soft', data['images'], data['labels'], syn_soft)
    
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================
# STEP 4: TM Distillation (improved)
# =====================================================
print("\n" + "="*60)
print("STEP 4: Trajectory Matching (TM) - Improved")
print("="*60)

from distill_tm import train_expert_trajectories, trajectory_matching

# Train more experts with more epochs
expert_dir = 'expert_trajectories_v2'
if not os.path.exists(expert_dir) or len(os.listdir(expert_dir)) < 3:
    print("Training expert trajectories (10 experts, 50 epochs)...")
    start = time.time()
    train_expert_trajectories(
        train_images, train_labels,
        num_classes=100, device=device,
        num_experts=10, expert_epochs=50,
        save_dir=expert_dir
    )
    print(f"  Experts trained in {time.time()-start:.0f}s")
else:
    print(f"Expert trajectories loaded from {expert_dir}")

for ipc in [10, 50]:
    cache_path = f'distilled_tm_ipc{ipc}_v2.pt'
    if os.path.exists(cache_path):
        print(f"TM IPC={ipc}: loaded from cache")
        data = torch.load(cache_path, weights_only=True)
    else:
        print(f"TM IPC={ipc}: distilling (5000 iterations)...")
        start = time.time()
        syn_img, syn_lbl = trajectory_matching(
            train_images, train_labels,
            num_classes=100, ipc=ipc, device=device,
            expert_dir=expert_dir,
            num_experts=10, iterations=5000, lr_img=1000.0,
            syn_steps=30, expert_epochs=3, max_start_epoch=25
        )
        data = {'images': syn_img, 'labels': syn_lbl}
        torch.save(data, cache_path)
        print(f"  Distilled in {time.time()-start:.0f}s")
    
    # Generate soft labels
    soft_cache = f'soft_labels_tm_ipc{ipc}_v2.pt'
    if os.path.exists(soft_cache):
        syn_soft = torch.load(soft_cache, weights_only=True)
    else:
        print(f"  Generating soft labels for TM IPC={ipc}...")
        syn_soft = generate_soft_labels_for_data(data['images'], teacher)
        torch.save(syn_soft, soft_cache)
    
    run_and_save('tm', ipc, 'hard', data['images'], data['labels'])
    run_and_save('tm', ipc, 'soft', data['images'], data['labels'], syn_soft)
    
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================
# Print summary
# =====================================================
print("\n" + "="*60)
print("RESULTS SUMMARY (v2)")
print("="*60)

# Paper reference values
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

print(f"{'Method':<12} {'IPC':<6} {'Label':<6} {'Ours':<10} {'Paper':<10} {'Diff':<10}")
print("-" * 55)
for method in ['dm', 'dc', 'tm', 'random', 'k_centers']:
    for ipc in [10, 50]:
        for lt in ['hard', 'soft']:
            key = f"{method}_ipc{ipc}_{lt}"
            ours = all_results.get(key, {}).get('mean', 0)
            paper = paper_values.get(key, 0)
            diff = ours - paper
            print(f"{method:<12} {ipc:<6} {lt:<6} {ours:<10.2f} {paper:<10.2f} {diff:+.2f}")

save_results()
print(f"\nAll results saved to {results_path}")
