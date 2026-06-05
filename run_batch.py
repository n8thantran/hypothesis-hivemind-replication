"""
Batch runner: distill all DD methods and evaluate all configs.
Designed to run in one go with minimal overhead.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import sys
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from convnet import ConvNet
from data_utils import get_cifar100_tensors, get_class_indices
from train_eval import train_and_evaluate
from dsa import DiffAugment

# Load data once
print("Loading CIFAR-100...")
train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
model_fn = lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))

# Load soft labels
soft_labels_all = torch.load('soft_labels.pt', weights_only=True)
print(f"Soft labels: {soft_labels_all.shape}")

# Load existing results
results_path = 'results/results.json'
os.makedirs('results', exist_ok=True)
if os.path.exists(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
else:
    all_results = {}

def save_results():
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

def evaluate_config(sub_images, sub_labels, label_type, sub_soft=None, num_runs=1):
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

def generate_soft_for_synthetic(syn_images, syn_labels):
    """Generate soft labels for synthetic images using a teacher trained on full data."""
    model = model_fn().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    model.train()
    for epoch in range(100):
        for batch_imgs, batch_labels in loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = nn.CrossEntropyLoss()(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        all_logits = []
        for i in range(0, len(syn_images), 256):
            batch = syn_images[i:i+256].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    return torch.cat(all_logits, dim=0)


def run_and_save(method, ipc, label_type, sub_images, sub_labels, sub_soft=None, num_runs=1):
    """Run eval and save results."""
    key = f"{method}_ipc{ipc}_{label_type}"
    if key in all_results:
        print(f"  SKIP {key} (already done)")
        return
    
    print(f"  Evaluating {key}...")
    start = time.time()
    mean, std, runs = evaluate_config(sub_images, sub_labels, label_type, sub_soft, num_runs)
    elapsed = time.time() - start
    
    all_results[key] = {
        'mean': mean, 'std': std, 'runs': runs,
        'method': method, 'ipc': ipc, 'label_type': label_type, 'time': elapsed
    }
    save_results()
    print(f"  {key}: {mean:.2f} ± {std:.2f}% ({elapsed:.0f}s)")
    return mean


# =====================================================
# STEP 1: DM Distillation
# =====================================================
print("\n" + "="*60)
print("STEP 1: Distribution Matching (DM)")
print("="*60)

from distill_dm import distribution_matching

for ipc in [10, 50]:
    cache_path = f'distilled_dm_ipc{ipc}.pt'
    if os.path.exists(cache_path):
        print(f"DM IPC={ipc}: loaded from cache")
        data = torch.load(cache_path, weights_only=True)
    else:
        print(f"DM IPC={ipc}: distilling...")
        start = time.time()
        syn_img, syn_lbl = distribution_matching(
            train_images, train_labels,
            num_classes=100, ipc=ipc, device=device,
            iterations=1000, lr_img=1.0, batch_real=64
        )
        data = {'images': syn_img, 'labels': syn_lbl}
        torch.save(data, cache_path)
        print(f"  Distilled in {time.time()-start:.0f}s")
    
    # Evaluate HL
    run_and_save('dm', ipc, 'hard', data['images'], data['labels'], num_runs=1)
    
    # Evaluate SL - need soft labels for synthetic data
    key_sl = f"dm_ipc{ipc}_soft"
    if key_sl not in all_results:
        soft_cache = f'soft_labels_dm_ipc{ipc}.pt'
        if os.path.exists(soft_cache):
            syn_soft = torch.load(soft_cache, weights_only=True)
        else:
            print(f"  Generating soft labels for DM IPC={ipc}...")
            syn_soft = generate_soft_for_synthetic(data['images'], data['labels'])
            torch.save(syn_soft, soft_cache)
        run_and_save('dm', ipc, 'soft', data['images'], data['labels'], syn_soft, num_runs=1)
    
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================
# STEP 2: DC Distillation  
# =====================================================
print("\n" + "="*60)
print("STEP 2: Dataset Condensation (DC)")
print("="*60)

from distill_dc import gradient_matching

for ipc in [10, 50]:
    cache_path = f'distilled_dc_ipc{ipc}.pt'
    if os.path.exists(cache_path):
        print(f"DC IPC={ipc}: loaded from cache")
        data = torch.load(cache_path, weights_only=True)
    else:
        print(f"DC IPC={ipc}: distilling...")
        start = time.time()
        syn_img, syn_lbl = gradient_matching(
            train_images, train_labels,
            num_classes=100, ipc=ipc, device=device,
            outer_loops=5, inner_loops=10, lr_img=1.0
        )
        data = {'images': syn_img, 'labels': syn_lbl}
        torch.save(data, cache_path)
        print(f"  Distilled in {time.time()-start:.0f}s")
    
    # Evaluate HL
    run_and_save('dc', ipc, 'hard', data['images'], data['labels'], num_runs=1)
    
    # Evaluate SL
    key_sl = f"dc_ipc{ipc}_soft"
    if key_sl not in all_results:
        soft_cache = f'soft_labels_dc_ipc{ipc}.pt'
        if os.path.exists(soft_cache):
            syn_soft = torch.load(soft_cache, weights_only=True)
        else:
            print(f"  Generating soft labels for DC IPC={ipc}...")
            syn_soft = generate_soft_for_synthetic(data['images'], data['labels'])
            torch.save(syn_soft, soft_cache)
        run_and_save('dc', ipc, 'soft', data['images'], data['labels'], syn_soft, num_runs=1)
    
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================
# STEP 3: TM Distillation
# =====================================================
print("\n" + "="*60)
print("STEP 3: Trajectory Matching (TM)")
print("="*60)

from distill_tm import train_expert_trajectories, trajectory_matching

# Train experts once
expert_dir = 'expert_trajectories'
if not os.path.exists(expert_dir) or len(os.listdir(expert_dir)) == 0:
    print("Training expert trajectories...")
    start = time.time()
    expert_dir = train_expert_trajectories(
        train_images, train_labels,
        num_classes=100, device=device,
        num_experts=5, expert_epochs=20,
        save_dir=expert_dir
    )
    print(f"  Experts trained in {time.time()-start:.0f}s")
else:
    print(f"Expert trajectories loaded from {expert_dir}")

for ipc in [10, 50]:
    cache_path = f'distilled_tm_ipc{ipc}.pt'
    if os.path.exists(cache_path):
        print(f"TM IPC={ipc}: loaded from cache")
        data = torch.load(cache_path, weights_only=True)
    else:
        print(f"TM IPC={ipc}: distilling...")
        start = time.time()
        syn_img, syn_lbl = trajectory_matching(
            train_images, train_labels,
            num_classes=100, ipc=ipc, device=device,
            expert_dir=expert_dir,
            match_iterations=1000, lr_img=0.1
        )
        data = {'images': syn_img, 'labels': syn_lbl}
        torch.save(data, cache_path)
        print(f"  Distilled in {time.time()-start:.0f}s")
    
    # Evaluate HL
    run_and_save('tm', ipc, 'hard', data['images'], data['labels'], num_runs=1)
    
    # Evaluate SL
    key_sl = f"tm_ipc{ipc}_soft"
    if key_sl not in all_results:
        soft_cache = f'soft_labels_tm_ipc{ipc}.pt'
        if os.path.exists(soft_cache):
            syn_soft = torch.load(soft_cache, weights_only=True)
        else:
            print(f"  Generating soft labels for TM IPC={ipc}...")
            syn_soft = generate_soft_for_synthetic(data['images'], data['labels'])
            torch.save(syn_soft, soft_cache)
        run_and_save('tm', ipc, 'soft', data['images'], data['labels'], syn_soft, num_runs=1)
    
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================
# Print summary
# =====================================================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Method':<12} {'IPC':<6} {'HL':<15} {'SL':<15}")
print("-" * 50)
for method in ['random', 'k_centers', 'dm', 'dc', 'tm']:
    for ipc in [10, 50]:
        hl_key = f"{method}_ipc{ipc}_hard"
        sl_key = f"{method}_ipc{ipc}_soft"
        hl = f"{all_results[hl_key]['mean']:.2f}" if hl_key in all_results else "N/A"
        sl = f"{all_results[sl_key]['mean']:.2f}" if sl_key in all_results else "N/A"
        print(f"{method:<12} {ipc:<6} {hl:<15} {sl:<15}")

save_results()
print(f"\nAll results saved to {results_path}")
