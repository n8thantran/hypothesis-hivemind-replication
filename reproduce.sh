#!/bin/bash
# reproduce.sh - Reproduce key results from
# "Rethinking Dataset Distillation: Hard Truths About Soft Labels"
#
# This script runs all 20 experiment configurations:
# - 3 DD methods (DM, DC, TM) x 2 IPC (10, 50) x 2 label types (HL, SL) = 12
# - 2 coreset methods (Random, K-Centers) x 2 IPC x 2 label types = 8
# Total: 20 configs
#
# Expected runtime: ~60-90 minutes on a single GPU
# Results are saved to results/results.json, results/table1.txt, results/analysis.txt

set -e

echo "============================================"
echo "Reproducing: Rethinking Dataset Distillation"
echo "============================================"

# Install dependencies
pip install datasets > /dev/null 2>&1

# Create results directory
mkdir -p results

# Step 1: Download and prepare CIFAR-100
echo ""
echo "Step 1: Preparing CIFAR-100 data..."
python -c "
from data_utils import get_cifar100_tensors
train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
print(f'Train: {train_images.shape}, Test: {test_images.shape}')
"

# Step 2: Train teacher model and generate soft labels for full dataset
echo ""
echo "Step 2: Training teacher model for soft labels..."
python -c "
import torch
import torch.nn as nn
from convnet import ConvNet
from data_utils import get_cifar100_tensors

train_images, train_labels, _, _ = get_cifar100_tensors()

model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
dataset = torch.utils.data.TensorDataset(train_images, train_labels)
loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

model.train()
for epoch in range(100):
    for batch_imgs, batch_labels in loader:
        batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda()
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(batch_imgs), batch_labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
    if (epoch + 1) % 25 == 0:
        print(f'  Teacher epoch {epoch+1}/100')

model.eval()
all_logits = []
with torch.no_grad():
    for i in range(0, len(train_images), 256):
        batch = train_images[i:i+256].cuda()
        all_logits.append(model(batch).cpu())
soft_labels_full = torch.cat(all_logits)
torch.save(soft_labels_full, 'soft_labels.pt')
torch.save(model.state_dict(), 'teacher_model.pt')
print(f'  Teacher soft labels saved: {soft_labels_full.shape}')
"

# Step 3: Distill datasets (DM, DC, TM)
echo ""
echo "Step 3: Distilling datasets..."

# DM IPC=10
echo "  DM IPC=10..."
python -c "
import torch
from data_utils import get_cifar100_tensors
from distill_dm import distribution_matching
train_images, train_labels, _, _ = get_cifar100_tensors()
syn_img, syn_lbl = distribution_matching(train_images, train_labels, num_classes=100, ipc=10, device='cuda', iterations=1000, batch_real=64, lr_img=1.0)
torch.save({'images': syn_img, 'labels': syn_lbl}, 'distilled_dm_ipc10.pt')
print(f'  Saved: {syn_img.shape}')
"

# DM IPC=50
echo "  DM IPC=50..."
python -c "
import torch
from data_utils import get_cifar100_tensors
from distill_dm import distribution_matching
train_images, train_labels, _, _ = get_cifar100_tensors()
syn_img, syn_lbl = distribution_matching(train_images, train_labels, num_classes=100, ipc=50, device='cuda', iterations=1000, batch_real=64, lr_img=1.0)
torch.save({'images': syn_img, 'labels': syn_lbl}, 'distilled_dm_ipc50.pt')
print(f'  Saved: {syn_img.shape}')
"

# DC IPC=10
echo "  DC IPC=10..."
python -c "
import torch
from data_utils import get_cifar100_tensors
from distill_dc import gradient_matching
train_images, train_labels, _, _ = get_cifar100_tensors()
syn_img, syn_lbl = gradient_matching(train_images, train_labels, num_classes=100, ipc=10, device='cuda', outer_loops=5, inner_loops=10, lr_img=1.0)
torch.save({'images': syn_img, 'labels': syn_lbl}, 'distilled_dc_ipc10.pt')
print(f'  Saved: {syn_img.shape}')
"

# DC IPC=50
echo "  DC IPC=50..."
python -c "
import torch
from data_utils import get_cifar100_tensors
from distill_dc import gradient_matching
train_images, train_labels, _, _ = get_cifar100_tensors()
syn_img, syn_lbl = gradient_matching(train_images, train_labels, num_classes=100, ipc=50, device='cuda', outer_loops=5, inner_loops=10, lr_img=1.0)
torch.save({'images': syn_img, 'labels': syn_lbl}, 'distilled_dc_ipc50.pt')
print(f'  Saved: {syn_img.shape}')
"

# TM: Train expert trajectories first
echo "  Training expert trajectories for TM..."
python -c "
import torch
import torch.nn as nn
from convnet import ConvNet
from data_utils import get_cifar100_tensors

train_images, train_labels, _, _ = get_cifar100_tensors()
import os; os.makedirs('expert_trajectories', exist_ok=True)

for exp_idx in range(3):
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    trajectory = [model.state_dict()]
    for epoch in range(15):
        model.train()
        for batch_imgs, batch_labels in loader:
            batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda()
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(batch_imgs), batch_labels)
            loss.backward()
            optimizer.step()
        trajectory.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
    
    torch.save(trajectory, f'expert_trajectories/expert_{exp_idx}.pt')
    print(f'  Expert {exp_idx} saved ({len(trajectory)} checkpoints)')
"

# TM IPC=10
echo "  TM IPC=10..."
python -c "
import torch
from data_utils import get_cifar100_tensors
from distill_tm import trajectory_matching
train_images, train_labels, _, _ = get_cifar100_tensors()
syn_img, syn_lbl = trajectory_matching(train_images, train_labels, num_classes=100, ipc=10, device='cuda', expert_dir='expert_trajectories', iterations=500, lr_img=1000.0, syn_steps=10)
torch.save({'images': syn_img, 'labels': syn_lbl}, 'distilled_tm_ipc10.pt')
print(f'  Saved: {syn_img.shape}')
"

# TM IPC=50
echo "  TM IPC=50..."
python -c "
import torch
from data_utils import get_cifar100_tensors
from distill_tm import trajectory_matching
train_images, train_labels, _, _ = get_cifar100_tensors()
syn_img, syn_lbl = trajectory_matching(train_images, train_labels, num_classes=100, ipc=50, device='cuda', expert_dir='expert_trajectories', iterations=100, lr_img=1000.0, syn_steps=10)
torch.save({'images': syn_img, 'labels': syn_lbl}, 'distilled_tm_ipc50.pt')
print(f'  Saved: {syn_img.shape}')
"

# Step 4: Generate soft labels for distilled datasets
echo ""
echo "Step 4: Generating soft labels for distilled datasets..."
python -c "
import torch
import torch.nn as nn
from convnet import ConvNet

model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).cuda()
model.load_state_dict(torch.load('teacher_model.pt', weights_only=True))
model.eval()

for method in ['dm', 'dc', 'tm']:
    for ipc in [10, 50]:
        data = torch.load(f'distilled_{method}_ipc{ipc}.pt', weights_only=True)
        with torch.no_grad():
            all_logits = []
            for i in range(0, len(data['images']), 256):
                batch = data['images'][i:i+256].cuda()
                all_logits.append(model(batch).cpu())
            logits = torch.cat(all_logits)
        torch.save(logits, f'soft_labels_{method}_ipc{ipc}.pt')
        print(f'  {method} IPC={ipc}: {logits.shape}')
"

# Step 5: Evaluate all configurations
echo ""
echo "Step 5: Evaluating all 20 configurations..."
python -c "
import torch, json
from convnet import ConvNet
from data_utils import get_cifar100_tensors
from train_eval import train_and_evaluate

_, _, test_images, test_labels = get_cifar100_tensors()
model_fn = lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))
results = {}

# DD methods
for method in ['dm', 'dc', 'tm']:
    for ipc in [10, 50]:
        data = torch.load(f'distilled_{method}_ipc{ipc}.pt', weights_only=True)
        syn_soft = torch.load(f'soft_labels_{method}_ipc{ipc}.pt', weights_only=True)
        
        # Hard labels
        acc = train_and_evaluate(data['images'], data['labels'], test_images, test_labels,
            model_fn, num_classes=100, device='cuda', label_type='hard', epochs=300, batch_size=256, seed=0, verbose=False)
        key = f'{method}_ipc{ipc}_hard'
        results[key] = {'mean': acc, 'std': 0.0, 'runs': [acc], 'method': method, 'ipc': ipc, 'label_type': 'hard'}
        print(f'  {key}: {acc:.2f}%')
        
        # Soft labels
        acc = train_and_evaluate(data['images'], data['labels'], test_images, test_labels,
            model_fn, num_classes=100, device='cuda', label_type='soft', soft_labels=syn_soft,
            epochs=300, batch_size=256, seed=0, verbose=False)
        key = f'{method}_ipc{ipc}_soft'
        results[key] = {'mean': acc, 'std': 0.0, 'runs': [acc], 'method': method, 'ipc': ipc, 'label_type': 'soft'}
        print(f'  {key}: {acc:.2f}%')

# Coreset methods
train_images, train_labels, _, _ = get_cifar100_tensors()
soft_labels_full = torch.load('soft_labels.pt', weights_only=True)

for method in ['random', 'k_centers']:
    for ipc in [10, 50]:
        # Select coreset
        selected_indices = []
        for c in range(100):
            class_idx = (train_labels == c).nonzero(as_tuple=True)[0]
            if method == 'random':
                torch.manual_seed(42)
                perm = torch.randperm(len(class_idx))[:ipc]
                selected_indices.append(class_idx[perm])
            else:  # k_centers
                class_imgs = train_images[class_idx].view(len(class_idx), -1)
                centers = []
                # First center: closest to mean
                mean = class_imgs.mean(0)
                dists = torch.cdist(class_imgs, mean.unsqueeze(0)).squeeze()
                centers.append(dists.argmin().item())
                for _ in range(ipc - 1):
                    center_feats = class_imgs[centers]
                    dists = torch.cdist(class_imgs, center_feats).min(dim=1).values
                    new_center = dists.argmax().item()
                    centers.append(new_center)
                selected_indices.append(class_idx[torch.tensor(centers)])
        
        selected_indices = torch.cat(selected_indices)
        syn_images = train_images[selected_indices]
        syn_labels = train_labels[selected_indices]
        syn_soft = soft_labels_full[selected_indices]
        
        # Hard labels (3 runs)
        accs = []
        for seed in range(3):
            acc = train_and_evaluate(syn_images, syn_labels, test_images, test_labels,
                model_fn, num_classes=100, device='cuda', label_type='hard', epochs=300, batch_size=256, seed=seed, verbose=False)
            accs.append(acc)
        key = f'{method}_ipc{ipc}_hard'
        results[key] = {'mean': sum(accs)/len(accs), 'std': (sum((a-sum(accs)/len(accs))**2 for a in accs)/len(accs))**0.5, 'runs': accs, 'method': method, 'ipc': ipc, 'label_type': 'hard'}
        print(f'  {key}: {results[key][\"mean\"]:.2f}% ± {results[key][\"std\"]:.2f}')
        
        # Soft labels
        acc = train_and_evaluate(syn_images, syn_labels, test_images, test_labels,
            model_fn, num_classes=100, device='cuda', label_type='soft', soft_labels=syn_soft,
            epochs=300, batch_size=256, seed=0, verbose=False)
        key = f'{method}_ipc{ipc}_soft'
        results[key] = {'mean': acc, 'std': 0.0, 'runs': [acc], 'method': method, 'ipc': ipc, 'label_type': 'soft'}
        print(f'  {key}: {acc:.2f}%')

with open('results/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\\nAll results saved to results/results.json')
"

# Step 6: Generate tables and analysis
echo ""
echo "Step 6: Generating results tables..."
python generate_results.py

echo ""
echo "============================================"
echo "DONE! Results saved to results/"
echo "  - results/results.json"
echo "  - results/table1.txt"
echo "  - results/analysis.txt"
echo "============================================"
