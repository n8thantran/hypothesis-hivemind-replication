"""
Run a single experiment (one method, one IPC, HL+SL).
Usage: python run_single.py <method> <ipc> [seed]
Methods: dm, dc, tm, random, kcenters
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

DEVICE = 'cuda'
NUM_CLASSES = 100
DSA_STRATEGY = 'color_crop_cutout_flip_scale_rotate'


def load_cifar100():
    data = torch.load('cifar100_tensors.pt', map_location='cpu')
    return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']


def evaluate_test(model, test_images, test_labels):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for j in range(0, len(test_images), 512):
            imgs = test_images[j:j+512].to(DEVICE)
            labels = test_labels[j:j+512].to(DEVICE)
            pred = model(imgs).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_hl(images, labels, test_images, test_labels, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConvNet(num_classes=100).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
    n = len(images)
    for epoch in range(300):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            imgs = images[idx].to(DEVICE)
            labs = labels[idx].to(DEVICE)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labs)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return evaluate_test(model, test_images, test_labels)


def train_sl(images, soft_labels, test_images, test_labels, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConvNet(num_classes=100).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    n = len(images)
    T = 20.0
    for epoch in range(300):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            imgs = images[idx].to(DEVICE)
            soft = soft_labels[idx].to(DEVICE)
            imgs = DiffAugment(imgs, strategy=DSA_STRATEGY)
            optimizer.zero_grad()
            log_p = F.log_softmax(model(imgs) / T, dim=1)
            target_p = F.softmax(soft / T, dim=1)
            loss = F.kl_div(log_p, target_p, reduction='batchmean') * (T ** 2)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return evaluate_test(model, test_images, test_labels)


def get_random_subset(train_images, train_labels, ipc, seed=0):
    np.random.seed(seed)
    indices = []
    for c in range(100):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0].numpy()
        sel = np.random.choice(cls_idx, ipc, replace=False)
        indices.extend(sel.tolist())
    return torch.tensor(indices)


def get_kcenters_subset(train_images, train_labels, ipc, seed=0):
    torch.manual_seed(seed)
    model = ConvNet(num_classes=100).to(DEVICE)
    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(DEVICE)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    all_features = torch.cat(all_features, dim=0)
    
    indices = []
    for c in range(100):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        cls_feat = all_features[cls_idx]
        selected = []
        mean_feat = cls_feat.mean(0)
        dists = torch.cdist(cls_feat.unsqueeze(0), mean_feat.unsqueeze(0).unsqueeze(0)).squeeze()
        first = dists.argmin().item()
        selected.append(first)
        
        for _ in range(ipc - 1):
            if len(selected) == 1:
                dists_to_selected = torch.cdist(cls_feat.unsqueeze(0), cls_feat[selected].unsqueeze(0)).squeeze()
            else:
                dists_to_selected = torch.cdist(cls_feat.unsqueeze(0), cls_feat[selected].unsqueeze(0)).squeeze(0)
                dists_to_selected = dists_to_selected.min(dim=1)[0]
            for s in selected:
                dists_to_selected[s] = -1
            next_idx = dists_to_selected.argmax().item()
            selected.append(next_idx)
        
        for s in selected:
            indices.append(cls_idx[s].item())
    return torch.tensor(indices)


def main():
    method = sys.argv[1]
    ipc = int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    print(f"Running {method} IPC={ipc} seed={seed}")
    train_images, train_labels, test_images, test_labels = load_cifar100()
    full_soft_labels = torch.load('soft_labels.pt', map_location='cpu')
    
    if method in ['dm', 'dc', 'tm']:
        dd_data = torch.load(f'distilled_{method}_ipc{ipc}.pt', map_location='cpu')
        images = dd_data['images']
        labels = dd_data['labels']
        soft_labels = torch.load(f'soft_labels_{method}_ipc{ipc}_correct.pt', map_location='cpu')
    elif method == 'random':
        indices = get_random_subset(train_images, train_labels, ipc, seed=0)
        images = train_images[indices]
        labels = train_labels[indices]
        soft_labels = full_soft_labels[indices]
    elif method == 'kcenters':
        indices = get_kcenters_subset(train_images, train_labels, ipc, seed=0)
        images = train_images[indices]
        labels = train_labels[indices]
        soft_labels = full_soft_labels[indices]
    
    print(f"  Dataset: {images.shape[0]} images")
    
    t0 = time.time()
    hl_acc = train_hl(images, labels, test_images, test_labels, seed=seed)
    print(f"  HL: {hl_acc:.2f}% ({time.time()-t0:.0f}s)")
    
    t0 = time.time()
    sl_acc = train_sl(images, soft_labels, test_images, test_labels, seed=seed)
    print(f"  SL: {sl_acc:.2f}% ({time.time()-t0:.0f}s)")
    
    # Save result
    os.makedirs('results', exist_ok=True)
    result_file = f'results/result_{method}_ipc{ipc}_seed{seed}.json'
    with open(result_file, 'w') as f:
        json.dump({'method': method, 'ipc': ipc, 'seed': seed, 
                   'hl': hl_acc, 'sl': sl_acc}, f)
    print(f"  Saved to {result_file}")


if __name__ == '__main__':
    main()
