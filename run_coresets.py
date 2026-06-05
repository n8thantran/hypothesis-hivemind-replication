"""Run coreset experiments (Random + K-centers) with proper evaluation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from convnet import ConvNet
from data_utils import get_cifar100_tensors, random_select
from dsa import DiffAugment
from train_eval import train_and_evaluate

device = 'cuda'

def kmeans_select(train_images, train_labels, ipc, device='cuda'):
    """Select samples using K-means clustering in feature space."""
    print(f"K-means selection IPC={ipc}...")
    
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    train_images_gpu = train_images.to(device)
    train_labels_gpu = train_labels.to(device)
    n = len(train_images)
    
    # Quick training for feature extraction (15 epochs)
    for epoch in range(15):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 512):
            idx = perm[i:i+512]
            imgs = train_images_gpu[idx]
            labels = train_labels_gpu[idx]
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
    
    # Extract features
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, n, 1024):
            imgs = train_images_gpu[i:i+1024]
            feat = model.features(imgs)
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu())
    features = torch.cat(features, dim=0)
    
    # Per-class K-means
    selected_indices = []
    num_classes = 100
    
    for c in range(num_classes):
        class_mask = (train_labels == c)
        class_indices = torch.where(class_mask)[0]
        class_features = features[class_mask]
        
        if len(class_indices) <= ipc:
            selected_indices.extend(class_indices.tolist())
            continue
        
        n_c = len(class_features)
        
        # Initialize centroids randomly
        torch.manual_seed(42 + c)
        init_idx = torch.randperm(n_c)[:ipc]
        centroids = class_features[init_idx].clone()
        
        for _ in range(100):  # K-means iterations
            dists = torch.cdist(class_features, centroids)
            assignments = dists.argmin(dim=1)
            
            new_centroids = torch.zeros_like(centroids)
            for k in range(ipc):
                mask = (assignments == k)
                if mask.sum() > 0:
                    new_centroids[k] = class_features[mask].mean(dim=0)
                else:
                    new_centroids[k] = centroids[k]
            
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        
        # Select nearest sample to each centroid
        dists = torch.cdist(centroids, class_features)
        used = set()
        for k in range(ipc):
            sorted_idx = dists[k].argsort()
            for idx in sorted_idx:
                if idx.item() not in used:
                    used.add(idx.item())
                    selected_indices.append(class_indices[idx.item()].item())
                    break
    
    return selected_indices

def generate_soft_labels(images, teacher_path='teacher_best.pt'):
    """Generate soft labels for a subset using the teacher."""
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    checkpoint = torch.load(teacher_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), 512):
            batch = images[i:i+512].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    return torch.cat(all_logits, dim=0)

def eval_method(images, labels, test_images, test_labels, model_fn, 
                label_type='hard', soft_labels=None, num_runs=3):
    accs = []
    for run in range(num_runs):
        acc = train_and_evaluate(
            images, labels, test_images, test_labels,
            model_fn, num_classes=100, device='cuda',
            label_type=label_type, soft_labels=soft_labels,
            epochs=300, batch_size=256, seed=run, verbose=False
        )
        accs.append(acc)
        print(f"    Run {run+1}: {acc:.2f}%")
    return np.mean(accs), np.std(accs)

def main():
    print("Loading data...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    model_fn = lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))
    
    full_soft_labels = torch.load('soft_labels_v3.pt', map_location='cpu')
    
    results = {}
    
    for ipc in [10, 50]:
        # Random
        print(f"\n=== Random IPC={ipc} ===")
        selected = random_select(train_labels, ipc=ipc, seed=42)
        sub_images = train_images[selected]
        sub_labels = train_labels[selected]
        sub_soft = full_soft_labels[selected]
        
        print("  HL:")
        hl_mean, hl_std = eval_method(sub_images, sub_labels, test_images, test_labels, model_fn, 'hard')
        print("  SL:")
        sl_mean, sl_std = eval_method(sub_images, sub_labels, test_images, test_labels, model_fn, 'soft', sub_soft)
        
        results[f'random_ipc{ipc}'] = {'hl_mean': hl_mean, 'hl_std': hl_std, 'sl_mean': sl_mean, 'sl_std': sl_std}
        print(f"  => HL={hl_mean:.2f}±{hl_std:.2f}, SL={sl_mean:.2f}±{sl_std:.2f}")
        
        # K-centers
        print(f"\n=== K-centers IPC={ipc} ===")
        selected = kmeans_select(train_images, train_labels, ipc, device)
        sub_images = train_images[selected]
        sub_labels = train_labels[selected]
        sub_soft = generate_soft_labels(sub_images)
        
        print("  HL:")
        hl_mean, hl_std = eval_method(sub_images, sub_labels, test_images, test_labels, model_fn, 'hard')
        print("  SL:")
        sl_mean, sl_std = eval_method(sub_images, sub_labels, test_images, test_labels, model_fn, 'soft', sub_soft)
        
        results[f'kcenters_ipc{ipc}'] = {'hl_mean': hl_mean, 'hl_std': hl_std, 'sl_mean': sl_mean, 'sl_std': sl_std}
        print(f"  => HL={hl_mean:.2f}±{hl_std:.2f}, SL={sl_mean:.2f}±{sl_std:.2f}")
    
    # Save
    with open('results/results_coresets_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("CORESET RESULTS")
    print("=" * 60)
    for key, r in results.items():
        print(f"{key}: HL={r['hl_mean']:.2f}±{r['hl_std']:.2f}, SL={r['sl_mean']:.2f}±{r['sl_std']:.2f}")

if __name__ == '__main__':
    main()
