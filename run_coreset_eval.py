"""
Run coreset evaluations: Random and K-centers for IPC 10 and 50, HL and SL.
"""
import torch
import numpy as np
import json
import os
import time
from convnet import ConvNet
from data_utils import get_cifar100_tensors, get_class_indices, random_select
from train_eval import train_and_evaluate, run_experiment

device = 'cuda'
NUM_RUNS = 3
RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def feature_kcenters(train_images, train_labels, num_classes, ipc, device='cuda'):
    """K-centers selection in feature space using a pretrained model."""
    print(f"Computing K-centers (feature space) IPC={ipc}...")
    
    teacher_path = '/workspace/teacher_final.pt'
    if os.path.exists(teacher_path):
        model = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        ckpt = torch.load(teacher_path, map_location=device); model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        model.eval()
    else:
        model = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        model.eval()
    
    features_list = []
    with torch.no_grad():
        for i in range(0, len(train_images), 512):
            batch = train_images[i:i+512].to(device)
            feat = model.embed(batch)
            features_list.append(feat.cpu())
    features = torch.cat(features_list, dim=0)
    
    selected = []
    class_indices = get_class_indices(train_labels, num_classes)
    
    for c in range(num_classes):
        idx = class_indices[c]
        class_feat = features[idx]
        n = len(idx)
        
        chosen = []
        mean_feat = class_feat.mean(0)
        dists_to_mean = torch.cdist(class_feat.unsqueeze(0), mean_feat.unsqueeze(0).unsqueeze(0)).squeeze()
        first = dists_to_mean.argmin().item()
        chosen.append(first)
        
        min_dists = torch.cdist(class_feat.unsqueeze(0), class_feat[first].unsqueeze(0).unsqueeze(0)).squeeze()
        
        for _ in range(1, ipc):
            farthest = min_dists.argmax().item()
            chosen.append(farthest)
            new_dists = torch.cdist(class_feat.unsqueeze(0), class_feat[farthest].unsqueeze(0).unsqueeze(0)).squeeze()
            min_dists = torch.minimum(min_dists, new_dists)
        
        for j in chosen:
            selected.append(idx[j])
    
    del model
    torch.cuda.empty_cache()
    return selected


def main():
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    soft_labels_full = None
    if os.path.exists('/workspace/soft_labels_final.pt'):
        soft_labels_full = torch.load('/workspace/soft_labels_final.pt', map_location='cpu')
        print(f"Loaded full soft labels: {soft_labels_full.shape}")
    
    model_fn = lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))
    
    results = {}
    
    # Try to load existing results
    results_path = os.path.join(RESULTS_DIR, 'results_final.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    
    # ============================================================
    # RANDOM
    # ============================================================
    for ipc in [10, 50]:
        selected = random_select(train_labels, ipc=ipc, seed=0)
        sub_images = train_images[selected]
        sub_labels = train_labels[selected]
        
        # HL
        key = f'Random_IPC{ipc}_HL'
        if key not in results:
            print(f"\n--- {key} ---")
            mean_acc, std_acc = run_experiment(
                sub_images, sub_labels, test_images, test_labels,
                model_fn, label_type='hard', epochs=300, num_runs=NUM_RUNS, verbose=True
            )
            results[key] = {'mean': round(mean_acc, 2), 'std': round(std_acc, 2)}
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"Skipping {key} (already done: {results[key]})")
        
        # SL
        key = f'Random_IPC{ipc}_SL'
        if key not in results:
            print(f"\n--- {key} ---")
            if soft_labels_full is not None:
                sub_soft = soft_labels_full[selected]
            else:
                sub_soft = torch.zeros(len(sub_images), 100)
            
            mean_acc, std_acc = run_experiment(
                sub_images, sub_labels, test_images, test_labels,
                model_fn, label_type='soft', soft_labels=sub_soft,
                epochs=300, num_runs=NUM_RUNS, verbose=True
            )
            results[key] = {'mean': round(mean_acc, 2), 'std': round(std_acc, 2)}
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"Skipping {key} (already done: {results[key]})")
    
    # ============================================================
    # K-CENTERS
    # ============================================================
    for ipc in [10, 50]:
        selected = feature_kcenters(train_images, train_labels, 100, ipc, device)
        sub_images = train_images[selected]
        sub_labels = train_labels[selected]
        
        # HL
        key = f'Kcenter_IPC{ipc}_HL'
        if key not in results:
            print(f"\n--- {key} ---")
            mean_acc, std_acc = run_experiment(
                sub_images, sub_labels, test_images, test_labels,
                model_fn, label_type='hard', epochs=300, num_runs=NUM_RUNS, verbose=True
            )
            results[key] = {'mean': round(mean_acc, 2), 'std': round(std_acc, 2)}
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"Skipping {key} (already done: {results[key]})")
        
        # SL
        key = f'Kcenter_IPC{ipc}_SL'
        if key not in results:
            print(f"\n--- {key} ---")
            if soft_labels_full is not None:
                sub_soft = soft_labels_full[selected]
            else:
                sub_soft = torch.zeros(len(sub_images), 100)
            
            mean_acc, std_acc = run_experiment(
                sub_images, sub_labels, test_images, test_labels,
                model_fn, label_type='soft', soft_labels=sub_soft,
                epochs=300, num_runs=NUM_RUNS, verbose=True
            )
            results[key] = {'mean': round(mean_acc, 2), 'std': round(std_acc, 2)}
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"Skipping {key} (already done: {results[key]})")
    
    print("\n\nCoreset Results:")
    for k, v in sorted(results.items()):
        if 'Random' in k or 'Kcenter' in k:
            print(f"  {k}: {v['mean']:.2f} ± {v['std']:.2f}")


if __name__ == '__main__':
    main()
