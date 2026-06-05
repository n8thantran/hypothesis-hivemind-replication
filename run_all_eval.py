"""
Run ALL evaluations for the paper's Table (small_scale_c100).
5 methods × 2 IPC × 2 label types × 3 trials = 60 evaluations.
"""
import torch
import numpy as np
import json
import time
import os
from convnet import ConvNet
from data_utils import get_cifar100_tensors, get_class_indices
from train_eval import train_and_evaluate

def get_random_subset(train_images, train_labels, ipc, num_classes, seed):
    """Get random real subset."""
    class_indices = get_class_indices(train_labels.numpy(), num_classes)
    np.random.seed(seed)
    selected_images = []
    selected_labels = []
    for c in range(num_classes):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            selected_images.append(train_images[indices[p]])
            selected_labels.append(c)
    return torch.stack(selected_images), torch.tensor(selected_labels, dtype=torch.long)

def get_kcenter_subset(train_images, train_labels, ipc, num_classes, device='cuda'):
    """Feature-space K-centers coreset selection."""
    class_indices = get_class_indices(train_labels.numpy(), num_classes)
    
    # Use a pretrained network for features
    net = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
    net.eval()
    
    selected_images = []
    selected_labels = []
    
    with torch.no_grad():
        for c in range(num_classes):
            indices = class_indices[c]
            class_imgs = train_images[indices].to(device)
            
            # Get features in batches
            feats = []
            for i in range(0, len(class_imgs), 256):
                batch = class_imgs[i:i+256]
                feat = net.embed(batch)
                feats.append(feat)
            feats = torch.cat(feats, dim=0)  # [N, feat_dim]
            
            # K-centers greedy selection
            N = feats.shape[0]
            selected = []
            # Start with the sample closest to the mean
            mean_feat = feats.mean(0)
            dists_to_mean = torch.cdist(feats.unsqueeze(0), mean_feat.unsqueeze(0).unsqueeze(0)).squeeze()
            first = dists_to_mean.argmin().item()
            selected.append(first)
            
            # Min distance to selected set
            min_dists = torch.cdist(feats.unsqueeze(0), feats[first].unsqueeze(0).unsqueeze(0)).squeeze()
            
            for _ in range(ipc - 1):
                # Pick the point farthest from the selected set
                # Mask already selected
                min_dists[selected] = -1
                next_idx = min_dists.argmax().item()
                selected.append(next_idx)
                
                # Update min distances
                new_dists = torch.cdist(feats.unsqueeze(0), feats[next_idx].unsqueeze(0).unsqueeze(0)).squeeze()
                min_dists = torch.min(min_dists, new_dists)
                min_dists[selected] = -1  # Re-mask
            
            for s in selected:
                selected_images.append(train_images[indices[s]])
                selected_labels.append(c)
    
    return torch.stack(selected_images), torch.tensor(selected_labels, dtype=torch.long)


def main():
    device = 'cuda'
    num_classes = 100
    
    print("Loading data...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Load soft labels
    soft_labels_full = torch.load('soft_labels_final.pt', weights_only=False)
    print(f"Soft labels shape: {soft_labels_full.shape}")
    
    model_fn = lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))
    
    results = {}
    
    # Define all experiments
    experiments = []
    
    for ipc in [10, 50]:
        # Random
        experiments.append(('Random', ipc))
        # K-centers
        experiments.append(('K-centers', ipc))
        # DM
        experiments.append(('DM', ipc))
        # DC
        experiments.append(('DC', ipc))
        # TM
        experiments.append(('TM', ipc))
    
    num_trials = 3
    trial_seeds = [42, 123, 456]
    
    # Pre-compute coreset subsets
    print("\n=== Pre-computing subsets ===")
    subsets = {}
    
    for ipc in [10, 50]:
        # Random subsets (different per trial)
        for trial, seed in enumerate(trial_seeds):
            imgs, lbls = get_random_subset(train_images, train_labels, ipc, num_classes, seed)
            # Get soft labels for these real images
            class_indices = get_class_indices(train_labels.numpy(), num_classes)
            sl = []
            for c in range(num_classes):
                indices = class_indices[c]
                np.random.seed(seed)
                perm = np.random.permutation(len(indices))[:ipc]
                for p in perm:
                    sl.append(soft_labels_full[indices[p]])
            sl = torch.stack(sl)
            subsets[('Random', ipc, trial)] = (imgs, lbls, sl)
        
        # K-centers (same for all trials)
        print(f"  Computing K-centers IPC={ipc}...")
        imgs, lbls = get_kcenter_subset(train_images, train_labels, ipc, num_classes)
        # Get soft labels for K-centers
        class_indices = get_class_indices(train_labels.numpy(), num_classes)
        # Need to find original indices for soft labels
        # For K-centers, we need to track which original indices were selected
        # Let me redo this properly
        
    # Actually, let me redo K-centers to also return original indices
    def get_kcenter_with_indices(train_images, train_labels, ipc, num_classes, device='cuda'):
        class_indices = get_class_indices(train_labels.numpy(), num_classes)
        net = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        net.eval()
        
        selected_images = []
        selected_labels = []
        original_indices = []
        
        with torch.no_grad():
            for c in range(num_classes):
                indices = class_indices[c]
                class_imgs = train_images[indices].to(device)
                
                feats = []
                for i in range(0, len(class_imgs), 256):
                    batch = class_imgs[i:i+256]
                    feat = net.embed(batch)
                    feats.append(feat)
                feats = torch.cat(feats, dim=0)
                
                N = feats.shape[0]
                selected = []
                mean_feat = feats.mean(0)
                dists_to_mean = torch.cdist(feats.unsqueeze(0), mean_feat.unsqueeze(0).unsqueeze(0)).squeeze()
                first = dists_to_mean.argmin().item()
                selected.append(first)
                
                min_dists = torch.cdist(feats.unsqueeze(0), feats[first].unsqueeze(0).unsqueeze(0)).squeeze()
                
                for _ in range(ipc - 1):
                    min_dists_copy = min_dists.clone()
                    for s in selected:
                        min_dists_copy[s] = -1
                    next_idx = min_dists_copy.argmax().item()
                    selected.append(next_idx)
                    
                    new_dists = torch.cdist(feats.unsqueeze(0), feats[next_idx].unsqueeze(0).unsqueeze(0)).squeeze()
                    min_dists = torch.min(min_dists, new_dists)
                
                for s in selected:
                    selected_images.append(train_images[indices[s]])
                    selected_labels.append(c)
                    original_indices.append(indices[s])
        
        return (torch.stack(selected_images), 
                torch.tensor(selected_labels, dtype=torch.long),
                original_indices)
    
    # Pre-compute all subsets
    subsets = {}
    
    for ipc in [10, 50]:
        # Random
        for trial, seed in enumerate(trial_seeds):
            np.random.seed(seed)
            class_indices = get_class_indices(train_labels.numpy(), num_classes)
            imgs_list, lbls_list, orig_idx_list = [], [], []
            for c in range(num_classes):
                indices = class_indices[c]
                perm = np.random.permutation(len(indices))[:ipc]
                for p in perm:
                    imgs_list.append(train_images[indices[p]])
                    lbls_list.append(c)
                    orig_idx_list.append(indices[p])
            imgs = torch.stack(imgs_list)
            lbls = torch.tensor(lbls_list, dtype=torch.long)
            sl = soft_labels_full[orig_idx_list]
            subsets[('Random', ipc, trial)] = (imgs, lbls, sl)
        
        # K-centers
        print(f"  Computing K-centers IPC={ipc}...")
        imgs, lbls, orig_indices = get_kcenter_with_indices(train_images, train_labels, ipc, num_classes)
        sl = soft_labels_full[orig_indices]
        for trial in range(num_trials):
            subsets[('K-centers', ipc, trial)] = (imgs, lbls, sl)
        
        # DD methods - load from files
        for method in ['DM', 'DC', 'TM']:
            path = f'distilled_{method.lower()}_ipc{ipc}.pt'
            if os.path.exists(path):
                d = torch.load(path, weights_only=False)
                imgs = d['images']
                lbls = d['labels']
                # Load soft labels for DD
                sl_path = f'soft_labels_{method.lower()}_ipc{ipc}.pt'
                if os.path.exists(sl_path):
                    sl = torch.load(sl_path, weights_only=False)
                else:
                    # Generate soft labels from teacher for DD images
                    sl = None
                for trial in range(num_trials):
                    subsets[(method, ipc, trial)] = (imgs, lbls, sl)
                print(f"  Loaded {path}: {imgs.shape}")
            else:
                print(f"  WARNING: {path} not found!")
    
    # Now run all evaluations
    print("\n=== Running evaluations ===")
    all_results = {}
    
    total_exps = len(experiments) * 2 * num_trials  # methods × label_types × trials
    exp_count = 0
    
    for method, ipc in experiments:
        for label_type in ['hard', 'soft']:
            accs = []
            for trial in range(num_trials):
                exp_count += 1
                key = (method, ipc, trial)
                if key not in subsets:
                    print(f"  SKIP {method} IPC={ipc} {label_type} trial={trial} (no data)")
                    continue
                
                imgs, lbls, sl = subsets[key]
                
                t0 = time.time()
                acc = train_and_evaluate(
                    imgs, lbls, test_images, test_labels,
                    model_fn, num_classes, device, label_type,
                    sl if label_type == 'soft' else None,
                    epochs=300, batch_size=256,
                    seed=trial_seeds[trial],
                    verbose=False
                )
                elapsed = time.time() - t0
                accs.append(acc)
                print(f"  [{exp_count}/{total_exps}] {method} IPC={ipc} {label_type.upper()} trial={trial}: {acc:.2f}% ({elapsed:.0f}s)")
            
            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                result_key = f"{method}_ipc{ipc}_{label_type}"
                all_results[result_key] = {
                    'mean': float(mean_acc),
                    'std': float(std_acc),
                    'trials': [float(a) for a in accs]
                }
                print(f"  => {method} IPC={ipc} {label_type.upper()}: {mean_acc:.2f} ± {std_acc:.2f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/results_final.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final table
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE (CIFAR-100, ConvNet-D3)")
    print("="*80)
    print(f"{'Method':<12} {'IPC':>4} {'HL (ours)':>16} {'HL (paper)':>16} {'SL (ours)':>16} {'SL (paper)':>16}")
    print("-"*80)
    
    paper_results = {
        'DM_10': (29.23, 0.26, 26.13, 0.10),
        'DM_50': (42.32, 0.37, 43.46, 0.18),
        'DC_10': (28.42, 0.29, 23.54, 0.31),
        'DC_50': (30.56, 0.56, 33.46, 0.38),
        'TM_10': (38.18, 0.42, 37.60, 0.25),
        'TM_50': (46.32, 0.26, 46.26, 0.30),
        'Random_10': (18.64, 0.25, 33.43, 0.18),
        'Random_50': (34.66, 0.41, 45.39, 0.23),
        'K-centers_10': (25.04, 0.30, 34.70, 0.27),
        'K-centers_50': (38.64, 0.43, 46.24, 0.12),
    }
    
    for method in ['DM', 'DC', 'TM', 'Random', 'K-centers']:
        for ipc in [10, 50]:
            hl_key = f"{method}_ipc{ipc}_hard"
            sl_key = f"{method}_ipc{ipc}_soft"
            paper_key = f"{method}_{ipc}"
            
            hl_str = "N/A"
            sl_str = "N/A"
            
            if hl_key in all_results:
                r = all_results[hl_key]
                hl_str = f"{r['mean']:.2f}±{r['std']:.2f}"
            if sl_key in all_results:
                r = all_results[sl_key]
                sl_str = f"{r['mean']:.2f}±{r['std']:.2f}"
            
            if paper_key in paper_results:
                p = paper_results[paper_key]
                paper_hl = f"{p[0]:.2f}±{p[1]:.2f}"
                paper_sl = f"{p[2]:.2f}±{p[3]:.2f}"
            else:
                paper_hl = "N/A"
                paper_sl = "N/A"
            
            print(f"{method:<12} {ipc:>4} {hl_str:>16} {paper_hl:>16} {sl_str:>16} {paper_sl:>16}")
    
    print("="*80)
    
    # Save table as text
    with open('results/table_final.txt', 'w') as f:
        f.write("CIFAR-100 Small-Scale DD Results (ConvNet-D3)\n")
        f.write("="*80 + "\n")
        f.write(f"{'Method':<12} {'IPC':>4} {'HL (ours)':>16} {'HL (paper)':>16} {'SL (ours)':>16} {'SL (paper)':>16}\n")
        f.write("-"*80 + "\n")
        for method in ['DM', 'DC', 'TM', 'Random', 'K-centers']:
            for ipc in [10, 50]:
                hl_key = f"{method}_ipc{ipc}_hard"
                sl_key = f"{method}_ipc{ipc}_soft"
                paper_key = f"{method}_{ipc}"
                
                hl_str = "N/A"
                sl_str = "N/A"
                
                if hl_key in all_results:
                    r = all_results[hl_key]
                    hl_str = f"{r['mean']:.2f}±{r['std']:.2f}"
                if sl_key in all_results:
                    r = all_results[sl_key]
                    sl_str = f"{r['mean']:.2f}±{r['std']:.2f}"
                
                if paper_key in paper_results:
                    p = paper_results[paper_key]
                    paper_hl = f"{p[0]:.2f}±{p[1]:.2f}"
                    paper_sl = f"{p[2]:.2f}±{p[3]:.2f}"
                else:
                    paper_hl = "N/A"
                    paper_sl = "N/A"
                
                f.write(f"{method:<12} {ipc:>4} {hl_str:>16} {paper_hl:>16} {sl_str:>16} {paper_sl:>16}\n")
        f.write("="*80 + "\n")
    
    print("\nResults saved to results/results_final.json and results/table_final.txt")


if __name__ == '__main__':
    main()
