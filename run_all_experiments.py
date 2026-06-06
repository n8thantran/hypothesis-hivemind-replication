"""
Complete evaluation pipeline for replicating Table: tab:small_scale_c100
"Rethinking Dataset Distillation: Hard Truths About Soft Labels"

Evaluates: DM, DC, TM, Random, K-centers on CIFAR-100 with ConvNet-D3
Both HL (hard label) and SL (soft label) settings, IPC={10, 50}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import sys
from convnet import ConvNet, get_convnet_d3
from dsa import DiffAugment
from data_utils import get_cifar100_tensors


# ============ CIFAR-100 normalization ============
CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
CIFAR100_STD = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)


def normalize_images(images):
    """Normalize [0,1] images to standard CIFAR-100 normalization."""
    return (images - CIFAR100_MEAN) / CIFAR100_STD


# ============ Data loading ============
def load_test_data():
    """Load and normalize CIFAR-100 test data."""
    _, _, test_images, test_labels = get_cifar100_tensors()
    test_images = normalize_images(test_images)
    return test_images, test_labels


def load_full_train_data():
    """Load and normalize CIFAR-100 training data."""
    train_images, train_labels, _, _ = get_cifar100_tensors()
    train_images = normalize_images(train_images)
    return train_images, train_labels


def load_dcbench_dd(method, ipc):
    """Load distilled data from DCBench (DC, DM, DSA)."""
    path = f'dcbench_data/data/condensed/{method}/CIFAR100/res_{method}_CIFAR100_ConvNet_{ipc}ipc.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    images = data['data'][0][0]  # Already normalized
    labels = data['data'][0][1].long()
    return images, labels


def load_dcbench_coreset(method, ipc):
    """Load coreset data from DCBench (random, kmeans-emb)."""
    base = 'dcbench_data/data/condensed'
    if method == 'random':
        images = torch.load(f'{base}/random/CIFAR100/CIFAR100_IPC{ipc}_normalize_images.pt', 
                          map_location='cpu', weights_only=False)
        labels = torch.load(f'{base}/random/CIFAR100/CIFAR100_IPC{ipc}_normalize_labels.pt',
                          map_location='cpu', weights_only=False).long()
    elif method == 'kmeans-emb':
        images = torch.load(f'{base}/kmeans-emb/CIFAR100/CIFAR100_IPC{ipc}_images.pt',
                          map_location='cpu', weights_only=False)
        labels = torch.load(f'{base}/kmeans-emb/CIFAR100/CIFAR100_IPC{ipc}_labels.pt',
                          map_location='cpu', weights_only=False).long()
    return images, labels


def load_our_tm(ipc):
    """Load our TM distilled data."""
    data = torch.load(f'distilled_tm_ipc{ipc}.pt', map_location='cpu', weights_only=False)
    images = data['images']  # Already normalized
    labels = data['labels'].long()
    return images, labels


# ============ Soft label generation ============
def generate_soft_labels(images, teacher_state, device='cuda', temperature=20.0):
    """Generate soft labels using teacher model for given images."""
    teacher = get_convnet_d3().to(device)
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            logits = teacher(batch)
            all_logits.append(logits.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    return logits  # Return raw logits; KL div will use them with temperature


# ============ Training ============
def train_and_eval(train_images, train_labels, test_images, test_labels,
                   label_type='hard', soft_labels=None,
                   epochs=300, batch_size=256, seed=0, device='cuda',
                   verbose=True):
    """Train ConvNet-D3 and evaluate. Returns test accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    model = get_convnet_d3().to(device)
    
    n_train = len(train_images)
    
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
        perm = torch.randperm(n_train)
        bs = min(batch_size, n_train)
        
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            batch_imgs = train_images[idx].to(device)
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            
            if label_type == 'hard':
                batch_labels = train_labels[idx].to(device)
                loss = criterion(outputs, batch_labels)
            else:
                batch_soft = soft_labels[idx].to(device)
                log_probs = F.log_softmax(outputs / temperature, dim=1)
                targets = F.softmax(batch_soft / temperature, dim=1)
                loss = F.kl_div(log_probs, targets, reduction='batchmean') * (temperature ** 2)
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            acc = quick_eval(model, test_images, test_labels, device)
            print(f"  Epoch {epoch+1}/{epochs}, Acc: {acc:.2f}%")
    
    acc = quick_eval(model, test_images, test_labels, device)
    return acc


def quick_eval(model, test_images, test_labels, device='cuda'):
    """Evaluate test accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), 512):
            x = test_images[i:i+512].to(device)
            y = test_labels[i:i+512].to(device)
            out = model(x)
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
    model.train()
    return 100.0 * correct / total


# ============ Main pipeline ============
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Parse args
    num_runs = 3
    if '--fast' in sys.argv:
        num_runs = 1
    if '--runs' in sys.argv:
        idx = sys.argv.index('--runs')
        num_runs = int(sys.argv[idx + 1])
    
    # Load test data
    print("Loading test data...")
    test_images, test_labels = load_test_data()
    print(f"Test: {test_images.shape}")
    
    # Load teacher for soft labels
    print("Loading teacher model...")
    teacher_data = torch.load('teacher.pt', map_location='cpu', weights_only=False)
    teacher_state = teacher_data['model_state_dict']
    print(f"Teacher accuracy: {teacher_data['accuracy']}%")
    
    # Also load full train data (for generating soft labels for coresets)
    print("Loading training data...")
    train_images_full, train_labels_full = load_full_train_data()
    
    results = {}
    
    # Define all experiments
    experiments = [
        # (name, method_type, source, ipcs)
        ('DM', 'dd', 'DM', [10, 50]),
        ('DC', 'dd', 'DC', [10, 50]),
        ('TM', 'tm', None, [10, 50]),
        ('Random', 'coreset', 'random', [10, 50]),
        ('K-centers', 'coreset', 'kmeans-emb', [10, 50]),
    ]
    
    for name, method_type, source, ipcs in experiments:
        for ipc in ipcs:
            key = f"{name}_IPC{ipc}"
            print(f"\n{'='*60}")
            print(f"Evaluating: {key}")
            print(f"{'='*60}")
            
            # Load data
            if method_type == 'dd':
                images, labels = load_dcbench_dd(source, ipc)
            elif method_type == 'tm':
                images, labels = load_our_tm(ipc)
            elif method_type == 'coreset':
                images, labels = load_dcbench_coreset(source, ipc)
            
            print(f"  Data: images={images.shape}, labels={labels.shape}")
            print(f"  Image stats: min={images.min():.3f}, max={images.max():.3f}, mean={images.mean():.3f}")
            print(f"  Unique labels: {len(torch.unique(labels))}")
            
            # Generate soft labels for this dataset
            print("  Generating soft labels...")
            soft_logits = generate_soft_labels(images, teacher_state, device)
            
            # HL evaluation
            print(f"\n  --- HL evaluation ({num_runs} runs) ---")
            hl_accs = []
            for run in range(num_runs):
                acc = train_and_eval(images, labels, test_images, test_labels,
                                    label_type='hard', epochs=300, batch_size=256,
                                    seed=run, device=device, verbose=(run == 0))
                hl_accs.append(acc)
                print(f"  HL Run {run+1}: {acc:.2f}%")
            
            hl_mean = np.mean(hl_accs)
            hl_std = np.std(hl_accs)
            
            # SL evaluation
            print(f"\n  --- SL evaluation ({num_runs} runs) ---")
            sl_accs = []
            for run in range(num_runs):
                acc = train_and_eval(images, labels, test_images, test_labels,
                                    label_type='soft', soft_labels=soft_logits,
                                    epochs=300, batch_size=256,
                                    seed=run, device=device, verbose=(run == 0))
                sl_accs.append(acc)
                print(f"  SL Run {run+1}: {acc:.2f}%")
            
            sl_mean = np.mean(sl_accs)
            sl_std = np.std(sl_accs)
            
            results[key] = {
                'hl_mean': round(hl_mean, 2),
                'hl_std': round(hl_std, 2),
                'sl_mean': round(sl_mean, 2),
                'sl_std': round(sl_std, 2),
                'hl_runs': [round(a, 2) for a in hl_accs],
                'sl_runs': [round(a, 2) for a in sl_accs],
            }
            
            # Save after each experiment
            os.makedirs('results', exist_ok=True)
            with open('results/main_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n  => {key}: HL={hl_mean:.2f}±{hl_std:.2f}, SL={sl_mean:.2f}±{sl_std:.2f}")
    
    # Print final table
    print("\n\n" + "="*80)
    print("FINAL RESULTS TABLE (tab:small_scale_c100)")
    print("="*80)
    print(f"{'Method':<12} {'IPC':<6} {'HL':<16} {'SL':<16}")
    print("-"*50)
    
    for name in ['DM', 'DC', 'TM', 'Random', 'K-centers']:
        for ipc in [10, 50]:
            key = f"{name}_IPC{ipc}"
            if key in results:
                r = results[key]
                print(f"{name:<12} {ipc:<6} {r['hl_mean']:.2f}±{r['hl_std']:.2f}   {r['sl_mean']:.2f}±{r['sl_std']:.2f}")
    
    print("\n" + "="*80)
    print("Paper target values:")
    print(f"{'Method':<12} {'IPC':<6} {'HL':<16} {'SL':<16}")
    print("-"*50)
    paper = {
        'DM_10': (29.23, 26.13), 'DM_50': (42.32, 43.46),
        'DC_10': (28.42, 23.54), 'DC_50': (30.56, 33.46),
        'TM_10': (38.18, 37.60), 'TM_50': (46.32, 46.26),
        'Random_10': (18.64, 33.43), 'Random_50': (34.66, 45.39),
        'K-centers_10': (25.04, 34.70), 'K-centers_50': (38.64, 46.24),
    }
    for name in ['DM', 'DC', 'TM', 'Random', 'K-centers']:
        for ipc in [10, 50]:
            pkey = f"{name}_{ipc}"
            if pkey in paper:
                print(f"{name:<12} {ipc:<6} {paper[pkey][0]:.2f}          {paper[pkey][1]:.2f}")
    
    # Save formatted table
    with open('results/table_output.txt', 'w') as f:
        f.write("CIFAR-100, ConvNet-D3 Results\n")
        f.write(f"{'Method':<12} {'IPC':<6} {'HL':<16} {'SL':<16}\n")
        f.write("-"*50 + "\n")
        for name in ['DM', 'DC', 'TM', 'Random', 'K-centers']:
            for ipc in [10, 50]:
                key = f"{name}_IPC{ipc}"
                if key in results:
                    r = results[key]
                    f.write(f"{name:<12} {ipc:<6} {r['hl_mean']:.2f}±{r['hl_std']:.2f}   {r['sl_mean']:.2f}±{r['sl_std']:.2f}\n")
    
    print("\nResults saved to results/main_results.json and results/table_output.txt")


if __name__ == '__main__':
    main()
