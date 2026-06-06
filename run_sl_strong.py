"""
Run all SL experiments with the strong ResNet-18 teacher (78.49% accuracy).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
import time

from convnet import get_convnet_d3
from dsa import DiffAugment


def get_resnet18_cifar100():
    import torchvision.models as models
    model = models.resnet18(weights=None, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_dcbench_data(method, ipc):
    base = 'dcbench_data/data/condensed'
    if method in ['DC', 'DM', 'DSA']:
        path = f'{base}/{method}/CIFAR100/res_{method}_CIFAR100_ConvNet_{ipc}ipc.pt'
        d = torch.load(path, map_location='cpu', weights_only=False)
        images = d['data'][0][0]
        labels = d['data'][0][1]
    elif method == 'TM':
        images = torch.load(f'{base}/TM/CIFAR100/IPC{ipc}/images_best.pt', map_location='cpu', weights_only=False)
        labels = torch.load(f'{base}/TM/CIFAR100/IPC{ipc}/labels_best.pt', map_location='cpu', weights_only=False)
    elif method == 'Random':
        images = torch.load(f'{base}/random/CIFAR100/CIFAR100_IPC{ipc}_normalize_images.pt', map_location='cpu', weights_only=False)
        labels = torch.load(f'{base}/random/CIFAR100/CIFAR100_IPC{ipc}_normalize_labels.pt', map_location='cpu', weights_only=False)
    elif method == 'K-centers':
        images = torch.load(f'{base}/kmeans-emb/CIFAR100/CIFAR100_IPC{ipc}_images.pt', map_location='cpu', weights_only=False)
        labels = torch.load(f'{base}/kmeans-emb/CIFAR100/CIFAR100_IPC{ipc}_labels.pt', map_location='cpu', weights_only=False)
    else:
        raise ValueError(f"Unknown method: {method}")
    return images.float(), labels.long()


def load_test_data():
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)


def generate_soft_labels_strong(images, device='cuda'):
    """Generate soft labels using strong RN18 teacher."""
    teacher = get_resnet18_cifar100().to(device)
    ckpt = torch.load('teacher_rn18_strong.pt', map_location=device, weights_only=False)
    teacher.load_state_dict(ckpt['model_state_dict'])
    teacher.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            logits = teacher(batch)
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


def evaluate(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def train_sl(images, soft_labels, testloader, device='cuda', seed=0, T=20.0):
    """SL training: 300 epochs, AdamW, lr=1e-3, cosine, KL(T=20), DSA."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3(num_classes=100).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    
    images_gpu = images.to(device)
    sl_gpu = soft_labels.to(device)
    n = len(images)
    batch_size = min(256, n)
    
    for epoch in range(300):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x = images_gpu[idx]
            sl = sl_gpu[idx]
            x = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate')
            optimizer.zero_grad()
            out = model(x)
            log_student = F.log_softmax(out / T, dim=1)
            teacher_probs = F.softmax(sl / T, dim=1)
            loss = F.kl_div(log_student, teacher_probs, reduction='batchmean') * (T * T)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return evaluate(model, testloader, device)


def main():
    device = 'cuda'
    testloader = load_test_data()
    
    methods = ['DC', 'DM', 'TM', 'Random', 'K-centers']
    ipcs = [10, 50]
    num_runs = 3
    
    # Load existing results
    results_path = 'results/table1_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        if isinstance(results, list):
            # Convert to dict
            results = {f"{r['method']}_IPC{r['ipc']}_{r['label_type']}": r for r in results}
    else:
        results = {}
    
    for method in methods:
        for ipc in ipcs:
            key = f"{method}_IPC{ipc}_SL_strong"
            if key in results:
                print(f"Skipping {key} (already done)")
                continue
            
            print(f"\n{'='*60}")
            print(f"Method: {method}, IPC: {ipc}, Labels: SL (strong RN18 teacher)")
            print(f"{'='*60}")
            
            images, labels = load_dcbench_data(method, ipc)
            print(f"Loaded {method} IPC{ipc}: {images.shape}")
            
            soft_labels = generate_soft_labels_strong(images, device)
            print(f"Generated soft labels: {soft_labels.shape}")
            
            accs = []
            for run in range(num_runs):
                seed = run * 42
                t0 = time.time()
                acc = train_sl(images, soft_labels, testloader, device=device, seed=seed)
                elapsed = time.time() - t0
                accs.append(round(acc, 2))
                print(f"  Run {run+1}/{num_runs}: {acc:.2f}% ({elapsed:.1f}s)")
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"  Result: {mean_acc:.2f} ± {std_acc:.2f}%")
            
            results[key] = {
                'method': method,
                'ipc': ipc,
                'label_type': 'SL_strong',
                'accs': accs,
                'mean': round(float(mean_acc), 2),
                'std': round(float(std_acc), 2),
            }
            
            # Save incrementally
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Print summary
    print("\n\n" + "="*80)
    print("SL Results with Strong RN18 Teacher (78.49%)")
    print("="*80)
    print(f"{'Method':<12} {'IPC':>4} {'SL (Ours)':>14} {'SL (Paper)':>12}")
    print("-"*60)
    
    paper_sl = {
        ('DC', 10): 23.54, ('DC', 50): 33.46,
        ('DM', 10): 26.13, ('DM', 50): 43.46,
        ('TM', 10): 37.60, ('TM', 50): 46.26,
        ('Random', 10): 33.43, ('Random', 50): 45.39,
        ('K-centers', 10): 34.70, ('K-centers', 50): 46.24,
    }
    
    for method in methods:
        for ipc in ipcs:
            key = f"{method}_IPC{ipc}_SL_strong"
            if key in results:
                r = results[key]
                paper = paper_sl.get((method, ipc), 0)
                print(f"{method:<12} {ipc:>4} {r['mean']:>10.2f}±{r['std']:.2f} {paper:>10.2f}")


if __name__ == '__main__':
    main()
