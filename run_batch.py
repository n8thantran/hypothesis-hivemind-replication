"""
Run experiments in batches for Table 1 replication.
Saves results incrementally to avoid losing progress.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
import sys
import time

from convnet import get_convnet_d3
from dsa import DiffAugment

# ============================================================
# Data Loading (cached)
# ============================================================

_test_data_cache = None

def load_test_data():
    global _test_data_cache
    if _test_data_cache is not None:
        return _test_data_cache
    
    import torchvision
    import torchvision.transforms as transforms
    
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    _test_data_cache = testloader
    return testloader


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
    
    return images.float(), labels.long()


def generate_soft_labels(images, device='cuda'):
    teacher = get_convnet_d3(num_classes=100).to(device)
    checkpoint = torch.load('teacher.pt', map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['model_state_dict'])
    else:
        teacher.load_state_dict(checkpoint)
    teacher.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            logits = teacher(batch)
            all_logits.append(logits.cpu())
    
    return torch.cat(all_logits, dim=0)


# ============================================================
# Training Functions
# ============================================================

def evaluate(model, testloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_hl(images, labels, testloader, device='cuda', seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3(num_classes=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    images_gpu = images.to(device)
    labels_gpu = labels.to(device)
    n = len(images)
    batch_size = min(256, n)
    
    for epoch in range(300):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x = DiffAugment(images_gpu[idx], strategy='color_crop_cutout_flip_scale_rotate')
            optimizer.zero_grad()
            loss = criterion(model(x), labels_gpu[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return evaluate(model, testloader, device)


def train_sl(images, soft_labels, testloader, device='cuda', seed=0, T=20.0):
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
            x = DiffAugment(images_gpu[idx], strategy='color_crop_cutout_flip_scale_rotate')
            out = model(x)
            log_student = F.log_softmax(out / T, dim=1)
            teacher_probs = F.softmax(sl_gpu[idx] / T, dim=1)
            loss = F.kl_div(log_student, teacher_probs, reduction='batchmean') * (T * T)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return evaluate(model, testloader, device)


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda'
    num_runs = 3
    
    # Load results file if exists (for incremental progress)
    results_file = 'results/table1_results.json'
    os.makedirs('results', exist_ok=True)
    
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    testloader = load_test_data()
    
    # Define experiments: method, ipc, label_type
    methods = ['DC', 'DM', 'TM', 'Random', 'K-centers']
    ipcs = [10, 50]
    
    # If command line args specify which to run
    if len(sys.argv) > 1:
        label_type = sys.argv[1]  # 'HL' or 'SL'
    else:
        label_type = 'HL'
    
    for method in methods:
        for ipc in ipcs:
            key = f"{method}_IPC{ipc}_{label_type}"
            
            if key in all_results:
                print(f"SKIP {key} (already done: {all_results[key]['mean']:.2f}±{all_results[key]['std']:.2f})")
                continue
            
            print(f"\n{'='*50}")
            print(f"Running: {method} IPC{ipc} {label_type}")
            print(f"{'='*50}")
            
            images, labels = load_dcbench_data(method, ipc)
            print(f"  Data: {images.shape}")
            
            soft_labels = None
            if label_type == 'SL':
                soft_labels = generate_soft_labels(images, device)
            
            accs = []
            for run in range(num_runs):
                seed = run * 42
                t0 = time.time()
                
                if label_type == 'HL':
                    acc = train_hl(images, labels, testloader, device, seed)
                else:
                    acc = train_sl(images, soft_labels, testloader, device, seed)
                
                elapsed = time.time() - t0
                accs.append(acc)
                print(f"  Run {run+1}: {acc:.2f}% ({elapsed:.1f}s)")
            
            result = {
                'method': method,
                'ipc': ipc,
                'label_type': label_type,
                'accs': accs,
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
            }
            
            all_results[key] = result
            
            # Save after each experiment
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"  => {np.mean(accs):.2f}±{np.std(accs):.2f}%")
    
    print("\n\nDone! Results saved to", results_file)


if __name__ == '__main__':
    main()
