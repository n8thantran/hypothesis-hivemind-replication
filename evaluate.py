"""
Clean evaluation module matching paper's exact hyperparameters.

Paper's Table (tab:stage3_hyper) for Small-scale (CIFAR-100):
- HL: 300 epochs, SGD, lr=0.01, StepLR@151, batch=256, DSA, CE loss
- SL: 300 epochs, AdamW, lr=1e-3, Cosine scheduler, batch=256, DSA, KL-Div(T=20)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from convnet import get_convnet_d3
from dsa import DiffAugment


def evaluate_hl(images, labels, test_images, test_labels, 
                num_classes=100, epochs=300, device='cuda', seed=0):
    """
    Evaluate with Hard Labels (HL) setting.
    300 epochs, SGD, lr=0.01, StepLR@151 (gamma=0.5), batch=256, DSA, CE loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.5)
    
    images = images.to(device)
    labels = labels.to(device)
    
    n = len(images)
    batch_size = min(256, n)
    
    model.train()
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch_imgs = images[idx]
            batch_labels = labels[idx]
            
            # Apply DSA augmentation
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = F.cross_entropy(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), 256):
            batch = test_images[i:i+256].to(device)
            batch_labels = test_labels[i:i+256].to(device)
            outputs = model(batch)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_labels).sum().item()
            total += batch_labels.size(0)
    
    acc = 100.0 * correct / total
    return acc


def evaluate_sl(images, soft_labels, test_images, test_labels,
                num_classes=100, epochs=300, temperature=20.0, device='cuda', seed=0):
    """
    Evaluate with Soft Labels (SL) setting.
    300 epochs, AdamW, lr=1e-3, Cosine scheduler, batch=256, DSA, KL-Div(T=20).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = get_convnet_d3(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    images = images.to(device)
    soft_labels = soft_labels.to(device)
    
    # Convert soft labels (logits) to soft targets with temperature
    soft_targets = F.softmax(soft_labels / temperature, dim=1)
    
    n = len(images)
    batch_size = min(256, n)
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch_imgs = images[idx]
            batch_targets = soft_targets[idx]
            
            # Apply DSA augmentation
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            
            # KL-Div loss with temperature
            log_probs = F.log_softmax(outputs / temperature, dim=1)
            loss = F.kl_div(log_probs, batch_targets, reduction='batchmean') * (temperature ** 2)
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), 256):
            batch = test_images[i:i+256].to(device)
            batch_labels = test_labels[i:i+256].to(device)
            outputs = model(batch)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_labels).sum().item()
            total += batch_labels.size(0)
    
    acc = 100.0 * correct / total
    return acc


def evaluate_multiple_runs(images, labels_or_soft, test_images, test_labels,
                           mode='hl', num_runs=3, **kwargs):
    """Run evaluation multiple times and return mean ± std."""
    accs = []
    for run in range(num_runs):
        if mode == 'hl':
            acc = evaluate_hl(images, labels_or_soft, test_images, test_labels,
                            seed=run*42, **kwargs)
        else:
            acc = evaluate_sl(images, labels_or_soft, test_images, test_labels,
                            seed=run*42, **kwargs)
        accs.append(acc)
        print(f"  Run {run+1}/{num_runs}: {acc:.2f}%")
    
    mean = np.mean(accs)
    std = np.std(accs)
    return mean, std, accs


if __name__ == '__main__':
    # Quick test
    print("Testing evaluation module...")
    model = get_convnet_d3()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"Model output shape: {out.shape}")
    print("Evaluation module loaded successfully.")
