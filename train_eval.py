"""
Training and evaluation pipeline for dataset distillation experiments.
Supports both Hard Label (HL) and Soft Label (SL) settings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from dsa import DiffAugment


def train_and_evaluate(train_images, train_labels, test_images, test_labels,
                       model_fn, num_classes=100, device='cuda',
                       label_type='hard', soft_labels=None,
                       epochs=300, batch_size=256, seed=0,
                       verbose=True):
    """
    Train a model on the given training data and evaluate on test data.
    
    Args:
        train_images: (N, C, H, W) tensor
        train_labels: (N,) tensor of hard labels
        test_images: (N_test, C, H, W) tensor
        test_labels: (N_test,) tensor
        model_fn: callable that returns a new model
        label_type: 'hard' or 'soft'
        soft_labels: (N, num_classes) tensor of soft labels (for SL setting)
        epochs: number of training epochs
        batch_size: training batch size
        seed: random seed
        verbose: print progress
    
    Returns:
        test_accuracy (float)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model_fn().to(device)
    
    if label_type == 'hard':
        # HL setting: SGD, lr=1e-2, StepLR@151, CE loss
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
    else:
        # SL setting: AdamW, lr=1e-3, CosineAnnealing, KL-Div with T=20
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        temperature = 20.0
    
    # Create dataset
    n_train = len(train_images)
    
    # For small datasets, we may need to iterate multiple times per epoch
    # to match the effective number of gradient updates
    steps_per_epoch = max(1, n_train // batch_size)
    if n_train < batch_size:
        effective_batch_size = n_train
    else:
        effective_batch_size = batch_size
    
    model.train()
    for epoch in range(epochs):
        # Shuffle indices
        perm = torch.randperm(n_train)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_train, effective_batch_size):
            idx = perm[i:i+effective_batch_size]
            batch_imgs = train_images[idx].to(device)
            
            # Apply DSA augmentation
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            
            if label_type == 'hard':
                batch_labels = train_labels[idx].to(device)
                loss = criterion(outputs, batch_labels)
            else:
                # KL-Div loss with temperature
                batch_soft = soft_labels[idx].to(device)
                log_probs = F.log_softmax(outputs / temperature, dim=1)
                targets = F.softmax(batch_soft / temperature, dim=1)
                loss = F.kl_div(log_probs, targets, reduction='batchmean') * (temperature ** 2)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            # Quick eval
            acc = evaluate(model, test_images, test_labels, device)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/max(n_batches,1):.4f}, Test Acc: {acc:.2f}%")
    
    # Final evaluation
    acc = evaluate(model, test_images, test_labels, device)
    return acc


def evaluate(model, test_images, test_labels, device='cuda', batch_size=512):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            batch_imgs = test_images[i:i+batch_size].to(device)
            batch_labels = test_labels[i:i+batch_size].to(device)
            outputs = model(batch_imgs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_labels).sum().item()
            total += batch_labels.size(0)
    
    model.train()
    return 100.0 * correct / total


def run_experiment(train_images, train_labels, test_images, test_labels,
                   model_fn, num_classes=100, device='cuda',
                   label_type='hard', soft_labels=None,
                   epochs=300, batch_size=256,
                   num_runs=3, verbose=True):
    """
    Run multiple trials and return mean ± std accuracy.
    """
    accs = []
    for run in range(num_runs):
        if verbose:
            print(f"\n--- Run {run+1}/{num_runs} ---")
        acc = train_and_evaluate(
            train_images, train_labels, test_images, test_labels,
            model_fn, num_classes, device, label_type, soft_labels,
            epochs, batch_size, seed=run, verbose=verbose
        )
        accs.append(acc)
        if verbose:
            print(f"Run {run+1} accuracy: {acc:.2f}%")
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    if verbose:
        print(f"\nFinal: {mean_acc:.2f} ± {std_acc:.2f}%")
    return mean_acc, std_acc


if __name__ == '__main__':
    from convnet import ConvNet
    from data_utils import get_cifar100_tensors, random_select
    
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Quick test: random subset IPC=10 with hard labels, few epochs
    print("\nQuick test: Random IPC=10, 10 epochs...")
    selected = random_select(train_labels, ipc=10, seed=0)
    sub_images = train_images[selected]
    sub_labels = train_labels[selected]
    
    model_fn = lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))
    acc = train_and_evaluate(
        sub_images, sub_labels, test_images, test_labels,
        model_fn, epochs=10, verbose=True, device='cuda'
    )
    print(f"Test accuracy (10 epochs): {acc:.2f}%")
