"""
Train the best possible ConvNet-D3 teacher on CIFAR-100.
Uses standard augmentation (RandomCrop, HFlip) + longer training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from convnet import ConvNet
from data_utils import get_cifar100_tensors
from dsa import DiffAugment
import time

def train_teacher():
    device = 'cuda'
    
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    
    # Use SGD with momentum, standard training setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 256
    n_train = len(train_images)
    best_acc = 0
    
    for epoch in range(2000):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch_imgs = train_images[idx]
            batch_labels = train_labels[idx]
            
            # Apply DSA augmentation (same as used in evaluation)
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i in range(0, len(test_images), 512):
                    batch = test_images[i:i+512]
                    labels = test_labels[i:i+512]
                    outputs = model(batch)
                    _, pred = outputs.max(1)
                    correct += pred.eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100.0 * correct / total
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/2000, Loss: {epoch_loss/n_batches:.4f}, Test Acc: {acc:.2f}%, LR: {lr:.6f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'teacher_best_v2.pt')
                print(f"  -> New best: {best_acc:.2f}%")
    
    print(f"\nBest teacher accuracy: {best_acc:.2f}%")
    
    # Load best model and generate logits
    model.load_state_dict(torch.load('teacher_best_v2.pt'))
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, n_train, 512):
            batch = train_images[i:i+512]
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    torch.save(all_logits, 'soft_labels_best.pt')
    print(f"Saved soft labels: {all_logits.shape}")
    
    # Verify
    probs = F.softmax(all_logits, dim=1)
    preds = probs.argmax(dim=1)
    train_acc = (preds == train_labels.cpu()).float().mean().item() * 100
    print(f"Teacher train accuracy: {train_acc:.2f}%")

if __name__ == '__main__':
    train_teacher()
