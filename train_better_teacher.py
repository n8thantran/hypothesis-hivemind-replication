"""Train a better teacher model for soft label generation."""
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
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    
    # Use SGD with cosine annealing for better convergence
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 256
    n_train = len(train_images)
    best_acc = 0
    
    for epoch in range(500):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch_imgs = train_images[idx].to(device)
            batch_labels = train_labels[idx].to(device)
            
            # Apply DSA augmentation
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
                    batch = test_images[i:i+512].to(device)
                    labels = test_labels[i:i+512].to(device)
                    outputs = model(batch)
                    _, pred = outputs.max(1)
                    correct += pred.eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100.0 * correct / total
            print(f"Epoch {epoch+1}/500, Loss: {epoch_loss/n_batches:.4f}, Test Acc: {acc:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                torch.save({'state_dict': model.state_dict(), 'accuracy': acc}, 'teacher_best.pt')
                print(f"  -> New best: {acc:.2f}%")
    
    print(f"\nBest teacher accuracy: {best_acc:.2f}%")
    
    # Generate soft labels with best model
    checkpoint = torch.load('teacher_best.pt', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, n_train, 512):
            batch = train_images[i:i+512].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    soft_labels = torch.cat(all_logits, dim=0)
    torch.save(soft_labels, 'soft_labels_v3.pt')
    print(f"Saved soft labels: {soft_labels.shape}")
    
    return best_acc

if __name__ == '__main__':
    acc = train_teacher()
