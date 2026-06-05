"""Train teacher model quickly with good accuracy."""
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
    
    # Move all data to GPU for speed
    train_images_gpu = train_images.to(device)
    train_labels_gpu = train_labels.to(device)
    test_images_gpu = test_images.to(device)
    test_labels_gpu = test_labels.to(device)
    
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    
    # Higher LR with cosine annealing
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 512
    n_train = len(train_images)
    best_acc = 0
    
    start = time.time()
    for epoch in range(300):
        model.train()
        perm = torch.randperm(n_train, device=device)
        
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch_imgs = train_images_gpu[idx]
            batch_labels = train_labels_gpu[idx]
            
            # Apply DSA augmentation
            batch_imgs = DiffAugment(batch_imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % 25 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i in range(0, len(test_images_gpu), 1024):
                    batch = test_images_gpu[i:i+1024]
                    labels = test_labels_gpu[i:i+1024]
                    outputs = model(batch)
                    _, pred = outputs.max(1)
                    correct += pred.eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100.0 * correct / total
            elapsed = time.time() - start
            print(f"Epoch {epoch+1}/300, Acc: {acc:.2f}%, Time: {elapsed:.0f}s")
            
            if acc > best_acc:
                best_acc = acc
                torch.save({'state_dict': model.state_dict(), 'accuracy': acc}, 'teacher_best.pt')
    
    print(f"\nBest teacher accuracy: {best_acc:.2f}%")
    
    # Generate soft labels with best model
    checkpoint = torch.load('teacher_best.pt', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, n_train, 1024):
            batch = train_images_gpu[i:i+1024]
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    soft_labels = torch.cat(all_logits, dim=0)
    torch.save(soft_labels, 'soft_labels_v3.pt')
    print(f"Saved soft labels: {soft_labels.shape}")
    
    return best_acc

if __name__ == '__main__':
    acc = train_teacher()
