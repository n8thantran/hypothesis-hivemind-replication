"""
Train a strong teacher model on CIFAR-100 for soft label generation.
Uses ConvNet-D3 with proper augmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from convnet import ConvNet
from data_utils import get_cifar100_tensors
from dsa import DiffAugment

def train_teacher(device='cuda', epochs=500, lr=0.01, batch_size=256, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    
    n_train = len(train_images)
    print(f"Training on {n_train} images for {epochs} epochs")
    
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = train_images[idx]
            labs = train_labels[idx]
            
            # Apply DSA augmentation
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i in range(0, len(test_images), 512):
                    imgs = test_images[i:i+512].to(device)
                    labs = test_labels[i:i+512].to(device)
                    outputs = model(imgs)
                    _, pred = outputs.max(1)
                    correct += pred.eq(labs).sum().item()
                    total += labs.size(0)
            acc = 100.0 * correct / total
            
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches:.4f}, "
                  f"Test Acc: {acc:.2f}%, Best: {best_acc:.2f}%, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}, Time: {elapsed:.0f}s")
    
    # Load best model
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Generate soft labels (logits) for the entire training set
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, n_train, 512):
            imgs = train_images[i:i+512]
            logits = model(imgs)
            all_logits.append(logits.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    
    # Save
    save_dict = {
        'model_state_dict': best_state,
        'logits': all_logits,
        'accuracy': best_acc,
    }
    torch.save(save_dict, '/workspace/teacher.pt')
    print(f"\nSaved teacher model with {best_acc:.2f}% accuracy")
    print(f"Logits shape: {all_logits.shape}")
    
    return model, all_logits

if __name__ == '__main__':
    train_teacher()
