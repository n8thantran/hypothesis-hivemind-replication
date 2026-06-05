"""
Train a high-quality teacher model on CIFAR-100 for soft label generation.
Uses DSA augmentation and longer training to get a strong teacher.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from convnet import ConvNet
from data_utils import get_cifar100_tensors
from dsa import DiffAugment
from train_eval import evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_teacher(train_images, train_labels, test_images, test_labels,
                  epochs=300, lr=0.01, seed=0):
    """Train a teacher model on full CIFAR-100 with DSA augmentation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=151, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    n_train = len(train_images)
    batch_size = 256
    
    best_acc = 0
    best_state = None
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch_imgs = train_images[idx].to(device)
            batch_labels = train_labels[idx].to(device)
            
            # Apply DSA augmentation (same as student training)
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
            acc = evaluate(model, test_images, test_labels, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches:.4f}, Test Acc: {acc:.2f}% (best: {best_acc:.2f}%)")
    
    # Final eval
    acc = evaluate(model, test_images, test_labels, device)
    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Load best model
    model.load_state_dict(best_state)
    print(f"Teacher final best accuracy: {best_acc:.2f}%")
    return model, best_acc


def generate_soft_labels_for_images(teacher, images, batch_size=256):
    """Generate soft labels (logits) for given images using teacher model."""
    teacher.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            logits = teacher(batch)
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


if __name__ == '__main__':
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Train ensemble of 3 teachers
    num_teachers = 3
    all_logits_train = []
    
    for t in range(num_teachers):
        print(f"\n{'='*60}")
        print(f"Training teacher {t+1}/{num_teachers}")
        print(f"{'='*60}")
        teacher, acc = train_teacher(train_images, train_labels, test_images, test_labels,
                                     epochs=300, seed=t)
        
        # Generate logits for training set
        logits = generate_soft_labels_for_images(teacher, train_images)
        all_logits_train.append(logits)
        
        # Save individual teacher
        torch.save(teacher.state_dict(), f'teacher_{t}.pt')
        print(f"Teacher {t+1} saved (acc={acc:.2f}%)")
        
        del teacher
        torch.cuda.empty_cache()
    
    # Average logits across teachers
    avg_logits = torch.stack(all_logits_train).mean(dim=0)
    torch.save(avg_logits, 'soft_labels_v2.pt')
    print(f"\nSoft labels saved: {avg_logits.shape}")
    print(f"Logit range: [{avg_logits.min():.2f}, {avg_logits.max():.2f}]")
