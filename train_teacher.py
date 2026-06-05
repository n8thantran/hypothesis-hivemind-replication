"""
Train a strong ConvNet-D3 teacher model on CIFAR-100 for soft label generation.
Uses standard CIFAR augmentation (RandomCrop + HFlip) for better generalization.
Target: 65%+ accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from convnet import ConvNet

def train_teacher(epochs=2000, lr=0.1, batch_size=128, device='cuda'):
    """Train a strong teacher model on CIFAR-100."""
    
    # Standard CIFAR-100 augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='/workspace/data', train=True, 
                                              download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                               shuffle=True, num_workers=4, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR100(root='/workspace/data', train=False, 
                                             download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, 
                                              shuffle=False, num_workers=4, pin_memory=True)
    
    # ConvNet-D3 teacher
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    
    # SGD with momentum, cosine annealing
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_imgs, batch_labels in trainloader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = F.cross_entropy(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        scheduler.step()
        train_acc = 100. * correct / total
        
        # Evaluate every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_imgs, batch_labels in testloader:
                    batch_imgs = batch_imgs.to(device)
                    batch_labels = batch_labels.to(device)
                    outputs = model(batch_imgs)
                    _, predicted = outputs.max(1)
                    total += batch_labels.size(0)
                    correct += predicted.eq(batch_labels).sum().item()
            test_acc = 100. * correct / total
            
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), '/workspace/teacher_best.pt')
                print(f"  -> New best: {best_acc:.2f}%")
    
    print(f"\nBest test accuracy: {best_acc:.2f}%")
    return model, best_acc


def generate_soft_labels_from_teacher(teacher_path='/workspace/teacher_best.pt', device='cuda'):
    """Generate soft labels (logits) for all training images using the teacher."""
    from data_utils import get_cifar100_tensors
    
    # Load teacher
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    model.load_state_dict(torch.load(teacher_path, map_location=device))
    model.eval()
    
    # Load data (normalized)
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Generate logits
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(train_images), 256):
            batch = train_images[i:i+256].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    
    # Verify teacher accuracy on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), 256):
            batch = test_images[i:i+256].to(device)
            labels = test_labels[i:i+256]
            outputs = model(batch)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    test_acc = 100. * correct / total
    print(f"Teacher test accuracy (on normalized data): {test_acc:.2f}%")
    
    # Save logits
    torch.save(logits, '/workspace/teacher_logits.pt')
    print(f"Saved teacher logits: {logits.shape}")
    
    return logits


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'labels':
        # Just generate labels from existing teacher
        generate_soft_labels_from_teacher()
    else:
        # Train teacher
        model, best_acc = train_teacher(epochs=2000, lr=0.1)
        # Generate labels
        generate_soft_labels_from_teacher()
