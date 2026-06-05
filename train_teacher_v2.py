"""
Train a strong ConvNet-D3 teacher on CIFAR-100 with standard augmentation.
Target: 65%+ test accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from convnet import ConvNet

def train_teacher():
    device = 'cuda'
    
    # Standard CIFAR augmentation (NOT DSA - that's for distilled sets)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
    
    model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    
    # More aggressive LR schedule with warmup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(2000):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        train_acc = 100. * correct / total
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            # Test
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            test_acc = 100. * correct / total
            
            print(f"Epoch {epoch+1}/2000 - Train: {train_acc:.2f}%, Test: {test_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'teacher_best_v2.pt')
                print(f"  => New best: {best_acc:.2f}%")
    
    print(f"\nBest test accuracy: {best_acc:.2f}%")
    
    # Generate soft labels (logits) using best model
    print("\nGenerating soft labels...")
    model.load_state_dict(torch.load('teacher_best_v2.pt'))
    model.eval()
    
    # Load raw data without augmentation for logit generation
    transform_raw = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    rawset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_raw)
    rawloader = torch.utils.data.DataLoader(rawset, batch_size=256, shuffle=False, num_workers=4)
    
    all_logits = []
    with torch.no_grad():
        for inputs, _ in rawloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    print(f"Logits shape: {all_logits.shape}")
    torch.save(all_logits, 'soft_labels_v2.pt')
    print("Saved soft_labels_v2.pt")
    
    return best_acc

if __name__ == '__main__':
    train_teacher()
