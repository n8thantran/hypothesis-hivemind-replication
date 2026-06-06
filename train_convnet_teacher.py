"""
Train a strong ConvNet-D3 teacher on CIFAR-100 with better hyperparameters.
The original teacher only got 59%. Let's try harder.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from convnet import get_convnet_d3
from dsa import DiffAugment

def main():
    device = 'cuda'
    
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    # Standard augmentation (not DSA)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    model = get_convnet_d3(num_classes=100).to(device)
    
    # Try SGD with cosine annealing for 500 epochs
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    start_time = time.time()
    
    for epoch in range(500):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 25 == 0 or epoch == 499:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/500: {acc:.2f}% (best: {best_acc:.2f}%) [{elapsed:.0f}s]")
            
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'accuracy': acc,
                    'epoch': epoch
                }, 'teacher_convnet_strong.pt')
    
    print(f"\nBest accuracy: {best_acc:.2f}%")
    
    # Generate soft labels
    print("\nGenerating soft labels...")
    model.load_state_dict(torch.load('teacher_convnet_strong.pt', weights_only=False)['model_state_dict'])
    model.eval()
    
    train_noaug = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=test_transform)
    loader = DataLoader(train_noaug, batch_size=256, shuffle=False, num_workers=4)
    
    all_logits = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
    
    soft_labels = torch.cat(all_logits, dim=0)
    torch.save(soft_labels, 'soft_labels_convnet_strong.pt')
    print(f"Saved soft labels: {soft_labels.shape}")


if __name__ == '__main__':
    main()
