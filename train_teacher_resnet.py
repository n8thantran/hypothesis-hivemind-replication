"""Train ResNet-18 teacher on CIFAR-100 for soft label generation."""
import torch, torchvision, torchvision.transforms as transforms
import torch.nn as nn, torch.optim as optim
import os

device = 'cuda'
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

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

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=train_transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=test_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

model = torchvision.models.resnet18(num_classes=100).to(device)
# Adapt for 32x32 images  
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
model.maxpool = nn.Identity()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 150], gamma=0.2)

best_acc = 0
for epoch in range(170):
    model.train()
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    if epoch % 20 == 0 or epoch >= 160:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = 100*correct/total
        print(f'Epoch {epoch}: {acc:.2f}%')
        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state_dict': model.state_dict(), 'acc': acc}, 'teacher_resnet18.pt')

print(f'Best teacher accuracy: {best_acc:.2f}%')
