"""
Data utilities for CIFAR-100 dataset distillation experiments.
Includes: data loading, coreset selection (Random, K-centers), soft label generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import defaultdict
from sklearn.metrics import pairwise_distances
import os


# CIFAR-100 mean and std
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


def get_cifar100(data_path='./data'):
    """Load CIFAR-100 train and test sets."""
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, 
                                              download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, 
                                             download=True, transform=transform_test)
    return trainset, testset


def get_cifar100_tensors(data_path='./data'):
    """Get CIFAR-100 as tensors (images, labels) for the full training set."""
    trainset, testset = get_cifar100(data_path)
    
    # Convert to tensors
    train_images = []
    train_labels = []
    for img, label in trainset:
        train_images.append(img)
        train_labels.append(label)
    
    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)
    
    test_images = []
    test_labels = []
    for img, label in testset:
        test_images.append(img)
        test_labels.append(label)
    
    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels)
    
    return train_images, train_labels, test_images, test_labels


def get_class_indices(labels, num_classes=100):
    """Get indices for each class."""
    class_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_indices[int(label)].append(i)
    return class_indices


def random_select(labels, ipc, num_classes=100, seed=0):
    """Random coreset selection: IPC samples per class."""
    np.random.seed(seed)
    class_indices = get_class_indices(labels, num_classes)
    selected = []
    for c in range(num_classes):
        indices = class_indices[c]
        chosen = np.random.choice(indices, size=ipc, replace=False)
        selected.extend(chosen.tolist())
    return sorted(selected)


def k_centers_select(images, labels, ipc, num_classes=100, seed=0):
    """
    K-Centers coreset selection: greedy farthest-first traversal per class.
    Uses pixel-space features (flattened images).
    """
    np.random.seed(seed)
    class_indices = get_class_indices(labels, num_classes)
    selected = []
    
    for c in range(num_classes):
        indices = np.array(class_indices[c])
        # Use flattened images as features
        features = images[indices].reshape(len(indices), -1).numpy()
        
        # Greedy farthest-first traversal
        chosen = []
        # Start with random point
        first = np.random.randint(len(indices))
        chosen.append(first)
        
        if ipc == 1:
            selected.append(indices[first])
            continue
        
        # Compute distances from first point
        dists = np.full(len(indices), np.inf)
        
        for _ in range(ipc - 1):
            # Update distances
            last_chosen = chosen[-1]
            new_dists = np.sum((features - features[last_chosen:last_chosen+1]) ** 2, axis=1)
            dists = np.minimum(dists, new_dists)
            # Select farthest point
            next_idx = np.argmax(dists)
            chosen.append(next_idx)
        
        selected.extend(indices[chosen].tolist())
    
    return sorted(selected)


def generate_soft_labels(train_images, train_labels, model_fn, num_classes=100, 
                         device='cuda', num_models=3, epochs=200, seed=42):
    """
    Generate soft labels by training teacher models on full CIFAR-100 and 
    averaging their softmax outputs.
    
    For the SL setting, we need fixed soft labels from a teacher.
    We train ConvNet-D3 teachers on the full dataset and use their predictions.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    all_logits = []
    
    for m_idx in range(num_models):
        print(f"Training teacher model {m_idx+1}/{num_models}...")
        model = model_fn().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(train_images, train_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, 
                                              num_workers=0, pin_memory=True)
        
        model.train()
        for epoch in range(epochs):
            for batch_imgs, batch_labels in loader:
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_imgs)
                loss = F.cross_entropy(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}")
        
        # Get soft labels
        model.eval()
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(train_images), 256):
                batch = train_images[i:i+256].to(device)
                logits = model(batch)
                logits_list.append(logits.cpu())
        
        all_logits.append(torch.cat(logits_list, dim=0))
    
    # Average logits across teachers
    avg_logits = torch.stack(all_logits).mean(dim=0)
    
    # Convert to soft labels (probabilities)
    # Use temperature=20 as specified in the paper for SL training
    # But store raw logits so we can apply temperature during training
    return avg_logits


if __name__ == '__main__':
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    print(f"Labels range: {train_labels.min()}-{train_labels.max()}")
    
    # Test random selection
    selected = random_select(train_labels, ipc=10)
    print(f"Random select IPC=10: {len(selected)} samples")
    
    # Test K-centers selection
    selected = k_centers_select(train_images, train_labels, ipc=10)
    print(f"K-centers select IPC=10: {len(selected)} samples")
