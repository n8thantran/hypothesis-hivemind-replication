"""
Data utilities for CIFAR-100 dataset distillation experiments.
Includes: data loading, coreset selection (Random, K-centers), soft label generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms


# CIFAR-100 mean and std
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


def get_cifar100_tensors(data_path='/workspace/data/hf_cache'):
    """Get CIFAR-100 as tensors (images, labels) using HuggingFace datasets."""
    ds = load_dataset('uoft-cs/cifar100', cache_dir=data_path)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    
    def process_split(split):
        images = []
        labels = []
        for example in split:
            img = example['img']
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            images.append(transform(img))
            labels.append(example['fine_label'])
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)
    
    print("Processing train split...")
    train_images, train_labels = process_split(ds['train'])
    print("Processing test split...")
    test_images, test_labels = process_split(ds['test'])
    
    return train_images, train_labels, test_images, test_labels


def get_class_indices(labels, num_classes=100):
    """Get indices for each class."""
    class_indices = defaultdict(list)
    for i in range(len(labels)):
        class_indices[int(labels[i])].append(i)
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


def k_centers_select(images, labels, ipc, num_classes=100, seed=0,
                     use_features=True, feature_model=None, device='cuda'):
    """
    K-Centers coreset selection: greedy farthest-first traversal per class.
    
    If use_features=True, extracts features from a ConvNet model (random init if 
    feature_model is None, or pretrained if provided) - much better coverage.
    If use_features=False, uses pixel-space (flattened images).
    """
    np.random.seed(seed)
    class_indices = get_class_indices(labels, num_classes)
    
    # Extract features if requested
    if use_features:
        features_all = _extract_features(images, feature_model, device)
    else:
        features_all = images.reshape(len(images), -1).numpy()
    
    selected = []
    for c in range(num_classes):
        indices = np.array(class_indices[c])
        
        if isinstance(features_all, np.ndarray):
            features = features_all[indices]
        else:
            features = features_all[indices].numpy()
        
        # Greedy farthest-first traversal
        chosen = []
        # Start with random point
        first = np.random.randint(len(indices))
        chosen.append(first)
        
        if ipc == 1:
            selected.append(int(indices[first]))
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
        
        selected.extend([int(indices[c_idx]) for c_idx in chosen])
    
    return sorted(selected)


def _extract_features(images, model=None, device='cuda'):
    """Extract features from images using a ConvNet model."""
    from convnet import ConvNet
    
    if model is None:
        # Use a randomly initialized model for feature extraction
        # This still gives better features than raw pixels due to architecture inductive biases
        # A pretrained model gives even better features
        model = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            feat = model.embed(batch)
            all_features.append(feat.cpu())
    
    features = torch.cat(all_features, dim=0)
    return features.numpy()


def generate_soft_labels(train_images, train_labels, model_fn, num_classes=100, 
                         device='cuda', num_models=3, epochs=200, seed=42):
    """
    Generate soft labels by training teacher models on full CIFAR-100 and
    averaging their softmax outputs.
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
    return avg_logits


if __name__ == '__main__':
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    print(f"Labels range: {train_labels.min()}-{train_labels.max()}")
    
    # Test random selection
    selected = random_select(train_labels, ipc=10)
    print(f"Random select IPC=10: {len(selected)} samples")
    
    # Test K-centers selection (feature-space)
    selected = k_centers_select(train_images, train_labels, ipc=10, use_features=True)
    print(f"K-centers select (feature) IPC=10: {len(selected)} samples")
    
    # Test K-centers selection (pixel-space)
    selected = k_centers_select(train_images, train_labels, ipc=10, use_features=False)
    print(f"K-centers select (pixel) IPC=10: {len(selected)} samples")
