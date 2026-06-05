"""
Distribution Matching (DM) - Zhao & Bilen (2023)
Synthesizes a distilled dataset by matching feature distributions between
real and synthetic data using randomly initialized networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices


def distribution_matching(train_images, train_labels, num_classes=100, ipc=10,
                          channel=3, im_size=(32, 32), device='cuda',
                          iterations=20000, lr_img=1.0, batch_real=256,
                          dsa_strategy='color_crop_cutout_flip_scale_rotate',
                          seed=0):
    """
    DM: Match distributions of real and synthetic data in feature space
    of randomly initialized networks.
    
    Args:
        train_images: (N, C, H, W) full training set
        train_labels: (N,) labels
        num_classes: number of classes
        ipc: images per class
        iterations: number of optimization iterations
        lr_img: learning rate for synthetic images
    
    Returns:
        syn_images: (num_classes * ipc, C, H, W) synthetic images
        syn_labels: (num_classes * ipc,) labels
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize synthetic images from random real images
    class_indices = get_class_indices(train_labels, num_classes)
    
    syn_images = []
    syn_labels = []
    for c in range(num_classes):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images.append(train_images[indices[p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=device)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    print(f"DM: Synthesizing {num_classes * ipc} images ({ipc} per class)...")
    
    for it in range(iterations):
        # Sample a new random network each iteration
        net = ConvNet(num_classes=num_classes, channel=channel, im_size=im_size).to(device)
        net.eval()  # Use random init, no training
        
        loss = torch.tensor(0.0, device=device)
        
        for c in range(num_classes):
            # Get real images for this class
            real_indices = class_indices[c]
            perm = np.random.permutation(len(real_indices))[:batch_real]
            real_batch = train_images[np.array(real_indices)[perm]].to(device)
            
            # Get synthetic images for this class
            syn_mask = syn_labels == c
            syn_batch = syn_images[syn_mask]
            
            # Apply DSA augmentation
            real_aug = DiffAugment(real_batch, strategy=dsa_strategy)
            syn_aug = DiffAugment(syn_batch, strategy=dsa_strategy)
            
            # Get features (embeddings before classifier)
            with torch.no_grad():
                real_feat = net.embed(real_aug)
            syn_feat = net.embed(syn_aug)
            
            # Match mean features
            loss += torch.mean((real_feat.mean(0) - syn_feat.mean(0)) ** 2)
        
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        
        if (it + 1) % 1000 == 0:
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


if __name__ == '__main__':
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Quick test with fewer iterations
    syn_images, syn_labels = distribution_matching(
        train_images, train_labels, ipc=10, iterations=100
    )
    print(f"Synthetic images shape: {syn_images.shape}")
    print(f"Synthetic labels shape: {syn_labels.shape}")
