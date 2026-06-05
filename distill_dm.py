"""
Distribution Matching (DM) - Zhao & Bilen (2023)
Synthesizes a distilled dataset by matching feature distributions between
real and synthetic data using randomly initialized networks.

Optimized for speed: batch all classes together, minimize overhead.
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
    DM: Match distributions of real and synthetic data in feature space.
    Optimized: process all classes in a single forward pass.
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
    
    # Pre-organize real images by class (keep on CPU to save GPU memory)
    real_by_class = []
    for c in range(num_classes):
        real_by_class.append(train_images[class_indices[c]])
    
    print(f"DM: Synthesizing {num_classes * ipc} images ({ipc} per class)...")
    
    for it in range(iterations):
        # Sample a new random network
        net = ConvNet(num_classes=num_classes, channel=channel, im_size=im_size).to(device)
        net.eval()
        
        # Sample real images for each class and batch them
        real_samples = []
        real_class_sizes = []
        batch_per_class = min(batch_real, min(len(c) for c in real_by_class))
        
        for c in range(num_classes):
            perm = torch.randperm(len(real_by_class[c]))[:batch_per_class]
            real_samples.append(real_by_class[c][perm])
            real_class_sizes.append(batch_per_class)
        
        # Batch all real images together for one forward pass
        all_real = torch.cat(real_samples, dim=0).to(device)
        all_real_aug = DiffAugment(all_real, strategy=dsa_strategy)
        
        with torch.no_grad():
            all_real_feat = net.embed(all_real_aug)
        
        # Compute mean features per class for real data
        real_mean_feats = []
        offset = 0
        for c in range(num_classes):
            real_mean_feats.append(all_real_feat[offset:offset+real_class_sizes[c]].mean(0))
            offset += real_class_sizes[c]
        real_mean_feats = torch.stack(real_mean_feats)  # (num_classes, feat_dim)
        
        # Forward pass on all synthetic images
        all_syn_aug = DiffAugment(syn_images, strategy=dsa_strategy)
        all_syn_feat = net.embed(all_syn_aug)
        
        # Compute mean features per class for synthetic data
        syn_mean_feats = []
        for c in range(num_classes):
            syn_mask = syn_labels == c
            syn_mean_feats.append(all_syn_feat[syn_mask].mean(0))
        syn_mean_feats = torch.stack(syn_mean_feats)  # (num_classes, feat_dim)
        
        # Loss: MSE between mean features
        loss = torch.mean((real_mean_feats - syn_mean_feats) ** 2)
        
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        
        # Cleanup to free GPU memory
        del net, all_real, all_real_aug, all_real_feat, all_syn_aug, all_syn_feat
        
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
