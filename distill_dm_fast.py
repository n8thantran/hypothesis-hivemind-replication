"""
Fast Distribution Matching (DM) - optimized for speed.
Key optimizations:
- Reuse network for multiple iterations before re-sampling
- Smaller batch sizes
- No DSA during distillation (optional)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_class_indices


def distribution_matching_fast(train_images, train_labels, num_classes=100, ipc=10,
                               channel=3, im_size=(32, 32), device='cuda',
                               iterations=5000, lr_img=1.0, batch_real=64,
                               net_reuse=10,
                               dsa_strategy='color_crop_cutout_flip_scale_rotate',
                               seed=0):
    """
    Fast DM: reuse networks for multiple iterations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels, num_classes)
    
    # Initialize synthetic images from random real images
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
    
    # Pre-organize real images by class on CPU
    real_by_class = []
    for c in range(num_classes):
        real_by_class.append(train_images[class_indices[c]])
    
    print(f"DM Fast: Synthesizing {num_classes * ipc} images ({ipc} per class), {iterations} iters...")
    
    net = None
    for it in range(iterations):
        # Sample a new random network every net_reuse iterations
        if it % net_reuse == 0:
            net = ConvNet(num_classes=num_classes, channel=channel, im_size=im_size).to(device)
            net.eval()
        
        # Sample real images for each class
        real_samples = []
        batch_per_class = batch_real
        for c in range(num_classes):
            n = len(real_by_class[c])
            perm = torch.randperm(n)[:batch_per_class]
            real_samples.append(real_by_class[c][perm])
        
        all_real = torch.cat(real_samples, dim=0).to(device)
        all_real_aug = DiffAugment(all_real, strategy=dsa_strategy)
        
        with torch.no_grad():
            all_real_feat = net.embed(all_real_aug)
        
        # Compute mean features per class for real data
        real_mean_feats = []
        offset = 0
        for c in range(num_classes):
            real_mean_feats.append(all_real_feat[offset:offset+batch_per_class].mean(0))
            offset += batch_per_class
        real_mean_feats = torch.stack(real_mean_feats)
        
        # Forward pass on all synthetic images
        all_syn_aug = DiffAugment(syn_images, strategy=dsa_strategy)
        all_syn_feat = net.embed(all_syn_aug)
        
        # Compute mean features per class for synthetic data
        syn_mean_feats = []
        for c in range(num_classes):
            syn_mask = syn_labels == c
            syn_mean_feats.append(all_syn_feat[syn_mask].mean(0))
        syn_mean_feats = torch.stack(syn_mean_feats)
        
        # Loss: MSE between mean features
        loss = torch.mean((real_mean_feats - syn_mean_feats) ** 2)
        
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        
        if (it + 1) % 500 == 0:
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()
