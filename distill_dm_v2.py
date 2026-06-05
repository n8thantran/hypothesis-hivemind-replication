"""
Distribution Matching (DM) - Corrected implementation.
Key fix: process classes in smaller batches to avoid OOM,
use proper batch sizes per class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices


def distribution_matching(train_images, train_labels, num_classes=100, ipc=10,
                          channel=3, im_size=(32, 32), device='cuda',
                          iterations=20000, lr_img=1.0,
                          dsa_strategy='color_crop_cutout_flip_scale_rotate',
                          seed=0):
    """
    DM: Match distributions of real and synthetic data in feature space.
    Uses random networks sampled each iteration.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize synthetic images from random real images
    class_indices = get_class_indices(train_labels, num_classes)
    
    syn_images_list = []
    syn_labels_list = []
    for c in range(num_classes):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images_list.append(train_images[indices[p]].clone())
            syn_labels_list.append(c)
    
    syn_images = torch.stack(syn_images_list).to(device).requires_grad_(True)
    syn_labels = torch.tensor(syn_labels_list, dtype=torch.long, device=device)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    # Pre-organize real images by class on CPU
    real_by_class = []
    for c in range(num_classes):
        real_by_class.append(train_images[class_indices[c]])
    
    n_syn = num_classes * ipc
    print(f"DM: Synthesizing {n_syn} images ({ipc} per class), {iterations} iterations...")
    
    start_time = time.time()
    
    for it in range(iterations):
        # Sample a new random network each iteration
        net = ConvNet(num_classes=num_classes, channel=channel, im_size=im_size).to(device)
        net.eval()
        
        loss = torch.tensor(0.0, device=device)
        
        # Process classes in groups to manage memory
        # Sample a subset of classes each iteration for efficiency
        n_classes_per_iter = min(num_classes, 10)  # Process 10 classes at a time
        class_order = np.random.permutation(num_classes)
        
        for group_start in range(0, num_classes, n_classes_per_iter):
            group_classes = class_order[group_start:group_start + n_classes_per_iter]
            
            # Get real samples for this group
            real_batch = []
            real_sizes = []
            batch_per_class = 256  # samples per class from real data
            
            for c in group_classes:
                n_available = len(real_by_class[c])
                n_sample = min(batch_per_class, n_available)
                perm = torch.randperm(n_available)[:n_sample]
                real_batch.append(real_by_class[c][perm])
                real_sizes.append(n_sample)
            
            all_real = torch.cat(real_batch, dim=0).to(device)
            all_real_aug = DiffAugment(all_real, strategy=dsa_strategy)
            
            with torch.no_grad():
                real_feat = net.embed(all_real_aug)
            
            # Get synthetic samples for this group
            syn_batch = []
            for c in group_classes:
                mask = syn_labels == c
                syn_batch.append(syn_images[mask])
            
            all_syn = torch.cat(syn_batch, dim=0)
            all_syn_aug = DiffAugment(all_syn, strategy=dsa_strategy)
            syn_feat = net.embed(all_syn_aug)
            
            # Compute per-class mean matching loss
            real_offset = 0
            syn_offset = 0
            for i, c in enumerate(group_classes):
                real_mean = real_feat[real_offset:real_offset + real_sizes[i]].mean(0)
                syn_mean = syn_feat[syn_offset:syn_offset + ipc].mean(0)
                loss = loss + torch.mean((real_mean - syn_mean) ** 2)
                real_offset += real_sizes[i]
                syn_offset += ipc
            
            del all_real, all_real_aug, real_feat, all_syn_aug, syn_feat
        
        loss = loss / num_classes
        
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        
        del net
        
        if (it + 1) % 500 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (it + 1) * (iterations - it - 1)
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.6f}, "
                  f"Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")
    
    total_time = time.time() - start_time
    print(f"DM distillation complete in {total_time:.0f}s")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


if __name__ == '__main__':
    import sys
    
    ipc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    
    print(f"Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    syn_images, syn_labels = distribution_matching(
        train_images, train_labels, ipc=ipc, iterations=iterations
    )
    
    save_path = f'/workspace/distilled_dm_ipc{ipc}.pt'
    torch.save({'images': syn_images, 'labels': syn_labels}, save_path)
    print(f"Saved to {save_path}")
