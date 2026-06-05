"""
Optimized Distribution Matching distillation.
Key: batch all classes together in single forward passes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_cifar100_tensors

DEVICE = 'cuda'
NUM_CLASSES = 100
DSA_STRATEGY = 'color_crop_cutout_flip_scale_rotate'


def distill_dm(train_images, train_labels, ipc, num_iters=20000, lr=1.0, seed=0):
    """
    Distribution Matching with optimized batched processing.
    Matches the original DM paper: 20000 iterations, lr=1.0 for images.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize synthetic data from random real images
    syn_images_list = []
    syn_labels_list = []
    for c in range(NUM_CLASSES):
        cls_idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(cls_idx))[:ipc]
        syn_images_list.append(train_images[cls_idx[perm]].clone())
        syn_labels_list.append(torch.full((ipc,), c, dtype=torch.long))
    
    syn_images = torch.cat(syn_images_list, dim=0).to(DEVICE).requires_grad_(True)
    syn_labels = torch.cat(syn_labels_list, dim=0).to(DEVICE)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr, momentum=0.5)
    
    # Organize real data by class for efficient sampling
    class_indices = {}
    for c in range(NUM_CLASSES):
        class_indices[c] = (train_labels == c).nonzero(as_tuple=True)[0]
    
    print(f"DM distillation: {num_iters} iterations, IPC={ipc}, lr={lr}")
    t0 = time.time()
    
    for it in range(num_iters):
        # Sample a new random network each iteration
        model = ConvNet(num_classes=NUM_CLASSES, channel=3, im_size=(32, 32)).to(DEVICE)
        model.eval()
        
        # Sample real data: batch_real per class
        batch_real = 256
        real_batch_list = []
        real_labels_list = []
        for c in range(NUM_CLASSES):
            idx = class_indices[c]
            perm = torch.randperm(len(idx))[:batch_real]
            real_batch_list.append(train_images[idx[perm]])
            real_labels_list.append(torch.full((min(batch_real, len(idx)),), c, dtype=torch.long))
        
        # Concatenate all real data
        real_batch = torch.cat(real_batch_list, dim=0).to(DEVICE)
        real_labels_batch = torch.cat(real_labels_list, dim=0).to(DEVICE)
        
        # Apply DSA augmentation
        real_aug = DiffAugment(real_batch, strategy=DSA_STRATEGY)
        syn_aug = DiffAugment(syn_images, strategy=DSA_STRATEGY)
        
        # Forward pass - all at once
        with torch.no_grad():
            real_feat = model.embed(real_aug)
        syn_feat = model.embed(syn_aug)
        
        # Per-class mean matching loss
        loss = torch.tensor(0.0, device=DEVICE)
        for c in range(NUM_CLASSES):
            real_mask = (real_labels_batch == c)
            syn_mask = (syn_labels == c)
            real_mean = real_feat[real_mask].mean(0)
            syn_mean = syn_feat[syn_mask].mean(0)
            loss += torch.sum((real_mean - syn_mean) ** 2)
        
        loss /= NUM_CLASSES
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 2000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (it + 1) * (num_iters - it - 1)
            print(f"  Iter {it+1}/{num_iters}, Loss: {loss.item():.6f}, "
                  f"Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
    
    print(f"DM distillation complete in {(time.time()-t0)/60:.1f} minutes")
    return syn_images.detach().cpu(), syn_labels.cpu()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipc', type=int, default=10)
    parser.add_argument('--iters', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1.0)
    args = parser.parse_args()
    
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    images, labels = distill_dm(train_images, train_labels, args.ipc,
                                 num_iters=args.iters, lr=args.lr)
    
    save_path = f'/workspace/distilled_dm_ipc{args.ipc}_final.pt'
    torch.save({'images': images, 'labels': labels}, save_path)
    print(f"Saved to {save_path}")
