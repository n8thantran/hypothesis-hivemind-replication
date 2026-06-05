"""
Dataset Condensation via Gradient Matching (DC) - Zhao et al. (2021)
Synthesizes a distilled dataset by matching gradients of real and synthetic data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices


def match_loss(gw_syn, gw_real, dis_metric='ours'):
    """Compute gradient matching loss."""
    dis = torch.tensor(0.0, device=gw_syn[0].device)
    
    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)
    elif dis_metric == 'mse':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += F.mse_loss(gwr, gws)
    
    return dis


def distance_wb(gwr, gws):
    """Distance between weight/bias gradients (from DC paper)."""
    shape = gwr.shape
    if len(shape) == 4:  # Conv layer
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:
        pass  # Already 2D
    elif len(shape) == 1:
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
    
    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / 
                           (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 1e-6))
    return dis_weight


def gradient_matching(train_images, train_labels, num_classes=100, ipc=10,
                      channel=3, im_size=(32, 32), device='cuda',
                      outer_loops=10, inner_loops=50, lr_img=1.0,
                      batch_real=256, dis_metric='ours',
                      dsa_strategy='color_crop_cutout_flip_scale_rotate',
                      seed=0):
    """
    DC: Match gradients of real and synthetic data.
    
    The outer loop samples a new network, the inner loop optimizes synthetic images
    to match gradients.
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
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"DC: Synthesizing {num_classes * ipc} images ({ipc} per class)...")
    
    for ol in range(outer_loops):
        # Initialize a new network for each outer loop
        net = ConvNet(num_classes=num_classes, channel=channel, im_size=im_size).to(device)
        net.train()
        optimizer_net = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        
        for il in range(inner_loops):
            loss = torch.tensor(0.0, device=device)
            
            for c in range(num_classes):
                # Real data gradients
                real_indices = class_indices[c]
                perm = np.random.permutation(len(real_indices))[:batch_real]
                real_batch = train_images[np.array(real_indices)[perm]].to(device)
                real_labels_batch = torch.full((len(perm),), c, dtype=torch.long, device=device)
                
                # Apply DSA
                real_aug = DiffAugment(real_batch, strategy=dsa_strategy)
                
                output_real = net(real_aug)
                loss_real = criterion(output_real, real_labels_batch)
                gw_real = torch.autograd.grad(loss_real, net.parameters(), create_graph=False)
                gw_real = list((_.detach().clone() for _ in gw_real))
                
                # Synthetic data gradients
                syn_mask = syn_labels == c
                syn_batch = syn_images[syn_mask]
                syn_labels_batch = torch.full((syn_batch.shape[0],), c, dtype=torch.long, device=device)
                
                syn_aug = DiffAugment(syn_batch, strategy=dsa_strategy)
                
                output_syn = net(syn_aug)
                loss_syn = criterion(output_syn, syn_labels_batch)
                gw_syn = torch.autograd.grad(loss_syn, net.parameters(), create_graph=True)
                
                loss += match_loss(gw_syn, gw_real, dis_metric=dis_metric)
            
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            
            # Update network on synthetic data
            if il < inner_loops - 1:
                # Train network on current synthetic data
                all_syn_aug = DiffAugment(syn_images.detach(), strategy=dsa_strategy)
                output = net(all_syn_aug)
                loss_net = criterion(output, syn_labels)
                optimizer_net.zero_grad()
                loss_net.backward()
                optimizer_net.step()
        
        if (ol + 1) % 1 == 0:
            print(f"  Outer loop {ol+1}/{outer_loops}, Loss: {loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


if __name__ == '__main__':
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Quick test
    syn_images, syn_labels = gradient_matching(
        train_images, train_labels, ipc=10, outer_loops=2, inner_loops=5
    )
    print(f"Synthetic images shape: {syn_images.shape}")
