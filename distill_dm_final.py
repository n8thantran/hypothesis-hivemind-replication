"""
Distribution Matching (DM) distillation - optimized batched implementation.
Matches the original DM paper (Zhao & Bilen, 2023) algorithm.
"""
import torch
import numpy as np
import time
from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices


def distill_dm(ipc=10, num_iters=10000, lr_img=1.0, batch_real=64,
               num_classes=100, seed=0, device='cuda', verbose=True):
    """
    DM distillation for CIFAR-100.
    
    Key: Each iteration uses a NEW randomly initialized frozen network.
    Loss = sum over classes of mean((mean_real_feat - mean_syn_feat)^2)
    """
    train_images, train_labels, _, _ = get_cifar100_tensors()
    class_indices = get_class_indices(train_labels.numpy(), num_classes)
    
    # Initialize synthetic images from real images
    np.random.seed(seed)
    torch.manual_seed(seed)
    
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
    
    # Move real data to GPU, organized by class
    real_by_class = [train_images[class_indices[c]].to(device) for c in range(num_classes)]
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    if verbose:
        print(f'DM distillation: IPC={ipc}, iters={num_iters}, lr={lr_img}, batch_real={batch_real}')
    
    t0 = time.time()
    for it in range(num_iters):
        # New random network each iteration
        net = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        
        # Sample real batch: batch_real per class, concatenated
        real_batch_list = []
        for c in range(num_classes):
            n = real_by_class[c].shape[0]
            perm = torch.randperm(n, device=device)[:batch_real]
            real_batch_list.append(real_by_class[c][perm])
        all_real = torch.cat(real_batch_list, dim=0)  # [num_classes * batch_real, 3, 32, 32]
        
        # Apply DSA to both
        all_real_aug = DiffAugment(all_real, strategy='color_crop_cutout_flip_scale_rotate')
        all_syn_aug = DiffAugment(syn_images, strategy='color_crop_cutout_flip_scale_rotate')
        
        # Get embeddings
        with torch.no_grad():
            real_feat = net.embed(all_real_aug)  # [num_classes * batch_real, feat_dim]
        syn_feat = net.embed(all_syn_aug)  # [num_classes * ipc, feat_dim]
        
        # Compute loss: mean over embedding dims of (mean_real - mean_syn)^2 per class
        loss = torch.tensor(0.0, device=device)
        for c in range(num_classes):
            real_mean = real_feat[c * batch_real:(c + 1) * batch_real].mean(0)
            syn_mean = syn_feat[c * ipc:(c + 1) * ipc].mean(0)
            loss += torch.mean((real_mean - syn_mean) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and (it + 1) % 2000 == 0:
            elapsed = time.time() - t0
            grad_norm = syn_images.grad.abs().mean().item()
            print(f'  Iter {it+1}/{num_iters}, Loss: {loss.item():.4f}, '
                  f'Grad: {grad_norm:.6f}, Time: {elapsed:.0f}s')
    
    total_time = time.time() - t0
    if verbose:
        print(f'DM distillation complete. Total time: {total_time:.0f}s')
    
    return syn_images.detach().cpu(), syn_labels.cpu()


if __name__ == '__main__':
    import sys
    ipc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    
    syn_images, syn_labels = distill_dm(ipc=ipc, num_iters=num_iters)
    torch.save({'images': syn_images, 'labels': syn_labels}, 
               f'distilled_dm_ipc{ipc}_final.pt')
    print(f'Saved distilled_dm_ipc{ipc}_final.pt')
