"""
Fast dataset distillation implementations.
Key optimization: batch all classes in single forward pass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os

from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_class_indices

NUM_CLASSES = 100
CHANNEL = 3
IM_SIZE = (32, 32)
device = 'cuda'


def init_syn_data(train_images, train_labels, ipc, seed=0):
    """Initialize synthetic data from real samples."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    
    syn_images = []
    syn_labels = []
    for c in range(NUM_CLASSES):
        indices = class_indices[c]
        perm = np.random.permutation(len(indices))[:ipc]
        for p in perm:
            syn_images.append(train_images[indices[p]].clone())
            syn_labels.append(c)
    
    syn_images = torch.stack(syn_images)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long)
    return syn_images, syn_labels


def dm_distill_fast(train_images, train_labels, ipc=10, iterations=20000,
                    lr_img=1.0, batch_real=256, save_path=None):
    """
    Distribution Matching - optimized version.
    Batches all classes in single forward pass.
    """
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        return data['images'], data['labels']
    
    print(f"DM Fast: IPC={ipc}, {iterations} iters")
    
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    syn_images, syn_labels = init_syn_data(train_images, train_labels, ipc)
    syn_images = syn_images.to(device).requires_grad_(True)
    syn_labels = syn_labels.to(device)
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    # Pre-organize real data by class for fast sampling
    real_by_class = {}
    for c in range(NUM_CLASSES):
        real_by_class[c] = train_images[class_indices[c]]
    
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    
    for it in range(iterations):
        # Sample batch_real images per class, concatenate
        real_batches = []
        real_class_ids = []
        for c in range(NUM_CLASSES):
            n_c = len(real_by_class[c])
            perm = torch.randperm(n_c)[:batch_real]
            real_batches.append(real_by_class[c][perm])
            real_class_ids.extend([c] * len(perm))
        
        real_all = torch.cat(real_batches, dim=0).to(device)
        real_class_ids = torch.tensor(real_class_ids, device=device)
        
        net = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        net.train()
        for p in net.parameters():
            p.requires_grad = False
        
        # Single shared augmentation seed
        seed_aug = int(time.time() * 1000) % (2**31)
        
        # Forward pass on real data (single batched pass)
        torch.manual_seed(seed_aug)
        real_aug = DiffAugment(real_all, strategy=dsa_strategy)
        with torch.no_grad():
            real_feat = net.embed(real_aug)
        
        # Forward pass on synthetic data (single pass)
        torch.manual_seed(seed_aug)
        syn_aug = DiffAugment(syn_images, strategy=dsa_strategy)
        syn_feat = net.embed(syn_aug)
        
        # Compute per-class mean feature matching loss
        loss = torch.tensor(0.0, device=device)
        for c in range(NUM_CLASSES):
            real_mask = real_class_ids == c
            syn_mask = syn_labels == c
            real_mean = real_feat[real_mask].mean(0)
            syn_mean = syn_feat[syn_mask].mean(0)
            loss += torch.sum((real_mean - syn_mean) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        del net, real_feat, syn_feat, real_aug, syn_aug, real_all
        
        if (it + 1) % 2000 == 0:
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.4f}")
    
    result_images = syn_images.detach().cpu()
    result_labels = syn_labels.cpu()
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels


def dc_distill_fast(train_images, train_labels, ipc=10, outer_iters=1000,
                    outer_loops=10, inner_loops=50, lr_img=1.0,
                    batch_real=256, save_path=None):
    """
    Dataset Condensation (Gradient Matching) - optimized.
    """
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        return data['images'], data['labels']
    
    print(f"DC Fast: IPC={ipc}, {outer_iters} outer iters, {outer_loops}x{inner_loops}")
    
    class_indices = get_class_indices(train_labels, NUM_CLASSES)
    syn_images, syn_labels = init_syn_data(train_images, train_labels, ipc)
    syn_images = syn_images.to(device).requires_grad_(True)
    syn_labels = syn_labels.to(device)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    
    real_by_class = {}
    for c in range(NUM_CLASSES):
        real_by_class[c] = train_images[class_indices[c]]
    
    for it in range(outer_iters):
        net = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        net.train()
        optimizer_net = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        
        for ol in range(outer_loops):
            # Gradient matching: match gradients class by class
            loss = torch.tensor(0.0, device=device)
            seed_aug = int(time.time() * 1000) % (2**31)
            
            for c in range(NUM_CLASSES):
                n_c = len(real_by_class[c])
                perm = torch.randperm(n_c)[:batch_real]
                real_batch = real_by_class[c][perm].to(device)
                real_labs = torch.full((len(perm),), c, dtype=torch.long, device=device)
                
                torch.manual_seed(seed_aug + c)
                real_aug = DiffAugment(real_batch, strategy=dsa_strategy)
                out_real = net(real_aug)
                loss_real = criterion(out_real, real_labs)
                gw_real = torch.autograd.grad(loss_real, net.parameters(), create_graph=False)
                gw_real = [g.detach().clone() for g in gw_real]
                
                syn_mask = syn_labels == c
                syn_batch = syn_images[syn_mask]
                syn_labs = torch.full((syn_batch.shape[0],), c, dtype=torch.long, device=device)
                
                torch.manual_seed(seed_aug + c)
                syn_aug = DiffAugment(syn_batch, strategy=dsa_strategy)
                out_syn = net(syn_aug)
                loss_syn = criterion(out_syn, syn_labs)
                gw_syn = torch.autograd.grad(loss_syn, net.parameters(), create_graph=True)
                
                # Gradient distance (cosine)
                for gwr, gws in zip(gw_real, gw_syn):
                    shape = gwr.shape
                    if len(shape) <= 1:
                        continue
                    gwr_flat = gwr.reshape(shape[0], -1) if len(shape) > 2 else gwr
                    gws_flat = gws.reshape(shape[0], -1) if len(shape) > 2 else gws
                    loss += torch.sum(1 - F.cosine_similarity(gwr_flat, gws_flat, dim=1))
            
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            
            # Inner loop: train network on synthetic data
            if ol < outer_loops - 1:
                for il in range(inner_loops):
                    syn_aug_net = DiffAugment(syn_images.detach(), strategy=dsa_strategy)
                    out_net = net(syn_aug_net)
                    loss_net = criterion(out_net, syn_labels)
                    optimizer_net.zero_grad()
                    loss_net.backward()
                    optimizer_net.step()
        
        del net
        
        if (it + 1) % 100 == 0:
            print(f"  Iter {it+1}/{outer_iters}, Loss: {loss.item():.4f}")
    
    result_images = syn_images.detach().cpu()
    result_labels = syn_labels.cpu()
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels


def tm_distill_fast(train_images, train_labels, ipc=10,
                    expert_dir='/workspace/expert_trajectories',
                    iterations=5000, lr_img=1000.0, syn_steps=30,
                    expert_epochs=3, max_start_epoch=25,
                    save_path=None):
    """
    Trajectory Matching distillation.
    """
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        return data['images'], data['labels']
    
    print(f"TM Fast: IPC={ipc}, {iterations} iters, syn_steps={syn_steps}")
    
    expert_files = sorted([f for f in os.listdir(expert_dir) if f.startswith('expert_')])
    expert_trajectories = []
    for f in expert_files:
        traj = torch.load(os.path.join(expert_dir, f), map_location='cpu', weights_only=False)
        expert_trajectories.append(traj)
    print(f"  Loaded {len(expert_trajectories)} experts, {len(expert_trajectories[0])} checkpoints each")
    
    syn_images, syn_labels = init_syn_data(train_images, train_labels, ipc)
    syn_images = syn_images.to(device).requires_grad_(True)
    syn_labels = syn_labels.to(device)
    
    syn_lr = torch.tensor(0.01, device=device, requires_grad=True)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=1e-5, momentum=0.5)
    
    criterion = nn.CrossEntropyLoss()
    dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    
    for it in range(iterations):
        exp_idx = np.random.randint(len(expert_trajectories))
        traj = expert_trajectories[exp_idx]
        max_start = min(max_start_epoch, len(traj) - expert_epochs - 1)
        if max_start < 1:
            max_start = 1
        start_epoch = np.random.randint(0, max_start)
        
        start_params = traj[start_epoch]
        target_params = traj[start_epoch + expert_epochs]
        
        student = ConvNet(num_classes=NUM_CLASSES, channel=CHANNEL, im_size=IM_SIZE).to(device)
        student.load_state_dict({k: v.to(device) for k, v in start_params.items()})
        student.train()
        
        for step in range(syn_steps):
            perm = torch.randperm(len(syn_images), device=device)
            batch_imgs = syn_images[perm]
            batch_labels = syn_labels[perm]
            batch_imgs_aug = DiffAugment(batch_imgs, strategy=dsa_strategy)
            
            out = student(batch_imgs_aug)
            loss_s = criterion(out, batch_labels)
            
            grads = torch.autograd.grad(loss_s, student.parameters(), create_graph=True)
            with torch.no_grad():
                for param, grad in zip(student.parameters(), grads):
                    param.sub_(syn_lr * grad)
        
        # Trajectory matching loss
        trainable_student = []
        trainable_target = []
        for name, param in student.named_parameters():
            trainable_student.append(param.reshape(-1))
            if name in target_params:
                trainable_target.append(target_params[name].to(device).reshape(-1))
        
        flat_student = torch.cat(trainable_student)
        flat_target = torch.cat(trainable_target)
        
        # Normalized matching
        student_dir = flat_student - torch.cat([start_params[name].to(device).reshape(-1) 
                                                 for name, _ in student.named_parameters()])
        target_dir = flat_target - torch.cat([start_params[name].to(device).reshape(-1) 
                                               for name, _ in student.named_parameters()])
        
        student_norm = student_dir / (torch.norm(student_dir) + 1e-6)
        target_norm = target_dir / (torch.norm(target_dir) + 1e-6)
        loss = torch.sum((student_norm - target_norm) ** 2)
        
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        loss.backward()
        optimizer_img.step()
        optimizer_lr.step()
        
        with torch.no_grad():
            syn_lr.clamp_(min=1e-6)
        
        del student
        
        if (it + 1) % 500 == 0:
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.6f}, lr: {syn_lr.item():.6f}")
    
    result_images = syn_images.detach().cpu()
    result_labels = syn_labels.cpu()
    
    if save_path:
        torch.save({'images': result_images, 'labels': result_labels}, save_path)
    
    return result_images, result_labels


if __name__ == '__main__':
    from data_utils import get_cifar100_tensors
    
    print("Loading data...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Quick test: DM IPC=10, 100 iterations
    print("\nTesting DM Fast (100 iters)...")
    t0 = time.time()
    imgs, labs = dm_distill_fast(train_images, train_labels, ipc=10, 
                                  iterations=100, save_path=None)
    t1 = time.time()
    print(f"DM 100 iters: {t1-t0:.1f}s, images shape: {imgs.shape}")
    print(f"Estimated time for 20000 iters: {20000*(t1-t0)/100/60:.1f} min")
