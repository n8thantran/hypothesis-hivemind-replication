"""
Distill datasets using DM, DC, and TM methods.
Optimized for correctness and reasonable runtime.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices


def distribution_matching(train_images, train_labels, num_classes=100, ipc=10,
                          device='cuda', iterations=10000, lr_img=1.0,
                          batch_per_class=64, seed=0):
    """DM: Match feature distributions between real and synthetic data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels.numpy(), num_classes)
    
    # Initialize from random real images
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
    
    optimizer = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    
    # Pre-organize real by class
    real_by_class = [train_images[class_indices[c]] for c in range(num_classes)]
    
    n_syn = num_classes * ipc
    print(f"DM: {n_syn} images, {iterations} iterations, lr={lr_img}")
    
    for it in range(iterations):
        net = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        net.eval()
        
        # Sample real images
        real_samples = []
        for c in range(num_classes):
            perm = torch.randperm(len(real_by_class[c]))[:batch_per_class]
            real_samples.append(real_by_class[c][perm])
        all_real = torch.cat(real_samples, dim=0).to(device)
        
        # Augment
        all_real_aug = DiffAugment(all_real, strategy='color_crop_cutout_flip_scale_rotate')
        syn_aug = DiffAugment(syn_images, strategy='color_crop_cutout_flip_scale_rotate')
        
        # Get features
        with torch.no_grad():
            real_feats = []
            for i in range(0, len(all_real_aug), 2000):
                real_feats.append(net.embed(all_real_aug[i:i+2000]))
            real_feat = torch.cat(real_feats, dim=0)
        
        syn_feat = net.embed(syn_aug)
        
        # Compute per-class mean matching loss
        loss = torch.tensor(0.0, device=device)
        for c in range(num_classes):
            real_mean = real_feat[c*batch_per_class:(c+1)*batch_per_class].mean(0)
            syn_mean = syn_feat[c*ipc:(c+1)*ipc].mean(0)
            loss = loss + ((real_mean - syn_mean) ** 2).sum()
        loss = loss / num_classes
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        del net, all_real, all_real_aug, real_feat, syn_aug, syn_feat
        if (it + 1) % 500 == 0:
            torch.cuda.empty_cache()
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def gradient_matching(train_images, train_labels, num_classes=100, ipc=10,
                      device='cuda', outer_loops=100, inner_loops=50,
                      lr_img=1.0, batch_real=256, seed=0):
    """DC: Match gradients between real and synthetic data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels.numpy(), num_classes)
    
    # Initialize from random real images
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
    
    real_by_class = [train_images[class_indices[c]] for c in range(num_classes)]
    
    n_syn = num_classes * ipc
    print(f"DC: {n_syn} images, {outer_loops} outer × {inner_loops} inner, lr={lr_img}")
    
    for outer in range(outer_loops):
        # New random network each outer loop
        net = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        net.train()
        optimizer_net = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        
        for inner in range(inner_loops):
            # Compute gradient on real data (sample per class)
            loss_real = torch.tensor(0.0, device=device)
            for c in range(num_classes):
                perm = torch.randperm(len(real_by_class[c]))[:batch_real // num_classes + 1]
                real_imgs = real_by_class[c][perm].to(device)
                real_imgs = DiffAugment(real_imgs, strategy='color_crop_cutout_flip_scale_rotate')
                real_lbls = torch.full((len(real_imgs),), c, dtype=torch.long, device=device)
                out = net(real_imgs)
                loss_real = loss_real + criterion(out, real_lbls)
            loss_real = loss_real / num_classes
            
            grad_real = torch.autograd.grad(loss_real, net.parameters(), create_graph=False)
            
            # Compute gradient on synthetic data
            syn_aug = DiffAugment(syn_images, strategy='color_crop_cutout_flip_scale_rotate')
            out_syn = net(syn_aug)
            loss_syn = criterion(out_syn, syn_labels)
            grad_syn = torch.autograd.grad(loss_syn, net.parameters(), create_graph=True)
            
            # Match gradients (cosine distance)
            loss_match = torch.tensor(0.0, device=device)
            for g_real, g_syn in zip(grad_real, grad_syn):
                g_real = g_real.detach()
                cos_sim = F.cosine_similarity(g_real.flatten().unsqueeze(0), 
                                               g_syn.flatten().unsqueeze(0))
                loss_match = loss_match + (1 - cos_sim)
            
            optimizer_img.zero_grad()
            loss_match.backward()
            optimizer_img.step()
            
            # Update network
            optimizer_net.zero_grad()
            syn_aug_detach = syn_images.detach()
            syn_aug_d = DiffAugment(syn_aug_detach, strategy='color_crop_cutout_flip_scale_rotate')
            out_d = net(syn_aug_d)
            loss_net = criterion(out_d, syn_labels)
            loss_net.backward()
            optimizer_net.step()
        
        del net, optimizer_net
        torch.cuda.empty_cache()
        
        if (outer + 1) % 10 == 0:
            print(f"  Outer {outer+1}/{outer_loops}, Match Loss: {loss_match.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def trajectory_matching(train_images, train_labels, num_classes=100, ipc=10,
                        device='cuda', iterations=5000, lr_img=1000.0,
                        expert_epochs=50, num_experts=3, seed=0):
    """TM: Match training trajectories of expert models."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    class_indices = get_class_indices(train_labels.numpy(), num_classes)
    
    # Step 1: Train expert trajectories (save checkpoints)
    print("TM: Training expert trajectories...")
    expert_trajectories = []
    
    for exp_idx in range(num_experts):
        traj_path = f'expert_trajectories/expert_{exp_idx}.pt'
        if os.path.exists(traj_path):
            print(f"  Loading existing expert {exp_idx}")
            traj = torch.load(traj_path, weights_only=False)
            expert_trajectories.append(traj)
            continue
            
        print(f"  Training expert {exp_idx}...")
        torch.manual_seed(seed + exp_idx * 100)
        net = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        trajectory = [net.state_dict()]
        
        n_train = len(train_images)
        for epoch in range(expert_epochs):
            perm = torch.randperm(n_train)
            for i in range(0, n_train, 256):
                idx = perm[i:i+256]
                imgs = train_images[idx].to(device)
                imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
                lbls = train_labels[idx].to(device)
                
                optimizer.zero_grad()
                out = net(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
            
            trajectory.append({k: v.cpu().clone() for k, v in net.state_dict().items()})
        
        os.makedirs('expert_trajectories', exist_ok=True)
        torch.save(trajectory, traj_path)
        expert_trajectories.append(trajectory)
        del net, optimizer
        torch.cuda.empty_cache()
        print(f"  Expert {exp_idx} done ({len(trajectory)} checkpoints)")
    
    # Step 2: Optimize synthetic images to match trajectories
    print(f"TM: Optimizing {num_classes * ipc} synthetic images...")
    
    # Initialize from random real images
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
    
    # Training trajectory matching
    max_start_epoch = expert_epochs - 2  # Leave room for target
    
    for it in range(iterations):
        # Pick random expert and starting point
        exp_idx = np.random.randint(num_experts)
        traj = expert_trajectories[exp_idx]
        start_epoch = np.random.randint(0, max_start_epoch)
        
        # Load starting parameters
        start_params = traj[start_epoch]
        target_params = traj[start_epoch + 1]
        
        # Create student network from start params
        student = ConvNet(num_classes=num_classes, channel=3, im_size=(32, 32)).to(device)
        student.load_state_dict({k: v.to(device) for k, v in start_params.items()})
        student.train()
        
        # Train student on synthetic data for a few steps
        student_optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
        
        num_steps = max(1, len(syn_images) // 256)
        perm = torch.randperm(len(syn_images))
        
        for step in range(num_steps):
            idx = perm[step*256:(step+1)*256]
            if len(idx) == 0:
                continue
            imgs = syn_images[idx]
            imgs = DiffAugment(imgs, strategy='color_crop_cutout_flip_scale_rotate')
            lbls = syn_labels[idx]
            
            student_optimizer.zero_grad()
            out = student(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            student_optimizer.step()
        
        # Compute trajectory matching loss
        loss_match = torch.tensor(0.0, device=device)
        target_dict = {k: v.to(device) for k, v in target_params.items()}
        
        for (name, param), (_, target) in zip(student.named_parameters(), target_dict.items()):
            if 'weight' in name or 'bias' in name:
                loss_match = loss_match + F.mse_loss(param, target, reduction='sum')
        
        optimizer_img.zero_grad()
        loss_match.backward()
        optimizer_img.step()
        
        del student, student_optimizer
        
        if (it + 1) % 500 == 0:
            torch.cuda.empty_cache()
            print(f"  Iter {it+1}/{iterations}, Match Loss: {loss_match.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


def generate_soft_labels(images, teacher_path='teacher_final.pt', device='cuda'):
    """Generate soft labels using the teacher model."""
    teacher = ConvNet(num_classes=100, channel=3, im_size=(32, 32)).to(device)
    ckpt = torch.load(teacher_path, weights_only=False)
    teacher.load_state_dict(ckpt['state_dict'])
    teacher.eval()
    
    logits = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            out = teacher(batch)
            logits.append(out.cpu())
    
    return torch.cat(logits, dim=0)


if __name__ == '__main__':
    import sys
    
    method = sys.argv[1] if len(sys.argv) > 1 else 'dm'
    ipc = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Loading CIFAR-100...")
    train_images, train_labels, _, _ = get_cifar100_tensors()
    
    t0 = time.time()
    
    if method == 'dm':
        iters = 10000 if ipc == 10 else 5000
        syn_images, syn_labels = distribution_matching(
            train_images, train_labels, ipc=ipc, iterations=iters
        )
    elif method == 'dc':
        outer = 100 if ipc == 10 else 50
        inner = 50
        syn_images, syn_labels = gradient_matching(
            train_images, train_labels, ipc=ipc, 
            outer_loops=outer, inner_loops=inner
        )
    elif method == 'tm':
        iters = 5000 if ipc == 10 else 3000
        syn_images, syn_labels = trajectory_matching(
            train_images, train_labels, ipc=ipc, iterations=iters
        )
    
    elapsed = time.time() - t0
    print(f"\n{method.upper()} IPC={ipc} done in {elapsed/60:.1f} min")
    print(f"Synthetic images: {syn_images.shape}, range=[{syn_images.min():.3f}, {syn_images.max():.3f}]")
    
    # Save
    torch.save({'images': syn_images, 'labels': syn_labels}, f'distilled_{method}_ipc{ipc}.pt')
    
    # Generate soft labels
    sl = generate_soft_labels(syn_images)
    torch.save(sl, f'soft_labels_{method}_ipc{ipc}.pt')
    
    print(f"Saved distilled_{method}_ipc{ipc}.pt and soft_labels_{method}_ipc{ipc}.pt")
