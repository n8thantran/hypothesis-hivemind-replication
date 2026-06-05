"""
Trajectory Matching (TM) - Cazenavette et al. (2022)
Synthesizes a distilled dataset by matching training trajectories of models
trained on real vs synthetic data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from convnet import ConvNet
from dsa import DiffAugment
from data_utils import get_cifar100_tensors, get_class_indices


def train_expert_trajectories(train_images, train_labels, num_classes=100,
                               channel=3, im_size=(32, 32), device='cuda',
                               num_experts=100, expert_epochs=50,
                               lr=0.01, batch_size=256,
                               save_dir='/workspace/expert_trajectories',
                               dsa_strategy='color_crop_cutout_flip_scale_rotate',
                               seed=0):
    """
    Train expert models on full dataset and save their parameter trajectories.
    Each expert starts from a different random init and trains for expert_epochs.
    We save checkpoints at each epoch.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_train = len(train_images)
    
    for exp_idx in range(num_experts):
        print(f"Training expert {exp_idx+1}/{num_experts}...")
        
        # Random init
        torch.manual_seed(seed + exp_idx * 1000)
        model = ConvNet(num_classes=num_classes, channel=channel, im_size=im_size).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Save trajectory (list of state dicts)
        trajectory = [model.state_dict()]
        
        model.train()
        for epoch in range(expert_epochs):
            perm = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i+batch_size]
                batch_imgs = train_images[idx].to(device)
                batch_labels = train_labels[idx].to(device)
                
                # Apply DSA
                batch_imgs = DiffAugment(batch_imgs, strategy=dsa_strategy)
                
                optimizer.zero_grad()
                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            
            # Save checkpoint after each epoch
            trajectory.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        
        # Save trajectory
        torch.save(trajectory, os.path.join(save_dir, f'expert_{exp_idx}.pt'))
        print(f"  Expert {exp_idx+1} done, saved {len(trajectory)} checkpoints")
    
    return save_dir


def trajectory_matching(train_images, train_labels, num_classes=100, ipc=10,
                        channel=3, im_size=(32, 32), device='cuda',
                        expert_dir='/workspace/expert_trajectories',
                        num_experts=100, iterations=5000, lr_img=1000.0,
                        lr_lr=1e-5, syn_steps=30, expert_epochs=3,
                        max_start_epoch=25,
                        dsa_strategy='color_crop_cutout_flip_scale_rotate',
                        seed=0):
    """
    TM: Match training trajectories.
    
    For each iteration:
    1. Sample a random expert trajectory and a starting epoch
    2. Initialize student model from expert's starting checkpoint
    3. Train student on synthetic data for syn_steps
    4. Minimize distance between student params and expert's target checkpoint
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load expert trajectories
    expert_files = sorted([f for f in os.listdir(expert_dir) if f.startswith('expert_')])
    num_experts = min(num_experts, len(expert_files))
    print(f"Loading {num_experts} expert trajectories...")
    
    expert_trajectories = []
    for f in expert_files[:num_experts]:
        traj = torch.load(os.path.join(expert_dir, f), map_location='cpu', weights_only=False)
        expert_trajectories.append(traj)
    
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
    
    # Learnable learning rate for student
    syn_lr = torch.tensor(0.01, device=device, requires_grad=True)
    
    optimizer_img = torch.optim.SGD([syn_images], lr=lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=lr_lr, momentum=0.5)
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"TM: Synthesizing {num_classes * ipc} images ({ipc} per class)...")
    
    for it in range(iterations):
        # Sample random expert and starting epoch
        exp_idx = np.random.randint(num_experts)
        traj = expert_trajectories[exp_idx]
        max_start = min(max_start_epoch, len(traj) - expert_epochs - 1)
        if max_start < 1:
            max_start = 1
        start_epoch = np.random.randint(0, max_start)
        
        # Get starting and target parameters
        start_params = traj[start_epoch]
        target_params = traj[start_epoch + expert_epochs]
        
        # Initialize student from starting checkpoint
        student = ConvNet(num_classes=num_classes, channel=channel, im_size=im_size).to(device)
        student.load_state_dict({k: v.to(device) for k, v in start_params.items()})
        student.train()
        
        # Train student on synthetic data for syn_steps
        n_syn = len(syn_images)
        for step in range(syn_steps):
            # Random permutation of synthetic data
            perm = torch.randperm(n_syn, device=device)
            batch_imgs = syn_images[perm]
            batch_labels = syn_labels[perm]
            
            # Apply DSA
            batch_imgs_aug = DiffAugment(batch_imgs, strategy=dsa_strategy)
            
            outputs = student(batch_imgs_aug)
            loss_student = criterion(outputs, batch_labels)
            
            # Manual SGD step (to keep computation graph)
            grads = torch.autograd.grad(loss_student, student.parameters(), create_graph=True)
            
            with torch.no_grad():
                for param, grad in zip(student.parameters(), grads):
                    param.sub_(syn_lr * grad)
        
        # Compute trajectory matching loss
        loss = torch.tensor(0.0, device=device)
        target_dict = {k: v.to(device) for k, v in target_params.items()}
        
        for (name, param), (_, target) in zip(student.named_parameters(), target_dict.items()):
            if 'weight' in name or 'bias' in name:
                loss += F.mse_loss(param, target, reduction='sum')
        
        # Normalize by parameter count
        num_params = sum(p.numel() for p in student.parameters())
        loss = loss / num_params
        
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        loss.backward()
        optimizer_img.step()
        optimizer_lr.step()
        
        # Clamp learning rate to be positive
        with torch.no_grad():
            syn_lr.clamp_(min=1e-6)
        
        if (it + 1) % 500 == 0:
            print(f"  Iter {it+1}/{iterations}, Loss: {loss.item():.8f}, syn_lr: {syn_lr.item():.6f}")
    
    return syn_images.detach().cpu(), syn_labels.cpu()


if __name__ == '__main__':
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    
    # Quick test: train 2 experts for 3 epochs
    print("\nTraining expert trajectories (quick test)...")
    train_expert_trajectories(
        train_images, train_labels, num_experts=2, expert_epochs=3,
        save_dir='/workspace/expert_trajectories_test'
    )
    
    # Quick test TM
    syn_images, syn_labels = trajectory_matching(
        train_images, train_labels, ipc=10, iterations=10,
        expert_dir='/workspace/expert_trajectories_test',
        num_experts=2, expert_epochs=2, syn_steps=5
    )
    print(f"Synthetic images shape: {syn_images.shape}")
