"""
Re-distill all DD methods with proper iteration counts.
Run this before the final evaluation.
"""
import torch
import numpy as np
import time
import sys
import os

from data_utils import get_cifar100_tensors

def main():
    print("Loading CIFAR-100...")
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    device = 'cuda'
    
    method = sys.argv[1] if len(sys.argv) > 1 else 'dm'
    ipc_str = sys.argv[2] if len(sys.argv) > 2 else '10'
    ipc = int(ipc_str)
    
    if method == 'dm':
        from distill_dm import distribution_matching
        iters = 5000  # Reduced from 20000 but still much better than 1000
        print(f"\n=== DM IPC={ipc}, {iters} iterations ===")
        t0 = time.time()
        syn_img, syn_lbl = distribution_matching(
            train_images, train_labels, 
            num_classes=100, ipc=ipc, device=device,
            iterations=iters, lr_img=1.0, batch_real=64,
            seed=0
        )
        dt = time.time() - t0
        print(f"Done in {dt:.0f}s")
        path = f'/workspace/distilled_dm_ipc{ipc}.pt'
        torch.save({'images': syn_img, 'labels': syn_lbl}, path)
        print(f"Saved to {path}")
    
    elif method == 'dc':
        from distill_dc import gradient_matching
        outer = 10
        inner = 10
        print(f"\n=== DC IPC={ipc}, outer={outer}, inner={inner} ===")
        t0 = time.time()
        syn_img, syn_lbl = gradient_matching(
            train_images, train_labels,
            num_classes=100, ipc=ipc, device=device,
            outer_loops=outer, inner_loops=inner, lr_img=1.0,
            batch_real=256,
            seed=0
        )
        dt = time.time() - t0
        print(f"Done in {dt:.0f}s")
        path = f'/workspace/distilled_dc_ipc{ipc}.pt'
        torch.save({'images': syn_img, 'labels': syn_lbl}, path)
        print(f"Saved to {path}")
    
    elif method == 'tm':
        from distill_tm import train_expert_trajectories, trajectory_matching
        
        expert_dir = '/workspace/expert_trajectories'
        if not os.path.exists(expert_dir) or len(os.listdir(expert_dir)) == 0:
            print("Training expert trajectories...")
            train_expert_trajectories(
                train_images, train_labels,
                num_experts=3, epochs=50,
                save_dir=expert_dir, device=device
            )
        
        iters = 3000
        print(f"\n=== TM IPC={ipc}, {iters} iterations, lr_img=1000 ===")
        t0 = time.time()
        syn_img, syn_lbl = trajectory_matching(
            train_images, train_labels,
            num_classes=100, ipc=ipc, device=device,
            expert_dir=expert_dir,
            match_iterations=iters, lr_img=1000.0,
            seed=0
        )
        dt = time.time() - t0
        print(f"Done in {dt:.0f}s")
        path = f'/workspace/distilled_tm_ipc{ipc}.pt'
        torch.save({'images': syn_img, 'labels': syn_lbl}, path)
        print(f"Saved to {path}")


if __name__ == '__main__':
    main()
