"""
Run a single experiment configuration and append results to results.json.
Usage: python run_single.py <method> <ipc> <label_type>
  method: random, k_centers, dm, dc, tm
  ipc: 10 or 50
  label_type: hard or soft
"""
import sys
import os
import json
import torch
import numpy as np
import time

def main():
    if len(sys.argv) < 4:
        print("Usage: python run_single.py <method> <ipc> <label_type> [num_runs]")
        sys.exit(1)
    
    method = sys.argv[1]
    ipc = int(sys.argv[2])
    label_type = sys.argv[3]
    num_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from convnet import ConvNet
    from data_utils import get_cifar100_tensors, random_select, k_centers_select
    from train_eval import train_and_evaluate
    
    print(f"Running: {method} IPC={ipc} {label_type} ({num_runs} runs)")
    start = time.time()
    
    # Load data
    train_images, train_labels, test_images, test_labels = get_cifar100_tensors()
    model_fn = lambda: ConvNet(num_classes=100, channel=3, im_size=(32, 32))
    
    # Load soft labels if needed
    soft_labels_all = None
    if label_type == 'soft':
        if os.path.exists('soft_labels.pt'):
            soft_labels_all = torch.load('soft_labels.pt', weights_only=True)
            print(f"Loaded soft labels: {soft_labels_all.shape}")
        else:
            print("Generating soft labels...")
            from data_utils import generate_soft_labels
            soft_labels_all = generate_soft_labels(
                train_images, train_labels, model_fn, 
                num_classes=100, device=device, num_models=1, epochs=100
            )
            torch.save(soft_labels_all, 'soft_labels.pt')
    
    # Get training data based on method
    if method == 'random':
        selected = random_select(train_labels, ipc=ipc, seed=42)
        sub_images = train_images[selected]
        sub_labels = train_labels[selected]
        sub_soft = soft_labels_all[selected] if soft_labels_all is not None else None
        
    elif method == 'k_centers':
        selected = k_centers_select(train_images, train_labels, ipc=ipc, seed=0)
        sub_images = train_images[selected]
        sub_labels = train_labels[selected]
        sub_soft = soft_labels_all[selected] if soft_labels_all is not None else None
        
    elif method == 'dm':
        cache_path = f'distilled_dm_ipc{ipc}.pt'
        if os.path.exists(cache_path):
            data = torch.load(cache_path, weights_only=True)
            sub_images = data['images']
            sub_labels = data['labels']
        else:
            from distill_dm import distill_dm
            sub_images, sub_labels = distill_dm(
                train_images, train_labels, model_fn,
                ipc=ipc, num_classes=100, device=device,
                iterations=3000, lr_img=1.0
            )
            torch.save({'images': sub_images, 'labels': sub_labels}, cache_path)
        sub_soft = soft_labels_all[torch.arange(len(sub_labels))] if soft_labels_all is not None else None
        # For DD methods with soft labels, generate soft labels for synthetic data
        if label_type == 'soft':
            sub_soft = _generate_soft_for_synthetic(sub_images, sub_labels, model_fn, device)
        
    elif method == 'dc':
        cache_path = f'distilled_dc_ipc{ipc}.pt'
        if os.path.exists(cache_path):
            data = torch.load(cache_path, weights_only=True)
            sub_images = data['images']
            sub_labels = data['labels']
        else:
            from distill_dc import distill_dc
            sub_images, sub_labels = distill_dc(
                train_images, train_labels, model_fn,
                ipc=ipc, num_classes=100, device=device,
                outer_loops=5, inner_loops=10, lr_img=1.0
            )
            torch.save({'images': sub_images, 'labels': sub_labels}, cache_path)
        if label_type == 'soft':
            sub_soft = _generate_soft_for_synthetic(sub_images, sub_labels, model_fn, device)
        else:
            sub_soft = None
        
    elif method == 'tm':
        cache_path = f'distilled_tm_ipc{ipc}.pt'
        if os.path.exists(cache_path):
            data = torch.load(cache_path, weights_only=True)
            sub_images = data['images']
            sub_labels = data['labels']
        else:
            from distill_tm import distill_tm
            sub_images, sub_labels = distill_tm(
                train_images, train_labels, model_fn,
                ipc=ipc, num_classes=100, device=device,
                num_experts=5, expert_epochs=20,
                match_iterations=1000, lr_img=0.1
            )
            torch.save({'images': sub_images, 'labels': sub_labels}, cache_path)
        if label_type == 'soft':
            sub_soft = _generate_soft_for_synthetic(sub_images, sub_labels, model_fn, device)
        else:
            sub_soft = None
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)
    
    print(f"Training data: {sub_images.shape}, Labels: {sub_labels.shape}")
    
    # Run evaluations
    accs = []
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        acc = train_and_evaluate(
            sub_images, sub_labels, test_images, test_labels,
            model_fn, num_classes=100, device=device,
            label_type=label_type, soft_labels=sub_soft,
            epochs=300, batch_size=256, seed=run, verbose=True
        )
        accs.append(acc)
        print(f"Run {run+1}: {acc:.2f}%")
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    elapsed = time.time() - start
    
    print(f"\n{'='*50}")
    print(f"Result: {method} IPC={ipc} {label_type}: {mean_acc:.2f} ± {std_acc:.2f}%")
    print(f"Time: {elapsed:.1f}s")
    
    # Save to results.json
    results_path = 'results/results.json'
    os.makedirs('results', exist_ok=True)
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    key = f"{method}_ipc{ipc}_{label_type}"
    all_results[key] = {
        'mean': mean_acc,
        'std': std_acc,
        'runs': accs,
        'method': method,
        'ipc': ipc,
        'label_type': label_type,
        'time': elapsed
    }
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved to {results_path}")


def _generate_soft_for_synthetic(syn_images, syn_labels, model_fn, device):
    """Generate soft labels for synthetic images using a teacher trained on full data."""
    from data_utils import get_cifar100_tensors
    train_images, train_labels, _, _ = get_cifar100_tensors()
    
    # Train a quick teacher
    import torch.nn as nn
    model = model_fn().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    model.train()
    for epoch in range(100):
        for batch_imgs, batch_labels in loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = nn.CrossEntropyLoss()(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    # Get soft labels for synthetic images
    model.eval()
    with torch.no_grad():
        all_logits = []
        for i in range(0, len(syn_images), 256):
            batch = syn_images[i:i+256].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    
    return torch.cat(all_logits, dim=0)


if __name__ == '__main__':
    main()
