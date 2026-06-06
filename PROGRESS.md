# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**EXECUTION** - Writing unified evaluation script, then running all experiments

## Status Summary (Checkpoint Turn 2550)
### What's Done
- ✅ ConvNet-D3 architecture (convnet.py) - tested, correct
- ✅ DSA augmentation (dsa.py) - tested, correct  
- ✅ Teacher model trained: 59.19% accuracy (teacher.pt)
- ✅ Soft labels generated for full training set (soft_labels.pt, shape [50000,100], raw logits)
- ✅ DCBench distilled datasets downloaded for CIFAR-100:
  - DC IPC10/50, DM IPC10/50, DSA IPC10/50, TM IPC10/50, Random IPC10/50, K-centers IPC10/50
  - All stored in dcbench_data/data/condensed/
  - All pre-normalized (mean≈0, std≈1), no additional normalization needed
- ✅ Verified HL Random IPC10 ≈ 18.64% (matches paper exactly)

### What's NOT Done
- [ ] Write unified evaluation script for DCBench data
- [ ] Generate soft labels for DCBench distilled datasets using teacher
- [ ] Run all 20 experiments (5 methods × 2 IPCs × 2 label types)
- [ ] Generate results table
- [ ] Write reproduce.sh
- [ ] Write REPORT.md
- [ ] Final commit

## DCBench Data Format
- DC/DM/DSA: `res_{method}_CIFAR100_ConvNet_{ipc}ipc.pt` → dict with 'data' key → data[0][0]=images, data[0][1]=labels
- TM: `IPC{ipc}/images_best.pt` and `IPC{ipc}/labels_best.pt` → direct tensors
- Random: `CIFAR100_IPC{ipc}_normalize_images.pt` and `CIFAR100_IPC{ipc}_normalize_labels.pt`
- K-centers: `CIFAR100_IPC{ipc}_images.pt` and `CIFAR100_IPC{ipc}_labels.pt`
- All images are [N, 3, 32, 32] float32, already channel-normalized
- Labels are integer class indices

## Target Table: tab:small_scale_c100 (CIFAR-100, ConvNet-D3)
| Method | IPC | HL | SL |
|--------|-----|------|------|
| DM | 10 | 29.23±0.26 | 26.13±0.10 |
| DM | 50 | 42.32±0.37 | 43.46±0.18 |
| DC | 10 | 28.42±0.29 | 23.54±0.31 |
| DC | 50 | 30.56±0.56 | 33.46±0.38 |
| TM | 10 | 38.18±0.42 | 37.60±0.25 |
| TM | 50 | 46.32±0.26 | 46.26±0.30 |
| Random | 10 | 18.64±0.25 | 33.43±0.18 |
| Random | 50 | 34.66±0.41 | 45.39±0.23 |
| K-centers | 10 | 25.04±0.30 | 34.70±0.27 |
| K-centers | 50 | 38.64±0.43 | 46.24±0.12 |

## Evaluation Hyperparameters (EXACT from paper)
### HL Setting (Small-scale)
- 300 epochs, SGD, lr=0.01, momentum=0.9, weight_decay=5e-4
- StepLR@epoch151 (gamma=0.1), batch=256, DSA augmentation, CE loss

### SL Setting (Small-scale)  
- 300 epochs, AdamW, lr=1e-3, weight_decay=0.01, Cosine scheduler
- batch=256, DSA augmentation, KL-Div(T=20), NO warmup

## Key Paper Claims to Demonstrate
1. In HL: DD methods (especially TM) >> coresets (clear gap)
2. In SL: gap narrows substantially - coresets competitive with DD
3. TM is the best DD method in both settings
4. K-centers > Random in HL, but similar in SL

## Known Issues
- SL numbers may be ~5% lower than paper due to teacher quality (59.19% vs possibly higher)
- DSA is not in paper's Table 1 (paper uses DC, DM, TM, Random, K-centers)
- The RELATIVE trends should still hold

## Key Files
- /workspace/convnet.py - ConvNet-D3 architecture
- /workspace/dsa.py - DSA augmentation
- /workspace/data_utils.py - Data loading utilities
- /workspace/train_eval.py - Training/evaluation functions
- /workspace/teacher.pt - Trained teacher model (59.19%)
- /workspace/soft_labels.pt - Full training set soft labels
- /workspace/dcbench_data/ - Downloaded DCBench distilled datasets
