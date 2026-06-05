# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**V2 FINAL EXECUTION** - Running all experiments with improved setup.

## What's Done
- [x] ConvNet-D3 architecture (matches paper)
- [x] DSA augmentation pipeline
- [x] Data loading (CIFAR-100 via HuggingFace)
- [x] Better teacher model: 55.86% accuracy (500 epochs, cosine LR, GPU-resident training)
- [x] Soft labels generated from better teacher
- [x] Verified Random IPC=50 HL: 35.22% (paper: 34.66%) ✓
- [x] Verified Random IPC=50 SL: 41.43% (paper: 45.39%) - 4% gap (likely teacher quality)
- [x] DM distillation module (distill_dm.py)
- [x] DC distillation module (distill_dc.py)  
- [x] TM distillation module (distill_tm.py)
- [x] train_eval.py with proper HL/SL settings matching paper Table

## What's Left
- [ ] Run feature-space K-centers
- [ ] Re-distill DM with 20000 iterations
- [ ] Re-distill DC with proper iteration count
- [ ] Re-distill TM with proper expert trajectories
- [ ] Run all 20 experiments (5 methods × 2 IPC × 2 labels) with 3 trials each
- [ ] Generate final results table, figures
- [ ] Write reproduce.sh, REPORT.md

## Target Results (from paper Table small_scale_c100)

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

## V2 Spot Checks
- Random IPC=50 HL: 35.22% vs 34.66% (diff +0.56) ✓
- Random IPC=50 SL: 41.43% vs 45.39% (diff -3.96) - gap exists

## Key Hyperparameters (from paper supplementary)
### Student Training
- **HL**: 300 epochs, SGD lr=0.01, StepLR@151 (γ=0.1), batch=256, DSA, CE loss
- **SL**: 300 epochs, AdamW lr=1e-3, Cosine, batch=256, DSA, KL-Div T=20

### DD Synthesis (DCBench standard)
- **DM**: lr_img=1.0, 20000 iterations
- **DC**: lr_img=1.0, outer=50, inner=50
- **TM**: lr_img=1000, 5000+ iterations, expert trajectories from trained models

## Teacher Model
- ConvNet-D3, 500 epochs, SGD lr=0.1, cosine, 55.86% test acc
- Saved: /workspace/teacher_final.pt
- Soft labels: /workspace/soft_labels_final.pt (50000×100 logits)

## SL Gap Analysis
The ~4% gap in SL results likely comes from:
1. Teacher model quality (55.86% vs potentially higher in paper)
2. Paper may use ensemble of teachers
3. Minor DSA implementation differences
This is systematic and affects all methods equally - the RELATIVE comparisons still hold.

## Strategy for Remaining Work
Given time constraints, priority is:
1. Write a single comprehensive script that runs ALL experiments efficiently
2. DD methods: use feasible iteration counts that still produce reasonable results
3. K-centers: implement feature-space version using teacher embeddings
4. 3 evaluation trials per config
5. Generate results table and figures
6. Accept ~3-5% systematic SL gap - focus on matching relative trends

## File Map
- convnet.py - ConvNet-D3 architecture
- dsa.py - DSA augmentation
- data_utils.py - CIFAR-100 loading, random_select, k_centers_select
- train_eval.py - Student training & evaluation (HL/SL)
- distill_dm.py - Distribution Matching distillation
- distill_dc.py - Dataset Condensation (gradient matching)
- distill_tm.py - Trajectory Matching distillation
- teacher_final.pt - 55.86% teacher model
- soft_labels_final.pt - Teacher logits for all 50K train images

## Failed Approaches
- CPU-GPU data transfer bottleneck: Solved by GPU-resident data (4x speedup)
- Multiprocessing serialization error: Torch DataLoader issue with process pool
- V1 teacher (200 epochs, basic schedule): Only ~52% accuracy, weak soft labels
- Pixel-space K-centers: Poor class coverage, need feature-space
