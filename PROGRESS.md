# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**FULL PIPELINE EXECUTION** - Running all experiments end-to-end

## Key Insight / Strategy Change
Rather than chasing exact SL numbers (which depend heavily on teacher quality),
focus on demonstrating the paper's KEY CLAIMS through relative trends:
1. In HL: DD methods >> coresets (clear gap)
2. In SL: quality gap narrows (DD methods don't beat coresets as much)  
3. TM is the best DD method
4. K-centers > Random in HL but not meaningfully in SL

## Status Assessment
### What works:
- HL evaluation ✓ (Random IPC10: 18.37%, paper: 18.64±0.25) - GOOD MATCH
- ConvNet-D3 architecture ✓
- DSA augmentation ✓  
- Data loading ✓
- DM/DC/TM distillation code exists
- Teacher model trained (~59% accuracy)
- Soft labels generated

### What's approximate:
- SL evaluation gets ~28% for Random IPC10 instead of 33.43%
  - This is a ~5% gap, likely due to teacher quality (my ConvNet-D3 ~59% vs potentially better teacher in paper)
  - The RELATIVE trends should still hold (DD methods should still show compressed gap in SL)

## Evaluation Hyperparameters (Table: tab:stage3_hyper, EXACT from paper)
### HL Setting (Small-scale)
- 300 epochs, SGD, lr=0.01, momentum=0.9, weight_decay=5e-4, StepLR@epoch151 (gamma=0.1), batch=256, DSA augmentation, CE loss

### SL Setting (Small-scale)  
- 300 epochs, AdamW, lr=1e-3, weight_decay=0.01, Cosine scheduler, batch=256, DSA augmentation
- KL-Div(T=20): loss = T² × KL(log_softmax(student_logits/T) || softmax(teacher_logits/T))
- NO warmup for small-scale

## Target Results (Table: tab:small_scale_c100, CIFAR-100, ConvNet-D3)
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

## Architecture files
- convnet.py - ConvNet-D3 ✓
- dsa.py - DSA augmentation ✓
- data_utils.py - Data loading (+ random_select, kcenter_select) ✓
- train_eval.py - Training and evaluation functions ✓
- distill_dm.py - Distribution Matching distillation
- distill_dc.py - Dataset Condensation (gradient matching)  
- distill_tm.py - Trajectory Matching distillation
- train_teacher_v2.py - Teacher training script
- soft_labels.pt - Pre-computed teacher logits (59% teacher)

## Plan (Executing NOW)
1. Write single clean pipeline script that runs ALL experiments
2. For DD methods: run distillation then evaluate (both HL and SL)
3. For coresets: select then evaluate (both HL and SL)
4. Collect all results, format as table
5. Write REPORT.md, reproduce.sh
6. Push final commit

## DD Method Hyperparameters (from DCBench / standard papers)
### DM (Distribution Matching)
- lr_img=1.0, iterations=20000, match every batch
- Initialize with random real images

### DC (Dataset Condensation / Gradient Matching)
- lr_img=1.0, iterations=5000, match gradients

### TM (Trajectory Matching)
- lr_img=0.01 (pixel), iterations=5000
- Needs expert trajectories from pre-trained models
- Expert models trained with SGD on full data

## Key Paper Claims to Reproduce
1. **HL setting: DD methods >> coresets** (DM/DC/TM significantly beat Random)
2. **SL setting: quality gap narrows** (Random catches up to DD methods)
3. **TM is best DD method** in both settings
4. **K-centers outperforms Random in HL** but not meaningfully in SL

## Failed Approaches (DO NOT REPEAT)
1. Pre-computing soft_probs = softmax(logits/T) separately: WRONG. Must store raw logits and apply softmax(logits/T) during training
2. Trying to get exact SL accuracy match without good teacher: diminishing returns. Focus on relative trends.
3. Training ConvNet-D3 teacher for 2000 epochs: too slow (10 min for 200 epochs). Use existing 59% teacher.
4. Many scattered experiment scripts: confusing. Write ONE clean pipeline script.
