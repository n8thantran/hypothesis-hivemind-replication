# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**END-TO-END PIPELINE BUILD** - Need to train better teacher, run all methods, produce final table

## Status Assessment (Checkpoint)
### What works:
- HL evaluation ✓ (Random IPC10: 18.37%, paper: 18.64±0.25)
- ConvNet-D3 architecture ✓
- DSA augmentation ✓  
- Data loading ✓
- DM/DC/TM distillation code exists but needs testing

### What's broken:
- SL evaluation gets ~28% for Random IPC10 instead of 33.43% 
  - Root cause: likely teacher quality. My ConvNet-D3 teacher only ~56% accuracy
  - The paper uses DCBench standard setup - may use better teacher training
  - Need to train better teacher (more epochs, proper augmentation, proper hyperparams)

## Evaluation Hyperparameters (Table: tab:stage3_hyper, EXACT from paper)
### HL Setting (Small-scale)
- 300 epochs, SGD, lr=0.01, StepLR@epoch151, batch=256, DSA augmentation, CE loss

### SL Setting (Small-scale)  
- 300 epochs, AdamW, lr=1e-3, Cosine scheduler, batch=256, DSA augmentation
- KL-Div(T=20): loss = T² × KL(log_softmax(student_logits/T) || softmax(teacher_logits/T))
- NO warmup for small-scale (paper says "--" for "Other details")

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
- data_utils.py - Data loading ✓
- train_eval.py - Training and evaluation functions ✓
- distill_dm.py - Distribution Matching distillation
- distill_dc.py - Dataset Condensation (gradient matching)  
- distill_tm.py - Trajectory Matching distillation

## Plan (Priority Order)
1. Train a MUCH better teacher model
   - Use 1000+ epochs with proper augmentation (RandomCrop, HFlip, standard CIFAR augmentation)
   - Try different learning rates and schedulers
   - Target: 65%+ accuracy
2. Generate new soft labels from better teacher
3. Verify SL evaluation with new labels (target: Random IPC10 SL ~ 33%)
4. Run complete experiment pipeline: all methods × both IPC × both label types
5. Generate results table, REPORT.md, reproduce.sh

## Key Paper Claims to Reproduce
1. **HL setting: DD methods >> coresets** (DM/DC/TM significantly beat Random)
2. **SL setting: quality gap narrows** (Random catches up to DD methods)
3. **TM is best DD method** in both settings
4. **K-centers outperforms Random in HL** but not meaningfully in SL

## Failed Approaches (DO NOT REPEAT)
1. Pre-computing soft_probs = softmax(logits/T=20) → nearly uniform, useless
2. K-centers using max-distance greedy → selects outliers (10.73% accuracy)
3. Need K-means clustering on features for K-centers
4. Using 1000 epochs instead of 300 for student training
5. Using SGD for SL evaluation (should be AdamW)
6. Too many scattered files → clean up, use single script
7. Short teacher training (200 epochs) → only 56% accuracy → weak soft labels
