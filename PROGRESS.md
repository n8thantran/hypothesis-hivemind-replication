# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**CLEAN REBUILD** - Creating single clean implementation with correct evaluation

## Critical Findings
1. **Soft labels ARE stored as logits** (not probabilities) - verified: shape [50000, 100], range [-22, +31]
2. **SL eval uses KL-div with T=20**: loss = T^2 * KL(log_softmax(student/T) || softmax(teacher_logits/T))
3. **HL eval works correctly**: Random IPC10 HL = 18.37% (paper: 18.64±0.25) ✓
4. **SL eval was broken** because old code pre-computed soft_probs = softmax(logits/T=20) which made near-uniform
5. **Correct approach**: Store teacher LOGITS, apply temperature during training in the loss function

## Evaluation Hyperparameters (Table: tab:stage3_hyper)
### HL Setting (Small-scale)
- 300 epochs, SGD, lr=0.01, StepLR@epoch151 (halve LR), batch=256, DSA augmentation, CE loss

### SL Setting (Small-scale)
- 300 epochs, AdamW, lr=1e-3, Cosine scheduler, batch=256, DSA augmentation
- KL-Div(T=20): loss = T² × KL(log_softmax(student_logits/T) || softmax(teacher_logits/T))

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

## Key architecture files
- convnet.py - ConvNet-D3 ✓
- dsa.py - DSA augmentation ✓
- data_utils.py - Data loading ✓
- soft_labels.pt - Teacher logits (50000, 100) ✓
- teacher_model.pt - ConvNet-D3 teacher state dict ✓

## Plan
1. ✅ Verify soft labels are logits with good dynamic range
2. Create clean replicate_final.py with correct HL+SL evaluation
3. Test on Random+K-centers first (coresets, no distillation needed)
4. Then run DD methods (DM, DC, TM)
5. Generate results table, report, reproduce.sh

## Failed Approaches (DO NOT REPEAT)
1. Pre-computing soft_probs = softmax(logits/T=20) → nearly uniform, useless
2. K-centers using max-distance greedy → selects outliers (10.73% accuracy)
3. Need K-means clustering on features for K-centers
4. Using 1000 epochs instead of 300
5. Using SGD for SL (should be AdamW)
6. Too many scattered files → clean up, use single script
