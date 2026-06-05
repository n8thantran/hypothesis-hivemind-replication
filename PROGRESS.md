# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**REBUILDING** - Found correct eval hyperparams from paper. Need to rebuild evaluation pipeline.

## Critical Discovery: Paper's Evaluation Setup
From paper's Table (tab:stage3_hyper):
- **Small-scale HL**: 300 epochs, SGD, lr=0.01, StepLR@epoch151 (halve LR), batch=256, DSA augmentation, CE loss
- **Small-scale SL**: 300 epochs, AdamW, lr=1e-3, Cosine scheduler, batch=256, DSA augmentation, KL-Div(T=20) loss
- **Paper uses DCBench pre-distilled datasets** - "adopt the standard setup provided by DCBench"
- **HL numbers = DCBench "best augmentation" column** (which is DSA)

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

## Strategy 
1. Fix evaluation to match paper's exact hyperparameters
2. Coreset methods (Random, K-centers) use real images - should match once eval is correct
3. For DD methods, need proper high-quality distillation (DCBench-level)
4. K-centers: Paper cites DeepCore; uses feature extraction + K-center algorithm

## Key Files
- convnet.py - ConvNet-D3 architecture ✓
- dsa.py - DSA augmentation ✓
- data_utils.py - Data loading ✓
- train_eval.py - Training/evaluation - NEEDS UPDATE to match paper hyperparams
- distill_dm.py, distill_dc.py, distill_tm.py - DD methods
- soft_labels.pt - Teacher soft labels

## What Needs to Change in Evaluation
1. HL: 300 epochs (not 1000), SGD lr=0.01, StepLR@151, batch=256, DSA aug, CE loss
2. SL: 300 epochs, AdamW lr=1e-3, Cosine scheduler, batch=256, DSA aug, KL-Div T=20
3. K-centers: Need proper implementation (DeepCore-style, not my broken version)

## Previous Results With Wrong Eval Settings
- Random HL IPC10: 18.83 (paper: 18.64) ✓ - close even with wrong settings
- Random HL IPC50: 35.25 (paper: 34.66) ✓ - close 
- K-centers HL IPC10: 10.73 (paper: 25.04) ✗✗ - K-centers algorithm broken
- DD methods: ~random level because distillation quality is poor

## Failed Approaches
1. K-centers using max-distance greedy: selects outliers, not representative points
2. DD with too few iterations: DM needs 20K+ iters for CIFAR-100
3. Using 1000 epochs for eval instead of 300
4. Using wrong optimizer for SL (SGD instead of AdamW)

## Plan
1. Write clean eval script with correct paper hyperparameters
2. Fix K-centers (use DeepCore-style: train model, extract features, K-means clustering)
3. Run coresets first (Random + K-centers) with correct eval → should match paper
4. Run DD methods with extended distillation
5. Generate final table + report
