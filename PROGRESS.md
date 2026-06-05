# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**EXECUTING EXPERIMENTS** - Running distillation and evaluation pipeline. Must work efficiently.

## Completed Results (8 of 20 configs)
| Method | IPC | Label | Mean Acc | Std | Paper Target |
|--------|-----|-------|----------|-----|-------------|
| Random | 10 | HL | 17.87 | 0.15 | 18.64±0.25 |
| Random | 10 | SL | 28.19 | 0.28 | 33.43±0.18 |
| Random | 50 | HL | 34.61 | 0.40 | 34.66±0.41 |
| Random | 50 | SL | 40.04 | 0.06 | 45.39±0.23 |
| K-centers | 10 | HL | 12.62 | 0.13 | 25.04±0.30 |
| K-centers | 10 | SL | 26.88 | 0.29 | 34.70±0.27 |
| K-centers | 50 | HL | 29.81 | 0.37 | 38.64±0.43 |
| K-centers | 50 | SL | 39.38 | 0.00 | 46.24±0.12 |

### Key Observations on Results So Far
- Random HL results are close to paper ✓ 
- K-centers HL results are LOWER than paper (12.62 vs 25.04 at IPC 10) - pixel-space k-centers may differ from paper's DeepCore k-centers
- SL results are systematically lower (~5 pts) - may need teacher quality improvement or hyperparameter tuning
- The TREND is correct: SL improves coreset performance relative to HL

### Cached Distillation Artifacts
- distilled_dm_ipc10.pt (1000 iters, batch_real=64) ✓
- Need: dm_ipc50, dc_ipc10, dc_ipc50, tm_ipc10, tm_ipc50

## Remaining Experiments (12 configs)
DD methods (DM, DC, TM) × 2 IPCs × 2 label types = 12 configs
Each needs: distillation + eval(HL) + eval(SL)

## Timing Budget
- DM distillation: ~80s (IPC 10), ~150s (IPC 50) with reduced iters
- DC distillation: ~60-120s per IPC
- TM: experts ~120s, matching ~60-120s per IPC
- Eval IPC 10: ~25s per run, Eval IPC 50: ~140s per run
- Total remaining estimate: ~30-40 min

## Strategy for Remaining Time
1. Run all distillations first (cache .pt files)
2. Run 1-run evaluations for all 12 DD configs
3. Add more runs if time permits
4. Generate results table, reproduce.sh, REPORT.md

## Key Paper Claims to Reproduce
1. **Table small_scale_c100**: DD methods (DM, DC, TM) vs coresets under HL and SL
2. Key insight: In HL, DD >> coresets; In SL, gap closes dramatically
3. Even if absolute numbers don't match, trends should be reproduced

## Paper Results Reference (CIFAR-100, ConvNet-D3)
### HL Setting:
| Method | IPC 10 | IPC 50 |
|--------|--------|--------|
| DM | 29.23±0.26 | 42.32±0.37 |
| DC | 28.42±0.29 | 30.56±0.56 |
| TM | 38.18±0.42 | 46.32±0.26 |
| Random | 18.64±0.25 | 34.66±0.41 |
| K-centers | 25.04±0.30 | 38.64±0.43 |

### SL Setting:
| Method | IPC 10 | IPC 50 |
|--------|--------|--------|
| DM | 26.13±0.10 | 43.46±0.18 |
| DC | 23.54±0.31 | 33.46±0.38 |
| TM | 37.60±0.25 | 46.26±0.30 |
| Random | 33.43±0.18 | 45.39±0.23 |
| K-centers | 34.70±0.27 | 46.24±0.12 |

## Hyperparameters 
### Student training:
- HL: 300 epochs, CE loss, SGD, lr=1e-2, StepLR@epoch 151, batch=256, DSA
- SL: 300 epochs, KL-Div (T=20), AdamW, lr=1e-3, Cosine schedule, batch=256, DSA

### DD method parameters (reduced for compute):
- DM: 1000 iterations (paper: 20000), lr_img=1.0, batch_real=64
- DC: 5 outer × 10 inner loops, lr_img=1.0
- TM: 5 experts × 20 epochs, 1000 matching iters, lr_img=0.1

## Known Issues / Failed Approaches
- K-centers in pixel space gives lower results than paper's DeepCore feature-space k-centers
- SL results ~5 points below paper - likely due to single teacher vs ensemble or shorter teacher training
- DM with 3000 iters + batch_real=256 was too slow (timeout at ~5min for IPC 10)
- Reduced to 1000 iters + batch_real=64 for speed

## Files
- convnet.py: ConvNet-D3 architecture
- dsa.py: Differentiable Siamese Augmentation
- data_utils.py: Data loading, coreset selection, soft label generation
- train_eval.py: Training and evaluation pipeline (HL and SL)
- distill_dm.py: Distribution Matching
- distill_dc.py: Dataset Condensation (gradient matching)
- distill_tm.py: Trajectory Matching
- run_single.py: Single experiment runner
- run_all.py: Full experiment orchestrator (not yet used successfully)
- results/results.json: Accumulated results
- soft_labels.pt: Pre-computed soft labels for training data
