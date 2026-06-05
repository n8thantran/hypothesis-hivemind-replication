# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
Starting fresh implementation. Previous code was for wrong paper.

## Key Paper Claims to Reproduce
1. **Small-scale DD vs coresets (Table: small_scale_c100)**: On CIFAR-100 with ConvNet-D3, TM outperforms random/K-centers in HL, but gap closes in SL setting
2. **CAD-Prune**: Compute-aware pruning metric that selects samples of optimal difficulty
3. **CA2D**: Compute-aware DD method using RDED-style patches from CAD-Prune selected samples
4. **DCS metric**: Correlation between distillation loss and downstream generalization

## Feasible Scope (given compute constraints - no ImageNet-1K)
Focus on CIFAR-100 small-scale experiments:
- Table small_scale_c100: DM, DC, TM vs Random, K-centers on CIFAR-100 (ConvNet-D3, HL+SL)
- CAD-Prune on CIFAR-100
- DCS metric computation

## Implementation Plan
- [ ] Set up environment (install packages)
- [ ] Implement ConvNet-D3 architecture
- [ ] Download CIFAR-100
- [ ] Implement training pipeline (HL and SL settings)
- [ ] Implement coreset methods: Random, K-centers
- [ ] Implement EL2N scoring
- [ ] Implement CAD-Prune
- [ ] Implement DD methods: DC, DM, TM
- [ ] Run small-scale experiments (CIFAR-100, ConvNet-D3)
- [ ] Implement DCS metric
- [ ] Generate results tables
- [ ] Write reproduce.sh and REPORT.md

## Key Hyperparameters (from Table stage3_hparams)
### Small-scale (CIFAR-100):
- HL: 300 epochs, CE loss, SGD, lr=1e-2, StepLR@151, batch=256, DSA augmentation
- SL: 300 epochs, KL-Div (T=20), AdamW, lr=1e-3, Cosine, batch=256, DSA augmentation

## Paper Results to Match (CIFAR-100, ConvNet-D3, HL setting)
| Method | IPC 10 | IPC 50 |
|--------|--------|--------|
| DM | 29.23±0.26 | 42.32±0.37 |
| DC | 28.42±0.29 | 30.56±0.56 |
| TM | 38.18±0.42 | 46.32±0.26 |
| Random | 18.64±0.25 | 34.66±0.41 |
| K-centers | 25.04±0.30 | 38.64±0.43 |

## Paper Results (CIFAR-100, ConvNet-D3, SL setting)
| Method | IPC 10 | IPC 50 |
|--------|--------|--------|
| DM | 26.13±0.10 | 43.46±0.18 |
| DC | 23.54±0.31 | 33.46±0.38 |
| TM | 37.60±0.25 | 46.26±0.30 |
| Random | 33.43±0.18 | 45.39±0.23 |
| K-centers | 34.70±0.27 | 46.24±0.12 |

## Completed Work
(none yet)

## Failed Approaches
(none yet)
