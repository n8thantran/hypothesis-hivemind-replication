# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**RUNNING EXPERIMENTS** - All code is implemented and tested. Need to execute full experiment pipeline now.

## Key Paper Claims to Reproduce
1. **Table small_scale_c100**: On CIFAR-100/ConvNet-D3, DD methods (DM, DC, TM) outperform coresets (Random, K-centers) in HL setting, but gap closes dramatically in SL setting
2. **CAD-Prune**: Compute-aware difficulty-based pruning  
3. **DCS metric**: Correlation between distillation loss and downstream generalization

## Implementation Status
- [x] ConvNet-D3 architecture (convnet.py) - tested, correct shapes
- [x] DSA augmentation (dsa.py) - tested
- [x] Data loading from HuggingFace (data_utils.py) - tested, CIFAR-100 cached
- [x] Coreset methods: Random, K-centers (data_utils.py) - tested
- [x] Training pipeline with HL and SL settings (train_eval.py) - tested
- [x] DM distillation (distill_dm.py) - tested with small iterations
- [x] DC distillation (distill_dc.py) - tested with small iterations
- [x] TM distillation (distill_tm.py) - tested with small iterations
- [x] Main experiment runner (run_all.py) - ready to run
- [ ] Execute full experiments
- [ ] CAD-Prune implementation
- [ ] DCS metric
- [ ] Generate results tables and plots
- [ ] Write reproduce.sh and REPORT.md

## Key Hyperparameters (from paper)
### Training on distilled/coreset data:
- HL: 300 epochs, CE loss, SGD, lr=1e-2, StepLR@epoch 151, batch=256, DSA
- SL: 300 epochs, KL-Div (T=20), AdamW, lr=1e-3, Cosine schedule, batch=256, DSA

### DD method parameters (reduced for compute):
- DM: 3000 iterations (paper uses 20000), lr_img=1.0
- DC: 5 outer × 10 inner loops (paper uses more), lr_img=1.0  
- TM: 5 experts × 20 epochs, 1000 matching iters (paper uses 100 experts × 50 epochs, 5000 iters)

## Paper Results to Match (CIFAR-100, ConvNet-D3)
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

## Key Insight to Reproduce
In HL: DD methods >> coresets (e.g., TM 38.18 vs Random 18.64 at IPC 10)
In SL: Gap closes dramatically (e.g., TM 37.60 vs Random 33.43 at IPC 10; at IPC 50: TM 46.26 vs K-centers 46.24)

## Timing Estimates
- Soft label generation: ~5 min (1 teacher, 100 epochs)
- DM per IPC: ~4.5 min (3000 iters)
- DC per IPC: ~2.5 min (5×10)
- TM experts: ~6 min (5×20 epochs), matching per IPC: ~2.5 min
- Training eval: ~24s per run (300 epochs), 3 runs per config = ~72s
- Total for 20 configs: ~60-90 min

## Files
- convnet.py: ConvNet-D3 architecture
- dsa.py: Differentiable Siamese Augmentation
- data_utils.py: CIFAR-100 loading, Random/K-centers selection
- train_eval.py: Training with HL/SL, evaluation pipeline
- distill_dm.py: Distribution Matching
- distill_dc.py: Dataset Condensation (gradient matching)
- distill_tm.py: Trajectory Matching (expert training + matching)
- run_all.py: Main experiment runner with caching
- run_experiments.py: Alternative experiment runner (unused)

## Failed Approaches
- Previous implementation was for wrong paper (API evaluation paper) - completely restarted

## Next Steps
1. Run coreset experiments first (fast, ~10 min)
2. Run DD experiments (slower, ~30 min)
3. Generate results table
4. Implement CAD-Prune if time permits
5. Write reproduce.sh and REPORT.md
