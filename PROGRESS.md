# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**V2 EXECUTION** - Running improved experiments to match paper values.

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

## V1 Results (baseline, under-trained)

| Method | IPC | Label | Ours | Paper | Diff |
|--------|-----|-------|------|-------|------|
| DM | 10 | HL | 17.56 | 29.23 | -11.67 |
| DM | 10 | SL | 27.96 | 26.13 | +1.83 |
| DM | 50 | HL | 34.46 | 42.32 | -7.86 |
| DM | 50 | SL | 41.09 | 43.46 | -2.37 |
| DC | 10 | HL | 16.27 | 28.42 | -12.15 |
| DC | 10 | SL | 15.24 | 23.54 | -8.30 |
| DC | 50 | HL | 29.68 | 30.56 | -0.88 |
| DC | 50 | SL | 37.61 | 33.46 | +4.15 |
| TM | 10 | HL | 17.36 | 38.18 | -20.82 |
| TM | 10 | SL | 28.70 | 37.60 | -8.90 |
| TM | 50 | HL | 34.35 | 46.32 | -11.97 |
| TM | 50 | SL | 40.90 | 46.26 | -5.36 |
| Random | 10 | HL | 17.87 | 18.64 | -0.77 |
| Random | 10 | SL | 28.19 | 33.43 | -5.24 |
| Random | 50 | HL | 34.61 | 34.66 | -0.05 |
| Random | 50 | SL | 40.04 | 45.39 | -5.35 |
| K-centers | 10 | HL | 12.62 | 25.04 | -12.42 |
| K-centers | 10 | SL | 26.88 | 34.70 | -7.82 |
| K-centers | 50 | HL | 29.81 | 38.64 | -8.83 |
| K-centers | 50 | SL | 39.38 | 46.24 | -6.86 |

## V2 Improvement Plan
1. [ ] Train better teacher model (300 epochs, proper schedule) → better soft labels
2. [ ] Fix K-centers to use feature-space embeddings from trained model
3. [ ] Re-run DM with 20000 iterations (paper standard)
4. [ ] Re-run DC with 50 outer × 50 inner loops (2500 total steps)
5. [ ] Re-run TM with 10 experts, 50 epochs, 5000 iterations
6. [ ] 3 evaluation runs for all configs
7. [ ] Update results, analysis, reproduce.sh, REPORT.md

## Root Causes for V1 Gaps
- **SL gap (~5% across all methods)**: Teacher only trained 200 epochs with basic schedule. Paper uses well-tuned teacher.
- **K-centers HL gap (-12.42)**: Pixel-space distance → bad coverage. Need feature-space.
- **DD HL gaps**: DM 1000 iter (need 20000), DC 50 loops (need 2500), TM 3 experts/1000 iter
- **Only 1 eval run** for DD methods → noisy

## Hyperparameters from Paper
### Student Training (small-scale CIFAR-100)
- **HL**: Epochs=300, CE loss, SGD lr=0.01, StepLR@151 (γ=0.1), batch=256, DSA aug
- **SL**: Epochs=300, KL-Div T=20, AdamW lr=0.001, Cosine schedule, batch=256, DSA aug

### DD Synthesis (from DCBench standard)
- **DM**: lr_img=1.0, 20000 iterations, batch_real=256
- **DC**: lr_img=1.0, outer=50, inner=50, batch_real=256
- **TM**: 100 experts, 50 epochs each, lr_img=1000, syn_steps=30, 5000 iterations

## Key Files
- convnet.py: ConvNet-D3 architecture
- dsa.py: Differentiable Siamese Augmentation
- data_utils.py: CIFAR-100 data loading (+ k_centers_select)
- train_eval.py: Training/evaluation with HL and SL
- distill_dm.py: Distribution Matching
- distill_dc.py: Dataset Condensation (gradient matching)
- distill_tm.py: Trajectory Matching
- run_v2.py: V2 comprehensive runner (NEW)
- results/results.json: V1 results
- REPORT.md: Final report (needs update)

## Failed Approaches
- K-centers in pixel space: gives 12.62% HL IPC10 vs paper 25.04% (-12.42 gap)
- DC with 5×10 loops: severely under-trained, 16.27% vs 28.42%
- TM with only 3 experts, 20 epochs: under-trained
- Teacher with 200 epochs, basic schedule: SL results ~5% below paper across all methods
