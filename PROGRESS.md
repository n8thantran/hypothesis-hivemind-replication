# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**V2 IMPROVEMENTS** - Improving results to better match paper values.

## V1 Results (baseline, under-trained)

| Method | IPC | Label | Ours | Paper | Diff | Notes |
|--------|-----|-------|------|-------|------|-------|
| DM | 10 | HL | 17.56 | 29.23 | -11.67 | 1000 iter, 1 run |
| DM | 10 | SL | 27.96 | 26.13 | +1.83 | |
| DM | 50 | HL | 34.46 | 42.32 | -7.86 | |
| DM | 50 | SL | 41.09 | 43.46 | -2.37 | |
| DC | 10 | HL | 16.27 | 28.42 | -12.15 | 5×10 loops, 1 run |
| DC | 10 | SL | 15.24 | 23.54 | -8.30 | |
| DC | 50 | HL | 29.68 | 30.56 | -0.88 | |
| DC | 50 | SL | 37.61 | 33.46 | +4.15 | |
| TM | 10 | HL | 17.36 | 38.18 | -20.82 | 3 experts, 1000 iter |
| TM | 10 | SL | 28.70 | 37.60 | -8.90 | |
| TM | 50 | HL | 34.35 | 46.32 | -11.97 | |
| TM | 50 | SL | 40.90 | 46.26 | -5.36 | |
| Random | 10 | HL | 17.87 | 18.64 | -0.77 | 3 runs ✓ |
| Random | 10 | SL | 28.19 | 33.43 | -5.24 | 3 runs ✓ |
| Random | 50 | HL | 34.61 | 34.66 | -0.05 | 3 runs ✓ |
| Random | 50 | SL | 40.04 | 45.39 | -5.35 | 3 runs ✓ |
| K-centers | 10 | HL | 12.62 | 25.04 | -12.42 | pixel-space, 3 runs |
| K-centers | 10 | SL | 26.88 | 34.70 | -7.82 | |
| K-centers | 50 | HL | 29.81 | 38.64 | -8.83 | |
| K-centers | 50 | SL | 39.38 | 46.24 | -6.86 | |

## V2 Improvement Plan
1. [x] Created run_improved.py with better params
2. [ ] Fix K-centers to use feature-space embeddings  
3. [ ] Train stronger teacher (300 epochs) for better soft labels
4. [ ] Re-run DM with 5000 iterations
5. [ ] Re-run DC with 20×20 outer×inner loops
6. [ ] Re-run TM with 10 experts, 50 epochs, 5000 iterations
7. [ ] 3 evaluation runs for all DD configs  
8. [ ] Update results, analysis, reproduce.sh, REPORT.md

## V1 Root Causes
- **K-centers**: pixel-space distance instead of feature-space → bad class coverage
- **DD methods under-trained**: DM 1000 iter (need 5000+), DC 50 loops (need 400+), TM 3 experts / 1000 iter
- **SL gap**: Teacher model quality / soft label temperature may differ
- **Only 1 eval run** for DD methods → noisy estimates

## Timing Estimates (from benchmarks)
- DM: 0.34s/iter → 5000 iters = 28 min/IPC → 56 min total
- DC: 1.9s/inner loop → 20×20=400 loops = 12.9 min/IPC → 26 min total 
- TM experts: 10 experts × 50 epochs × ~20s/epoch = ~2.8 hours (too slow)
- TM distill: ~5000 iters each → need to benchmark
- Training eval: ~25s/run × 4 IPC × 3 runs × 5 methods = ~25 min
- Total feasible time: ~2-3 hours for improved DM+DC, skip TM re-training

## Hyperparameters from Paper (Table in Appendix)
### Student Training (small-scale CIFAR-100)
- **HL**: Epochs=300, CE loss, SGD lr=0.01, StepLR@151 (γ=0.1), batch=256, DSA aug
- **SL**: Epochs=300, KL-Div T=20, AdamW lr=0.001, Cosine schedule, batch=256, DSA aug

## Key Files
- convnet.py: ConvNet-D3 architecture
- dsa.py: Differentiable Siamese Augmentation
- data_utils.py: CIFAR-100 data loading (+ k_centers_select)
- train_eval.py: Training/evaluation with HL and SL
- distill_dm.py: Distribution Matching
- distill_dc.py: Dataset Condensation (gradient matching)
- distill_tm.py: Trajectory Matching
- run_improved.py: V2 improved runner (NEW)
- generate_results.py: Results table generator
- reproduce.sh: Full reproduction script
- results/results.json: V1 results
- REPORT.md: Final report (needs update)

## Failed Approaches
- K-centers in pixel space: gives 12.62% HL IPC10 vs paper 25.04% (-12.42 gap)
- DC with 5×10 loops: severely under-trained, 16.27% vs 28.42%
- TM with only 3 experts, 20 epochs: under-trained
