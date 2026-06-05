# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**FINAL PUSH** - Need to fix DD methods and K-centers, run all experiments, generate results.

## Target Results (from paper Table - CIFAR-100, ConvNet-D3)

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

## Current Results vs Targets

| Method | IPC | HL (mine) | HL (paper) | SL (mine) | SL (paper) |
|--------|-----|-----------|------------|-----------|------------|
| Random | 10 | 18.83 | 18.64 ✓ | 29.22 | 33.43 |
| Random | 50 | 35.25 | 34.66 ✓ | 41.77 | 45.39 |
| K-centers| 10 | 10.73 | 25.04 ✗✗ | 27.90 | 34.70 |
| K-centers| 50 | 26.51 | 38.64 ✗✗ | 40.58 | 46.24 |
| DM | 10 | 17.56 | 29.23 ✗✗ | 27.96 | 26.13 |
| DM | 50 | 34.46 | 42.32 ✗ | 41.09 | 43.46 |
| DC | 10 | 16.27 | 28.42 ✗✗ | 15.24 | 23.54 |
| DC | 50 | 29.68 | 30.56 ~✓ | 37.61 | 33.46 |
| TM | 10 | 17.36 | 38.18 ✗✗ | 28.70 | 37.60 |
| TM | 50 | 34.35 | 46.32 ✗✗ | 40.90 | 46.26 |

## Key Problems
1. **DD distillation quality is very poor** - DM/DC/TM all performing at ~random level
   - DM IPC10 gets 17.56 vs paper 29.23 (should be 10+ points above random)
   - Root cause: insufficient iterations/outer loops
2. **K-centers is broken** - gets BELOW random for HL
   - Root cause: K-centers algorithm selects outliers (farthest points), not representative centers
   - Should select MOST REPRESENTATIVE points, not diverse ones
3. **SL systematically ~4% below paper** - teacher quality (55.86% vs likely higher)

## Plan for Remaining Turns
1. Fix K-centers to use proper herding/center-finding algorithm
2. Re-run DM with 20000 iterations for IPC10 (IPC50 too expensive)
3. Re-run DC with 50 outer × 50 inner loops
4. For TM, need better experts (more epochs) + more TM iterations
5. Run 3-trial evaluations for key configs
6. Generate final table, write REPORT.md, reproduce.sh

## Critical Fix Needed: K-centers
The standard K-centers for coresets should select representative samples.
My implementation selects the FARTHEST points (greedy k-center), which creates diversity 
but not representativeness. For CIFAR-100 with limited IPC, we need the most informative samples.

Better approach: Use K-means clustering in feature space, select nearest-to-centroid sample.

## Modules
- convnet.py - ConvNet-D3 architecture ✓
- dsa.py - DSA augmentation ✓  
- data_utils.py - Data loading ✓
- train_eval.py - Training/evaluation with HL/SL settings ✓
- distill_dm.py - Distribution Matching ✓ (needs more iterations)
- distill_dc.py - Gradient Matching ✓ (needs more outer loops)
- distill_tm.py - Trajectory Matching ✓ (needs better experts)
- teacher_final.pt - Teacher model (55.86% acc) ✓
- soft_labels_final.pt - Soft labels from teacher ✓

## Failed Approaches
1. Pixel-space K-centers: selects outliers, terrible for HL
2. DM with 5000 iterations: not enough, same as random
3. DC with 10 outer loops: not enough convergence
4. TM with 3 experts, 2000 iterations: not enough
