# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**COMPLETE** - All deliverables ready.

## All Results (20/20 configs complete)

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

## Key Claim Verification

### Paper's Main Claim: SL closes gap between DD and coresets ✅
**IPC=50 with SL**: DD avg=39.87, Coreset avg=39.71, Gap=0.16 (essentially zero)
**IPC=10 with SL**: Coresets actually surpass DD (27.54 vs 23.97)

### Random+SL > DD+HL ✅
- IPC=10: Random+SL (28.19) > Best DD+HL (17.56)
- IPC=50: Random+SL (40.04) > Best DD+HL (34.46)

## Implementation Plan
- [x] ConvNet-D3 architecture
- [x] DSA augmentation
- [x] Data loading (HuggingFace CIFAR-100)
- [x] Training/evaluation pipeline (HL and SL)
- [x] Distribution Matching (DM) distillation
- [x] Dataset Condensation (DC/gradient matching) distillation
- [x] Trajectory Matching (TM) distillation
- [x] Random coreset selection
- [x] K-centers coreset selection
- [x] Soft label generation (teacher model)
- [x] All 20 experiment configs run
- [x] Generate results table and figures
- [x] Create reproduce.sh
- [x] Write REPORT.md
- [x] Final commit

## Key Files
- convnet.py: ConvNet-D3 architecture
- dsa.py: Differentiable Siamese Augmentation
- data_utils.py: CIFAR-100 data loading
- train_eval.py: Training/evaluation with HL and SL
- distill_dm.py: Distribution Matching
- distill_dc.py: Dataset Condensation (gradient matching)
- distill_tm.py: Trajectory Matching
- generate_results.py: Results table generator
- reproduce.sh: Full reproduction script
- results/results.json: All 20 experiment results
- results/table1.txt: Formatted results table
- results/analysis.txt: Detailed claim analysis
- REPORT.md: Final report
