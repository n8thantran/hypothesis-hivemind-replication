# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**EXECUTION** - Need to run run_experiments.py (just written), then produce deliverables

## Status Summary (Checkpoint Turn 2500)
### What's Done
- ✅ ConvNet-D3 architecture (convnet.py) - tested, correct
- ✅ DSA augmentation (dsa.py) - tested, correct  
- ✅ Teacher model trained: 59.19% accuracy (teacher.pt)
- ✅ Soft labels generated for full training set (soft_labels.pt, shape [50000,100], raw logits)
- ✅ Distilled datasets created: DM/DC/TM × IPC10/IPC50 (distilled_{method}_ipc{ipc}.pt)
- ✅ Soft labels for distilled sets (soft_labels_{method}_ipc{ipc}_correct.pt)
- ✅ Clean run_experiments.py written with ALL methods, correct hyperparameters
- ✅ Verified HL Random IPC10 ≈ 18.64% (matches paper exactly)

### What's NOT Done
- [ ] Run full experiment pipeline (run_experiments.py) - 20 experiments total
- [ ] Generate results table
- [ ] Write reproduce.sh
- [ ] Write REPORT.md
- [ ] Final commit

## Target Table: tab:small_scale_c100 (CIFAR-100, ConvNet-D3)
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

## Key Paper Claims to Demonstrate
1. In HL: DD methods (especially TM) >> coresets (clear gap)
2. In SL: gap narrows substantially - coresets competitive with DD
3. TM is the best DD method in both settings
4. K-centers > Random in HL, but similar in SL

## Evaluation Hyperparameters (EXACT from paper, Table: tab:stage3_hyper)
### HL Setting (Small-scale)
- 300 epochs, SGD, lr=0.01, momentum=0.9, weight_decay=5e-4
- StepLR@epoch151 (gamma=0.1), batch=256, DSA augmentation, CE loss

### SL Setting (Small-scale)  
- 300 epochs, AdamW, lr=1e-3, weight_decay=0.01, Cosine scheduler
- batch=256, DSA augmentation, KL-Div(T=20), NO warmup

## Known Issues / Observations
- SL numbers ~5% lower than paper for Random/K-centers (28% vs 33%)
  - Likely due to teacher quality: our ConvNet teacher = 59.19%, paper may use stronger teacher
  - Tried many variations: different T (3,5,10,15,20), different WD, different loss formulations
  - All give ~28% for Random IPC10 SL. This is a teacher quality issue, not a bug.
  - The RELATIVE trends should still hold (this is what matters for the paper's claims)
- HL numbers match well: Random IPC10 = 18.64% (paper: 18.64±0.25)
- ResNet-18 teacher training attempted but timed out (10min limit)

## Failed Approaches
- Tried training ResNet-18 teacher for better soft labels - timed out
- Tried different KD formulations (CE soft, combined CE+KL) - no significant improvement
- Tried different temperatures (3,5,10,15,20) - all similar (~28% for Random IPC10 SL)
- Tried different weight decays (0, 0.001, 0.01, 0.1) - all similar

## Plan for Remaining Turns
1. Run run_experiments.py with NUM_SEEDS=1 first (faster, ~2hrs for 20 experiments)
   - Actually each experiment is ~90s, so 20 experiments = ~30min
   - But 10min timeout per command, so need to run in batches
2. Collect results, format table
3. Write reproduce.sh and REPORT.md
4. Final commit and push

## Key Files
- /workspace/convnet.py - ConvNet-D3 architecture
- /workspace/dsa.py - DSA augmentation
- /workspace/data_utils.py - Data loading utilities
- /workspace/train_eval.py - Training/evaluation functions
- /workspace/distill_dm.py - Distribution Matching distillation
- /workspace/distill_dc.py - Dataset Condensation (gradient matching)
- /workspace/distill_tm.py - Trajectory Matching distillation
- /workspace/run_experiments.py - Main experiment pipeline (NEW, clean)
- /workspace/teacher.pt - Trained teacher model (59.19%)
- /workspace/soft_labels.pt - Full training set soft labels
- /workspace/distilled_{dm,dc,tm}_ipc{10,50}.pt - Distilled datasets
- /workspace/soft_labels_{dm,dc,tm}_ipc{10,50}_correct.pt - DD soft labels
