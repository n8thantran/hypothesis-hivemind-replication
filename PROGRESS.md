# PROGRESS - Dataset Distillation Paper Replication

## Paper
"Rethinking Dataset Distillation: Hard Truths About Soft Labels" (CVPR 2026)

## Current Phase
**FINAL EXECUTION** - Writing clean pipeline to run all experiments and produce results table

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

## Assets Available
- Teacher model: 59.19% accuracy (teacher.pt)
- Soft labels for full training set: soft_labels.pt (50000, 100)
- Distilled sets: dm/dc/tm × ipc10/ipc50 (all .pt files)
- Soft labels for DD sets: soft_labels_{method}_{ipc}_v2.pt
- Core modules: convnet.py, dsa.py, data_utils.py, train_eval.py

## Plan
1. ✅ Write clean final_pipeline.py that evaluates everything
2. [ ] Run HL evaluations for all methods × IPCs (1 run each for speed)
3. [ ] Run SL evaluations for all methods × IPCs  
4. [ ] Format results table
5. [ ] Write REPORT.md and reproduce.sh
6. [ ] Final commit and push

## Known Issues
- SL numbers may be ~5% lower than paper due to teacher quality (59% vs potentially higher)
- The RELATIVE trends should still hold (this is what matters)
- Random HL IPC10 matches paper well: 18.64% vs 18.64±0.25

## Architecture
- ConvNet-D3: see convnet.py, 3 blocks, 128 channels, instance norm, avg pool, ReLU
