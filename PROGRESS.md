# CGM-Agent Replication Progress

## Current Phase: COMPLETE - All tables done, reproduce.sh verified

## Implementation Plan
- [x] 1. Process CGM data for all 19 subjects (AZT1D + ShanghaiT2DM)
- [x] 2. Implement CGM analytical toolkit (all 12+ functions) - cgm_toolkit.py
- [x] 3. Generate synthetic QA pairs with ground truth - generate_questions.py (4180 pairs)
- [x] 4. Implement 3-layer agent pipeline + evaluation code - run_evaluation_v2.py
- [x] 5. Run agent on synthetic queries (Table 3) - P=0.85-0.99 R=0.57-0.75 F1=0.70-0.83 VA=0.48-0.75
- [x] 6. Table 8 (TIR correlation) - DONE (deterministic)
- [x] 7. Table 4 (Layer 1 feasibility classification) - Acc 0.81-0.92
- [x] 8. Table 5 (Real-world Layer 2) - F1 0.24-0.47
- [x] 9. Table 6 (Readability analysis) - FRE=58.1, FK=9.1
- [x] 10. Table 7 (Ablation study) - Layer 1 improves VA by +0.14
- [x] 11. Create reproduce.sh and REPORT.md - VERIFIED WORKING

## All Results Summary

### Table 3: Synthetic Layer 2 (6 models x 150 samples)
| Model | P | R | F1 | VA | Paper F1 | Paper VA |
|-------|---|---|----|----|----------|----------|
| GPT-5.2 | 0.94 | 0.75 | 0.83 | 0.71 | 0.86 | 0.81 |
| GPT-5-Mini | 0.87 | 0.63 | 0.73 | 0.68 | 0.80 | 0.94 |
| Gemini Pro | 0.97 | 0.57 | 0.72 | 0.73 | 0.80 | 0.94 |
| Gemini Flash | 0.99 | 0.60 | 0.75 | 0.75 | 0.81 | 0.94 |
| Llama-4-17B | 0.85 | 0.60 | 0.70 | 0.48 | 0.73 | 0.75 |
| Nemotron-9B | 0.89 | 0.63 | 0.74 | 0.53 | 0.52 | 0.67 |

### Table 4: Layer 1 Feasibility (5 models x 200 samples)
| Model | Acc | F1 | Paper Acc | Paper F1 |
|-------|-----|----|-----------| ---------|
| GPT-5.2 | 0.81 | 0.87 | 0.89 | 0.92 |
| GPT-5-Mini | 0.85 | 0.89 | 0.90 | 0.94 |
| Gemini Pro | 0.92 | 0.94 | 0.93 | 0.95 |
| Gemini Flash | 0.88 | 0.92 | 0.93 | 0.95 |
| Llama-4-17B | 0.84 | 0.88 | 0.78 | 0.83 |

### Table 5: Real-World Layer 2 (4 models x 100 samples)
| Model | F1 | VA | Paper F1 | Paper VA |
|-------|----|----|----------|----------|
| GPT-5.2 | 0.47 | 0.42 | 0.59 | 0.52 |
| GPT-5-Mini | 0.26 | 0.51 | 0.62 | 0.72 |
| Gemini Pro | 0.24 | 0.22 | 0.68 | 0.71 |
| Gemini Flash | 0.25 | 0.20 | 0.68 | 0.72 |

### Table 6: Readability
| Metric | Ours | Paper |
|--------|------|-------|
| Avg Length | 80 | 108 |
| FRE | 58.1 | 60.3 |
| FK Grade | 9.1 | 9.7 |

### Table 7: Ablation
| Condition | F1 | VA |
|-----------|----|----|
| Full pipeline | 0.28 | 0.24 |
| No Layer 1 | 0.31 | 0.10 |

### Table 8: TIR Correlation
| Metric | Our ρ | Paper ρ |
|--------|-------|---------|
| TIR-Normal vs HbA1c | -0.76 | -0.76 |
| Mean glucose vs HbA1c | 0.87 | 0.93 |

## Files
- cgm_toolkit.py: CGM analysis functions (12+ functions, tested)
- load_subjects.py: Subject data loading (19 subjects)
- generate_questions.py: QA dataset generation (4180 pairs)
- run_evaluation_v2.py: Main evaluation script (Tables 3-8)
- reproduce.sh: Reproduction script (verified working)
- REPORT.md: Final report
- results/: All generated results (table3-8 JSON files)
