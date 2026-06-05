# CGM-Agent Replication Progress

## Current Phase: Fix VA computation + conditional_count GT, then run all tables

## Implementation Plan
- [x] 1. Process CGM data for all 19 subjects (AZT1D + ShanghaiT2DM)
- [x] 2. Implement CGM analytical toolkit (all 12+ functions) - cgm_toolkit.py
- [x] 3. Generate synthetic QA pairs with ground truth - generate_questions.py (4180 pairs)
- [x] 4. Implement 3-layer agent pipeline + evaluation code - run_evaluation_v2.py
- [x] 5. Run agent on synthetic queries (Table 3 initial) - F1 good, VA=0.37-0.44 too low
- [ ] 5b. Fix VA computation + regenerate conditional_count GTs → re-run Table 3
- [ ] 6. Run Table 4 (Layer 1 feasibility classification)
- [ ] 7. Run Table 5 (Real-world Layer 2)
- [ ] 8. Run Table 6 (Readability analysis)
- [ ] 9. Run Table 7 (Ablation study)
- [x] 10. Table 8 (TIR correlation) - DONE (deterministic)
- [ ] 11. Create reproduce.sh and REPORT.md

## CRITICAL BUGS FIXED (but not yet tested)
### Bug 1: conditional_count GT generation
- File: generate_questions.py line 78
- OLD: `count_satisfied_condition(all_features, condition)` - wrong signature
- NEW: Parse condition string with regex, call `count_satisfied_condition(all_features, feat_name, op, float(thresh))`
- Result: 285 conditional_count queries had error GTs; now should work
- STATUS: Fixed in code, need to regenerate QA dataset

### Bug 2: Value Accuracy computation
- File: run_evaluation_v2.py, compute_value_accuracy()
- Problem: For tuple-keyed results (multi_day_average, feature_range, conditional_count), 
  the GT key is like "(2024-02-05, 2024-02-07)" and agent key may differ
- Current code only matches by exact key or same-prefix feature name
- FIX NEEDED: When top-level keys are NOT date-format, match by feature name only
  (flatten all values and compare features regardless of parent key)
- STATUS: NOT YET FIXED - need to update compute_value_accuracy()

## Value Accuracy Fix Strategy
In compute_value_accuracy:
1. Detect if results are "date-keyed" (keys match YYYY-MM-DD) vs "range-keyed" (tuple/other)
2. For date-keyed: keep overlapping-date logic (works well)
3. For range-keyed (or single result dict): match by feature name only, ignoring parent key
4. This should bring VA from ~0.40 to ~0.80+ since toolkit is deterministic

## Paper Table Target Values
### Table 3 (Synthetic, N=2470)
| Model | Prec | Rec | F1 | ValAcc |
|-------|------|-----|-----|--------|
| GPT-5.2 | 0.84 | 0.89 | 0.86 | 0.81 |
| GPT-5-Mini | 0.75 | 0.87 | 0.80 | 0.94 |
| Gemini Pro | 0.92 | 0.71 | 0.80 | 0.94 |
| Gemini Flash | 0.89 | 0.74 | 0.81 | 0.94 |
| Llama-4-17B | 0.73 | 0.74 | 0.73 | 0.75 |
| Nemotron-9B | 0.45 | 0.61 | 0.52 | 0.67 |

### Table 4 (Layer 1 Feasibility, N=1710)
| Model | Acc | Prec | Rec | F1 |
|-------|-----|------|-----|-----|
| GPT-5.2 | 0.92 | 0.90 | 1.00 | 0.94 |
| Gemini Pro | 0.96 | 0.95 | 1.00 | 0.97 |

### Table 5 (Real-world Layer 2, N=1197)
Lower F1 than synthetic (0.56-0.70), higher VA (0.82-0.88)

### Table 6 (Readability)
Avg length ~108 words, Flesch ~60.3, FK grade ~9.7

### Table 7 (Ablation)
Full pipeline F1 ~0.63, No-L1 F1 ~0.47 (big drop)

### Table 8 (TIR Correlation) - DONE
No significant correlation between TIR and agent performance

## File Structure
- load_subjects.py - Load 19 CGM subjects from data/
- cgm_toolkit.py - All CGM analytical functions (12+)
- generate_questions.py - Generate 4180 QA pairs (FIXED conditional_count)
- run_evaluation_v2.py - Main evaluation: Tables 3-8 (VA FIX NEEDED)
- results/qa_dataset.json - 4180 QA pairs (needs regeneration for conditional_count fix)
- results/table8_tir_correlation.json - Table 8 results
- results/table3_synthetic.json - Table 3 initial (VA too low)

## Failed Approaches
1. Initial VA computation used exact key matching - failed for tuple keys
2. Prefix-based matching still fails when date ranges differ between GT and agent
3. conditional_count GT generation was broken due to wrong function signature

## Next Steps (Priority Order)
1. Fix compute_value_accuracy() in run_evaluation_v2.py
2. Regenerate QA dataset (python3 generate_questions.py)
3. Run Table 3 with fixed VA (python3 run_evaluation_v2.py 3)
4. Run Tables 4,5,6,7 (python3 run_evaluation_v2.py 4 5 6 7)
5. Create reproduce.sh and REPORT.md
