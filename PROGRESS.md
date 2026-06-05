# CGM-Agent Replication Progress

## Current Phase: Building question generation + agent pipeline

## Implementation Plan
- [x] 1. Process CGM data for all 19 subjects (AZT1D + ShanghaiT2DM) - DONE
- [x] 2. Implement CGM analytical toolkit (all 12+ functions) - DONE in cgm_toolkit.py
- [ ] 3. Generate synthetic QA pairs with ground truth (220/subject × 19 subjects = 4180)
- [ ] 4. Implement 3-layer agent pipeline (Input Processor, Analytical Agent, Response Generator)
- [ ] 5. Run agent on synthetic queries (Table 3 replication)
- [ ] 6. Implement Layer 1 feasibility classification (Table 4)
- [ ] 7. Implement evaluation metrics (Precision, Recall, F1, Value Accuracy)
- [ ] 8. Compute readability analysis (Table 6 - deterministic, uses textstat)
- [ ] 9. Compute TIR correlation (Table 8 - deterministic)
- [ ] 10. Create reproduce.sh and final report

## Available LLM APIs (via OpenRouter)
- openai/gpt-4o-mini (proxy for GPT-5-Mini)
- openai/gpt-4o (proxy for GPT-5.2)
- google/gemini-2.5-flash (proxy for Gemini 3.0 Flash)
- meta-llama/llama-3.1-8b-instruct (proxy for Llama-4 Maverick)

## Key Tables to Reproduce
- Table 3: Synthetic query results (Layer 2) - 6 models × Prec/Rec/F1/ValAcc
- Table 4: Layer 1 feasibility classification - 5 models × Acc/Prec/Rec/F1
- Table 5: Layer 2 real-world query results
- Table 6: Readability analysis (deterministic: Flesch-Kincaid etc.)
- Table 7: Ablation study
- Table 8: TIR correlation (deterministic: Pearson r, p-value)

## Data Status
- All 19 subjects loaded and verified
- AZT1D: P1-P11, 5-min sampling, ~45-49 days each
- ShanghaiT2DM: P12-P19, 15-min sampling, ~24-41 days each

## Key Files
- cgm_toolkit.py: All 12+ analytical functions (615 lines)
- load_subjects.py: Loads all 19 subjects, verified working
- results/subject_summaries.json: Subject metadata

## Question Template Types (from paper Section 4.3)
Synthetic templates cover 8 categories:
1. Basic retrieval (single day features)
2. Multi-day averages (with weartime filtering)
3. Conditional counting
4. Feature range (min/max across days)
5. Period comparison
6. Time-window analysis
7. Glucose excursion detection
8. Trend visualization

## Evaluation Metrics (paper Section 5.2)
- Precision = matched_features / agent_features
- Recall = matched_features / gt_features
- F1 = harmonic mean
- Value Accuracy = values_within_1pct / total_matched_features

## Failed Approaches
(none yet)
