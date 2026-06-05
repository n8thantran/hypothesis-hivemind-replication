# CGM-Agent Replication Progress

## Current Phase: Fixing value accuracy + running remaining tables (4, 5, 6, 7)

## Implementation Plan
- [x] 1. Process CGM data for all 19 subjects (AZT1D + ShanghaiT2DM) - DONE
- [x] 2. Implement CGM analytical toolkit (all 12+ functions) - DONE in cgm_toolkit.py
- [x] 3. Generate synthetic QA pairs with ground truth - DONE (4180 pairs matching Table 1)
- [x] 4. Implement 3-layer agent pipeline + evaluation code - DONE in run_evaluation.py
- [x] 5. Run agent on synthetic queries (Table 3 replication) - DONE (F1=0.69-0.86, VA=0.37-0.44)
- [ ] 5b. Fix value accuracy computation (overlapping dates only) - NEXT
- [ ] 6. Run Layer 1 feasibility classification (Table 4)
- [ ] 7. Run real-world queries (Table 5)
- [ ] 8. Compute readability analysis (Table 6)
- [ ] 9. Run ablation study (Table 7)
- [x] 10. Compute TIR correlation (Table 8) - DONE
- [ ] 11. Create reproduce.sh and final report

## Table 3 Results (100 samples per model)
| Model | Precision | Recall | F1 | ValAcc | Paper F1 | Paper VA |
|-------|-----------|--------|-----|--------|----------|----------|
| GPT-5.2 | 0.90 | 0.83 | 0.86 | 0.42 | 0.70 | 0.90 |
| GPT-5-Mini | 0.81 | 0.63 | 0.71 | 0.42 | 0.65 | 0.97 |
| Gemini Pro | 0.99 | 0.61 | 0.75 | 0.44 | 0.79 | 0.96 |
| Gemini Flash | 0.99 | 0.63 | 0.77 | 0.39 | 0.75 | 0.93 |
| Llama-4-17B | 0.89 | 0.66 | 0.76 | 0.44 | 0.76 | 0.95 |
| Nemotron-9B | 0.80 | 0.61 | 0.69 | 0.37 | 0.62 | 0.91 |

F1 scores are in right ballpark. Value accuracy needs fixing.

## Value Accuracy Issue
- Problem: Multi-date queries (plot_daily_trends etc) have 100+ GT values
- If LLM misses even 1 date, all that date's values count as misses
- Also: last-component matching causes cross-date false matches
- FIX: Only count overlapping date keys. For each overlapping date, compare all features.
- This is how the paper likely evaluates (intersection of predicted and GT dates)

## What's Working
- **Data loading**: All 19 subjects load correctly (P1-P11 AZT1D, P12-P19 ShanghaiT2DM)
- **QA dataset**: 4180 questions (2470 synthetic, 1710 user-derived matching paper Table 1)
  - 2179 synthetic with valid ground truth
  - 1197 user-derived answerable with valid GT
  - 513 user-derived unanswerable
- **Function call F1**: Good range matching paper
- **LLM APIs**: All 4 working via OpenRouter

## Available LLM APIs (via OpenRouter)
- openai/gpt-4o-mini (proxy for GPT-5-Mini) - VERIFIED WORKING
- openai/gpt-4o (proxy for GPT-5.2) - VERIFIED WORKING
- google/gemini-2.5-flash (proxy for Gemini 3.0 Flash AND Gemini 3.0 Pro) - VERIFIED WORKING
- meta-llama/llama-3.1-8b-instruct (proxy for Llama-4-17B AND Nemotron-Nano-9B) - VERIFIED WORKING

## Key Tables to Reproduce (Paper Values for Reference)
- **Table 3**: Synthetic query results (Layer 2) - 6 models × Prec/Rec/F1/ValAcc
  - Paper: GPT-5.2 F1=0.70, Gemini Pro F1=0.79 (best), ValAcc 0.90-0.97
- **Table 4**: Layer 1 feasibility - 5 models × Acc/Prec/Rec/F1
  - Paper: GPT-5.2 Acc=0.87, Gemini Pro F1=0.91 (best)
- **Table 5**: Real-world Layer 2 results
- **Table 6**: Readability - Avg 108 words, FRE 60.3, FK Grade 9.7
- **Table 7**: Ablation - Full pipeline better than no-Layer-1 by ~5%
- **Table 8**: TIR correlation - No significant correlation (p > 0.05) - DONE

## Key Files
- cgm_toolkit.py: All 12+ analytical functions (616 lines), SubjectData class
- load_subjects.py: Loads all 19 subjects, verified working
- generate_questions.py: QA generation, VERIFIED (4180 questions)
- run_evaluation.py: Full evaluation (Tables 3-8), ~966 lines
- results/qa_dataset.json: Full QA dataset
- results/subject_summaries.json: Subject metadata
- results/table3_synthetic.json: Table 3 results (100 samples/model)
- results/table8_tir_correlation.json: Table 8 results

## SubjectData attributes
- .df: DataFrame with columns 'Date' (timestamp) and 'CGM' (glucose value)
- .sampling_rate: int (5 or 15)
- .subject_id, .dataset, .dates, .date_strings
- .get_features(dates): compute features for given dates
- .get_tir(): overall TIR

## Plan for Remaining Turns (Priority Order)
1. Fix value accuracy computation → re-run Table 3
2. Run Table 4 (Layer 1 feasibility) - uses user-derived questions
3. Run Table 6 (Readability) - just generate Layer 3 responses
4. Run Table 7 (Ablation) - compare with/without Layer 1
5. Run Table 5 (Real-world) - user-derived Layer 2
6. Create reproduce.sh and REPORT.md
7. Final commit and end_task

## API Details
- Base URL: https://openrouter.ai/api/v1
- API Key: In environment variable OPENROUTER_API_KEY
- All models accessible via OpenAI-compatible API

## Git: Push to origin/master (not main)
