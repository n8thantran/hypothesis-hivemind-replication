# CGM-Agent Replication Progress

## Current Phase: Fix VA computation, then run all remaining tables (3-7)

## Implementation Plan
- [x] 1. Process CGM data for all 19 subjects (AZT1D + ShanghaiT2DM) - DONE
- [x] 2. Implement CGM analytical toolkit (all 12+ functions) - DONE in cgm_toolkit.py
- [x] 3. Generate synthetic QA pairs with ground truth - DONE (4180 pairs matching Table 1)
- [x] 4. Implement 3-layer agent pipeline + evaluation code - DONE in run_evaluation_v2.py
- [x] 5. Run agent on synthetic queries (Table 3 initial) - DONE (F1 good, VA=0.37-0.44 too low)
- [ ] 5b. Fix value accuracy computation → re-run Table 3 - IN PROGRESS
- [ ] 6. Run Table 4 (Layer 1 feasibility classification)
- [ ] 7. Run Table 5 (Real-world Layer 2)
- [ ] 8. Run Table 6 (Readability analysis)
- [ ] 9. Run Table 7 (Ablation study)
- [x] 10. Table 8 (TIR correlation) - DONE (deterministic)
- [ ] 11. Create reproduce.sh and REPORT.md

## Value Accuracy Bug Analysis (CRITICAL)
Root cause identified: When toolkit runs with same dates, values match 100%.
The VA issue is in key matching:
1. For multi_day_average/feature_range/conditional_count: GT key is "(2024-02-05, 2024-02-07)" 
   - This tuple key doesn't match date regex \d{4}-\d{2}-\d{2}
   - If LLM predicts different date range, tuple key changes entirely
2. For trend/basic_retrieval: GT has date keys like "2024-02-05"
   - If LLM predicts same dates, values match perfectly
   - Overlapping-date logic works but many dates may not overlap
3. For empty features (570/2179 queries): ALL 18+ features compared
   - This is correct since toolkit returns all features

FIX STRATEGY: 
- For tuple-keyed results (multi_day_average etc): Compare values regardless of key name
  since there's only one result dict. Just flatten all values and compare by feature name.
- For date-keyed results: Keep overlapping-date logic but be more flexible on key matching
- Key insight: If function call is correct AND dates overlap, VA should be ~100%

## Paper Table Values (for reference)
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
| GPT-5-Mini | 0.91 | 0.90 | 0.97 | 0.94 |
| Gemini Pro | 0.96 | 0.95 | 1.00 | 0.97 |
| Gemini Flash | 0.95 | 0.93 | 1.00 | 0.96 |
| Llama-4-17B | 0.86 | 0.87 | 0.95 | 0.91 |

### Table 5 (Real-world Layer 2, N=1197)
| Model | Prec | Rec | F1 | ValAcc |
|-------|------|-----|-----|--------|
| GPT-5.2 | 0.65 | 0.76 | 0.70 | 0.82 |
| GPT-5-Mini | 0.56 | 0.68 | 0.62 | 0.86 |
| Gemini Pro | 0.65 | 0.62 | 0.64 | 0.88 |
| Gemini Flash | 0.65 | 0.58 | 0.61 | 0.86 |
| Llama-4-17B | 0.44 | 0.66 | 0.53 | 0.44 |

### Table 6 (Readability)
Avg Length: 108 words, FRE: 60.3, FK Grade: 9.7

### Table 7 (Ablation, Gemini Pro, N=1197)
Full Pipeline: F1=0.64, VA=0.88
w/o Input Processor: F1=0.45, VA=0.82

### Table 8 (TIR Correlation)
Overall: TIR vs F1 r=0.385 p=0.104, TIR vs VA r=0.095 p=0.699
T1D: r=0.358 p=0.280, r=-0.044 p=0.898
T2D: r=0.622 p=0.100, r=0.181 p=0.667

## Key Files
- cgm_toolkit.py: All 12+ analytical functions (616 lines), SubjectData class
- load_subjects.py: Loads all 19 subjects, verified working
- generate_questions.py: QA generation, VERIFIED (4180 questions)
- run_evaluation_v2.py: Full evaluation (Tables 3-8), ~1170 lines
- results/qa_dataset.json: Full QA dataset
- results/subject_summaries.json: Subject metadata
- results/table3_synthetic.json: Table 3 initial results
- results/table8_tir_correlation.json: Table 8 results

## Available LLM APIs (via OpenRouter)
- openai/gpt-4o (proxy for GPT-5.2) - VERIFIED
- openai/gpt-4o-mini (proxy for GPT-5-Mini) - VERIFIED
- google/gemini-2.5-flash (proxy for Gemini 3.0 Flash AND Pro) - VERIFIED
- meta-llama/llama-3.1-8b-instruct (proxy for Llama-4-17B AND Nemotron-9B) - VERIFIED

## SubjectData attributes
- .df: DataFrame with columns 'Date' (timestamp) and 'CGM' (glucose value)
- .sampling_rate: int (5 or 15)
- .subject_id, .dataset, .dates, .date_strings
- .get_features(dates): compute features for given dates
- .get_tir(): overall TIR

## API Details
- Base URL: https://openrouter.ai/api/v1
- API Key: In environment variable OPENROUTER_API_KEY
- All models accessible via OpenAI-compatible API

## Git: Push to origin/master (not main)
