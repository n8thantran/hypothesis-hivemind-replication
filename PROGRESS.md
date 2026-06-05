# CGM-Agent Replication Progress

## Current Phase: Question generation + agent pipeline (Steps 3-10)

## Implementation Plan
- [x] 1. Process CGM data for all 19 subjects (AZT1D + ShanghaiT2DM) - DONE
- [x] 2. Implement CGM analytical toolkit (all 12+ functions) - DONE in cgm_toolkit.py
- [~] 3. Generate synthetic QA pairs with ground truth - IN PROGRESS (bug: SubjectData is object not dict, need `.df` not `['df']`)
- [ ] 4. Implement 3-layer agent pipeline (Input Processor, Analytical Agent, Response Generator)
- [ ] 5. Run agent on synthetic queries (Table 3 replication)
- [ ] 6. Implement Layer 1 feasibility classification (Table 4)
- [ ] 7. Implement evaluation metrics (Precision, Recall, F1, Value Accuracy)
- [ ] 8. Compute readability analysis (Table 6 - deterministic, uses textstat)
- [ ] 9. Compute TIR correlation (Table 8 - deterministic)
- [ ] 10. Create reproduce.sh and final report

## Available LLM APIs (via OpenRouter)
- openai/gpt-4o-mini (proxy for GPT-5-Mini) - VERIFIED WORKING
- openai/gpt-4o (proxy for GPT-5.2) - VERIFIED WORKING
- google/gemini-2.5-flash (proxy for Gemini 3.0 Flash) - VERIFIED WORKING
- meta-llama/llama-3.1-8b-instruct (proxy for Llama-4 Maverick) - VERIFIED WORKING
- NOTE: google/gemini-2.0-flash-001 does NOT work (404)

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
- SubjectData objects have attributes: .df, .sampling_rate, .subject_id, .dataset, etc.

## Key Files
- cgm_toolkit.py: All 12+ analytical functions (615 lines), SubjectData class
- load_subjects.py: Loads all 19 subjects, verified working
- generate_questions.py: QA generation (NEEDS FIX: use subj_data.df not subj_data['df'])
- results/subject_summaries.json: Subject metadata

## SubjectData attributes (important!)
- .df: DataFrame with columns 'Date' (timestamp) and 'CGM' (glucose value)
- .sampling_rate: int (5 or 15)
- .subject_id: str (P1-P19)
- .dataset: str ('AZT1D' or 'ShanghaiT2DM')
- .dates: list of date objects
- .date_strings: list of date strings
- .get_features(dates): compute features for given dates
- .get_tir(): overall TIR

## DataFrame columns
- cgm_toolkit functions expect: 'timestamp' and 'glucose' columns
- load_subjects creates: 'Date' and 'CGM' columns
- NEED TO CHECK: may need to rename columns for toolkit compatibility

## Question Template Types (from paper Section 4.3)
8 categories: basic retrieval, multi-day averages, conditional counting, feature range, 
period comparison, time-window analysis, glucose excursion detection, trend visualization

## Evaluation Metrics (paper Section 5.2)
- Precision = matched_features / agent_features
- Recall = matched_features / gt_features
- F1 = harmonic mean
- Value Accuracy = values_within_1pct / total_matched_features

## Strategy for Remaining Work
1. Fix generate_questions.py (attribute access + column naming)
2. Generate all QA pairs deterministically
3. Build agent pipeline using LLM APIs
4. Run evaluation on manageable subset (~200-500 queries per model)
5. Compute deterministic metrics (readability, TIR correlation) 
6. Package everything in reproduce.sh

## Failed Approaches
- google/gemini-2.0-flash-001 returned 404, use google/gemini-2.5-flash instead
