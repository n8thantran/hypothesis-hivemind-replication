# CGM-Agent Replication Progress

## Current Phase: Implementing agent pipeline + evaluation (Steps 4-10)

## Implementation Plan
- [x] 1. Process CGM data for all 19 subjects (AZT1D + ShanghaiT2DM) - DONE
- [x] 2. Implement CGM analytical toolkit (all 12+ functions) - DONE in cgm_toolkit.py
- [x] 3. Generate synthetic QA pairs with ground truth - DONE (4180 pairs matching Table 1)
- [ ] 4. Implement 3-layer agent pipeline (Input Processor, Analytical Agent, Response Generator)
- [ ] 5. Run agent on synthetic queries (Table 3 replication)
- [ ] 6. Implement Layer 1 feasibility classification (Table 4)
- [ ] 7. Implement evaluation metrics (Precision, Recall, F1, Value Accuracy)
- [ ] 8. Compute readability analysis (Table 6 - deterministic, uses textstat)
- [ ] 9. Compute TIR correlation (Table 8 - deterministic once per-subject metrics available)
- [ ] 10. Create reproduce.sh and final report

## Available LLM APIs (via OpenRouter)
- openai/gpt-4o-mini (proxy for GPT-5-Mini) - VERIFIED WORKING
- openai/gpt-4o (proxy for GPT-5.2) - VERIFIED WORKING
- google/gemini-2.5-flash (proxy for Gemini 3.0 Flash) - VERIFIED WORKING
- meta-llama/llama-3.1-8b-instruct (proxy for Llama-4 Maverick) - VERIFIED WORKING

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
- QA dataset: 4180 questions saved to results/qa_dataset.json

## Key Files
- cgm_toolkit.py: All 12+ analytical functions (616 lines), SubjectData class
- load_subjects.py: Loads all 19 subjects, verified working
- generate_questions.py: QA generation, VERIFIED (4180 questions generated)
- results/qa_dataset.json: Full QA dataset
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

## Implementation Strategy
1. Implement agent pipeline (Layer 1 + Layer 2 + Layer 3) with function calling
2. For Table 3 (synthetic): Run Layer 2 directly (no Layer 1 needed since intent is clear)
3. For Table 4 (Layer 1): Run feasibility classification on user-derived questions
4. Use deterministic evaluation where possible, LLM judge for complex cases
5. For readability (Table 6): Use textstat library on generated responses
6. For TIR correlation (Table 8): Pearson correlation on per-subject metrics

## API Details
- Base URL: https://openrouter.ai/api/v1
- API Key: In environment variable OPENROUTER_API_KEY
- All models accessible via OpenAI-compatible API

## Git: Push to origin/master (not main)
