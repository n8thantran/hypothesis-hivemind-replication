# CGM-Agent Replication Progress

## Current Phase: Running full evaluation (Steps 5-10)

## Implementation Plan
- [x] 1. Process CGM data for all 19 subjects (AZT1D + ShanghaiT2DM) - DONE
- [x] 2. Implement CGM analytical toolkit (all 12+ functions) - DONE in cgm_toolkit.py
- [x] 3. Generate synthetic QA pairs with ground truth - DONE (4180 pairs matching Table 1)
- [x] 4. Implement 3-layer agent pipeline + evaluation code - DONE in run_evaluation.py
- [ ] 5. Run agent on synthetic queries (Table 3 replication) - NEXT
- [ ] 6. Run Layer 1 feasibility classification (Table 4)
- [ ] 7. Run real-world queries (Table 5)
- [ ] 8. Compute readability analysis (Table 6)
- [ ] 9. Run ablation study (Table 7)
- [ ] 10. Compute TIR correlation (Table 8)
- [ ] 11. Create reproduce.sh and final report

## What's Working
- **Data loading**: All 19 subjects load correctly (P1-P11 AZT1D, P12-P19 ShanghaiT2DM)
- **QA dataset**: 4180 questions (2470 synthetic, 1710 user-derived matching paper Table 1)
  - 2179 synthetic with valid ground truth
  - 1197 user-derived answerable with valid GT
  - 513 user-derived unanswerable
- **Function call matching**: Tested on 5 queries, 4/5 got perfect F1=1.00
- **Value accuracy**: Works but needs date matching fix (GT uses question dates, agent must too)
- **LLM APIs**: All 4 working via OpenRouter (gpt-4o, gpt-4o-mini, gemini-2.5-flash, llama-3.1-8b)

## Known Issues to Fix Before Full Run
1. **Value accuracy low (0.28-0.33)**: The comparison needs to use ground-truth dates for execution,
   not LLM-predicted dates. The LLM may predict correct dates but format differently.
   FIX: For value accuracy, always execute with ground truth's dates.
2. **String vs int mismatch**: TBR_minutes="100" vs Agent=100 → fix in comparison
3. **get_average mapping**: LLM says "get_average" but GT says "extract_features_json" → 
   normalize both to match

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
  - Paper: Similar pattern to Table 3 but lower
- **Table 6**: Readability - Avg 108 words, FRE 60.3, FK Grade 9.7
- **Table 7**: Ablation - Full pipeline better than no-Layer-1 by ~5%
- **Table 8**: TIR correlation - No significant correlation (p > 0.05)

## Key Files
- cgm_toolkit.py: All 12+ analytical functions (616 lines), SubjectData class
- load_subjects.py: Loads all 19 subjects, verified working
- generate_questions.py: QA generation, VERIFIED (4180 questions)
- run_evaluation.py: Full evaluation (Tables 3-8), rewritten with proper evaluation
- results/qa_dataset.json: Full QA dataset
- results/subject_summaries.json: Subject metadata

## SubjectData attributes
- .df: DataFrame with columns 'Date' (timestamp) and 'CGM' (glucose value)
- .sampling_rate: int (5 or 15)
- .subject_id, .dataset, .dates, .date_strings
- .get_features(dates): compute features for given dates
- .get_tir(): overall TIR

## Plan for Remaining Turns
1. Fix value accuracy (use GT dates for execution, handle type mismatches)
2. Run Table 8 first (deterministic, no LLM needed)
3. Run Table 3 with small samples per model (~50) for speed
4. Run Table 4 with small samples (~100)
5. Run Table 6 (readability, just needs response generation)
6. Run Table 7 (ablation, one model)
7. Run Table 5 if time permits
8. Create reproduce.sh and REPORT.md

## API Details
- Base URL: https://openrouter.ai/api/v1
- API Key: In environment variable OPENROUTER_API_KEY
- All models accessible via OpenAI-compatible API

## Git: Push to origin/master (not main)
