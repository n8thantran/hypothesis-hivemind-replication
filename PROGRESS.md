# Progress Tracking

## Current Phase
COMPLETE - All figures generated, reproduce.sh tested, REPORT.md written. Ready to call end_task.

## Implementation Plan
- [x] Set up environment and check available resources
- [x] Download all 50 papers from OpenReview
- [x] Verify API access (OpenRouter: all 6 models + text-embedding-3-small confirmed)
- [x] Extract text from PDFs using PyMuPDF
- [x] Task 1 Step 1: Generate experiment summaries (1 per paper, using Claude Sonnet 4.5)
- [x] Task 1 Step 2: Generate underlying hypotheses (10 samples × 6 models × 50 papers = 3000 calls)
- [x] Task 2: Generate novel hypotheses (10 samples × 6 models × 50 papers = 3000 calls)
- [x] Compute embeddings using text-embedding-3-small via OpenRouter
- [x] Compute cosine similarity matrices
- [x] Generate Figure 1A: Inter-model heatmap for Task 1
- [x] Generate Figure 1B: Inter-model heatmap for Task 2
- [x] Generate Figure 2: Intra-model similarity plots (both tasks)
- [x] Generate Figure 3: KDE distribution plots (both tasks)
- [x] Create reproduce.sh (tested successfully)
- [x] Write REPORT.md
- [x] All files committed and pushed

## Key Results
- Task 1 inter-model mean off-diagonal: 0.768
- Task 2 inter-model mean off-diagonal: 0.635
- Both tasks show high inter-model similarity, confirming paper's "hivemind" finding
- KDE plots show clear separation between same-paper and different-paper distributions

## Completed Work
- `download_papers.py` - Downloads all 50 papers from OpenReview
- `extract_text.py` - Extracts text from PDFs using PyMuPDF
- `generate_summaries.py` - Generates experiment summaries (Claude Sonnet 4.5)
- `generate_hypotheses.py` - Generates hypotheses for both tasks (6 models × 10 samples × 50 papers)
- `compute_embeddings.py` - Computes embeddings via OpenRouter (text-embedding-3-small)
- `analyze_and_plot.py` - Generates all 4 figures + metrics
- `reproduce.sh` - Full pipeline script (tested, runs successfully with cached data)
- `REPORT.md` - Final report

## Failed Approaches
- Initial generate_hypotheses.py failed because reasoning models (GPT-5 Nano/Mini/5) need high max_tokens for reasoning budget. Fixed by setting max_tokens=16000.
- Some API calls returned null content; handled by checking for null and retrying.

## Evaluation Coverage
All quantitative results from the paper's Section 4 and Appendix A are replicated:
- Figure 1 (A + B): Inter-model similarity heatmaps
- Figure 2 (A + B): Intra-model similarity comparison  
- Figure 3 (A + B): Same vs different paper KDE distributions
- Core claim: Models form a "hivemind" with limited epistemic diversity
