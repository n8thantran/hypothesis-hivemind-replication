# Progress Tracking

## Current Phase
COMPLETE - All figures generated, reproduce.sh tested, REPORT.md written.

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

## Key Results
- Task 1 inter-model mean off-diagonal: 0.768
- Task 2 inter-model mean off-diagonal: 0.635
- Both tasks show high inter-model similarity, confirming paper's "hivemind" finding
- KDE plots show clear separation between same-paper and different-paper distributions

## Completed Work
- `download_papers.py` - Downloads all 50 papers
- `extract_text.py` - Extracts text from PDFs
- `generate_summaries.py` - Generates experiment summaries
- `generate_hypotheses.py` - Generates hypotheses for both tasks
- `compute_embeddings.py` - Computes embeddings via OpenRouter
- `analyze_and_plot.py` - Generates all 4 figures + metrics
- `reproduce.sh` - Full pipeline script
- `REPORT.md` - Final report
