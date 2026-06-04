# PROGRESS

## Current Phase
All data collection and embedding generation COMPLETE. Now writing analysis/visualization code to produce Figures 1-3.

## Paper Summary
Position paper arguing agentic AI scientists aren't built for autonomous scientific discovery. Key empirical contribution: "Hypothesis Hivemind" experiment (Section 4 / Appendix A) showing frontier LLMs produce semantically convergent hypotheses.

## Key Experiment: Hypothesis Hivemind

### Setup
- **Dataset**: 50 papers from NeurIPS 2025 AI4Mat track (URLs in appendix, lines ~382-431 of paper.tex)
- **Models**: 6 models from 2 providers
  - Anthropic: Claude Haiku 4.5, Claude Sonnet 4.5, Claude Sonnet 4.6
  - OpenAI: GPT-5 Nano, GPT-5 Mini, GPT-5
- **Tasks**:
  1. Recover underlying hypothesis from experiment summary (convergence baseline)
  2. Generate novel hypothesis from full paper text (diversity desired)
- **Samples**: 10 independent samples per model per task per paper
- **Embeddings**: text-embedding-3-small (OpenAI) via OpenRouter, 1536-dim
- **Metrics**: Cosine similarity between embedding groups

### Expected Outputs/Figures
1. **Figure 1(A)**: Heatmap of inter-model cosine similarities for Task 1 (convergence desired) → `plots/heatmap_A.pdf`
2. **Figure 1(B)**: Heatmap of inter-model cosine similarities for Task 2 (diversity desired) → `plots/heatmap_B.pdf`
3. **Figure 2(A)**: Intra-model similarities for Task 1 → `plots/intra_model_repetition_underlying_hypotheses.pdf`
4. **Figure 2(B)**: Intra-model similarities for Task 2 → `plots/intra_model_repetition_new_hypotheses.pdf`
5. **Figure 3(A)**: KDE same-paper vs different-paper for Task 1 → `plots/intra_inter_kde_pooled_A.pdf`
6. **Figure 3(B)**: KDE same-paper vs different-paper for Task 2 → `plots/intra_inter_kde_pooled_B.pdf`

### Key Finding
Inter-model similarities remain HIGH even for Task 2, showing models converge semantically even when diversity is desired.

## Implementation Plan
- [x] Read paper
- [x] Verify API access (OpenRouter works for all 6 models + embeddings)
- [x] Set up GitHub repo
- [x] Extract paper URLs from appendix and download PDFs (50 papers)
- [x] Extract text from PDFs using PyMuPDF 
- [x] Generate experiment summaries for Task 1 (50 papers, stored in cache/summaries.json)
- [x] Task 1: hypothesis recovery - COMPLETE (6 models × 50 papers × 10 samples = 3000 items)
- [x] Task 2: novel hypothesis generation - COMPLETE (6 models × 50 papers × 10 samples = 3000 items)
- [x] Generate embeddings for Task 1 hypotheses (3000 embeddings, 1536-dim)
- [x] Generate embeddings for Task 2 hypotheses (3000 embeddings, 1536-dim)
- [ ] Compute similarity metrics (inter-model and intra-model) → analysis.py
- [ ] Generate Figure 1 (A and B heatmaps)
- [ ] Generate Figure 2 (intra-model bar chart)
- [ ] Generate Figure 3 (KDE plots)
- [ ] Create reproduce.sh
- [ ] Write REPORT.md

## Key Technical Details
- OpenRouter API key: available as $OPENROUTER_API_KEY env var
- GitHub Push Token: available as $GITHUB_PUSH_TOKEN env var
- All cached data in /workspace/cache/:
  - summaries.json: 50 experiment summaries (Task 1 Stage A output)
  - task1_hypotheses.json: 3000 hypotheses (6 models × 50 papers × 10 samples)
  - task2_hypotheses.json: 3000 hypotheses (6 models × 50 papers × 10 samples)
  - task1_embeddings.json: 3000 embeddings (1536-dim each)
  - task2_embeddings.json: 3000 embeddings (1536-dim each)
- Paper PDFs + extracted text in /workspace/data/papers/

## Completed Code Files
- config.py: Model definitions, paper IDs, prompts, API keys
- download_papers.py: Downloads PDFs from OpenReview and extracts text
- api_utils.py: OpenRouter API calls (chat completion + embeddings)
- run_parallel.py: Parallelized experiment runner for all steps

## Failed Approaches
- GPT-5 frequently returns empty content (reasoning model issue) - worked around with retries and increased max_tokens/timeout
- JSON cache files got corrupted during timeout kills - fixed by finding last valid truncation point
- Initial sequential runner was too slow - switched to parallel ThreadPoolExecutor

## Metrics Computation Plan
For heatmaps (inter-model similarity):
- For each pair of models (i, j), for each paper p:
  - Compute mean cosine similarity between all 10×10=100 pairs of embeddings from model i and model j for paper p
  - Average across all 50 papers
- This gives a 6×6 symmetric matrix

For intra-model similarity:
- For each model, for each paper:
  - Compute mean cosine similarity of all C(10,2)=45 pairs within that model's 10 samples
  - Average across papers → gives one number per model per task
  
For KDE (same-paper vs different-paper):
- Same-paper: cosine similarities between embeddings from any two models for the SAME paper
- Different-paper: cosine similarities between embeddings from any two models for DIFFERENT papers
