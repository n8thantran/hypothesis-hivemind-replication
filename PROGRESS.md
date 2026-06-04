# PROGRESS

## Current Phase
Task 2 data collection nearly complete (GPT-5 has 11/50 papers, other 5 models done). Need to finish GPT-5 Task 2, then embeddings + analysis + figures.

## Paper Summary
Position paper arguing agentic AI scientists aren't built for autonomous scientific discovery. Key empirical contribution: "Hypothesis Hivemind" experiment (Section 4 / Appendix A) showing frontier LLMs produce semantically convergent hypotheses.

## Key Experiment: Hypothesis Hivemind

### Setup
- **Dataset**: 50 papers from NeurIPS 2025 AI4Mat track (URLs in appendix, lines ~382-431 of paper.tex)
- **Models**: 6 models from 2 providers
  - Anthropic: Claude Haiku 4.5, Claude Sonnet 4.5, Claude Sonnet 4.6
  - OpenAI: GPT-5 Nano, GPT-5 Mini, GPT-5
- **Tasks**:
  1. Recover underlying hypothesis from experiment summary (interpretive/convergence baseline)
  2. Generate novel hypothesis from full paper text (open-ended/diversity desired)
- **Samples**: 10 independent samples per model per task per paper
- **Embeddings**: text-embedding-3-small (OpenAI) via OpenRouter
- **Metrics**: Cosine similarity between embedding groups

### Prompts (from paper Box)
- Task 1 Stage A: "Given the following academic paper, provide a detailed summary of the experiments conducted..."
- Task 1 Stage B: "Given the following experiment summary, what is the underlying hypothesis being tested?"
- Task 2: "Given the following academic paper, generate a novel hypothesis..."

### Expected Outputs/Figures
1. **Figure 1(A)**: Heatmap of inter-model cosine similarities for Task 1 (convergence desired)
2. **Figure 1(B)**: Heatmap of inter-model cosine similarities for Task 2 (diversity desired)
3. **Figure 2**: Intra-model similarities for both tasks (bar chart)
4. **Figure 3**: KDE plots - same-paper vs different-paper cosine similarity distributions

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
- [~] Task 2: novel hypothesis generation - 5/6 models done, GPT-5 at 11/50 papers (GPT-5 returns empty content frequently as reasoning model)
- [ ] Generate embeddings for Task 1 hypotheses
- [ ] Generate embeddings for Task 2 hypotheses
- [ ] Compute similarity metrics (inter-model and intra-model)
- [ ] Generate Figure 1 (A and B heatmaps)
- [ ] Generate Figure 2 (intra-model bar chart)
- [ ] Generate Figure 3 (KDE plots)
- [ ] Create reproduce.sh
- [ ] Write REPORT.md

## Key Technical Details
- OpenRouter API key: available as $OPENROUTER_API_KEY env var
- GitHub PAT: available as $GITHUB_PAT env var
- GitHub user: n8thantran
- GitHub repo: hypothesis-hivemind-replication
- GPT-5 and GPT-5-nano are reasoning models - frequently return empty content, need retries with max_tokens=16000
- OpenRouter base URL: https://openrouter.ai/api/v1
- Model IDs on OpenRouter:
  - anthropic/claude-haiku-4.5
  - anthropic/claude-sonnet-4.5
  - anthropic/claude-sonnet-4.6
  - openai/gpt-5-nano
  - openai/gpt-5-mini
  - openai/gpt-5
- Embedding model: openai/text-embedding-3-small via OpenRouter
- MAX_PAPER_TEXT = 80000 chars (truncation for long papers)
- temperature = 1.0 for diversity
- NUM_WORKERS = 10 parallel threads

## Completed Work
- `config.py`: Configuration with model IDs, prompts, constants
- `api_utils.py`: API utilities for LLM calls and embeddings via OpenRouter
- `download_papers.py`: Downloads PDFs from OpenReview and extracts text
- `run_parallel.py`: Parallelized experiment runner with thread-safe caching
- `data/papers/`: 50 PDFs and extracted text files
- `data/paper_index.json`: Index of all papers
- `cache/summaries.json`: Experiment summaries for all 50 papers
- `cache/task1_hypotheses.json`: COMPLETE - 300 entries (6 models × 50 papers), each with 10 samples
- `cache/task2_hypotheses.json`: 261/300 entries (GPT-5 only 11/50 papers)

## Failed Approaches
- Sequential API calls too slow (10+ min timeout for one model)
- GPT-5-nano initially returned empty content - fixed by increasing max_tokens to 16000
- GPT-5 (full) frequently returns empty content even with retries - it's a reasoning model that sometimes produces only internal reasoning with no visible output

## Remaining Work Priority
1. **Finish GPT-5 Task 2** - keep retrying, or proceed with partial data if stuck
2. **Embeddings** - batch embed all hypotheses using text-embedding-3-small
3. **Analysis** - compute cosine similarity matrices
4. **Figures** - generate all 3 figures matching paper
5. **reproduce.sh and REPORT.md**

## Data Sizes
- Task 1: 3000 hypotheses across 300 cache entries
- Task 2: ~2610 hypotheses so far (will be 3000 when complete)
- Each hypothesis needs one embedding vector
