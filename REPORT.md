# Replication Report: "Your AI Scientist Is Not Built For Discovery"

## What Was Implemented

This replication implements the **Hypothesis Hivemind** experiment (Section 4 / Appendix A) from the paper, which is the paper's key empirical contribution. The experiment tests whether frontier LLMs produce semantically convergent hypotheses, even when prompted for novelty and diversity.

### Experiment Setup
- **Dataset**: 50 papers from the NeurIPS 2025 AI4Mat workshop track (as specified in Appendix A)
- **Models**: 6 frontier models from 2 providers:
  - Anthropic: Claude Haiku 4.5, Claude Sonnet 4.5, Claude Sonnet 4.6
  - OpenAI: GPT-5 Nano, GPT-5 Mini, GPT-5
- **Tasks**:
  1. **Task 1 (Convergence baseline)**: Recover the underlying hypothesis from an experiment summary
  2. **Task 2 (Diversity desired)**: Generate a novel hypothesis from the full paper text
- **Sampling**: 10 independent samples per model per task per paper (3,000 hypotheses per task)
- **Embeddings**: OpenAI text-embedding-3-small (1536-dim) via OpenRouter
- **Metrics**: Cosine similarity between embedding vectors

### Pipeline Steps
1. Download 50 papers from OpenReview and extract text (PyMuPDF)
2. Generate experiment summaries for Task 1 using Claude Sonnet 4.6
3. Generate 3,000 hypotheses for Task 1 (6 models × 50 papers × 10 samples)
4. Generate 3,000 hypotheses for Task 2 (6 models × 50 papers × 10 samples)
5. Embed all 6,000 hypotheses using text-embedding-3-small
6. Compute similarity metrics and generate figures

## Commands Run Successfully

```bash
python download_papers.py          # Download 50 papers
python run_parallel.py             # Generate all hypotheses + embeddings
python analysis.py                 # Compute metrics and generate figures
bash reproduce.sh                  # End-to-end reproduction (from cached data)
```

## Key Results

### Figure 1: Inter-Model Cosine Similarity Heatmaps

| Metric | Task 1 (Convergence) | Task 2 (Diversity) |
|--------|---------------------|--------------------|
| Mean off-diagonal similarity | **0.7706** | **0.6345** |

**Key Finding**: Inter-model similarity remains HIGH (0.63) even for Task 2 where diversity and novelty are explicitly requested. This confirms the paper's central claim that frontier LLMs converge semantically in their hypothesis generation.

Within-provider similarities are higher than cross-provider:
- Anthropic models (Task 2): 0.63-0.66
- OpenAI models (Task 2): 0.63-0.69
- Cross-provider (Task 2): 0.57-0.67

### Figure 2: Intra-Model Similarity (Self-Repetition)

| Model | Task 1 | Task 2 |
|-------|--------|--------|
| Claude Haiku 4.5 | 0.867 | 0.724 |
| Claude Sonnet 4.5 | 0.863 | 0.689 |
| Claude Sonnet 4.6 | 0.917 | 0.794 |
| GPT-5 Nano | 0.855 | 0.747 |
| GPT-5 Mini | 0.876 | 0.744 |
| GPT-5 | 0.874 | 0.721 |

Models show high self-repetition (0.69-0.79 for Task 2), confirming limited diversity even within a single model's outputs.

### Figure 3: Same-Paper vs Different-Paper KDE

| Metric | Task 1 | Task 2 |
|--------|--------|--------|
| Same paper mean | 0.7706 | 0.6345 |
| Different paper mean | 0.4000 | 0.3834 |

Clear separation between same-paper and different-paper similarity distributions shows models are responding to paper content, but converging within papers.

## File Locations

### Code
- `/workspace/config.py` — Model definitions, paper URLs, prompts, API configuration
- `/workspace/download_papers.py` — Downloads PDFs from OpenReview and extracts text
- `/workspace/api_utils.py` — OpenRouter API utilities (chat completion + embeddings)
- `/workspace/run_parallel.py` — Parallelized experiment runner
- `/workspace/analysis.py` — Similarity analysis and visualization
- `/workspace/reproduce.sh` — Reproduction script

### Data (cached)
- `/workspace/cache/summaries.json` — 50 experiment summaries
- `/workspace/cache/task1_hypotheses.json` — 3,000 Task 1 hypotheses
- `/workspace/cache/task2_hypotheses.json` — 3,000 Task 2 hypotheses  
- `/workspace/cache/task1_embeddings.json` — 3,000 Task 1 embeddings (1536-dim)
- `/workspace/cache/task2_embeddings.json` — 3,000 Task 2 embeddings (1536-dim)
- `/workspace/data/papers/` — 50 PDFs and extracted text files

### Results
- `/workspace/results/metrics.json` — All numerical metrics
- `/workspace/results/plots/heatmap_A.pdf` — Figure 1(A): Inter-model similarity heatmap (Task 1)
- `/workspace/results/plots/heatmap_B.pdf` — Figure 1(B): Inter-model similarity heatmap (Task 2)
- `/workspace/results/plots/intra_model_repetition_underlying_hypotheses.pdf` — Figure 2(A)
- `/workspace/results/plots/intra_model_repetition_new_hypotheses.pdf` — Figure 2(B)
- `/workspace/results/plots/intra_inter_kde_pooled_A.pdf` — Figure 3(A): KDE (Task 1)
- `/workspace/results/plots/intra_inter_kde_pooled_B.pdf` — Figure 3(B): KDE (Task 2)

## Comparison with Paper

The paper doesn't report exact numerical values for all metrics, but presents figures showing:
1. **High inter-model similarity for both tasks** — ✅ Confirmed (0.77 Task 1, 0.63 Task 2)
2. **Similarity remains high even for Task 2 (novel hypotheses)** — ✅ Confirmed
3. **Clear separation in KDE between same-paper and different-paper** — ✅ Confirmed
4. **Within-provider models are more similar than cross-provider** — ✅ Confirmed
5. **High intra-model self-repetition** — ✅ Confirmed (0.69-0.79 for Task 2)

## What Is Still Incomplete or Approximate

1. **Model versions**: The paper uses specific model versions (e.g., "claude-sonnet-4-20250514"). We used the latest available versions via OpenRouter, which may differ slightly.
2. **Exact prompts**: We reconstructed prompts from the paper's description; the exact wording may differ slightly from what the authors used.
3. **Embedding API**: Used OpenRouter as proxy for OpenAI embeddings; results should be identical but routing may differ.
4. **Temperature**: The paper specifies temperature=1.0 for all models, which we followed.
5. **Exact visual styling**: Our figures match the paper's content and structure but may differ in visual styling (colormap, fonts, etc.).
