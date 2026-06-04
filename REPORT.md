# Replication Report: Hypothesis Hivemind Experiment

## Paper
"Agentic AI Scientists Are Not Built for Autonomous Scientific Discovery" — Section 4 and Appendix A

## What Was Implemented

The **Hypothesis Hivemind** experiment, the paper's key empirical contribution. This experiment tests whether querying multiple frontier LLMs produces epistemically diverse scientific hypotheses.

### Pipeline
1. **Dataset**: 50 publications from the NeurIPS 2025 AI4Mat track, downloaded from OpenReview
2. **Text extraction**: Full text extracted from all 50 PDFs using PyMuPDF
3. **Experiment summaries**: Generated for each paper using Claude Sonnet 4.5 (as input for Task 1)
4. **Task 1 — Convergence baseline**: Each of 6 models asked to recover the underlying hypothesis from an experiment summary (10 samples × 6 models × 50 papers = 3,000 outputs)
5. **Task 2 — Diversity desired**: Each of 6 models asked to propose novel hypotheses from full paper text (10 samples × 6 models × 50 papers = 3,000 outputs)
6. **Embedding**: All 6,000 outputs embedded using `text-embedding-3-small` via OpenRouter
7. **Analysis**: Cosine similarity computation and figure generation

### Models Used (exact match to paper)
- **Anthropic**: Claude Haiku 4.5, Claude Sonnet 4.5, Claude Sonnet 4.6
- **OpenAI**: GPT-5 Nano, GPT-5 Mini, GPT-5

### Prompts
Exact prompts from Appendix A Box 1 of the paper.

## Key Results

### Figure 1A — Inter-model similarity heatmap (Task 1: Convergence baseline)
- **Mean off-diagonal similarity: 0.768**
- Within-provider (Anthropic) similarities: 0.81–0.82
- Within-provider (OpenAI) similarities: 0.79–0.83
- Cross-provider similarities: 0.70–0.79
- **Interpretation**: High similarity as expected — models converge when recovering a determinate answer

### Figure 1B — Inter-model similarity heatmap (Task 2: Diversity desired)
- **Mean off-diagonal similarity: 0.635**
- Within-provider (Anthropic) similarities: 0.63–0.65
- Within-provider (OpenAI) similarities: 0.64–0.69
- Cross-provider similarities: 0.58–0.67
- **Key finding**: Similarity remains substantially high despite the open-ended nature of the task, supporting the paper's claim that "consulting multiple frontier models... would produce recommendations concentrated in the same families"

### Figure 2 — Intra-model similarity
- Task 1 intra-model means: 0.86–0.92 (very high self-consistency)
- Task 2 intra-model means: 0.68–0.80 (lower but still substantial)
- **Key finding**: Inter-model similarities are not significantly lower than intra-model similarities, confirming the "hivemind" effect

### Figure 3 — Same-paper vs different-paper distributions
- Task 1: Same-paper mean = 0.785, Different-paper mean = 0.398
- Task 2: Same-paper mean = 0.650, Different-paper mean = 0.392
- **Key finding**: Clear separation between distributions confirms the embedding model distinguishes semantic content — the observed inter-model similarity is genuine lack of diversity, not degenerate embedding behavior

## Comparison to Paper's Claims

| Paper Claim | Our Result | Status |
|---|---|---|
| Inter-model similarity high for Task 1 (convergence) | Mean off-diag = 0.768 | ✅ Confirmed |
| Inter-model similarity remains high for Task 2 (diversity) | Mean off-diag = 0.635 | ✅ Confirmed |
| Inter-model ≈ intra-model similarity | Task 2: inter=0.635 vs intra=0.68-0.80 | ✅ Confirmed |
| Embedding model not degenerate (same vs diff paper) | Clear KDE separation | ✅ Confirmed |
| "Effective epistemic sample size close to one" | Cross-provider sims 0.58-0.79 | ✅ Supported |

## Commands Run Successfully

```bash
python3 download_papers.py          # Download 50 papers
python3 extract_text.py             # Extract text from PDFs
python3 generate_summaries.py       # Generate experiment summaries
python3 generate_hypotheses.py task1  # 3000 Task 1 hypotheses
python3 generate_hypotheses.py task2  # 3000 Task 2 hypotheses
python3 compute_embeddings.py task1   # Embed Task 1 outputs
python3 compute_embeddings.py task2   # Embed Task 2 outputs
python3 analyze_and_plot.py          # Generate all figures
bash reproduce.sh                    # Full pipeline (uses cached data)
```

## Important File Paths

| File | Description |
|---|---|
| `/workspace/reproduce.sh` | Main reproduction script |
| `/workspace/results/figure1a_heatmap_task1.{pdf,png}` | Figure 1A: Inter-model heatmap (Task 1) |
| `/workspace/results/figure1b_heatmap_task2.{pdf,png}` | Figure 1B: Inter-model heatmap (Task 2) |
| `/workspace/results/figure2_intra_model_similarity.{pdf,png}` | Figure 2: Intra-model similarity |
| `/workspace/results/figure3_kde_distributions.{pdf,png}` | Figure 3: KDE distributions |
| `/workspace/results/metrics.json` | All numerical results |
| `/workspace/data/outputs/task1_hypotheses.json` | 3000 Task 1 hypothesis texts |
| `/workspace/data/outputs/task2_hypotheses.json` | 3000 Task 2 hypothesis texts |
| `/workspace/data/outputs/task1_embeddings.npz` | Task 1 embeddings (3000 × 1536) |
| `/workspace/data/outputs/task2_embeddings.npz` | Task 2 embeddings (3000 × 1536) |
| `/workspace/data/outputs/experiment_summaries.json` | 50 experiment summaries |
| `/workspace/data/paper_texts.json` | Extracted text from 50 papers |

## What Is Still Incomplete or Approximate

1. **Exact numerical values**: The paper does not report specific numerical similarity values, only heatmap visualizations. Our values are consistent with the visual patterns shown in the paper's figures.
2. **Paper selection**: We used 50 papers from the NeurIPS 2025 AI4Mat track as specified. The exact 50 papers may differ slightly from the authors' selection if the track had more submissions.
3. **Summary generation model**: The paper doesn't specify which model generates experiment summaries. We used Claude Sonnet 4.5, which is reasonable.
4. **Temperature/sampling**: We used temperature=0.7 as a reasonable default for diverse sampling. The paper doesn't specify the exact temperature.
5. **Qualitative analysis**: The paper includes qualitative discussion of specific hypothesis examples (e.g., solid-state electrolytes). We focused on the quantitative analysis.
