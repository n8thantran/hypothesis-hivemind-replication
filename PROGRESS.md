# PROGRESS

## Current Phase
Building the full pipeline. Setup is done - APIs verified, repo created.

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
- [ ] Extract paper URLs from appendix and download PDFs
- [ ] Extract text from PDFs using PyMuPDF
- [ ] Implement Task 1 (experiment summary → hypothesis recovery)
- [ ] Implement Task 2 (novel hypothesis generation)
- [ ] Implement embedding generation
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
- GPT-5-nano is a reasoning model - needs max_tokens=2000+ to get content output
- OpenRouter base URL: https://openrouter.ai/api/v1
- Model IDs on OpenRouter:
  - anthropic/claude-haiku-4.5
  - anthropic/claude-sonnet-4.5
  - anthropic/claude-sonnet-4.6
  - openai/gpt-5-nano
  - openai/gpt-5-mini
  - openai/gpt-5

## Completed Work
- PROGRESS.md: this file
- paper/paper.tex: the paper source

## Failed Approaches
(none yet)

## Cost Considerations
- 50 papers × 6 models × 10 samples × 2 tasks = 6000 LLM calls
- Plus 50 papers × 6 models × 1 experiment summary = 300 calls for Task 1 Stage A
- Total: ~6300 LLM calls + 6000 embedding calls
- May need to reduce sample count if budget is tight (e.g., 5 samples instead of 10)
