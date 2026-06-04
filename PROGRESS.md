# Progress Tracking

## Current Phase
Building the full pipeline. All 50 papers downloaded, API access confirmed for all 6 models + embedding model via OpenRouter.

## Paper Summary
Position paper arguing "agentic AI scientists are not built for autonomous scientific discovery." Key empirical contribution: **hypothesis hivemind experiment** (Section 4, Appendix A).

## Key Experiment: Hypothesis Hivemind
- **Dataset**: 50 publications from 2025 NeurIPS AI4Mat track (all downloaded)
- **Models**: 6 frontier models from 2 providers:
  - Anthropic: `anthropic/claude-haiku-4.5`, `anthropic/claude-sonnet-4.5`, `anthropic/claude-sonnet-4.6`
  - OpenAI: `openai/gpt-5-nano`, `openai/gpt-5-mini`, `openai/gpt-5`
- **Tasks**:
  - Task 1 (convergence baseline): Recover underlying hypothesis from experiment summary
  - Task 2 (diversity desired): Propose novel hypotheses from full paper text
- **Method**: 10 samples per model per paper, embed with text-embedding-3-small, compute cosine similarity
- **Key Results** (Figures):
  - Fig 1A: Inter-model heatmap for Task 1 (high similarity expected & observed)
  - Fig 1B: Inter-model heatmap for Task 2 (high similarity observed, challenging diversity)
  - Fig 2: Intra-model similarities for both tasks
  - Fig 3: KDE plots of same-paper vs different-paper cosine similarities

## Implementation Plan
- [x] Set up environment and check available resources
- [x] Download all 50 papers from OpenReview
- [x] Verify API access (OpenRouter: all 6 models + text-embedding-3-small confirmed)
- [ ] Extract text from PDFs using PyMuPDF
- [ ] Task 1 Step 1: Generate experiment summaries (1 per paper, using one model)
- [ ] Task 1 Step 2: Generate underlying hypotheses (10 samples × 6 models × 50 papers = 3000 calls)
- [ ] Task 2: Generate novel hypotheses (10 samples × 6 models × 50 papers = 3000 calls)
- [ ] Compute embeddings using text-embedding-3-small via OpenRouter
- [ ] Compute cosine similarity matrices
- [ ] Generate Figure 1A: Inter-model heatmap for Task 1
- [ ] Generate Figure 1B: Inter-model heatmap for Task 2
- [ ] Generate Figure 2: Intra-model similarity plots (both tasks)
- [ ] Generate Figure 3: KDE distribution plots (both tasks)
- [ ] Create reproduce.sh
- [ ] Write REPORT.md

## Key Decisions
- **API**: Using OpenRouter API (key available) for all 6 exact models from the paper
- **Embedding**: text-embedding-3-small via OpenRouter (exact match to paper)
- **Prompts**: Exact prompts from Appendix A Box 1
- **Experiment summaries**: Paper uses one model to generate summaries, then feeds to all 6 for hypothesis recovery
- **Cost concern**: 6000 LLM calls + 6000 embedding calls. May need to reduce if budget limited.

## Completed Work
- `download_papers.py` - Downloads all 50 papers (tested, working)
- `/workspace/data/papers/` - All 50 PDFs downloaded

## Failed Approaches
(none yet)

## Evaluation Coverage
- Main claim: Inter-provider output similarity remains high regardless of task diversity
- Need to reproduce: Figures 1A, 1B, 2, 3
- All infrastructure confirmed available
