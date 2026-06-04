# Progress Tracking

## Current Phase
Reading the paper and creating implementation plan.

## Paper Summary
This is a position paper arguing that "agentic AI scientists are not built for autonomous scientific discovery." The key empirical contribution is the **hypothesis hivemind experiment** (Section 4, Appendix A).

## Key Experiment: Hypothesis Hivemind
- **Dataset**: 50 publications from 2025 NeurIPS AI4Mat track
- **Models**: 6 frontier models from 2 providers (Anthropic: Claude Haiku 4.5, Claude Sonnet 4.5, Claude Sonnet 4.6; OpenAI: GPT-5 Nano, GPT-5 Mini, GPT-5)
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
- [ ] Set up environment and check available resources (GPU, API keys)
- [ ] Download the 50 papers from OpenReview
- [ ] Implement paper text extraction pipeline
- [ ] Generate experiment summaries (Task 1 prerequisite)
- [ ] Generate underlying hypotheses (Task 1) with multiple models
- [ ] Generate novel hypotheses (Task 2) with multiple models
- [ ] Compute embeddings using text-embedding-3-small (or equivalent)
- [ ] Compute cosine similarity matrices
- [ ] Generate heatmap plots (Fig 1A, 1B)
- [ ] Generate intra-model similarity plots (Fig 2)
- [ ] Generate KDE distribution plots (Fig 3)
- [ ] Create reproduce.sh
- [ ] Write REPORT.md

## Key Decisions
- Models: Need to substitute with available models since some listed are future/unavailable
- Embedding model: text-embedding-3-small (OpenAI) - need API key or substitute
- Papers: All 50 URLs listed in Appendix

## Completed Work
(none yet)

## Failed Approaches
(none yet)

## Evaluation Coverage
- Main claim: Inter-provider output similarity remains high regardless of task diversity
- Need to reproduce: Figures 1A, 1B, 2, 3
