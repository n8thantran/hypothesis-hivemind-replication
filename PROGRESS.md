# PROGRESS

## Current Phase
Reading paper and creating implementation plan.

## Paper Summary
This is a **position paper** arguing that agentic AI scientists are not built for autonomous scientific discovery. The key **empirical contribution** is the "Hypothesis Hivemind" experiment (Section 4 / Appendix A) which demonstrates that frontier LLMs from different providers produce semantically convergent hypotheses.

## Key Experiment: Hypothesis Hivemind

### Setup
- **Dataset**: 50 papers from NeurIPS 2025 AI4Mat track (URLs in appendix)
- **Models**: 6 models from 2 providers
  - Anthropic: Claude Haiku 4.5, Claude Sonnet 4.5, Claude Sonnet 4.6
  - OpenAI: GPT-5 Nano, GPT-5 Mini, GPT-5
- **Tasks**:
  1. Recover underlying hypothesis from experiment summary (interpretive/convergence baseline)
  2. Generate novel hypothesis from full paper text (open-ended/diversity desired)
- **Samples**: 10 independent samples per model per task per paper
- **Embeddings**: text-embedding-3-small (OpenAI)
- **Metrics**: Cosine similarity between embedding groups

### Prompts (from Box in paper)
- Task 1 has 2 stages: (a) generate experiment summary from paper, (b) generate underlying hypothesis from summary
- Task 2: generate novel hypothesis from full paper text

### Expected Outputs/Figures
1. **Figure 1(A)**: Heatmap of inter-model cosine similarities for Task 1 (convergence desired)
2. **Figure 1(B)**: Heatmap of inter-model cosine similarities for Task 2 (diversity desired)
3. **Figure 2**: Intra-model similarities for both tasks
4. **Figure 3**: KDE plots - same-paper vs different-paper cosine similarity distributions

### Key Finding
Inter-model similarities remain HIGH even for Task 2, showing models converge semantically even when diversity is desired.

## Implementation Plan
- [x] Read paper
- [ ] Download 50 papers from OpenReview
- [ ] Extract text from PDFs
- [ ] Set up API access for LLMs
- [ ] Implement experiment summary generation (Task 1 preprocessing)
- [ ] Implement hypothesis generation (Task 1 and Task 2)
- [ ] Implement embedding generation
- [ ] Compute similarity metrics
- [ ] Generate all 4 figures
- [ ] Create reproduce.sh
- [ ] Write REPORT.md

## Key Decisions
- Need to determine which models are available via API
- May need to substitute models if the paper's exact models aren't accessible
- Embedding model: text-embedding-3-small from OpenAI

## Completed Work
(none yet)

## Failed Approaches
(none yet)
