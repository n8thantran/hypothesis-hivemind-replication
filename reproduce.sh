#!/bin/bash
# Reproduce the Hypothesis Hivemind experiment from:
# "Agentic AI Scientists Are Not Built for Autonomous Scientific Discovery"
#
# This script reproduces Figures 1A, 1B, 2, and 3 from the paper.
#
# Prerequisites:
#   - OPENROUTER_API_KEY environment variable set
#   - Python 3 with pip
#
# The full pipeline takes ~2-3 hours due to 6000+ API calls.
# If data already exists, only the analysis step runs (~10 seconds).

set -e

echo "=== Hypothesis Hivemind Replication ==="
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q pymupdf aiohttp numpy matplotlib scikit-learn scipy

# Step 1: Download papers (50 from NeurIPS 2025 AI4Mat)
if [ ! -d "data/papers" ] || [ $(ls data/papers/*.pdf 2>/dev/null | wc -l) -lt 50 ]; then
    echo "Step 1: Downloading papers..."
    python3 download_papers.py
else
    echo "Step 1: Papers already downloaded (50 PDFs found)"
fi

# Step 2: Extract text from PDFs
if [ ! -f "data/paper_texts.json" ]; then
    echo "Step 2: Extracting text from PDFs..."
    python3 extract_text.py
else
    echo "Step 2: Paper texts already extracted"
fi

# Step 3: Generate experiment summaries (Task 1 input)
if [ ! -f "data/outputs/experiment_summaries.json" ]; then
    echo "Step 3: Generating experiment summaries..."
    python3 generate_summaries.py
else
    echo "Step 3: Experiment summaries already generated"
fi

# Step 4: Generate hypotheses for Task 1 (recover underlying hypothesis)
if [ ! -f "data/outputs/task1_hypotheses.json" ]; then
    echo "Step 4: Generating Task 1 hypotheses (3000 API calls)..."
    python3 generate_hypotheses.py task1
else
    echo "Step 4: Task 1 hypotheses already generated"
fi

# Step 5: Generate hypotheses for Task 2 (novel hypotheses)
if [ ! -f "data/outputs/task2_hypotheses.json" ]; then
    echo "Step 5: Generating Task 2 hypotheses (3000 API calls)..."
    python3 generate_hypotheses.py task2
else
    echo "Step 5: Task 2 hypotheses already generated"
fi

# Step 6: Compute embeddings
if [ ! -f "data/outputs/task1_embeddings.npz" ]; then
    echo "Step 6a: Computing Task 1 embeddings..."
    python3 compute_embeddings.py task1
else
    echo "Step 6a: Task 1 embeddings already computed"
fi

if [ ! -f "data/outputs/task2_embeddings.npz" ]; then
    echo "Step 6b: Computing Task 2 embeddings..."
    python3 compute_embeddings.py task2
else
    echo "Step 6b: Task 2 embeddings already computed"
fi

# Step 7: Analyze and generate figures
echo "Step 7: Analyzing embeddings and generating figures..."
python3 analyze_and_plot.py

echo ""
echo "=== Done ==="
echo "Results saved to /workspace/results/"
echo "  - figure1a_heatmap_task1.{pdf,png}  (Fig 1A: Inter-model similarity, convergence task)"
echo "  - figure1b_heatmap_task2.{pdf,png}  (Fig 1B: Inter-model similarity, diversity task)"
echo "  - figure2_intra_model_similarity.{pdf,png}  (Fig 2: Intra-model similarity)"
echo "  - figure3_kde_distributions.{pdf,png}  (Fig 3: Same vs different paper distributions)"
echo "  - metrics.json  (All numerical results)"
