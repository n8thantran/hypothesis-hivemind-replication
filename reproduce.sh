#!/bin/bash
# Reproduce the Hypothesis Hivemind experiment from the paper:
# "Your AI Scientist Is Not Built For Discovery"
#
# This script regenerates ALL figures and metrics from cached data.
# To fully rerun from scratch (API calls), see comments below.
#
# Prerequisites: Python 3, pip packages (numpy, matplotlib, scipy, PyMuPDF, requests)
# Environment variable: OPENROUTER_API_KEY (for API calls)

set -e

echo "========================================="
echo " Hypothesis Hivemind Replication"
echo "========================================="

# Install dependencies
echo "Installing dependencies..."
pip install numpy matplotlib scipy PyMuPDF requests -q

# ---- FULL REPLICATION (uncomment to rerun from scratch) ----
# WARNING: This requires an OpenRouter API key and takes ~2-3 hours
# and costs ~$20-50 in API credits.
#
# # 1. Download papers
# python download_papers.py
#
# # 2. Generate experiment summaries (Task 1 Stage A)
# python run_parallel.py --step summary
#
# # 3. Generate hypotheses for Task 1 (10 samples × 6 models × 50 papers)
# python run_parallel.py --step task1
#
# # 4. Generate hypotheses for Task 2 (10 samples × 6 models × 50 papers)
# python run_parallel.py --step task2
#
# # 5. Generate embeddings for Task 1
# python run_parallel.py --step embed_task1
#
# # 6. Generate embeddings for Task 2
# python run_parallel.py --step embed_task2
# ----- END FULL REPLICATION -----

# Generate analysis, metrics, and all figures from cached data
echo ""
echo "Running analysis and generating figures..."
python analysis.py

echo ""
echo "========================================="
echo " Results Summary"
echo "========================================="
echo ""
echo "Generated files:"
echo "  results/metrics.json            - All numerical metrics"
echo "  results/plots/heatmap_A.pdf     - Figure 1(A): Inter-model similarity (Task 1)"
echo "  results/plots/heatmap_B.pdf     - Figure 1(B): Inter-model similarity (Task 2)"
echo "  results/plots/intra_model_repetition_underlying_hypotheses.pdf - Figure 2(A)"
echo "  results/plots/intra_model_repetition_new_hypotheses.pdf        - Figure 2(B)"
echo "  results/plots/intra_inter_kde_pooled_A.pdf  - Figure 3(A): KDE (Task 1)"
echo "  results/plots/intra_inter_kde_pooled_B.pdf  - Figure 3(B): KDE (Task 2)"
echo ""

# Print key metrics
python -c "
import json
with open('results/metrics.json') as f:
    m = json.load(f)
print('KEY METRICS:')
print(f'  Inter-model similarity (Task 1, convergence desired): {m[\"inter_model_mean_offdiag_task1\"]:.4f}')
print(f'  Inter-model similarity (Task 2, diversity desired):   {m[\"inter_model_mean_offdiag_task2\"]:.4f}')
print()
print('  Intra-model similarities (Task 1):')
for model in m['model_order']:
    print(f'    {model}: {m[\"intra_model_similarity_task1\"][model]:.4f}')
print()
print('  Intra-model similarities (Task 2):')
for model in m['model_order']:
    print(f'    {model}: {m[\"intra_model_similarity_task2\"][model]:.4f}')
print()
print('  KDE Stats:')
print(f'    Task 1 - Same paper mean: {m[\"kde_stats\"][\"task1_same_paper_mean\"]:.4f}, Different paper mean: {m[\"kde_stats\"][\"task1_diff_paper_mean\"]:.4f}')
print(f'    Task 2 - Same paper mean: {m[\"kde_stats\"][\"task2_same_paper_mean\"]:.4f}, Different paper mean: {m[\"kde_stats\"][\"task2_diff_paper_mean\"]:.4f}')
"

echo ""
echo "========================================="
echo " Done! All results in results/"
echo "========================================="
