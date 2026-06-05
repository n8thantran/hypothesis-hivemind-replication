#!/bin/bash
# Reproduce all key results from the CGM-Agent paper
# Usage: bash reproduce.sh [table_number]
# If no table_number given, runs all tables

set -e

echo "=========================================="
echo "CGM-Agent Replication - Reproduce Script"
echo "=========================================="

# Install dependencies
pip install textstat aiohttp pandas numpy scipy -q

# Create results directory  
mkdir -p /workspace/results

# Step 1: Generate QA dataset (if not already generated)
if [ ! -f /workspace/results/qa_dataset.json ]; then
    echo ""
    echo "[Step 1] Generating QA dataset..."
    python3 /workspace/generate_questions.py
else
    echo "[Step 1] QA dataset already exists, skipping generation"
fi

# Determine which tables to run
TABLE_ARG=${1:-"all"}

if [ "$TABLE_ARG" = "all" ] || [ "$TABLE_ARG" = "3" ]; then
    echo ""
    echo "[Step 2] Running Table 3: Synthetic Layer 2 evaluation..."
    python3 /workspace/run_evaluation_v2.py 3
fi

if [ "$TABLE_ARG" = "all" ] || [ "$TABLE_ARG" = "4" ]; then
    echo ""
    echo "[Step 3] Running Table 4: Layer 1 feasibility classification..."
    python3 /workspace/run_evaluation_v2.py 4
fi

if [ "$TABLE_ARG" = "all" ] || [ "$TABLE_ARG" = "5" ]; then
    echo ""
    echo "[Step 4] Running Table 5: Real-world Layer 2 evaluation..."
    python3 /workspace/run_evaluation_v2.py 5
fi

if [ "$TABLE_ARG" = "all" ] || [ "$TABLE_ARG" = "6" ]; then
    echo ""
    echo "[Step 5] Running Table 6: Readability analysis..."
    python3 /workspace/run_evaluation_v2.py 6
fi

if [ "$TABLE_ARG" = "all" ] || [ "$TABLE_ARG" = "7" ]; then
    echo ""
    echo "[Step 6] Running Table 7: Ablation study..."    
    python3 /workspace/run_evaluation_v2.py 7
fi

if [ "$TABLE_ARG" = "all" ] || [ "$TABLE_ARG" = "8" ]; then
    echo ""
    echo "[Step 7] Running Table 8: TIR correlation analysis..."
    python3 /workspace/run_evaluation_v2.py 8
fi

echo ""
echo "=========================================="
echo "All results saved to /workspace/results/"
echo "=========================================="
echo ""
echo "Result files:"
ls -la /workspace/results/table*.json 2>/dev/null || echo "No result files found"
