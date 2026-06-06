#!/bin/bash
# reproduce.sh - Reproduce key results from
# "Rethinking Dataset Distillation: Hard Truths About Soft Labels"
#
# This script reproduces Table 1 (CIFAR-100, ConvNet-D3, IPC 10 & IPC 50)
# in both Hard Label (HL) and Soft Label (SL) settings.
#
# Expected runtime: ~4-6 hours on a single GPU (3 runs × 300 epochs × 10 configs × 2 settings)
# For a quick demo, set QUICK=1 to run 1 trial with fewer epochs.

set -e

QUICK=${QUICK:-0}

echo "=============================================="
echo "Dataset Distillation Replication"
echo "Table 1: CIFAR-100, ConvNet-D3, IPC 10 & 50"
echo "=============================================="

# Install dependencies
pip install datasets huggingface_hub scipy scikit-learn -q 2>/dev/null

# Download CIFAR-100 data
python -c "
import torchvision
torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
print('CIFAR-100 downloaded.')
"

# Download DCBench distilled datasets (pre-synthesized)
if [ ! -d "dcbench_data" ]; then
    echo "Downloading DCBench distilled datasets..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='justincui03/dc_benchmark', repo_type='dataset', local_dir='dcbench_data', allow_patterns=['data/condensed/*CIFAR100*'])
print('DCBench data downloaded.')
"
fi

# Ensure results directory exists
mkdir -p results

# ===== STEP 1: Run HL experiments =====
echo ""
echo "========== STEP 1: Hard Label (HL) Experiments =========="
echo ""

if [ "$QUICK" = "1" ]; then
    NUM_RUNS=1
    EPOCHS=50
    echo "(QUICK MODE: 1 run, $EPOCHS epochs)"
else
    NUM_RUNS=3
    EPOCHS=300
fi

python run_hl_experiments.py --num_runs $NUM_RUNS --epochs $EPOCHS

# ===== STEP 2: Train teacher for SL =====
echo ""
echo "========== STEP 2: Train Teacher Model =========="
echo ""

if [ ! -f "teacher.pt" ]; then
    python train_teacher_for_sl.py
else
    echo "Teacher model already exists (teacher.pt), skipping training."
fi

# ===== STEP 3: Run SL experiments =====
echo ""
echo "========== STEP 3: Soft Label (SL) Experiments =========="
echo ""

python run_sl_experiments.py --num_runs $NUM_RUNS --epochs $EPOCHS

# ===== STEP 4: Generate final results table =====
echo ""
echo "========== STEP 4: Generate Results Table =========="
echo ""

python generate_table.py

echo ""
echo "=============================================="
echo "DONE! Results saved to:"
echo "  results/table1_final.txt       (formatted table)"
echo "  results/table1_structured.json (structured data)"
echo "  results/table1_results.json    (raw experiment data)"
echo "=============================================="
