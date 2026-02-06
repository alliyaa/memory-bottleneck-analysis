#!/bin/bash
echo "========================================="
echo "Running All Memory Bottleneck Experiments"
echo "WITH BANDWIDTH ESTIMATION"
echo "========================================="

python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found!'"
if [ $? -ne 0 ]; then
    echo "ERROR: Need GPU!"
    exit 1
fi

mkdir -p results/logs results/plots

echo "\n1. Basic Training"
python train.py

echo "\n2. Batch Size Scaling (Bandwidth Bottleneck)"
python experiments/batch_scaling.py

echo "\n3. Sequence Length Scaling (Capacity Bottleneck)"
python experiments/sequence_scaling.py

echo "\n4. FP32 vs FP16 Precision"
python experiments/precision_test.py

echo "\n5. Generating Plots"
python utils/plotting.py

echo "\n========================================="
echo "DONE! Check:"
echo "  - Data: results/logs/*.csv"
echo "  - Plots: results/plots/*.png"
echo "========================================="
