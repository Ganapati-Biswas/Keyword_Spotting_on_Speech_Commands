#!/bin/bash
set -e

echo "Running full Keyword Spotting pipeline..."

echo "1. Training Baseline Model..."
make train

echo "2. Evaluating Best Model..."
make evaluate

echo "3. Exporting to ONNX..."
make export

echo "4. Quantizing Model..."
make quantize

echo "5. Running Benchmarks..."
make benchmark

echo "Pipeline completed successfully!"
