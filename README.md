# Keyword Spotting on Speech Commands

## Problem Statement
Always-on keyword detection for resource-constrained devices, targeting a 12-class classification problem (e.g., 10 keywords + silence + unknown).

## Success Criteria

| Metric | Target |
|---|---|
| Baseline CNN test accuracy | >= 90% |
| DS-CNN test accuracy | ~94-95% |
| Stretch (BC-ResNet) | 96%+ |
| int8 accuracy drop vs fp32 | <= 1 point |
| Model size | < 500 KB fp32 |
| Cross-hardware backends | >= 4 backends |

## Dataset Description
Google Speech Commands v0.02. We use standard speaker-independent splits for training, validation, and testing.

## Model Architectures
- **Baseline CNN**: A simple 1D or 2D CNN over spectrogram/MFCC features.
- **DS-CNN**: Depthwise-Separable CNN, optimized for low parameters while maintaining high accuracy.
- **BC-ResNet**: Broadcast-Residual Network, achieving state-of-the-art results for this dataset.

## Project Structure
```
.
├── configs/             # Configuration YAML files
├── data/                # Dataset directory (created via script)
├── experiments/         # Checkpoints, logs, and ONNX models
├── notebooks/           # Jupyter notebooks for exploration
├── scripts/             # Utility scripts (download data, pipeline)
├── web_demo/            # Web-based inference demo
├── Makefile             # Automation tasks
├── README.md            # This file
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore file
```

## Setup Instructions
1. Clone the repository: `git clone <repo_url>`
2. Create virtual environment: `python3 -m venv venv && source venv/bin/activate` (or run `make setup`)
3. Install requirements: `pip install -r requirements.txt` (or run `make setup`)
4. Download data: `make download-data`

## Usage Instructions
- **Train**: `make train` (Baseline) or `make train-dscnn` (DS-CNN)
- **Evaluate**: `make evaluate`
- **Export**: `make export`
- **Quantize**: `make quantize`
- **Benchmark**: `make benchmark`
- **Web Demo**: `make demo`

## Ablation Table
| Model / Augmentation | Epochs | Test Accuracy | Size (KB) |
|---|---|---|---|
| Baseline (no aug) | | | |
| Baseline (+ SpecAug) | | | |

## Benchmark Results
| Model | Backend | FP32 Latency | INT8 Latency | Accuracy |
|---|---|---|---|---|
| DS-CNN | ONNXRuntime | | | |

## Experiment Log
- *Date*: Experiment details...

## References
- Google Speech Commands Dataset: [arXiv:1804.03209](https://arxiv.org/abs/1804.03209)
- DS-CNN: [arXiv:1711.07128](https://arxiv.org/abs/1711.07128)
- BC-ResNet: [arXiv:2105.04715](https://arxiv.org/abs/2105.04715)

## License
MIT License
