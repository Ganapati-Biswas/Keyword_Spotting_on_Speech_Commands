# Keyword Spotting on Speech Commands: End-to-End Project Guide

**Goal:** Train a small keyword-spotting (KWS) model from scratch, evaluate it properly, export it to ONNX, quantize it, and run and benchmark the *same* model on every piece of hardware you can get your hands on (CPU, Apple Silicon, GPU, browser, Android).

**Estimated effort:** 2–3 weekends. Phases 0–4 take about one weekend, Phases 5–6 another, and Phases 7–10 a third.

**What you will actually learn (the point of the project):**
- Training craft: data pipelines, augmentation, debugging a training run, reading curves, ablations
- Honest evaluation: speaker-independent splits, per-class errors, false accepts
- Deployment under constraints: export parity, quantization accuracy loss, cross-backend benchmarking

You already know the deployment side of edge inference. Spend your extra time on Phases 3–6, where most of the new learning is.

---

## Phase 0: Understand the Problem (½ day)

### 0.1 What is keyword spotting?
An always-on model listens to audio and detects a small vocabulary of trigger words ("yes", "stop", "go", ...). It must be tiny (low power, low memory), low-latency, and robust to noise, accents, and silence.

### 0.2 The task you will solve
Clip-level classification of 1-second, 16 kHz audio clips into **12 classes**:

`yes, no, up, down, left, right, on, off, stop, go, _silence_, _unknown_`

- The 10 command words are the targets.
- `_unknown_` = the other 25 words in the dataset (bed, bird, cat, ...).
- `_silence_` = background noise only (generated from the provided noise files).

### 0.3 Questions to be able to answer before coding
- Why is a plain accuracy number misleading for an always-on detector? (Think: false accepts per hour.)
- Why must train/val/test be split **by speaker**?
- Why is the `_unknown_` class the hardest to handle?
- Why use log-mel features instead of raw waveforms for tiny models?
- Clip-level classification vs. streaming detection: what changes?

### 0.4 Define success up front (write this in your README now)

| Metric | Target |
|---|---|
| Baseline CNN, 12-class test accuracy | ≥ 90% |
| DS-CNN (or similar), 12-class test accuracy | ~94–95% |
| Stretch (BC-ResNet-style) | 96%+ |
| int8 accuracy drop vs. fp32 | ≤ 1 point |
| Model size | < 500 KB fp32 (small variants are far below this) |
| Same model runs on | ≥ 4 backends, with a benchmark table |

### Resources
- Dataset paper: Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition* — https://arxiv.org/abs/1804.03209
- *Hello Edge: Keyword Spotting on Microcontrollers* (DS-CNN origin) — https://arxiv.org/abs/1711.07128
- MLPerf Tiny (KWS is one of its benchmarks, so it is a good reference for "what is a reasonable model") — https://github.com/mlcommons/tiny
- Hugging Face Audio Course (audio basics: waveforms, spectrograms, mel scale) — https://huggingface.co/learn/audio-course

---

## Phase 1: Environment and Repo Setup (1–2 hours)

### 1.1 Tooling
- Python 3.10–3.12, a virtual environment (`venv` or `conda`)
- Install: `torch`, `torchaudio`, `numpy`, `soundfile`, `scikit-learn`, `matplotlib`, `onnx`, `onnxruntime`, `tqdm`, optionally `tensorboard` or `wandb`
- **Pin versions** in `requirements.txt`. The torchaudio I/O and dataset helpers have been changing between releases. Loading wavs with `soundfile` and writing your own `Dataset` is the more stable route.
- Training device: CUDA if you have it, **MPS** on Apple Silicon, otherwise CPU. These models are small enough that CPU or MPS is fine.

### 1.2 Suggested repo layout
```
kws/
├── README.md
├── requirements.txt
├── data/                  # downloaded dataset (gitignored)
├── src/
│   ├── dataset.py         # Dataset, silence/unknown handling, augmentation
│   ├── features.py        # log-mel front-end
│   ├── models.py          # CNN baseline, DS-CNN, (BC-ResNet)
│   ├── train.py           # training loop, config via argparse/yaml
│   ├── evaluate.py        # metrics, confusion matrix, error analysis
│   ├── export_onnx.py
│   ├── quantize.py
│   └── benchmark.py       # backend-agnostic harness
├── configs/
├── experiments/           # one folder per run: config, log, checkpoint, metrics.json
├── notebooks/             # exploration only
└── web_demo/              # onnxruntime-web demo (Phase 9)
```

### 1.3 Reproducibility habits (start now, not later)
- Fix seeds (Python, NumPy, PyTorch)
- Save the config and git commit hash with every run
- Keep a simple experiment log (template at the end of this document)

---

## Phase 2: Get and Explore the Data (2–3 hours)

### 2.1 Download
- Speech Commands **v0.02**: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz (about 2 GB, CC BY 4.0)
- Optional but handy: the pre-packaged official 12-class test set, `speech_commands_test_set_v0.02.tar.gz`, from the same location (balanced, with silence and unknown examples). It makes your results comparable to published ones.

### 2.2 Understand the folder layout
- One folder per word, containing 1-second `.wav` files at 16 kHz
- `validation_list.txt` and `testing_list.txt` define the speaker-independent split. **Everything not listed is training data.**
- `_background_noise_/` contains ~6 long noise recordings (used for the silence class and noise augmentation)

### 2.3 Explore (do not skip, this is where intuition is built)
- [ ] Count files per word; plot the class distribution
- [ ] Listen to ~20 clips, including bad ones (clipped, quiet, truncated)
- [ ] Plot waveforms and spectrograms for 3–4 words
- [ ] Check clip lengths. Some are shorter than 1 s and need padding.
- [ ] Verify there is no speaker overlap between train, val, and test splits (the speaker ID is the filename prefix before `_nohash_`)
- [ ] Compute how imbalanced the 12-class problem becomes once all non-target words fold into `_unknown_`

### Resources
- Speech Commands page on TensorFlow's site and the paper (section on splits) for exact conventions

---

## Phase 3: Build the Data Pipeline (½–1 day)

### 3.1 Label mapping
- 10 target words → classes 0–9
- Every other word → `_unknown_`
- Generate `_silence_` clips by cropping random 1-second windows from the background noise files, scaled by a random volume (including near-zero)

### 3.2 Class balance
A common recipe is to make `_silence_` and `_unknown_` each about 10% of the training set, by subsampling unknown words each epoch (different unknown words each time is a nice trick). Keep val and test distributions fixed.

### 3.3 Feature extraction (log-mel)
Reasonable starting config: 16 kHz, 30 ms window (`n_fft=480`), 10 ms hop (`hop_length=160`), 40 mel bins, log compression (`log(mel + eps)`), then per-utterance or global normalization. The output is about 40 × 98–101 frames depending on padding and centering.

**Design decision to make deliberately:** where does the front-end live?
- **Option A: outside the model.** Compute features in the data loader. It's simpler for training, but at deployment *every* runtime (Python, C++, JS, Android) must reimplement identical feature math.
- **Option B: inside the exported graph.** The model takes raw waveform `[B, 16000]`. This is better for portability, but you need an export-friendly STFT. A robust trick is to implement the STFT as a fixed-weight `Conv1d` using a DFT basis, then apply the mel filterbank as a fixed matmul.

Recommendation: train with Option A, then do Option B as an explicit exercise in Phase 7. Always verify that the two paths match numerically.

### 3.4 Augmentation (train only)
- Random time shift (about ±100 ms)
- Background noise mixing at random SNR (with probability ~0.8)
- Optional: speed or pitch perturbation, random gain
- **SpecAugment-style** time and frequency masking on the log-mel

### 3.5 DataLoader
- Pre-cache decoded waveforms in memory (the dataset is small enough) to speed up epochs
- `num_workers > 0`, `pin_memory` on CUDA
- Unit-test the pipeline: shapes, label distribution per batch, determinism with a fixed seed

### Resources
- PyTorch tutorial on speech command classification — https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html (read for ideas, but you are building a better pipeline)
- SpecAugment — https://arxiv.org/abs/1904.08779

---

## Phase 4: Baseline Model and a Trustworthy Training Loop (½ day)

### 4.1 Baseline model
A tiny CNN: 2–3 conv blocks (conv → batchnorm → ReLU → pool), global average pooling, linear head. Expect about 90% accuracy.

### 4.2 Training loop essentials
- Loss: cross-entropy (add label smoothing later)
- Optimizer: AdamW (lr ~1e-3, weight decay ~1e-2 as a starting point)
- LR schedule: cosine or one-cycle
- Metrics per epoch: train loss, val loss, val accuracy
- Save the best checkpoint by val accuracy (never by test)

### 4.3 Sanity checks before any real training (Karpathy-style)
- [ ] Initial loss ≈ `ln(12)` ≈ 2.48 at init
- [ ] The model can **overfit a single batch** to ~100% accuracy
- [ ] Shuffle the labels: accuracy should fall to chance (this proves there is no leakage)
- [ ] Look at augmented samples (spectrograms) with your own eyes
- [ ] Train for a few epochs and confirm curves move sensibly

### 4.4 Definition of done for this phase
A reproducible `train.py` run that reaches ≥ 90% validation accuracy, with the config, log, and checkpoint saved under `experiments/`.

### Resources
- Karpathy, *A Recipe for Training Neural Networks* — http://karpathy.github.io/2019/04/25/recipe/ (the single most useful read for this phase)
- PyTorch docs: `torch.optim.AdamW`, `lr_scheduler.OneCycleLR` / `CosineAnnealingLR`

---

## Phase 5: Make the Model Good (1 weekend day)

### 5.1 Better architectures
1. **DS-CNN** (depthwise-separable convolutions): the standard tiny KWS model, ~20–100K params depending on width
2. **BC-ResNet** (stretch): broadcasted residual learning, very strong at <100K params

Read the papers *before* implementing, and compare parameter count, MACs, and accuracy for each model in one table.

### 5.2 Run ablations, one change at a time
| Experiment | Question |
|---|---|
| No augmentation vs. noise mixing | How much does it help? |
| + time shift | |
| + SpecAugment | |
| Label smoothing 0 vs. 0.1 | |
| Model width multiplier (0.5×, 1×, 2×) | Accuracy vs. size curve |
| Number of mel bins (40 vs. 64) | |
| EMA of weights | |

Log each result in your experiment table. This habit is a bigger takeaway than any single number.

### 5.3 Things that often go wrong
- Overfitting to speakers if you accidentally mix splits
- Silence class that is too easy (the model learns the volume, not noise)
- Normalization computed on the wrong subset
- Inconsistent padding between train and eval

### Resources
- Hello Edge — https://arxiv.org/abs/1711.07128 (DS-CNN details)
- BC-ResNet — https://arxiv.org/abs/2106.04140
- Reference implementations on GitHub for comparison (search "BC-ResNet pytorch"). Compare *after* you have your own version.

---

## Phase 6: Evaluate Properly (½ day)

### 6.1 Rules
- Choose your final model using **validation** only
- Touch the test set **once** for the final number (and report it with the official test set if you downloaded it)

### 6.2 What to report
- [ ] 12-class accuracy
- [ ] Per-class precision and recall
- [ ] Confusion matrix (heatmap)
- [ ] Top confusions (for example "no" ↔ "go", "up" ↔ "off") and a look at the actual audio for the worst errors
- [ ] Accuracy by noise condition if you can synthesize it (clean vs. noisy test clips)

### 6.3 Detector-style metrics (what matters for always-on)
Treat "any target word detected" as a detection problem:
- False Accept Rate (FAR) and False Reject Rate (FRR)
- Plot an ROC or DET curve by sweeping a confidence threshold on the target-class softmax score
- Define an `_unknown_` / reject policy: below threshold means reject

### 6.4 Stretch: false accepts per hour
Build a long audio stream with no target words (concatenated unknown-word clips plus background noise), run a sliding window (for example a 1 s window with a 100 ms hop), and count detections over the total duration. Be honest in the write-up about how synthetic this negative stream is.

### 6.5 Definition of done
An `evaluate.py` that produces all of the above from a checkpoint, plus a short written error analysis.

---

## Phase 7: Export to ONNX and Verify Parity (½ day)

### 7.1 Export
```python
model.eval()
dummy = torch.randn(1, 1, 40, 98)          # match your real input shape
torch.onnx.export(
    model, dummy, "kws_fp32.onnx",
    input_names=["mel"], output_names=["logits"],
    dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
```
Note: recent PyTorch versions may default to the dynamo-based exporter, which has a different dynamic-shape API. If you hit export issues, try the other mode, and consult the current `torch.onnx` docs.

### 7.2 Validate the graph
```python
import onnx, onnxruntime as ort, numpy as np
onnx.checker.check_model(onnx.load("kws_fp32.onnx"))
sess = ort.InferenceSession("kws_fp32.onnx", providers=["CPUExecutionProvider"])
onnx_out = sess.run(None, {"mel": x.numpy()})[0]
np.testing.assert_allclose(torch_out.numpy(), onnx_out, rtol=1e-3, atol=1e-4)
```

### 7.3 Parity checklist
- [ ] Logits match PyTorch within tolerance on a batch of real data
- [ ] Argmax agreement is 100% on the full test set (fp32)
- [ ] Works for batch sizes 1 and N
- [ ] Inspect the graph in Netron (https://netron.app) to see the operators you will be asking each backend to support

### 7.4 Move the front-end into the graph (Option B from 3.3)
Export a second model that takes the raw waveform `[B, 16000]`. Verify it matches "Python features + model" on real clips. This is the version that makes browser and mobile deployment easy.

### Resources
- PyTorch ONNX export docs (search "torch.onnx")
- ONNX Runtime docs — https://onnxruntime.ai/docs/
- Netron — https://netron.app

---

## Phase 8: Quantize (½ day)

### 8.1 Post-training static quantization (int8) with ONNX Runtime
```python
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantFormat, QuantType)

class KWSCalib(CalibrationDataReader):
    def __init__(self, batches):                     # list of np.float32 arrays
        self.it = iter([{"mel": b} for b in batches])
    def get_next(self):
        return next(self.it, None)

quantize_static(
    "kws_fp32.onnx", "kws_int8.onnx", KWSCalib(calib_batches),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
)
```
Tips:
- Run ORT's quantization **pre-processing** (shape inference and graph optimization) first. See the quantization docs.
- Calibrate on a few hundred *training* clips, not test.
- If the front-end is in the graph, **exclude it from quantization** (`nodes_to_exclude`). Log and FFT math quantizes badly.
- If accuracy drops a lot, try per-tensor vs. per-channel, symmetric vs. asymmetric activations, and excluding the first and last layers.

### 8.2 Measure what quantization actually does
| Model | Size | Accuracy | Latency (CPU, batch 1) |
|---|---|---|---|
| fp32 | | | |
| int8 (PTQ) | | | |

Note: int8 is **not automatically faster**. It depends on the CPU's int8 instruction support and on the runtime kernels. Report what you measure.

### 8.3 Stretch: quantization-aware training (QAT)
Fine-tune in PyTorch with fake-quant, re-export, and compare against PTQ. This is where you see which layers (often depthwise convs) are sensitive.

### Resources
- ONNX Runtime quantization docs — https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- PyTorch quantization docs (for QAT)

---

## Phase 9: Run on Real Hardware and Benchmark (1 day)

The goal here is the headline deliverable: **one trained model, many backends, one comparable table.**

### 9.1 Backends to try (do as many as you can)
| Target | How |
|---|---|
| Any CPU | ONNX Runtime `CPUExecutionProvider` (baseline for everything) |
| Apple Silicon | `providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]` |
| NVIDIA GPU | `onnxruntime-gpu` with `CUDAExecutionProvider` (free Colab GPU works); optionally TensorRT EP |
| Windows GPU/NPU | DirectML execution provider (if you have access to such a machine) |
| Browser | `onnxruntime-web` with WebGPU and WASM backends, plus a live-microphone demo |
| Android | Convert via `ai-edge-torch` to LiteRT and run with CPU, GPU, or NNAPI delegates, or use ORT's Android build |

Note: tiny models often run *faster on CPU* than on GPU or NPU because of dispatch and transfer overhead. Seeing and explaining this is a learning goal, not a failure.

### 9.2 Benchmark methodology (get this right)
```python
import time, numpy as np
def bench(sess, x, warmup=50, iters=1000):
    for _ in range(warmup):
        sess.run(None, {"mel": x})
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, {"mel": x})
        ts.append((time.perf_counter() - t) * 1e3)
    return np.percentile(ts, [50, 95, 99])
```
- Batch size 1 (always-on use case), plus a larger batch for throughput
- Warm up first. First-run cost (graph compile, kernel selection) is its own number, so report it separately.
- Fix and report thread counts; run with the machine otherwise idle
- Watch for thermal throttling on laptops and phones: run a long loop and look at the trend
- Report both **model-only** latency and **end-to-end** (front-end + model) latency
- For *each* backend, also run the full test set and report accuracy (numerics differ across backends)

### 9.3 Results table (this is your deliverable)
| Backend | Device | Precision | Model size | p50 ms | p95 ms | p99 ms | Test acc | Notes |
|---|---|---|---|---|---|---|---|---|
| ORT CPU | | fp32 | | | | | | |
| ORT CPU | | int8 | | | | | | |
| ORT CoreML | | fp32 | | | | | | |
| ORT CUDA | | fp32 | | | | | | |
| ORT Web (WASM) | | fp32 | | | | | | |
| ORT Web (WebGPU) | | fp32 | | | | | | |
| LiteRT (Android) | | fp32/int8 | | | | | | |

### 9.4 Browser demo (highly recommended)
A static page that loads the ONNX model with `onnxruntime-web`, captures the mic via `getUserMedia`, runs a rolling 1-second buffer, and shows the predicted word. Use the in-graph front-end model (7.4) to avoid reimplementing log-mel in JavaScript. Host it on GitHub Pages so anyone can open it on any device.

### 9.5 Stretch: C++ runner
Write a minimal C++ program using the ONNX Runtime C/C++ API that loads the model, reads a wav file, and prints the prediction and latency. Compare its latency against the Python harness to quantify Python overhead.

### Resources
- ONNX Runtime execution providers — https://onnxruntime.ai/docs/execution-providers/
- CoreML EP — https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- ONNX Runtime Web tutorials — https://onnxruntime.ai/docs/tutorials/web/
- `ai-edge-torch` (PyTorch → LiteRT) — https://github.com/google-ai-edge/ai-edge-torch

---

## Phase 10: Package and Write Up (½ day)

### 10.1 README contents
1. Problem statement and success criteria (from Phase 0)
2. Dataset and split description
3. Model architecture and parameter count / MACs
4. Training recipe (augmentations, hyperparameters)
5. Ablation table (Phase 5)
6. Final results and error analysis (Phase 6)
7. Quantization results (Phase 8)
8. Cross-hardware benchmark table (Phase 9)
9. How to reproduce, with exact commands
10. Lessons learned and what you would do next

### 10.2 Final checklist
- [ ] Anyone can clone the repo and reproduce the headline numbers with documented commands
- [ ] The fp32 and int8 ONNX files are committed or attached to a release
- [ ] The benchmark table lists devices, versions, and methodology
- [ ] The browser demo link works
- [ ] Known limitations are stated (accents, noise types, synthetic false-accept test, etc.)

---

## Stretch Goals (after the core project)

- **Streaming inference:** a stateful or causal model (streaming conv with cached state) and a proper detector with smoothing (for example require N consecutive frames above threshold)
- **Your own wake word:** record yourself or friends saying a custom word and fine-tune the model, which shows how few examples are needed and why generalization is hard
- **Knowledge distillation:** train a larger teacher and distill into a smaller student; plot the size/accuracy frontier
- **Robustness suite:** test with added noise, reverberation, and different microphones
- **Next project:** apply the same ONNX export, quantization, and benchmark harness to an image model (for example an image quality classifier) and compare quantization sensitivity across modalities

---

## Suggested Timeline

| Block | Phases | Output |
|---|---|---|
| Weekend 1, day 1 | 0, 1, 2, 3 | Understanding, repo, data explored, pipeline working |
| Weekend 1, day 2 | 4 | Baseline trained at ≥ 90% with sanity checks done |
| Weekend 2, day 1 | 5 | DS-CNN trained, ablation table |
| Weekend 2, day 2 | 6 | Full evaluation and error analysis |
| Weekend 3, day 1 | 7, 8 | ONNX fp32 and int8 with parity and accuracy numbers |
| Weekend 3, day 2 | 9, 10 | Benchmark table, browser demo, README |

---

## Experiment Log Template

| ID | Date | Model | Params | Change vs. previous | Val acc | Notes / what I learned |
|---|---|---|---|---|---|---|
| e001 | | baseline CNN | | none | | |
| e002 | | baseline CNN | | + noise mixing | | |
| e003 | | DS-CNN-S | | new architecture | | |

---

## Master Resource List

**Papers**
- Speech Commands dataset — https://arxiv.org/abs/1804.03209
- Hello Edge (DS-CNN) — https://arxiv.org/abs/1711.07128
- BC-ResNet — https://arxiv.org/abs/2106.04140
- SpecAugment — https://arxiv.org/abs/1904.08779
- Streaming keyword spotting on mobile devices — https://arxiv.org/abs/2005.06720

**Training craft**
- Karpathy, *A Recipe for Training Neural Networks* — http://karpathy.github.io/2019/04/25/recipe/
- PyTorch speech command tutorial — https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
- Hugging Face Audio Course — https://huggingface.co/learn/audio-course

**Deployment**
- ONNX Runtime docs — https://onnxruntime.ai/docs/
- ORT quantization — https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ORT execution providers — https://onnxruntime.ai/docs/execution-providers/
- ORT Web — https://onnxruntime.ai/docs/tutorials/web/
- ai-edge-torch — https://github.com/google-ai-edge/ai-edge-torch
- Netron model viewer — https://netron.app

**Benchmarks and references**
- MLPerf Tiny — https://github.com/mlcommons/tiny