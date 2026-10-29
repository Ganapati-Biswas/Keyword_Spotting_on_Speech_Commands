# Keyword Spotting Project: Skills Gained and Future Scope

Companion to `keyword_spotting_project_guide.md` & `kws_model_architecture_guide.md`. This document covers what you will gain from finishing the project and what you can build on top of it.

---

## 1. Skills and Knowledge You Will Gain

### 1.1 Training craft (the main gap this project closes)
- Building a data pipeline from raw files: label mapping, class balancing, silence and unknown-class generation, caching
- Augmentation: noise mixing, time shift, SpecAugment-style masking
- Debugging a training run: overfit-one-batch, shuffled-label test, initial-loss check, reading loss and accuracy curves
- Optimizer and schedule choices (AdamW, cosine/one-cycle), label smoothing, EMA
- Running controlled ablations and judging results against seed-to-seed noise
- Reproducibility habits: seeds, pinned versions, config and commit tracking, experiment logs

### 1.2 Model design intuition
- Why depthwise-separable convolutions, time-only (1-D) convolutions, and broadcasted residuals exist
- Counting parameters, MACs, and activation memory by hand, and checking against tools
- Temporal receptive field reasoning
- Reading a paper, re-implementing it, and verifying the implementation against published parameter counts
- Trading accuracy against size, MACs, and **measured** latency (which can reorder the models)

### 1.3 Evaluation rigor
- Speaker-independent splits and leakage checks
- Per-class precision/recall, confusion matrices, error analysis on real audio
- Detector-style metrics: false accept rate, false reject rate, threshold sweeps, ROC/DET curves
- Knowing when an accuracy number is misleading, such as an easy `_silence_` class or an unbalanced `_unknown_` class

### 1.4 Audio and DSP basics
- Sampling, windowing, STFT, mel filterbanks, log compression, MFCC
- Front-end design: inside vs. outside the model, and an export-friendly STFT as a fixed-weight conv
- Why feature-extraction mismatches between runtimes cause silent accuracy loss

### 1.5 Portable deployment
- Exporting to ONNX, checking numerical parity, and inspecting graphs in Netron
- Post-training static quantization, calibration, and (as a stretch) quantization-aware training
- Running one model through multiple ONNX Runtime execution providers (CPU, CoreML, CUDA, WebGPU/WASM) and optionally LiteRT
- Benchmark methodology: warm-up, percentiles, thermal throttling, end-to-end vs. model-only latency
- How training choices (activation function, normalization, architecture) affect quantization and backend behaviour

### 1.6 Engineering and communication
- Structuring an ML repo so others can reproduce it
- Writing a results-driven README with limitations stated honestly
- Shipping a live browser demo

### 1.7 What this project does *not* teach
- Transformer and LLM internals
- Large-scale or distributed training
- GPU kernel programming
- Serving at scale (batching, KV cache, scheduling, multi-tenant systems)

If you are heading toward inference and model-serving systems, this project gives you the foundation of training literacy and quantization intuition, not the whole path.

---

## 2. How to Describe It (portfolio and resume angle)

- "Trained a sub-100K-parameter keyword-spotting model on Speech Commands v2 to ≥ XX% test accuracy; ran ablations across DS-CNN, TC-ResNet, and BC-ResNet families."
- "Exported to ONNX with verified parity; int8 quantization with ≤ X-point accuracy drop."
- "Benchmarked the same model across N backends (CPU, CoreML, CUDA, WebGPU, ...) with p50/p95/p99 latency; published a live in-browser demo."
- Link the repo, benchmark table, and demo in one place.

Fill in the real numbers once you have them.

---

## 3. What You Can Build Next

### 3.1 Short term (1–2 months), same skill set

| Project | What it is | Why it is a good next step |
|---|---|---|
| **Streaming voice pipeline** | Voice activity detection + wake word + command recognition on continuous audio, with a proper detector and false-accepts-per-hour measurement | Moves from clip classification to a real always-on system |
| **Open-source cross-backend benchmark harness** | A clean tool that takes an ONNX model and reports accuracy, latency percentiles, and memory across CPU, CoreML, CUDA, WebGPU, and mobile delegates | Reuses your Phase 9 work; your multi-SoC and delegate experience is the differentiator; useful to others |
| **Image-model version of the same pipeline** | Apply the same export, quantization, and benchmark harness to a vision model (for example an image quality classifier) | Shows quantization sensitivity across modalities and relates to ISP work |

### 3.2 Medium term (3–6 months)

| Project | What it is | New skills |
|---|---|---|
| **Fully local voice assistant** | Wake word → streaming speech recognition (Whisper-class models) → small on-device LLM → text-to-speech | LLM quantization, KV cache, latency budgeting across a multi-model pipeline, streaming systems |
| **Open-source contributions** | Audio-side work in whisper.cpp / llama.cpp, or operator support, quantization and benchmarks in ONNX Runtime | Real-world inference-engine codebases, code review, community visibility |
| **Distillation and compression study** | Systematically compare distillation, QAT, pruning, and width scaling on the KWS family | Model-compression depth, publishable-style results |

### 3.3 Longer term: product directions (hypotheses to validate, not validated ideas)

| Direction | Description | Honest caveat |
|---|---|---|
| **Indic-language on-device voice commands** | Keyword spotting and small-vocabulary recognition for Hindi, Bengali, Kannada, and other languages | Underserved space, but data collection and evaluation across dialects are the hard part |
| **Custom wake-word / voice-command SDK** | A toolkit for embedded and appliance makers to train and deploy small voice models | Existing players include Picovoice, openWakeWord, and Edge Impulse, so you would need a clear angle such as language coverage, footprint, or training workflow |
| **Edge-ML deploy-and-benchmark tooling** | A service or tool that runs a model across many real devices and reports comparable numbers | Addresses a real pain point, and OEM/SoC experience is a strong advantage; device access is the practical challenge |

Before building any product, talk to potential customers first and validate that the problem is real, painful, and paid for.

---

## 4. Suggested Path

1. Finish this project at the **Target** tier (see `kws_project_scope.md`).
2. Turn the Phase 9 benchmark code into an **open-source cross-backend benchmark harness**.
3. Build the **local voice assistant** as the larger project, which brings in LLM inference.
4. Use contributions to whisper.cpp, llama.cpp, or ONNX Runtime as the bridge into inference-engine work.
5. Revisit product ideas once you have users to talk to and real numbers to show.

This sequence covers training, edge deployment, and LLM inference, so it supports both a portfolio story and an inference-systems career direction.

---

## 5. Skills Checklist (tick off as you go)

**Training**
- [ ] Wrote a data pipeline and unit-tested it
- [ ] Ran sanity checks before real training
- [ ] Ran ablations with ≥ 3 seeds
- [ ] Explained why each architecture rung beat the previous one

**Evaluation**
- [ ] Produced per-class metrics and a confusion matrix
- [ ] Wrote an error analysis based on listening to real failures
- [ ] Measured a detector-style FAR/FRR curve

**Deployment**
- [ ] Verified ONNX parity with PyTorch
- [ ] Quantized to int8 and measured accuracy and latency changes
- [ ] Benchmarked on ≥ 4 backends with documented methodology
- [ ] Shipped the browser demo

**Communication**
- [ ] README that a stranger can use to reproduce your results
- [ ] Limitations section written honestly