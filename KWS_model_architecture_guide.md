# Model Architecture Guide for Keyword Spotting

Companion to `keyword_spotting_project_guide.md`. It expands **Phases 4–5** into a step-by-step architecture ladder: start with a simple CNN, then improve it one deliberate change at a time, measuring what each change buys you.

**How to use this guide**
1. Read Part 0 and Part 1 first (rules and measuring tools).
2. Climb the ladder in Part 2 in order. Do not skip rungs, because each one teaches why the next exists.
3. After every rung, fill in one row of the comparison table (Part 5).

**Input assumed throughout:** log-mel features shaped `[B, 1, F=40, T≈98]` (batch, channel, frequency, time), 12 output classes.

---

## Part 0: Ground Rules for Architecture Work

1. **Change one thing at a time.** If you swap the architecture and the augmentation together, you learn nothing.
2. **Fix everything else.** Same data pipeline, same optimizer recipe, same epochs, same seed set across architectures.
3. **Use ≥3 seeds for small differences.** Seed-to-seed noise on this task is typically a few tenths of a point. A 0.2-point "improvement" from one run is not evidence.
4. **Select on validation only.** The test set is touched once at the very end.
5. **Judge on three axes, not one:** accuracy, parameters/MACs, and deployed latency. A model that is 0.3 points better but 4× slower may be worse for an always-on detector.
6. **Decision rule to move up a rung:** accept the new architecture only if it gains ≥0.5 points of validation accuracy at similar cost, *or* matches the old accuracy with clearly lower cost.

---

## Part 1: Complexity Toolkit (learn this before comparing models)

### 1.1 Counting parameters and MACs by hand
For a conv layer with `C_in` input channels, `C_out` output channels, kernel `kh × kw`, groups `g`:

- **Weights:** `C_out × (C_in / g) × kh × kw`
- **MACs:** weights × `H_out × W_out` (each weight is used once per output position)
- **Depthwise conv:** `g = C_in = C_out`, so weights = `C × kh × kw`
- **Pointwise conv:** `kh = kw = 1`, so weights = `C_in × C_out`
- **Depthwise-separable vs. standard 3×3:** cost ratio ≈ `1/C_out + 1/9`. For 64+ channels that is roughly an 8× reduction.

**Exercise:** compute these by hand for your baseline, then confirm with a tool. Matching your hand count is the best check that you understand the model.

### 1.2 Tools
- `torchinfo.summary(model, input_size=(1,1,40,98))` for per-layer shapes and parameters
- `fvcore.nn.FlopCountAnalysis` or `ptflops` for MACs (note the MAC vs. FLOP convention: 1 MAC = 2 FLOPs)
- ONNX graph in Netron for what the *deployed* operators are

### 1.3 Two metrics people forget
- **Peak activation memory:** the largest sum of simultaneously live tensors, which often matters more than parameter count on small devices. Estimate it as max over layers of (input + output tensor bytes).
- **Receptive field in time:** it should cover roughly the duration of a spoken word (~0.5–1 s, i.e. 50–100 frames at a 10 ms hop). If your network's temporal receptive field is only 200 ms, it cannot see whole words.

---

## Part 2: The Architecture Ladder

Rough accuracy ranges below are **sanity ranges for your pipeline on 12-class v2**, not promises. Published numbers use varying versions and protocols (v1 vs. v2, 12 vs. 35 classes), so compare like with like.

| Rung | Model | Idea added | Approx. params | Rough 12-class acc |
|---|---|---|---|---|
| A0 | Tiny MLP on pooled features | Floor / sanity check | ~5–20K | 60–80% |
| A1 | Baseline CNN | Local 2-D patterns | ~25K | 88–92% |
| A2 | DS-CNN | Depthwise-separable convs: more capacity per MAC | ~25–40K (S) | 93–95% |
| A3 | TC-ResNet | Treat frequency as channels, convolve only in time; add residuals | ~65K | 94–96% |
| A4 | BC-ResNet | Broadcast 1-D temporal modelling back over 2-D features | <10K up to ~300K | 96–98% |
| A5 | Deployment-aware tweaks | Quantization- and NPU-friendly ops | same | same, with better int8 |
| A6 | Attention / streaming (reference) | Global context, causal inference | varies | for reading |

---

### Rung A0: The Floor (30 minutes)
**What:** average log-mel features over time (and maybe also take std over time) to get a `[80]` vector, then use a 2-layer MLP.
**Why:** it tells you how much of the task is solvable *without* temporal modelling. If the MLP gets 70%, you know the CNN's gains come from structure. It also catches data bugs, since a floor that is far too high hints at leakage.
**Exit criterion:** training works end-to-end; accuracy is clearly above chance (8.3%) and clearly below the CNN.

---

### Rung A1: Baseline CNN
**Idea:** three conv blocks (conv → BN → ReLU → max-pool), global average pooling, linear classifier.

```python
import torch, torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, n_classes=12, c=(16, 32, 64)):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, c[0]), block(c[0], c[1]), block(c[1], c[2]))
        self.head = nn.Linear(c[2], n_classes)

    def forward(self, x):                 # x: [B,1,40,98]
        x = self.features(x)              # [B,64,5,12]
        return self.head(x.mean(dim=(2, 3)))
```
About 24K parameters (verify by hand: 144 + 4,608 + 18,432 conv weights + BN + 780 head).

**Things to try on this rung:** width (8/16/32 base channels), 2 vs. 3 vs. 4 blocks, max-pool vs. strided conv, dropout before the head.
**What you learn:** a working training loop, BN behaviour, how capacity trades against accuracy.
**Limitation to notice:** every conv spends many MACs at high resolution on full 2-D maps, which is wasteful for spectrogram-like inputs.

---

### Rung A2: DS-CNN (the standard tiny KWS model)
**Idea:** replace standard 3×3 convs with **depthwise 3×3 + pointwise 1×1**. This gives the same receptive field growth at a fraction of the MACs, so you can afford more channels.

**Hello Edge layout (adapt to your input):** one standard conv stem (kernel ≈ 10×4, stride 2×2), then N depthwise-separable blocks at constant width, then average pooling and a linear layer. The paper's small/medium/large variants differ in width (≈64/172/276) and depth (4/4/5 blocks). Note that the paper used 10 MFCC × 49 frames, so your 40-mel input needs a stem that handles that shape.

```python
class DSBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))

class DSCNN(nn.Module):
    def __init__(self, n_classes=12, c=64, n_blocks=4, drop=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, (10, 4), stride=(2, 2), padding=(5, 1), bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True))          # [B,c,21,49]
        self.blocks = nn.Sequential(*[DSBlock(c, c) for _ in range(n_blocks)])
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(c, n_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.head(self.drop(x.mean(dim=(2, 3))))
```
About 23K parameters at `c=64, n_blocks=4`.

**Experiments on this rung**
- Width `c` ∈ {32, 64, 96, 128}, plot accuracy vs. params and MACs
- Depth 3 / 4 / 5 / 6 blocks
- Stem kernel and stride (bigger stride = fewer MACs but less detail)
- Input features: 40-mel vs. 10–13 MFCC (a DCT of the log-mel). MFCCs are smaller and were used in the original paper.
- Dropout 0 / 0.1 / 0.2

**What you learn:** the accuracy-vs-cost frontier, and why depthwise convs are the workhorse of tiny models.
**Limitation:** it treats time and frequency symmetrically, but speech structure is not symmetric. Words are defined by how spectra *change over time*.

---

### Rung A3: TC-ResNet (convolve over time only, add residuals)
**Idea:** reshape the input so **frequency bins become channels** (`[B, 40, 1, T]` or 1-D `[B, 40, T]`) and use 1-D temporal convolutions. This is a big reduction in MACs and latency, and 1-D convs are very hardware-friendly, which is why the paper emphasized real device latency. Add residual connections so you can go deeper.

```python
class TCBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 9, stride=stride, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 9, padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.skip = nn.Sequential(
            nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.skip(x))

class TCResNet8(nn.Module):
    def __init__(self, n_mels=40, n_classes=12, widths=(16, 24, 32, 48)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm1d(widths[0]), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            TCBlock(widths[0], widths[1], 2),
            TCBlock(widths[1], widths[2], 2),
            TCBlock(widths[2], widths[3], 2))
        self.head = nn.Linear(widths[3], n_classes)

    def forward(self, x):                      # x: [B,1,40,T] -> squeeze to [B,40,T]
        x = x.squeeze(1)
        x = self.blocks(self.stem(x))
        return self.head(x.mean(dim=2))
```
About 65K parameters (consistent with the ~66K the TC-ResNet8 paper reports).

**Experiments on this rung**
- Width multiplier (0.5×, 1×, 1.5×)
- Kernel size 3 / 5 / 9 / 15 (check the temporal receptive field each time)
- With and without residual skips (see how much depth helps)
- Stride schedule (where you downsample time)
- Depthwise-separable 1-D convs in place of full 1-D convs

**What you learn:** why input layout and the choice of "which axis becomes channels" matters as much as the operator type.
**Limitation:** frequency structure is collapsed immediately into channels, so the model has no explicit frequency-local modelling.

---

### Rung A4: BC-ResNet (best accuracy per parameter in this family)
**Idea:** keep a 2-D feature map but get the best of both worlds. A cheap **frequency-wise depthwise conv** handles frequency-local structure. The features are then **averaged over frequency** to a 1-D map, where **temporal depthwise convs (with dilation)** do the heavy temporal modelling cheaply. The result is **broadcast** back over the frequency axis and added to the residual. The block is described by:

`out = x + f2(x) + broadcast( f1( avgpool_freq( f2(x) ) ) )`

where `f2` is the frequency-wise depthwise conv with BN, and `f1` is temporal depthwise conv + BN + SiLU + pointwise conv + dropout.

```python
class BCBlock(nn.Module):
    """Simplified 'normal' BC-ResNet block (no channel change, no stride).
    Input [B,C,F,T]. Compare against the paper's figures for exact details."""
    def __init__(self, c, dilation=1, dropout=0.1):
        super().__init__()
        self.f2 = nn.Sequential(
            nn.Conv2d(c, c, (3, 1), padding=(1, 0), groups=c, bias=False),
            nn.BatchNorm2d(c))
        self.f1 = nn.Sequential(
            nn.Conv2d(c, c, (1, 3), padding=(0, dilation), dilation=(1, dilation),
                      groups=c, bias=False),
            nn.BatchNorm2d(c), nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False), nn.Dropout2d(dropout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        a = self.f2(x)                          # frequency-local features
        b = a.mean(dim=2, keepdim=True)         # [B,C,1,T] average over frequency
        b = self.f1(b)                          # temporal modelling on the 1-D map
        return self.act(x + a + b)              # b broadcasts across F
```

**Overall network (approximate, verify against Table 1 of the paper):**
stem 5×5 conv (stride 2 in frequency) → four groups of BC blocks with channels ≈ 8τ, 12τ, 16τ, 20τ, block counts ≈ 2/2/4/4, frequency strides ≈ 1/2/2/1 and temporal dilations ≈ 1/2/4/8 (transition blocks change channels and downsample) → depthwise 5×5 conv → 1×1 conv → average pool → 1×1 conv to 12 classes. The scale factor **τ** (1, 1.5, 2, 3, 6, 8) selects model size, from under 10K to a few hundred K parameters.

**How to build it without getting lost**
1. Read the paper and the official code once, end-to-end.
2. Write the normal block (above) and a shape-trace test.
3. Add the transition block (channel change + frequency stride; see the paper's block figure).
4. Assemble the stages, then **check your parameter count against the paper's** for τ=1. A mismatch means a layout mistake.
5. Train τ=1, then τ=3, then τ=6.

**Experiments on this rung**
- τ sweep: plot accuracy vs. params for BC-ResNet and for your DS-CNN/TC-ResNet curves on the same axes
- Dilation schedule on/off
- SiLU vs. ReLU (relevant for quantization, see A5)
- Frequency-wise normalization variants (search for "SubSpectral Normalization", a follow-up idea used with this family)
- Removing the broadcast path (what does the 1-D temporal branch buy you?)

**What you learn:** how a structural prior (time and frequency do different jobs) gives more than just scaling width.

---

### Rung A5: Deployment-Aware Architecture Tweaks
You already deploy models professionally, so treat this rung as *measuring* what you know on a model you built yourself.

| Choice | Why it matters | Experiment |
|---|---|---|
| SiLU/Swish vs. ReLU / ReLU6 | Smooth activations are often costlier or less friendly in int8 and on some accelerators | Retrain with ReLU6, compare fp32 and int8 accuracy |
| Depthwise convs | Great on MACs, but memory-bound; some delegates run them poorly | Benchmark depthwise vs. regular convs on each backend |
| Mean over frequency / global pooling | Reduction ops can have limited support on some runtimes | Check the ONNX op list in Netron on each target |
| Dilation | Some runtimes lack optimized dilated kernels | Compare latency with dilation replaced by stride or deeper layers |
| BN | Folds into conv at graph-optimization time; make sure it did | Inspect the exported graph for leftover BN nodes |
| Channel counts | Multiples of 8 (or 16) map better onto SIMD/NPU lanes | Round widths and compare latency |
| Input layout | Channel-first/last choice changes transposes | Check the converted graph for extra transposes |

**Quantization-friendliness test:** for each candidate model, record the int8 PTQ accuracy drop (Phase 8 of the main guide). Pick finalists by *deployed* accuracy and latency, not fp32 accuracy alone.

---

### Rung A6: Beyond Tiny CNNs (read, optionally try)
- **Keyword Transformer (KWT):** pure attention on spectrogram patches. Strong accuracy, but the sizes it needs for the best results are far above a tiny-model budget. Good for understanding the trade-off.
- **MatchboxNet / QuartzNet family:** 1-D time-channel separable convs with residual blocks. It is a good alternative reference to TC-ResNet.
- **Attention-RNN models:** a conv front-end plus a recurrent layer plus attention, an older but instructive design.
- **Streaming/causal models:** use causal convs or cached state so inference runs frame by frame on continuous audio. This is the right architecture family if you pursue the streaming stretch goal.

---

## Part 3: Ablation Matrix (run across rungs, not just one)

| Dimension | Values | Hypothesis to test |
|---|---|---|
| Input features | 40-mel, 64-mel, 10–13 MFCC | MFCC is cheaper; does it cost accuracy? |
| Normalization | per-utterance vs. global stats | Which is more robust to gain? |
| Pooling | avg vs. max vs. strided | Avg pooling for the head is usually the safer choice |
| Width | 0.5×–2× | Where does accuracy saturate? |
| Depth | 3–8 blocks | Does depth help without residuals? |
| Dropout | 0 / 0.1 / 0.3 | More regularization when the model is bigger |
| Activation | ReLU / ReLU6 / SiLU | Accuracy vs. int8 behaviour |
| Skip connections | on / off | Depth enabler |

---

## Part 4: Training-Side Boosters That Interact with Architecture

Add these *after* you have the architecture settled, one at a time:
- **Label smoothing (0.05–0.1):** often a small but consistent gain
- **EMA of weights:** cheap and usually helps final-epoch stability
- **Mixup / time-frequency masking:** more regularization for larger models
- **Knowledge distillation:** train a larger teacher (for example BC-ResNet τ=8), then distill into a small student (τ=1 or 2). This often recovers a good part of the gap and is a classic edge-ML technique.
- **Quantization-aware training:** fine-tune with fake-quant for the final deployment model
- **Longer schedules:** small models often keep improving with more epochs under strong augmentation

---

## Part 5: Selection and Reporting

### 5.1 Comparison table (fill one row per candidate)

| Model | Params | MACs | Peak act. mem | Val acc (mean ± std, 3 seeds) | fp32 latency (CPU, b=1) | int8 acc | int8 latency |
|---|---|---|---|---|---|---|---|
| A1 Baseline CNN | | | | | | | |
| A2 DS-CNN-S | | | | | | | |
| A2 DS-CNN-M | | | | | | | |
| A3 TC-ResNet8 | | | | | | | |
| A4 BC-ResNet-1 | | | | | | | |
| A4 BC-ResNet-3 | | | | | | | |
| A4 BC-ResNet-6 | | | | | | | |

### 5.2 Plots worth making
1. Accuracy vs. parameters (all families on one chart)
2. Accuracy vs. MACs
3. Accuracy vs. measured latency on your slowest target (this often reorders the models!)
4. fp32 vs. int8 accuracy per model

### 5.3 How to choose the final model
- Define your budget first (for example "≤100K params, ≤10 ms on the slowest target, ≤1 point int8 drop").
- Among models inside the budget, pick the most accurate.
- Report the Pareto frontier, not just the winner, so readers can see the trade-offs.

---

## Part 6: Resources

### Papers (read in this order)
1. *Hello Edge: Keyword Spotting on Microcontrollers* (DS-CNN, the cost-accuracy mindset) — https://arxiv.org/abs/1711.07128
2. *Temporal Convolution for Real-time Keyword Spotting on Mobile Devices* (TC-ResNet) — https://arxiv.org/abs/1904.03814
3. *Broadcasted Residual Learning for Efficient Keyword Spotting* (BC-ResNet) — https://arxiv.org/abs/2106.04140
4. *MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network Architecture for Speech Commands Recognition* — https://arxiv.org/abs/2004.08531
5. *Keyword Transformer: A Self-Attention Model for Keyword Spotting* — https://arxiv.org/abs/2104.11178
6. *Streaming keyword spotting on mobile devices* — https://arxiv.org/abs/2005.06720
7. *A neural attention model for speech command recognition* — https://arxiv.org/abs/1808.08929

### Reference code (use for comparison *after* writing your own)
- ARM ML-KWS-for-MCU (Hello Edge code) — https://github.com/ARM-software/ML-KWS-for-MCU
- Qualcomm AI Research BC-ResNet — https://github.com/Qualcomm-AI-research/bcresnet
- ARM Keyword Transformer — https://github.com/ARM-software/keyword-transformer

### Concepts to study alongside
- Depthwise-separable convolutions: the MobileNet paper (https://arxiv.org/abs/1704.04861)
- Residual learning (ResNet): https://arxiv.org/abs/1512.03385
- Receptive field arithmetic (search "receptive field arithmetic for convolutional networks")
- Knowledge distillation (Hinton et al., https://arxiv.org/abs/1503.02531)
- Training craft: Karpathy, *A Recipe for Training Neural Networks* — http://karpathy.github.io/2019/04/25/recipe/

### Verifying your implementation (cheap sanity checks)
- [ ] Parameter count matches the paper's (or your hand calculation)
- [ ] A shape-trace print shows each layer's output size as expected
- [ ] The model overfits one batch quickly
- [ ] The model exports to ONNX and matches PyTorch outputs
- [ ] Changing the number of time frames (for example 98 → 101) does not crash it

---

## Suggested Order of Work (about 3–5 evenings)

| Evening | Do | Output |
|---|---|---|
| 1 | Part 1 tools, A0 floor, A1 baseline | Working complexity counter, baseline numbers |
| 2 | A2 DS-CNN and its width/depth sweep | Accuracy-vs-cost curve |
| 3 | A3 TC-ResNet | Time-only convolution results and comparison |
| 4 | A4 BC-ResNet τ=1 and τ=3 | Best small model so far |
| 5 | A5 deployment tweaks, Part 5 table and plots | Final candidate list ready for Phase 7–9 |