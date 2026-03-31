# NSHT_Dual_Evo Technical Monograph

## 1. Scope and Contribution Framing

This document is a paper-grade architecture specification for **NSHT_Dual_Evo**, the Paper 3 dual-stream ECG classifier implementation. It formalizes:

1. NSHT_Dual_Evo dual-stream architecture (active codebase: `src/models/nsht_dual_evo.py`).
2. Learnable wavelet preprocessing and cross-modal attention fusion.
3. Training, balancing, explainability, and reproducibility protocol.
4. Historical note: NSHT-Tri extension (tri-stream variant with statistical features)—available for extension but not active in current deployment.

The focus is to ensure method claims are scientifically precise, implementation-accurate, and publication-ready.

---

## 2. Problem Setup and Mathematical Notation

Define heartbeat dataset:

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N},\quad x_i\in\mathbb{R}^{L},\ y_i\in\{0,1,2,3,4\}
$$

where:

- $L=216$ samples per R-peak segment.
- Classes $\{0,1,2,3,4\}$ correspond to AAMI types $\{N,S,V,F,Q\}$.

**NSHT_Dual_Evo** (primary implementation) learns a dual-stream mapping:

$$
f_\theta: \mathbb{R}^{216}\times\mathbb{R}^{H\times W\times3}\rightarrow\Delta^{5}
$$

where the first input is the 1D temporal signal and the second is a 2D spectral image.

**Historical note:** NSHT-Tri extension (unused in current deployment) would add a statistical feature vector $s_i\in\mathbb{R}^{d_s}$:

$$
f^{\mathrm{tri}}_\theta: \mathbb{R}^{216}\times\mathbb{R}^{H\times W\times3}\times\mathbb{R}^{d_s}\rightarrow\Delta^{5}
$$

---

## 3. Why NSHT Exists

Paper 3 addresses three bottlenecks in prior ECG systems:

1. Fixed preprocessing limits adaptation to noise/domain changes.
2. Unimodal encoders fail to jointly reason over time morphology and spectral texture.
3. CE-only objectives can under-structure latent class geometry.

NSHT targets these with:

1. Learnable wavelet front-end.
2. Cross-modal temporal-spectral fusion.
3. Prototype-consistency regularization.

---

## 3.1 Standard Hybrid ECG vs NSHT_Dual_Evo Baseline Comparison

| Aspect | Conventional Hybrid Baseline | NSHT_Dual_Evo (Current Implementation) |
|--------|------------------------------|----------------------------------------|
| Preprocessing | Fixed/hand-tuned wavelets | **Learnable Morlet parameters** |
| Temporal encoder | RNN or basic 1D CNN | **Inception multi-scale + residuals** |
| Spectral encoder | Standard 2D CNN | **Lightweight CWT + optimized blocks** |
| Modal fusion | Static concatenation | **Cross-modal attention (1D queries 2D)** |
| Latent structure | CE loss only | **CE + prototype consistency joint loss** |
| Low-frequency modeling | Handcrafted/Implicit | **Explicit multi-scale 1D temporal kernels** |
| Training stability | Standard AMP | **BF16 AMP + TF32** |
| Evaluation policy | Often global balancing | **Split-first option (leakage-safe)** |
| Explainability | Post-hoc only | **Structured XAI outputs (wavelet, attention, streams)** |
| XAI runtime | N/A | `scripts/explain_paper3.py` |

**Key Novelty:** NSHT_Dual_Evo combines learnable preprocessing, cross-modal fusion, prototype learning, and structured explainability into an end-to-end-optimized framework with only 706K parameters.

**Codebase Reference:** See [MODULAR_CODEBASE_README.md](MODULAR_CODEBASE_README.md#project-structure) for project structure and runtime setup.

---

## 4. High-Level Architecture

### 4.1 NSHT (Dual Stream)

Input beat drives two synchronized representations:

1. **Temporal branch:** 1D Conv → Multi-Scale Inception Block → Temporal MHSA
2. **Spectral branch:** Learnable Wavelet Front-End → Dual Coordinate Attention CNN Block

An **Evolutionary Fusion** module uses cross-attention and gating to align both branches into a shared latent vector for pooling and classification.

### 4.2 Legacy NSHT-Tri

> [!WARNING]
> Prior drafts included an `NSHT-Tri` variant that added a third branch for handcrafted statistical descriptors. This concept is structurally obsolete and fully superseded by the expressive power of the dual-stream feature fusion. Ensure future manuscript claims refer strictly to dual-stream `NSHT_Dual_Evo`.

---

## 5. Learnable Wavelet Front-End

### 5.1 Parametric Morlet Kernel

A simplified learnable real-valued Morlet-style atom:

$$
\psi(t;\sigma,\omega_0)=\exp\!\left(-\frac{t^2}{2\sigma^2}\right)\cos\!\left(\omega_0\frac{t}{\sigma}\right)
$$

Each filter has trainable $(\sigma,\omega_0)$ parameters.

### 5.2 Filter-Bank Convolution

For input signal $x$ and filter bank $\{\psi_k\}_{k=1}^{K}$:

$$
z_k = x * \psi_k
$$

$$
Z=[z_1,\dots,z_K]\in\mathbb{R}^{K\times L}
$$

This creates adaptive denoising and frequency-selective responses learned end-to-end.

### 5.3 Practical Benefit

Compared to fixed wavelet transforms, this front-end can retune to dataset-specific morphology and noise profiles during optimization.

---

## 6. Temporal Encoder (1D Stream)

The temporal encoder natively captures morphology and timing-sensitive cues sequentially:

1. **Initial 1D Convolution** for base feature projection.
2. **Multi-scale Inception Block** (Kernel sizes $k=9, 19, 39$) to natively capture varying frequency resolutions (such as wide low-frequency P-waves and sharp high-frequency QRS complexes simultaneously).
3. **Temporal Attention** (Multi-Head Self-Attention over the 1D sequence) to learn global beat-level phase relationships.

Output sequence feature map:

$$
T\in\mathbb{R}^{B\times C_t\times L_t}
$$

---

## 7. Spectral Encoder (2D Stream)

### 7.1 Spectral Representation

The spectral path encodes time-frequency texture and harmonic distribution cues from the active `AdaptiveMorletWavelet` layer output.

The subsequent structure applies lightweight hierarchical processing:
1. Primary `Conv2D` block.
2. `Coordinate Attention` mechanism to capture direction-aware cross-channel relationships.
3. Spatial downsampling `Conv2D` block (stride 2).
4. Secondary `Coordinate Attention` block.
5. Spatial reduction via `AdaptiveAvgPool2D(1, None)` to a 1D sequence mapping out the temporal distribution of spectral density.

Representative output interpolated to matching temporal sequence length $L_t$:

$$
S\in\mathbb{R}^{B\times C_s\times L_t}
$$

---

## 8. Evolutionary Fusion Loop

### 8.1 Core Operation

To combine the 1D and 2D insights, an `EvolutionaryFusion` block aligns evidence dynamically.

Let temporal features be Key/Value source and spectral features be Query source:

$$
Q=W_Q S',\ K=W_K T,\ V=W_V T
$$

$$
A=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)
$$

$$
S_{\mathrm{enhanced}}=AV
$$

### 8.2 Dynamic Gated Residual

To protect the original morphological integrity from noisy spectral queries, a dynamic gate is constructed over the stream global means to form a convex combination:

$$
g = \sigma( W_{\mathrm{gate}} [ \mathrm{Mean}(T) \;\|\; \mathrm{Mean}(S) ] )
$$

$$
F_{\mathrm{gated}} = g \cdot T + (1-g) \cdot S_{\mathrm{enhanced}}
$$

The matrices are then fully reduced to a single vector sequence via a residual-added $1\times1$ convolution block: `Refine(Fuse([T || S_enhanced]) + F_gated)`.

### 8.3 Why Evolutionary Fusion Instead of Concatenation

1. Dynamic relevance weighting per temporal sequence position.
2. Protection against spectral noise overriding clean temporal morphology.
3. Expressive latent structuring enabling deep extraction without manual hooks.

---

## 10. Prototype Consistency Objective

### 10.1 Prototype Definition

Each class $c$ has learnable prototype $p_c\in\mathbb{R}^{d}$.

For sample embedding $h_i$ with label $y_i$:

$$
\mathcal{L}_{\mathrm{proto}}=\frac{1}{N}\sum_{i=1}^{N}\|h_i-p_{y_i}\|_2^2
$$

### 10.2 Joint Loss

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{CE}}+\lambda\mathcal{L}_{\mathrm{proto}}
$$

with schedule $\lambda(t)$ typically increasing through training to avoid early over-constraining.

### 10.3 Effect

- Tightens intra-class compactness.
- Improves latent-space interpretability.
- Supports prototype-space visualization and diagnostics.

---

## 10. Legacy Architecture Concept (NSHT-Tri)

> [!WARNING]
> Prior theoretical drafts included an `NSHT-Tri` variant that added a third branch for handcrafted statistical descriptors. **This tri-stream concept is structurally obsolete.** The current `NSHT_Dual_Evo` effectively captures all desired baseline and rhythm statistical priors end-to-end via the `EvolutionaryFusion` loops and robust temporal MHSA, keeping parameter footprint low without requiring manual engineering.

---

## 11. Shape Trace and Branch Interfaces

Representative flow (batch size $B$):

1. Raw beat: $B\times1\times216$.
2. Wavelet front-end output: $B\times K\times216$.
3. Temporal encoded map: $B\times C_t\times L_t$.
4. Spectral encoded map: $B\times C_s\times L_t$.
5. Cross-modal fused sequence: $B\times d\times L_t$.
6. Pooled Latent vector (GAP + GMP): $B\times 2d$.
7. Logits: $B\times5$.

All branch projections must be dimensionally aligned before concatenation/attention.

---

## 13. Training Protocol and Data Integrity

### 13.1 Split-First Balancing Policy

The repository supports leakage-safe strategy:

1. Split train/val/test first.
2. Apply SMOTE/ADASYN only on training split if enabled.
3. Keep validation/test untouched.

This should be the default for claims of generalization validity.

### 13.2 Optimization Components

Common Paper 3 runtime ingredients:

- AdamW or equivalent adaptive optimizer.
- AMP mixed precision where supported.
- Early stopping and LR scheduling.
- Gradient clipping/accumulation depending on memory budget.

### 13.3 Evaluation

Always report:

- Overall accuracy.
- Macro F1 and per-class precision/recall/F1.
- Confusion matrix.
- Optional prototype-space diagnostics.

---

## 14. Complexity, Throughput, and Memory

### 14.1 Time Complexity Drivers

Total cost combines three terms:

$$
\mathcal{C}_{\mathrm{total}}\approx \mathcal{C}_{1D}+\mathcal{C}_{2D}+\mathcal{C}_{\mathrm{fusion}}
$$

with attention term scaling approximately with sequence/context lengths.

### 14.2 Parameter Efficiency

NSHT family remains relatively compact for a multi-branch system while adding interpretability hooks unavailable in many single-stream baselines.

### 14.3 Practical Runtime Notes

- Spectral path can dominate memory if resolution is high.
- Mixed precision significantly improves feasibility on constrained GPUs.
- Branch-level profiling is recommended to detect bottlenecks.

---

## 15. Novelty Matrix

Explicit novelty points for manuscript use:

1. Learnable wavelet denoising front-end.
2. Temporal-spectral cross-modal attention fusion.
3. Dedicated low-frequency/P-wave enhancement pathways.
4. Prototype-consistency regularization in latent space.
5. Leakage-safe split-first balancing support.
6. Multi-view Paper 3 XAI artifact suite.

---

## 16. Novelty Summary: NSHT_Dual_Evo

Key innovations in **NSHT_Dual_Evo** (current deployment):

1. **Learnable wavelet denoising** — Replaces fixed preprocessing with gradient-optimizable filter banks.
2. **Temporal-spectral cross-modal attention** — Dynamically aligns 1D morphology with 2D spectral features.
3. **Dedicated low-frequency/P-wave pathway** — Explicit modeling of subtle atrial patterns.
4. **Prototype-consistency regularization** — Combines CE loss with latent-space geometry regularization.
5. **Leakage-safe split-first balancing** — Enforces methodological integrity for robust claims.
6. **Structured XAI artifact suite** — Wavelet visualization, attention heatmaps, component energy metrics.

**Historical Note:** NSHT-Tri (tri-stream variant with statistical features) was explored but is not deployed in the active codebase.
| Statistical domain priors | Rare | Optional external | Native third-stream integration |
| Explainability | Usually single-view | Multi-artifact | Multi-artifact + stats context |

---

## 17. Explainability and Artifact Protocol

Active script:

- `scripts/explain_paper3.py`

Command:

```bash
python scripts/explain_paper3.py \
  --model-path checkpoints/paper3_nsht/best_model.pt \
  --config configs/paper3_nsht.yaml \
  --num-samples-per-class 1
```

Optional split-first override:

```bash
python scripts/explain_paper3.py \
  --model-path checkpoints/paper3_nsht/best_model.pt \
  --config configs/paper3_nsht.yaml \
  --num-samples-per-class 1 \
  --data.balance_after_split
```

Typical artifact set under `experiments/paper3_nsht/xai/`:

- `wavelet_params.png`.
- `cross_attention.png`.
- `stream_contributions.png`.
- `arrays.npz`.
- per-sample and global `summary.json`.
- optional prototype visualization exports.

Prototype extraction utility:

```bash
python scripts/extract_nsht_prototypes.py \
  --model-path checkpoints/paper3_nsht/best_model.pt \
  --config configs/paper3_nsht.yaml
```

---

## 18. Failure Modes and Diagnostic Strategy

### 18.1 Common Failure Modes

1. N/S boundary ambiguity under low P-wave visibility.
2. V/F overlap in ventricular morphology variants.
3. Prototype collapse or poor separation if $\lambda$ scheduling is misconfigured.
4. Misaligned branch preprocessing causing unstable fusion.

### 18.2 Diagnostics

1. Inspect attention maps for mode collapse.
2. Inspect learned wavelet parameter distributions.
3. Evaluate class-wise embedding distances to prototypes.
4. Verify balancing policy and split integrity before interpreting metrics.

---

## 19. Ablation Blueprint

Minimum publication-quality ablations:

1. Remove learnable wavelet front-end (replace with fixed transform).
2. Remove cross-modal attention (use naive concatenation).
3. Remove prototype loss ($\lambda=0$).
4. Remove low-frequency/P-wave branch.
5. Disable stats branch (NSHT-Tri to NSHT comparison).
6. Balance policy comparison (split-first vs pre-balanced).

All ablations should include repeated seeds and confidence intervals.

---

## 20. Reproducibility Standard

1. Persist random seeds and split indices.
2. Persist exact config snapshots for every run.
3. Persist checkpoint hashes and training logs.
4. Record software stack versions and GPU runtime settings.
5. Bundle metrics with linked artifact IDs.

This supports exact reruns and auditability.

---

## 21. Architecture and Runtime Flowcharts

### 21.1 NSHT Dual/Tri Core Architecture

```mermaid
flowchart TD
  A[ECG Beat 1D] --> B1[Temporal Stream]
  A --> B2[Spectral Stream]

  subgraph Temporal Stream
    B1 --> C1[Conv1D + Base Features]
    C1 --> C2[Multi-Scale Inception Block \n k=9, 19, 39]
    C2 --> C3[Temporal MHSA Self-Attention]
    C3 --> C4[Project to Hidden Dim]
  end

  subgraph Spectral Stream
    B2 --> D1[Adaptive Morlet Wavelet\nLearnable CWT]
    D1 --> D2[Conv2D Block]
    D2 --> D3[Coordinate Attention]
    D3 --> D4[Conv2D Block]
    D4 --> D5[Coordinate Attention]
    D5 --> D6[Reduce to 1D + Project]
  end

  C4 --> E[Evolutionary Fusion]
  D6 --> E

  subgraph Evolutionary Fusion
    E --> F1[Cross-Attention\nSpectral attends Temporal]
    E --> F2[Feature Gating\nSigmoid Gate on Pooled Streams]
    F1 --> F3[Combine Streams]
    F2 --> F3[Residual Gated Sum]
  end

  F3 --> G[Global Pooling \n GAP + GMP]
  G --> H[Classifier Head \n Linear + Dropout]
  H --> I[5-Class Logits]
  
  G --> J{use_prototypes?}
  J -->|Yes| K[Prototype Projector]
  K --> L[Latent Embedding vs Prototypes]
  L --> M[Prototype Consistency Loss]
```

### 21.2 Data Integrity and Evaluation Flow

```mermaid
flowchart TD
  A[Raw Segments] --> B{balance_after_split}
  B -->|true| C[Split Train Val Test]
  B -->|false| D[Load Prebalanced Arrays]
  C --> E[Apply SMOTE or ADASYN on Train]
  E --> F[Build NSHT DataLoaders]
  D --> F
  F --> G[Train NSHT Family]
  G --> H[Validation Model Selection]
  H --> I[Test Evaluation]
  I --> J[Run explain_paper3.py]
  J --> K[Wavelet and Attention Artifacts]
```

---

## 22. Manuscript Claim Guardrails

For scientific correctness:

1. Distinguish NSHT dual-stream results from NSHT-Tri results.
2. Separate architecture gains from balancing-policy gains.
3. Attribute prototype-space interpretability claims only when prototype head/loss is active.
4. Avoid claiming fixed-wavelet behavior when learnable front-end is used.

---

## 23. Equation Rendering Compatibility

Use standalone display blocks for robust markdown rendering:

$$
\psi(t;\sigma,\omega_0)=\exp\!\left(-\frac{t^2}{2\sigma^2}\right)\cos\!\left(\omega_0\frac{t}{\sigma}\right)
$$

$$
\mathcal{L}_{\mathrm{proto}}=\frac{1}{N}\sum_{i=1}^{N}\|h_i-p_{y_i}\|_2^2
$$

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{CE}}+\lambda\mathcal{L}_{\mathrm{proto}}
$$

Prefer explicit LaTeX operators and one equation per display block.

---

## 24. Reference Pointers

Core conceptual references:

- Wavelet theory and medical signal processing foundations.
- Attention mechanisms for cross-modal fusion.
- Prototype learning and metric-regularized classification.
- ECG benchmark literature for MIT-BIH and INCART contexts.

---

## 24. Reference Pointers

Core conceptual references and codebase entry points:

- **Architecture implementation**: `src/models/nsht_dual_evo.py`
- **Configuration template**: `configs/paper3_nsht.yaml`
- **Training entrypoint**: `scripts/train.py`
- **Evaluation**: `scripts/evaluate.py`
- **XAI/explainability**: `scripts/explain_paper3.py`
- **Prototype export**: `scripts/extract_nsht_prototypes.py`
- **Project overview**: [MODULAR_CODEBASE_README.md](MODULAR_CODEBASE_README.md)

This monograph is the canonical architecture specification for Paper 3 in this repository.
