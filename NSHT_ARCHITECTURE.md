
# Neuro-Spectral Hybrid Transformer (NSHT) & NSHT-Tri
---

## NSHT vs NSHT-Tri: Architectures & Usage Guide

### NSHT (ecg_nsht_model.py)
**Description:**
- Dual-stream model combining a learnable wavelet front-end, 1D temporal encoder, 2D spectral encoder, and cross-modal attention fusion.
- Designed for high-accuracy ECG arrhythmia classification using both time and frequency domain features.

**How to Use:**
1. Prepare your data (e.g., X_balanced.npy, y_balanced.npy in balanced_data/).
2. Run ecg_nsht_model.py to train or evaluate the model:
    ```bash
    python ecg_nsht_model.py
    ```
3. Model weights and results will be saved in the models/ directory (e.g., nsht_best.pth).
4. For inference, load the model and use the provided predict/evaluate functions.

**Key Features:**
- Learnable Morlet wavelet denoising
- 1D Inception encoder (temporal)
- 2D CNN encoder (spectral)
- Cross-modal attention fusion
- Prototype consistency loss

---

### NSHT-Tri (ecg_nsht_tri.py)
**Description:**
- Extension of NSHT with a third branch for direct statistical feature injection (e.g., skewness, kurtosis, entropy).
- Fuses neural and mathematical features for improved discrimination, especially for subtle class differences.

**How to Use:**
1. Prepare your data (as above).
2. Run ecg_nsht_tri.py to train or evaluate the tri-stream model:
    ```bash
    python ecg_nsht_tri.py
    ```
3. Model weights and results will be saved in the models/ directory (e.g., nsht_tri_best.pth).
4. For inference, load the model and use the provided predict/evaluate functions.

**Key Features:**
- All NSHT features, plus:
  - Statistical feature extractor (20+ features per beat)
  - MLP projector for stats
  - Late fusion of neural and statistical features

---

## When to Use Which?
- Use **NSHT** for standard dual-stream neuro-spectral modeling.
- Use **NSHT-Tri** if you want to inject domain knowledge/statistics for potentially higher accuracy, especially on challenging class boundaries (e.g., N vs S).

---

## A Novel End-to-End Architecture for ECG Arrhythmia Classification

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement & Motivation](#problem-statement--motivation)
3. [Literature Review & Pain Points](#literature-review--pain-points)
4. [Architecture Overview](#architecture-overview)
5. [Technical Components](#technical-components)
6. [Implementation Details](#implementation-details)
7. [Experimental Results](#experimental-results)
8. [Ablation Study Design](#ablation-study-design)
9. [Comparison with State-of-the-Art](#comparison-with-state-of-the-art)
10. [Future Improvements](#future-improvements)
11. [Conclusion](#conclusion)

---

## 1. Executive Summary

**NSHT (Neuro-Spectral Hybrid Transformer)** is a novel deep learning architecture designed for ECG arrhythmia classification that addresses three fundamental limitations in current approaches:

| Problem | Our Solution |
|---------|--------------|
| Static denoising (fixed wavelets) | **Learnable Wavelet Front-End** |
| Unimodal blindness (1D OR 2D only) | **Cross-Modal Attention Fusion** |
| Lack of structural constraints | **Prototype Consistency Loss** |

### Key Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 98.80% |
| **Best Validation Accuracy** | 99.18% |
| **Parameters** | 706,821 (~0.7M) |
| **Training Time** | ~25 minutes (50 epochs) |
| **Hardware** | RTX 3050 4GB VRAM |

---

## 2. Problem Statement & Motivation

### Clinical Need

Electrocardiogram (ECG) analysis is critical for diagnosing cardiac arrhythmias. Manual interpretation by cardiologists is:
- **Time-consuming**: A 24-hour Holter monitor can contain 100,000+ heartbeats
- **Error-prone**: Inter-observer variability of 10-20%
- **Inaccessible**: Shortage of trained cardiologists globally

### Technical Challenge

The "99.5% barrier" in ECG classification is caused by:

1. **Normal vs. Supraventricular (N↔S) confusion**: Differ only by subtle P-wave timing
2. **Ventricular vs. Fusion (V↔F) confusion**: Both involve ventricular activation
3. **Patient variability**: Normal beats vary significantly between individuals

### Our Goal

Design an architecture that:
- ✅ Learns optimal denoising parameters end-to-end
- ✅ Combines temporal (1D) and spectral (2D) information
- ✅ Enforces meaningful latent space structure
- ✅ Runs on consumer hardware (4GB VRAM)

---

## 3. Literature Review & Pain Points

### Current State-of-the-Art Methods

| Paper | Method | Accuracy | Limitation |
|-------|--------|----------|------------|
| FG-HSWIN-CFA (2024) | Swin Transformer + Frequency Grouping | 98.96% | Fixed frequency bands |
| DWTFrTV-DEAL (2024) | Wavelet + Total Variation Denoising | 99.7% | Non-differentiable denoising |
| InceptionTime (2020) | 1D CNN Ensemble | 98.88% | Ignores frequency domain |
| EfficientNet-B0 | 2D CNN on Scalograms | 97.91% | Loses temporal precision |

### Critical Pain Points Identified

#### Pain Point 1: Static Denoising (QAWT/DWT)

**Problem**: Papers use fixed wavelet transforms (Morlet, db4) with hand-tuned thresholds.

```
Traditional Pipeline:
Raw ECG → Fixed Wavelet → Hard Threshold → Denoised → CNN
           ↑                    ↑
           Not learnable        Not learnable
```

**Consequence**: If noise characteristics change (different ECG device, patient movement), the fixed denoising fails.

**Our Solution**: Learnable Morlet Wavelets with trainable scale (σ) and frequency (f₀) parameters.

---

#### Pain Point 2: Unimodal Blindness

**Problem**: Most methods use EITHER 1D signals OR 2D scalograms, not both.

| Approach | Strength | Weakness |
|----------|----------|----------|
| 1D CNN | Precise R-peak timing | Misses frequency patterns |
| 2D CNN | Rich spectral features | Blurs temporal location |

**Consequence**: The model cannot simultaneously know "when" (1D) and "what frequency" (2D) an event occurred.

**Our Solution**: Cross-Modal Attention Fusion where 1D features query 2D features.

---

#### Pain Point 3: Lack of Latent Structure

**Problem**: Standard cross-entropy loss doesn't enforce any structure in the latent space.

**Consequence**: 
- Clusters overlap in feature space
- Model relies on arbitrary decision boundaries
- Poor interpretability

**Our Solution**: Prototype Consistency Loss with learnable class centroids.

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NSHT Architecture                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw ECG (1, 216)                                                   │
│       │                                                              │
│       ├──────────────────────┐                                      │
│       ↓                      ↓                                      │
│  ┌─────────────┐      ┌─────────────┐                               │
│  │ Learnable   │      │  P-Wave     │                               │
│  │ Wavelet     │      │  Head       │                               │
│  │ (32 filters)│      │ (Low-freq)  │                               │
│  └──────┬──────┘      └──────┬──────┘                               │
│         │                    │                                       │
│         ↓                    │                                       │
│  ┌─────────────┐            │                                       │
│  │ 1D Encoder  │            │                                       │
│  │ (Inception) │            │                                       │
│  └──────┬──────┘            │                                       │
│         │                    │                                       │
│         │    ┌───────────────┼───────────────┐                      │
│         │    │               │               │                      │
│         │    │  Scalogram (3, 128, 128)      │                      │
│         │    │         │                     │                      │
│         │    │         ↓                     │                      │
│         │    │  ┌─────────────┐              │                      │
│         │    │  │ 2D Encoder  │──→ Low-Freq  │                      │
│         │    │  │ (CNN)       │    Features  │                      │
│         │    │  └──────┬──────┘              │                      │
│         │    │         │                     │                      │
│         ↓    ↓         ↓                     ↓                      │
│  ┌───────────────────────────────────────────────┐                  │
│  │         Cross-Modal Attention Fusion          │                  │
│  │    (1D queries 2D for spectral context)       │                  │
│  └───────────────────────┬───────────────────────┘                  │
│                          │                                          │
│                          ↓                                          │
│                   ┌─────────────┐                                   │
│                   │ Global Pool │                                   │
│                   │ + Concat    │                                   │
│                   └──────┬──────┘                                   │
│                          │                                          │
│                          ↓                                          │
│                   ┌─────────────┐                                   │
│                   │ Classifier  │ ←── Prototype Loss (λ)            │
│                   │ (5 classes) │                                   │
│                   └─────────────┘                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Technical Components

### 5.1 Learnable Morlet Wavelet Front-End

**Mathematical Formulation**:

The Morlet wavelet is defined as:
$$\psi(t) = e^{-\frac{t^2}{2\sigma^2}} \cos(2\pi f_0 \cdot \frac{t}{\sigma})$$

Where:
- $\sigma$ = scale (controls envelope width)
- $f_0$ = center frequency

**Innovation**: Both σ and f₀ are **learnable parameters** optimized via backpropagation.

```python
class LearnableMorletWavelet(nn.Module):
    def __init__(self, out_channels=32, kernel_size=31):
        # Initialize with log-spaced scales (warm start)
        self.scale = nn.Parameter(torch.logspace(0, 2, out_channels) * 0.5)
        self.freq = nn.Parameter(torch.linspace(0.5, 10.0, out_channels))
```

**Why This Matters**:
- Adapts to dataset-specific noise profiles
- Learns frequency bands most discriminative for each class
- Replaces hand-tuned wavelet selection

---

### 5.2 P-Wave Specialized Head

**Motivation**: The P-wave (0.5-5 Hz) is critical for N vs. S discrimination but easily lost in noise.

**Implementation**:
- Large kernels (41, 21) to capture slow P-wave morphology
- Separate processing path from main encoder
- Concatenated to final features

```python
class PWaveHead(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=41, padding=20)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=21, padding=10)
```

---

### 5.3 Cross-Modal Attention Fusion

**Concept**: Each 1D temporal position queries all 2D spectral positions to find relevant frequency information.

$$\text{Attention}(Q_{1D}, K_{2D}, V_{2D}) = \text{softmax}\left(\frac{Q_{1D} K_{2D}^T}{\sqrt{d_k}}\right) V_{2D}$$

**Why Not Simple Concatenation?**
- Concatenation treats all features equally
- Attention learns which 2D regions are relevant for each 1D timestep
- More parameter-efficient than fully-connected fusion

---

### 5.4 Prototype Consistency Loss

**Concept**: Each class has a learnable centroid (prototype) in feature space.

$$\mathcal{L}_{proto} = \frac{1}{N}\sum_{i=1}^{N} \|f_i - p_{y_i}\|^2$$

Where:
- $f_i$ = feature embedding of sample i
- $p_{y_i}$ = prototype of class $y_i$

**Dynamic λ Schedule**:
```python
lambda_schedule = np.linspace(0.01, 0.1, epochs)
# Start small (let model learn features first)
# Gradually increase (enforce clustering)
```

---

## 6. Implementation Details

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 8 | Memory constraint (4GB VRAM) |
| Gradient Accumulation | 4 | Effective batch = 32 |
| Learning Rate | 1e-3 | Standard for AdamW |
| Optimizer | AdamW | Weight decay for regularization |
| Scheduler | CosineAnnealing | Smooth LR decay |
| Mixed Precision | FP16 | 2x memory savings |
| Label Smoothing | 0.1 | Prevent overconfidence |

### Memory Optimization

```python
# Mixed precision training
with autocast():
    logits, proto_loss = model(x1d, x2d, targets)
    loss = ce_loss + lambda_proto * proto_loss

# Gradient accumulation
loss = loss / accumulation_steps
scaler.scale(loss).backward()

if (batch_idx + 1) % accumulation_steps == 0:
    scaler.step(optimizer)
```

---

## 7. Experimental Results

### Training Progression

| Epoch | Train Acc | Val Acc | λ |
|-------|-----------|---------|---|
| 1 | 85.90% | 93.40% | 0.010 |
| 10 | 97.65% | 97.86% | 0.027 |
| 20 | 99.17% | 98.71% | 0.045 |
| 30 | 99.71% | 98.95% | 0.063 |
| 40 | 99.94% | 99.08% | 0.082 |
| 50 | 99.99% | 99.13% | 0.100 |

**Best Validation**: 99.18% (Epoch 42)
**Final Test**: 98.80%

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **N** (Normal) | 97.92% | 98.00% | 97.96% | 1,200 |
| **S** (Supraventricular) | 98.34% | 98.75% | 98.54% | 1,200 |
| **V** (Ventricular) | 98.99% | 97.92% | 98.45% | 1,200 |
| **F** (Fusion) | 99.01% | 99.67% | 99.34% | 1,200 |
| **Q** (Unknown/Paced) | 99.75% | 99.67% | 99.71% | 1,200 |

### Confusion Matrix

```
Predicted →     N       S       V       F       Q
Actual ↓
    N        1176     14       5       3       2
    S          14   1185       0       1       0
    V          10      6    1175       8       1
    F           1      0       3    1196       0
    Q           0      0       4       0    1196
```

### Key Observations

1. **N↔S Confusion Reduced**: 14+14=28 errors (vs. 22+10=32 in InceptionTime)
2. **V↔F Confusion**: 8+3=11 errors (still a challenge)
3. **Q Class Nearly Perfect**: 99.67% recall

---

## 8. Ablation Study Design

To validate each component, the following experiments should be run:

### Experiment 1: Without Learnable Wavelets

Replace `LearnableMorletWavelet` with fixed Morlet CWT.

**Expected Result**: ~1% accuracy drop (unable to adapt denoising)

### Experiment 2: Without Cross-Modal Attention

Replace attention fusion with simple concatenation.

**Expected Result**: ~0.5% accuracy drop (loses dynamic feature alignment)

### Experiment 3: Without Prototype Loss

Set λ = 0 (standard cross-entropy only).

**Expected Result**: ~0.3% accuracy drop (weaker class separation)

### Experiment 4: Without P-Wave Head

Remove specialized low-frequency processing.

**Expected Result**: Increased N↔S confusion

---

## 9. Comparison with State-of-the-Art

| Method | Type | Accuracy | Parameters | Trainable Denoising |
|--------|------|----------|------------|---------------------|
| DWTFrTV-DEAL | Wavelet + ML | 99.7% | N/A | ❌ No |
| FG-HSWIN-CFA | Swin Transformer | 98.96% | ~25M | ❌ No |
| InceptionTime Ensemble | 1D CNN | 98.88% | ~8.5M | ❌ No |
| Multi-Meta-Learner | Stacking | 99.17% | ~15M | ❌ No |
| **NSHT (Ours)** | Hybrid Transformer | 98.80% | **0.7M** | ✅ **Yes** |

### Advantages of NSHT

1. **Smallest Model**: Only 706K parameters (10-30x smaller than alternatives)
2. **End-to-End Learnable**: No hand-tuned preprocessing
3. **Interpretable**: Prototype loss creates meaningful clusters
4. **Hardware Efficient**: Runs on 4GB VRAM

### Current Limitations

1. **Accuracy Gap**: 0.37% below Multi-Meta-Learner
2. **Scalogram Overhead**: 2D path adds preprocessing time
3. **Generalization**: Not yet tested on other ECG databases

---

## 10. Future Improvements

### Immediate (Can Improve to 99%+)

| Improvement | Expected Gain | Complexity |
|-------------|---------------|------------|
| Longer training (100 epochs) | +0.2-0.3% | Low |
| Larger embedding (256) | +0.1-0.2% | Medium |
| More attention heads (8) | +0.1% | Low |
| Focal Loss for N class | +0.1-0.2% | Low |

### Medium-Term (Publication Ready)

1. **Inter-Patient Split Validation**: Ensure no data leakage
2. **10-Fold Cross-Validation**: Robust performance estimate
3. **Noise Robustness Test**: Inject SNR 10dB noise
4. **T-SNE Visualization**: Show prototype clustering

### Long-Term (Beyond SOTA)

1. **Swin Transformer Backbone**: Replace 2D encoder
2. **Contrastive Prototype Loss**: Pull same class, push different class
3. **Multi-Task Learning**: Predict both arrhythmia and patient ID
4. **Self-Supervised Pre-training**: Use unlabeled ECG data

---

## 11. Conclusion

### Summary

The **Neuro-Spectral Hybrid Transformer (NSHT)** introduces three novel components:

1. **Learnable Wavelet Front-End**: Replaces fixed denoising with trainable filters
2. **Cross-Modal Attention Fusion**: Combines 1D temporal and 2D spectral features dynamically
3. **Prototype Consistency Loss**: Enforces meaningful latent space structure

### Results

- **98.80% accuracy** on MIT-BIH with only **706K parameters**
- **99.18% validation accuracy** demonstrates learning capacity
- All components are differentiable and end-to-end trainable

### Scientific Contribution

This architecture represents a **paradigm shift** from:
```
Fixed Preprocessing → Feature Extraction → Classification
```
To:
```
Learnable Preprocessing ↔ Feature Extraction ↔ Classification
              ↑                    ↑                 ↑
              All jointly optimized via backpropagation
```

### Publication Potential

With the planned ablation studies and inter-patient validation, NSHT could be submitted to:
- IEEE Transactions on Biomedical Engineering
- Computers in Biology and Medicine
- Medical Image Analysis

---

## 12. Current Repository XAI Workflow (March 2026)

The active NSHT explainability path is implemented in `scripts/explain_paper3.py` for `NSHT_Dual_Evo` checkpoints.

Run:

```bash
python scripts/explain_paper3.py \
    --model-path checkpoints/paper3_nsht/best_model.pt \
    --config configs/paper3_nsht.yaml \
    --num-samples-per-class 1
```

Optional leakage-safe override:

```bash
python scripts/explain_paper3.py \
    --model-path checkpoints/paper3_nsht/best_model.pt \
    --config configs/paper3_nsht.yaml \
    --data.balance_after_split
```

Main artifacts under `experiments/paper3_nsht/xai/`:
- `wavelet_params.png` (learned sigma, omega0, and wavelet scales)
- `cross_attention.png` (cross-modal attention heatmap per sample)
- `stream_contributions.png` (temporal vs spectral stream energy)
- `arrays.npz` plus per-sample and global `summary.json`
- optional `prototype_tsne.png` when prototype mode is enabled

For explicit prototype extraction and t-SNE export, run:

```bash
python scripts/extract_nsht_prototypes.py \
    --model-path checkpoints/paper3_nsht/best_model.pt \
    --config configs/paper3_nsht.yaml
```

---

## Appendix: File Structure

```
ClassficationECG/
├── ecg_nsht_model.py           # Complete NSHT implementation
├── NSHT_ARCHITECTURE.md        # This documentation
├── models/
│   └── nsht_best.pth           # Trained model weights
│   └── nsht_results.npy        # Test results and features
├── balanced_data/
│   ├── X_balanced.npy          # Input signals
│   └── y_balanced.npy          # Labels
```

---

*Document Version: 1.0*
*Date: January 24, 2026*
*Author: ECG Classification Research*
