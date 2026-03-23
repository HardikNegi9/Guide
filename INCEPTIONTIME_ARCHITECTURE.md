# InceptionTime Architecture (Paper 1)

## 1. Overview
This document describes the current Paper 1 architecture used in the repository, including model topology, tensor shapes, runtime training behavior, and explainability workflow.

The primary model family is ContextAwareInceptionTime, a 1D multi-branch convolutional architecture designed for ECG beat classification.

---

## 2. High-Level Architecture

```mermaid
flowchart TD
    A[Input ECG Segment Bx1xL] --> B[Stem Conv/Norm/Activation]
    B --> C[Inception Block 1]
    C --> D[Inception Block 2]
    D --> E[Inception Block 3]
    E --> F[Residual Add 1]
    F --> G[Inception Block 4]
    G --> H[Inception Block 5]
    H --> I[Inception Block 6]
    I --> J[Residual Add 2]
    J --> K[Global Average Pooling]
    K --> L[Dropout]
    L --> M[Linear Classifier]
    M --> N[Softmax 5 Classes]

    subgraph Inception Block
        B1[1x1 Bottleneck]
        B2[Conv k=11]
        B3[Conv k=21]
        B4[Conv k=41]
        B5[MaxPool + 1x1 Conv]
        B2 --> B6[Concat]
        B3 --> B6
        B4 --> B6
        B5 --> B6
        B6 --> B7[BatchNorm + GELU/ReLU]
    end
```

---

## 3. Tensor Shape Trace (Typical)

| Stage | Shape |
|------|------|
| Input | B x 1 x 216 |
| Stem | B x C1 x 216 |
| Inception stack output | B x C2 x T |
| Global average pooling | B x C2 |
| Classifier logits | B x 5 |

Notes:
- Exact channel counts depend on config variant.
- Temporal length T is reduced by stride/pooling choices in the implementation.

---

## 4. Why InceptionTime Works for ECG

1. Multi-scale kernels capture morphology at different durations.
2. Residual links stabilize deeper 1D stacks.
3. Global pooling reduces overfitting to absolute beat position.
4. Lightweight 1D path enables high-throughput training.

---

## 5. Training Pipeline (Current Runtime)

```mermaid
flowchart TD
    A[Raw R-peak Segments] --> B{balance_after_split?}
    B -->|true| C[Split train/val/test first]
    B -->|false| D[Load pre-balanced npy files]
    C --> E[Apply SMOTE/ADASYN only on train split]
    E --> F[Build DataLoaders]
    D --> F
    F --> G[Train ContextAwareInceptionTime]
    G --> H[Validate and Early Stop]
    H --> I[Save best checkpoint]
    I --> J[Evaluate on untouched test split]
```

Runtime behavior in this repository:
- BF16 mixed precision and TF32 are enabled on supported GPUs.
- Data loading is tuned for container-safe execution.
- Optional split-first balancing avoids train/test leakage.

---

## 6. Loss, Optimization, and Metrics

- Primary loss: cross-entropy.
- Optimizer: AdamW in current configs.
- Typical controls: scheduler, early stopping, gradient clipping (if enabled), and compile acceleration.
- Core report metrics: accuracy, macro precision/recall/F1, per-class metrics, confusion matrix.

---

## 7. Explainability Pipeline

Paper 1 XAI is generated with script-level explainability and does not require retraining.

Command:

```bash
python scripts/explain_paper1.py \
    --model-path checkpoints/paper1_inceptiontime/best_model.pt \
    --config configs/paper1_inceptiontime.yaml \
    --num-samples-per-class 1
```

Optional leakage-safe loading override:

```bash
--data.balance_after_split
```

Artifacts in experiments/paper1_inceptiontime/xai/:
- signal_attributions.png
- branch_summary.png
- attributions.npz
- branch_summary.json
- per-sample summary.json and global summary.json

---

## 8. Practical Failure Modes and Mitigations

| Failure Mode | Symptom | Mitigation |
|---|---|---|
| Dataset mismatch | missing balanced file errors | use split-first mode or generate required files |
| Overfitting | large train-test gap | increase regularization, monitor early stopping |
| Class instability | weak minority-class recall | tune balancing method and class-level diagnostics |
| Runtime stalls | workers hanging in containers | keep conservative num_workers and spawn-safe loader path |

---

## 9. Architecture Blocks Explained

Architecture diagram blocks:
1. Input ECG Segment: accepts one beat window per sample with shape B x 1 x L.
2. Stem Conv/Norm/Activation: builds a richer initial channel representation from the raw waveform.
3. Inception Blocks 1-6: extract features at multiple temporal scales in parallel.
4. 1x1 Bottleneck: compresses channels before wide kernels to reduce compute.
5. Conv k=11, k=21, k=41 branches: detect short, medium, and long temporal motifs.
6. MaxPool + 1x1 branch: adds a pooled context path and keeps branch diversity.
7. Concat + BatchNorm + GELU/ReLU: merges branch outputs and stabilizes activations.
8. Residual Add 1 and 2: improve gradient flow in deeper stacks.
9. Global Average Pooling: converts temporal feature maps into a fixed-length vector.
10. Dropout: regularizes the classifier input.
11. Linear Classifier: maps features to 5 class logits.
12. Softmax: converts logits to class probabilities.

## 10. Training Flowchart Blocks Explained

Training pipeline blocks:
1. Raw R-peak Segments: beat-level source input.
2. balance_after_split gate: chooses leakage-safe split-first or legacy pre-balanced path.
3. Split train/val/test first: creates clean evaluation splits.
4. Load pre-balanced npy files: legacy path that expects prepared arrays.
5. Apply SMOTE/ADASYN on train split: balances only training data in split-first mode.
6. Build DataLoaders: packages tensors for iteration.
7. Train ContextAwareInceptionTime: optimization loop across epochs.
8. Validate and Early Stop: monitors generalization and stops when improvement stalls.
9. Save best checkpoint: persists the best validation model.
10. Evaluate on test split: computes final held-out metrics.

## 11. Equation Rendering Compatibility

For Markdown preview compatibility, keep equations in multiline display blocks:

$$
\hat{y} = \mathrm{Softmax}(Wz + b)
$$

$$
\mathcal{L}_{\mathrm{CE}} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

Use `\times` in math instead of Unicode multiplication symbols where possible.

---

## 12. Reference
- Fawaz et al., InceptionTime: Finding AlexNet for Time Series Classification.
- Repository config: configs/paper1_inceptiontime.yaml
