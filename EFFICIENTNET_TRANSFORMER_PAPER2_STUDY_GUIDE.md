# Paper 2 Technical Monograph: EfficientNet Scalogram Pipeline (Legacy Hybrid Narrative vs Current AttentionEfficientNet Runtime)

## 1. Scope and Positioning

This document is a paper-grade technical specification for the Paper 2 branch of the repository. It explicitly separates:

1. The original research narrative: EfficientNet-B0 plus Transformer over CWT scalogram tokens.
2. The current repository runtime path: AttentionEfficientNet with CBAM-style attention and dedicated explainability outputs.

The goal is to preserve scientific clarity while keeping implementation truth exact and reproducible.

---

## 2. Problem Definition and Notation

Given ECG beat segments centered on R-peaks, define a dataset:

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N},\quad x_i\in\mathbb{R}^{L},\ y_i\in\{0,1,2,3,4\}
$$

where:

- $L=1080$ samples per beat segment.
- Classes $\{0,1,2,3,4\}$ correspond to AAMI types $\{N,S,V,F,Q\}$.

The Paper 2 representation map converts 1D to 2D:

$$
\Phi_{\text{CWT}}: \mathbb{R}^{1080}\rightarrow\mathbb{R}^{H\times W\times 3}    
$$

with $(H,W)=(224,224)$ and 3-channel pseudo-RGB replication.

The classifier maps images to class probabilities:

$$
f_\theta\!:\mathbb{R}^{224\times224\times3}\rightarrow\Delta^{5}
$$

where $\Delta^{5}$ is the probability simplex over 5 classes.

---

## 3. Clinical and Signal-Processing Motivation

Paper 2 investigates whether converting ECG beats to time-frequency images enables transfer learning from vision backbones. The central hypothesis is:

1. CWT improves visibility of arrhythmia-specific frequency signatures.
2. Pretrained image models accelerate learning in low-data classes.
3. Attention modules improve context integration across spectral regions.

This path trades exact temporal localization for richer local frequency-texture cues.

---

## 4. End-to-End Data Path

### 2.1 Preprocessing Phase Architecture (Modular Codebase Reality)

In strict adherence to the repository's src/data/download.py and src/data/dataset.py, the preprocessing pipeline avoids destructive filters (like Butterworth or high-pass denoising) to strictly preserve the inherent morphological fidelity of the QRS complexes. 

#### Native Data Pipeline Flowchart
```mermaid
flowchart TD
    Raw[Read WFDB Records & Annotations] --> Resample[Resample to 360 Hz <br> e.g. from INCART 257 Hz]
    Resample --> ZScore[Continuous Record Z-Score Normalization]
    ZScore --> Window[1080-Sample Extraction <br> Centered on R-Peak]
    Window --> Map[AAMI 5-Class Target Mapping]
    Map --> Split[Stratified Train/Val/Test Split]
    Split --> TrainOnly[Apply ADASYN/SMOTE <br> to Training Split Only]
```

**Architectural Flow of Data Preparation:**

A. **Data Ingestion (wfdb interface):**
   - **Source:** Opens .dat files and .atr annotations from PhysioNet.
   - **Extraction:** Isolates the primary diagnostic channel (typically MLII for MIT-BIH) and corresponding symbolic labels (e.g., 'N', 'V', 'R').

B. **Harmonization & Resampling:**
   - **Temporal Alignment:** Disparate datasets must share a spatial-frequency mapping. The INCART dataset (natively 257 Hz) is fundamentally upsampled to 360 Hz using scipy.signal.resample.
   - **Annotation Scaling:** Annotation indices are linearly scaled (index * (360/257)) to ensure the markers perfectly align with the newly interpolated R-peaks.

C. **Zero-Mean, Unit-Variance Normalization (Z-Score):**
   - Applied **globally per record** *before* segmentation. 
   - $x_{norm} = \frac{x - \mu}{\sigma + 1e^{-8}}$
   - This ensures global amplitude drift is bounded without destroying the localized amplitude variance of individual heartbeats, acting as a natural un-destructive baseline stabilizer.

D. **Multi-Beat Context Windowing (1080-Sample Extraction):**
   - **Center Anchor:** The pipeline iterates over annotated R-peaks.
   - **Extraction Boundary:** It precisely slices [-540, +540] samples relative to the R-peak index.
   - **Why 1080?** At 360 Hz, 1080 samples equate exactly to a **3.0-second temporal window**. This is a paradigm shift from traditional 1-second (360 samples) extraction because 3.0 seconds almost definitively captures overlapping anterior and posterior heartbeats. 
   - **Clinical Justification:** Capturing adjacent beats naturally embeds the **R-R interval** into the dataset, which is the mathematically required distinguishing factor for ambiguous classes like 'S' (Supraventricular) and 'F' (Fusion) that share intra-beat normal morphologies.

E. **AAMI Mapping & Stratified Leakage-Safe Splits:**
   - **Class Aggregation:** Raw annotations map perfectly to {0:'N', 1:'S', 2:'V', 3:'F', 4:'Q'} per the AAMI diagnostic standard.
   - **Training Protocol:** Driven by src/data/dataset.py, the pure unaugmented arrays follow an 80/15/5% Train/Val/Test random stratified split.
   - **Balancing via ADASYN/SMOTE:** Crucially, SMOTE or ADASYN upsampling is applied *exclusively* to the separated training subset. Overwriting happens *only* inside the train matrix, preventing synthesized data representations from leaking and contaminating the Test or Validation structures.

### 4.3 Scalogram Conversion

For each beat $x(t)$:

$$
W(a,b)=\frac{1}{\sqrt{a}}\int_{-\infty}^{\infty}x(t)\,\psi^*\!\left(\frac{t-b}{a}\right)dt
$$

Scalogram magnitude:

$$
S(a,b)=|W(a,b)|
$$

Practical settings:

- Wavelet: Morlet (`morl`).
- Scales: `np.arange(1,65)`.
- Native shape $64\times1080$, then interpolated to $224\times224$ (`img_size`).
- Channel format: 3-channel replication for vision backbones.

---

## 4.3 EfficientNet-B0 vs AttentionEfficientNet Comparison

| Dimension | Base EfficientNet-B0 | AttentionEfficientNet (Current Runtime) |
|-----------|---------------------|----------------------------------------|
| Backbone | MBConv only | MBConv with CBAM attention |
| Channel Attention | SE blocks only | SE + CBAM channel refinement |
| Spatial Attention | None | **CBAM spatial attention maps** |
| Input | 224Ã—224 RGB images | 224Ã—224 CWT scalogram (3-channel replication) |
| Head design | Global pooling â†’ FC | **Global pooling + CBAM â†’ FC** |
| Interpretability | Basic feature maps | **Grad-CAM + spatial/channel attention artifacts** |
| Training stability | Standard AMP | **BF16 mixed precision + TF32** |
| XAI Runtime | Not applicable | `scripts/explain_paper2.py` |
| Implementation | torchvision baseline | `src/models/efficientnet_scalogram.py` |
| Config reference | N/A | `configs/paper2_efficientnet.yaml` |

**Key Novelty:** CBAM attention augmentation replaces the legacy Transformer tokenization path, enabling efficient interpretability through explicit attention maps while maintaining transfer learning benefits.

**Codebase Reference:** See [MODULAR_CODEBASE_README.md](MODULAR_CODEBASE_README.md#project-structure) for runtime details.

---

## 5. Legacy Architecture Narrative (EfficientNet + Transformer)

> [!WARNING]
> This section captures the original conceptual architecture used in previous theoretical writeups limit testing. **The Vision Transformer (ViT) architecture is NO LONGER ACTIVE.** The current state-of-the-art runtime applies the multi-scale, purely convolutional `AttentionEfficientNet` mechanism defined below.

### 5.1 Backbone and Projection

EfficientNet-B0 feature map:

$$
F\in\mathbb{R}^{B\times1280\times7\times7}
$$

Projection:

$$
\tilde{F}=\mathrm{ReLU}(\mathrm{BN}(\mathrm{Conv}_{1\times1}(F)))\in\mathbb{R}^{B\times256\times7\times7}
$$

### 5.2 Tokenization and CLS Injection

Flatten spatial grid to tokens:

$$
T\in\mathbb{R}^{B\times49\times256}
$$

Add CLS token and positional embeddings:

$$
\hat{T}=[\mathrm{CLS};T_1;\dots;T_{49}] + E_{\mathrm{pos}}\in\mathbb{R}^{B\times50\times256}
$$

### 5.3 Transformer Encoder

Attention kernel:

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Encoder depth in prior guide: 4 layers, 8 heads, embedding 256.

### 5.4 Classification Head

Use CLS output with stacked MLP to logits in $\mathbb{R}^{5}$.

---

## 6. Current Runtime Architecture (AttentionEfficientNet + CBAM)

The repository now executes a multi-scale attention-augmented CNN path for Paper 2 training, evaluation, and XAI.

### 6.1 High-Level Computation

#### PyTorch Forward Pass Implementation
```python
# Pseudo-code Implementation: AttentionEfficientNet Forward Pass
def forward(self, scalogram):
    # scalogram shape: [Batch, 3, 224, 224] (CWT magnitude)
    
    # 1. Backbone Feature Hierarchy (Multi-scale Extract)
    f2, f3, f4 = self.efficientnet_backbone.extract_endpoints(scalogram)
    
    # 2. Independent CBAM Attention per Scale
    f2_attn = self.cbam_scale2(f2)
    f3_attn = self.cbam_scale3(f3)
    f4_attn = self.cbam_scale4(f4)
    
    # 3. Spatial Alignment (Upsampling to F2 resolution)
    f3_up = F.interpolate(f3_attn, size=f2_attn.shape[-2:], mode='bilinear')
    f4_up = F.interpolate(f4_attn, size=f2_attn.shape[-2:], mode='bilinear')
    
    # 4. Multi-Scale Fusion
    fused_features = torch.cat([f2_attn, f3_up, f4_up], dim=1)
    fused_features = F.relu(self.fusion_bn(self.fusion_conv(fused_features)))
    
    # 5. Global CBAM Refinement
    fused_attn = self.global_cbam(fused_features)
    
    # 6. Global Pooling & Classification
    gap = torch.mean(fused_attn, dim=[-2, -1])
    gmp = torch.max(fused_attn.view(fused_attn.size(0), fused_attn.size(1), -1), dim=-1)[0]
    
    features = torch.cat([gap, gmp], dim=1)
    features = self.dropout(features)
    
    logits = self.classifier(features)
    return logits
```

The backbone extracts pyramidal features by tapping hierarchical outputs at block indices (e.g., scales 2, 3, and 4 in EfficientNet-B0), generating synchronized multi-scale context mappings. 

$$
\hat{y}=g_\phi\!\left(\mathrm{FusionCBAM}\left( \text{Concat}\Big|_{i=2}^{4} \Big[ \mathrm{Upsample}\big( \mathrm{CBAM}_{i}\big(\mathrm{EffNet}_{i}(\Phi_{\mathrm{CWT}}(x))\big)\big) \Big] \right)\right)
$$

Where:

- $\mathrm{EffNet}_{i}$ represents the hierarchical CNN output at scale $i$.
- $\mathrm{CBAM}_{i}$ explicitly refines each scale output independently.
- $g_\phi$ is the final global-pooled dense classifier head.

### 6.2 Channel Attention

For feature tensor $U\in\mathbb{R}^{C\times H\times W}$:

$$
M_c(U)=\sigma\left(\mathrm{MLP}(\mathrm{GAP}(U))+\mathrm{MLP}(\mathrm{GMP}(U))\right)
$$

$$
U'=M_c(U)\odot U
$$

### 6.3 Spatial Attention

$$
M_s(U')=\sigma\left(f^{7\times7}([\mathrm{AvgPool}_c(U');\mathrm{MaxPool}_c(U')])\right)
$$

$$
U''=M_s(U')\odot U'
$$

This produces interpretable attention maps aligned with Paper 2 XAI artifacts.

---

### 6.4 Model Definition Pseudo-Code

The AttentionEfficientNet applies dual CBAM attention over multi-scale visual patches.

```python
# Pseudo-code Implementation: AttentionEfficientNet Architecture
class CBAM_Module(nn.Module):
    def forward(self, x):
        # Channel Attention isolates relevant frequency bands
        x_out = self.channel_attention(x) * x
        # Spatial Attention focuses on chronologically and texturally critical regions
        x_out = self.spatial_attention(x_out) * x_out
        return x_out

class AttentionEfficientNet(nn.Module):
    def forward(self, x_scalogram):
        # 1. Backbone Feature Hierarchy Extract (from blocks C2, C3, C4)
        f2, f3, f4 = self.efficientnet_backbone.extract_multiscale(x_scalogram)
        
        # 2. Independent CBAM Refinements (Scale-specific focus points)
        f2 = self.cbam2(f2)
        f3 = self.cbam3(f3)
        f4 = self.cbam4(f4)
        
        # 3. Multi-Scale Spatial Alignment & Fusion
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode='bilinear')
        f4_up = F.interpolate(f4, size=f2.shape[2:], mode='bilinear')
        
        # Dense convolutional fusion across the synchronized multi-scale stacks
        f_concat = torch.cat([f2, f3_up, f4_up], dim=1)
        f_fusion = self.fusion_conv(f_concat)
        f_final = self.fusion_cbam(f_fusion)
        
        # 4. Dense Classifier Output
        h_pool = torch.cat([self.gap(f_final), self.gmp(f_final)], dim=1)
        return self.classifier(h_pool)
```

## 7. Shape Trace and Interface Contracts

Representative tensor flow (batch size $B$):

1. Input scalogram: $B\times3\times224\times224$.
2. Backbone multi-scale blocks: 3 parallel downsampled feature maps $F_2, F_3, F_4$.
3. Per-scale attention: CBAM applied to each $F_i$ independently.
4. Scale fusion: $F_3, F_4$ bilinearly upsampled to matches dim of $F_2$, then concatenated.
5. Fused Attention: Processed through 512-channel Fusion CBAM.
6. Global pooling: $B\times 1024$ (GAP and GMP concatenated).
7. Classifier output: $B\times5$.

Contract guarantees:

- Input normalization policy must match training.
- Class index mapping must remain consistent across train/eval/XAI.
- Checkpoint architecture and config pairing must match exactly.

---

## 8. Objective Function and Optimization

### 8.1 Loss

Primary objective:

$$
\mathcal{L}_{\mathrm{CE}}=-\sum_{c=1}^{C}\tilde{y}_c\log\hat{y}_c
$$

Label smoothing form:

$$
\tilde{y}_c=(1-\alpha)y_c+\frac{\alpha}{C},\quad \alpha=0.1
$$

### 8.2 Softmax

$$
\hat{y}_c=\frac{e^{z_c}}{\sum_{k=1}^{C}e^{z_k}}
$$

### 8.3 Optimizer and Runtime

Typical Paper 2 path in this repository:
---

## 8A. Overfitting Prevention Strategy for Vision Models

Paper 2 applies specialized regularization suited to 2D image-based ECG representations (scalograms):

### 8A.1. Early Stopping by Validation Accuracy

Similar to Paper 1, early stopping monitors validation **accuracy** rather than loss alone:

```python
trainer.train(
    train_loader, val_loader,
    epochs=200,
    monitor='val_acc'  # Checkpoint at best val_acc epoch, not best val_loss
)
```

**Configuration:**
```yaml
training:
  monitor: val_acc  # Default for classification tasks
```

### 8A.2. Label Smoothing for Vision Models

Label smoothing is applied to the classification loss, softening one-hot targets:

$$
\tilde{y}_c=(1-\alpha)y_c+\frac{\alpha}{C}
$$

For Paper 2 (pretrained EfficientNet), label smoothing is typically **lower** ($\alpha=0.03$) than Paper 1, as the pretrained backbone already provides strong feature regularization:

**Configuration:**
```yaml
training:
  label_smoothing: 0.03  # Lower for vision, higher for 1D models
```

### 8A.3. Vision-Specific Augmentation Considerations

Paper 2 uses **CWT scalogram images** as input. Spatial augmentation (random crops, rotations) may distort time-frequency semantics, so **augmentation is typically disabled** for the 1D-derived 2D representation:

**Configuration:**
```yaml
training:
  augmentation_prob: 0.0  # Disabled for scalogram; frequency distortion harmful
```

If augmentation is desired, use conservative vision transforms (mild scale jitter only), not aggressive spatial transforms.

### 8A.4. Transfer Learning and Early Stopping

With a pretrained EfficientNet backbone, early stopping may trigger earlier (e.g., epoch 20â€“40 vs. Paper 1's 60â€“80), as the model quickly adapts learned visual features to ECG spectrograms:

**Recommendation:**
  - Use early stopping patience with LR reduction (reduce LR 3â€“4 times before stop).
  - Monitor validation accuracy peak to avoid stopping too early.
  - Allow at least 30â€“50 epochs for fine-tuning fine_tune phase stabilization.

### 8A.5. ADASYN + Class Weights Interaction

As in Paper 1, when using ADASYN oversampling, **disable class weights**:

```yaml
data:
  balancing_method: adasyn
training:
  use_class_weights: false  # ADASYN already boosts minorities
```

This prevents double-reweighting that can bias model toward minority classes.

### 8A.6. K-Fold Hyperparameter Passthrough

Ensure learning rate and weight decay are **explicitly passed** to K-fold trainer:

```python
kfold_trainer = KFoldTrainer(
    n_splits=10,
    lr=config.training.lr,  # Critical: explicitly pass configured LR
    weight_decay=config.training.weight_decay,
    label_smoothing=config.training.label_smoothing,
    use_class_weights=config.training.use_class_weights,
    early_stopping_monitor=config.training.monitor,
    # ... other parameters
)
```

**Historical Issue:** K-fold training ignored config hyperparameters, using hardcoded defaults. **Now fixed.**

### 8A.7. Attention Map Regularization via CBAM

CBAM (channel + spatial attention) modules in Paper 2 provide **implicit regularization**:

  - **Channel attention** suppresses noisy feature maps.
  - **Spatial attention** focuses on arrhythmia-relevant scalogram regions.

Together, attention mechanisms act as learned feature selectors, reducing overfitting risk without explicit regularization hyperparameters.

### 8A.8. Key Overfitting Signatures for Vision Models

| Signature | Cause | Remedy |
|-----------|-------|--------|
| Train loss â†’ 0.002, Val loss â†’ 0.5+ | Memorization | â†‘ label_smoothing to 0.05, use dropout in classifier |
| Val acc peaks epoch 35, val loss best epoch 50 | Checkpoint metric mismatch | Set `monitor: val_acc` |
| Fold-to-fold variance > 3% | Hyperparameter leakage | Verify lr/wd passthrough in K-fold |
| Poor minority F1-scores | ADASYN + class weights | Set `use_class_weights: false` |
| Noisy attention maps | Insufficient CBAM regularization | OK; attention naturally focuses; not a failure mode |

---
- AdamW.
- AMP (BF16 preferred where supported).
- LR reduction and early stopping.
- Non-blocking transfer and container-safe execution settings.

---

## 9. Complexity and Resource Analysis

### 9.1 CWT Overhead

CWT dominates preprocessing cost compared to direct 1D pipelines. For beat count $N$ and scale count $S$:

$$
\text{Cost}_{\mathrm{CWT}}\propto N\cdot S\cdot L
$$

This is a major throughput driver.

### 9.2 Model-Side Cost

Attention-enhanced CNN remains cheaper than full token-level global self-attention over high-resolution patches, while retaining spatial interpretability.

### 9.3 Practical Memory Notes

- Large batch sizes are constrained by scalogram tensors.
- Worker count should be tuned conservatively in containerized workflows.
- AMP is important for throughput and stability on limited VRAM.

---

## 10. Novelty Map

Novel contributions and where they appear:

1. Time-frequency representation shift: CWT scalogram conversion from 1D ECG.
2. Attention-augmented vision backbone: CBAM refinements in runtime path.
3. Leakage-safe balancing policy: split-first train-only synthesis.
4. Structured explainability suite: Grad-CAM + attention diagnostics bundle.

These should be explicitly called out in any manuscript section discussing methods.

---

## 11. Baseline vs Current Runtime Table

| Dimension | Legacy Paper 2 Narrative | Current Repository Runtime |
|---|---|---|
| Core design | EfficientNet + Transformer | AttentionEfficientNet + CBAM |
| Global relation model | Transformer self-attention | Attention-enhanced CNN hierarchy |
| Tokenization | Explicit 49+CLS tokens | Not used in active path |
| Balancing policy | Generic narrative | Split-first supported and documented |
| Explainability focus | General hybrid explanations | Grad-CAM + CBAM spatial/channel maps |
| Deployment profile | More sequence-oriented | Practical CNN-attention inference path |

---

## 12. Training and Evaluation Protocol

### 12.1 Data Split Policy

Recommended default for paper-grade reproducibility:

1. Fix seed and split indices.
2. Apply balancing only to train split when enabled.
3. Keep validation/test untouched.

### 12.2 Checkpointing

Use validation objective for checkpoint selection, never test objective.

### 12.3 Metrics

Report:

- Accuracy.
- Macro precision/recall/F1.
- Per-class metrics and confusion matrix.
- Optional calibration diagnostics.

---

## 13. Multi-scale Explainable AI (XAI) Protocol (Paper 2)

Paper 2 bridges computer vision techniques with time-series frequency diagnostics through two primary attribution modes applied directly to the $224 \times 224$ CWT Scalograms.

### 13.1. 2D Grad-CAM (Targeting Fusion Block)

To determine which time-frequency regions drove the global classification, 2D Grad-CAM is attached to the final `fusion` block preceding global pooling.

Let $A^k \in \mathbb{R}^{H \times W}$ be the $k$-th feature map in the final 512-channel fusion tensor, and $y^c$ the raw logit score for class $c$.

**Gradient Computation and Global Average Pooling:**
We compute the mean channel gradients $\alpha_k^c$ which dictate the clinical importance of the $k$-th frequency filter:

$$
\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\partial y^c}{\partial A_{i, j}^k}
$$
where $Z = H \times W$ represents spatial area.

**Spatial Activation Map:**
The final localization map $L_{Grad-CAM}^c$ is a ReLU-activated linear combination:

$$
L_{Grad-CAM}^c = \mathrm{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)
$$

This $L_{Grad-CAM}^c$ is bilinearly upsampled from its latent size (e.g., $14 \times 14$) to $224 \times 224$ and overlaid onto the optical CWT scalogram.

```python
# Pseudo-code Implementation: 2D Grad-CAM
def compute_gradcam_2d(model, scalogram_tensor, target_class):
    activations, gradients = [], []

    # Bind hooks to the fusion convolution block
    def fwd_hook(m, i, o): activations.append(o)
    def bwd_hook(m, i, o): gradients.append(o[0])

    h1 = model.fusion.register_forward_hook(fwd_hook)
    h2 = model.fusion.register_full_backward_hook(bwd_hook)

    # Trigger gradient flow
    logits = model(scalogram_tensor)
    logits[:, target_class].sum().backward()

    # Calculate alpha weights
    A_k = activations[0]
    grad_A = gradients[0]
    alpha_k = grad_A.mean(dim=(2, 3), keepdim=True)

    # ReLU combination
    cam = torch.relu((alpha_k * A_k).sum(dim=1)).squeeze(0)

    h1.remove()
    h2.remove()

    # Interpolate back to 224x224
    cam_up = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear').squeeze()
    return normalize_2d(cam_up.detach().cpu().numpy())
```

### 13.2. CBAM Spatial and Channel Attention Extraction

Unlike post-hoc Grad-CAM, the CBAM modules inherently compute attention natively. We extract these states during a specialized forward pass `forward_with_attention_maps`.

**Spatial Maps Extracted:**
For each backbone scale $S \in \{1, 2, 3\}$, the spatial attention map $M_s(U') \in [0,1]^{H_S \times W_S}$ highlights physical time-frequency bounds:

$$
M_{s, \text{extracted}} = \sigma\left(f^{7\times 7}\big([\mathrm{AvgPool_c}(U'); \mathrm{MaxPool_c}(U')]\big)\right)
$$

**Channel Maps Extracted:**
The channel attention $M_c(U) \in [0,1]^{C}$ isolates octave-filters:

$$
M_{c, \text{extracted}} = \sigma\left(\mathrm{MLP}(\mathrm{GAP}(U)) + \mathrm{MLP}(\mathrm{GMP}(U))\right)
$$

```python
# Pseudo-code Implementation: Intercepting Native CBAM States
def extract_cbam_maps(model, scalogram_tensor):
    model.eval()
    with torch.no_grad():
        # Specialized forward signature that yields intermediate bounds
        _, _, scale_maps, fusion_maps = model.forward_with_attention_maps(scalogram_tensor)

    # scale_maps contains 3 dictionaries of attention states for EfficientNet hooks
    spatial_heatmaps = [squeeze_to_2d(m['spatial']) for m in scale_maps]
    channel_vectors = [squeeze_to_1d(m['channel']) for m in scale_maps]
    
    fusion_spatial = squeeze_to_2d(fusion_maps['spatial'])
    fusion_channel = squeeze_to_1d(fusion_maps['channel'])
    
    return spatial_heatmaps, channel_vectors, fusion_spatial, fusion_channel
```

### 13.3. XAI Sub-System Architecture Flow

```mermaid
flowchart TD
    Raw[Input Scalogram B x 3 x 224 x 224] --> Model[Trained AttentionEfficientNet Checkpoint]
    
    subgraph Core XAI Pipeline
        Model --> FusionBlock[Capture Final Fusion Block Activations]
        Model --> ClassHead[Target Pathologic Class Score]
        ClassHead -->|Backward Pass| Grads[Compute Gradients]
        FusionBlock --> Cross[Global Multiply & Pool]
        Grads --> Cross
        Cross --> ScaledMask[2D Grad-CAM Mask B x 1 x 14 x 14]
        ScaledMask -->|Bilinear Upsample| CAM[2D Grad-CAM 224 x 224]
    end

    subgraph Native CBAM Extraction
        Model --> CBAM[Intercept Native Attention States]
        CBAM --> Spatial[Spatial Maps: 3 Scales + Fusion]
        CBAM --> Channel[Channel Maps: Top-32 Filters]
    end
    
    CAM --> Overlay(Attribution Overlay on Raw Optical Space)
    Spatial --> Output[Clinical XAI Output Artifacts: png, npz, json]
    Channel --> Output
    Overlay --> Output
```

### 13.4. Execution Command

Active evaluation script:

```bash
python scripts/explain_paper2.py \
  --model-path checkpoints/paper2_efficientnet/best_model.pt \
  --config configs/paper2_efficientnet.yaml \
  --num-samples-per-class 1
```

*(Note: Use `--data.balance_after_split` if required by the data ingestion protocol).*

Generated clinical artifacts under `experiments/paper2_efficientnet/xai/`:    
- `scalogram_gradcam.png`: Shows the raw CWT visually blended with the highly active Grad-CAM regions.
- `cbam_spatial_maps.png`: 4x grid comparing multi-scale receptive attention vs the final fusion spatial attention.
- `cbam_channel_attention.png`: Bar plots showing the Top-32 neural filters activated globally across the entire beat.
- `arrays.npz` and `summary.json`.

---

## 14. Failure Modes and Diagnostics

### 14.1 Known Error Modes

1. N/S confusion from subtle morphology overlap.
2. V/F confusion where ventricular morphology mixes.
3. Domain shift across acquisition conditions.

### 14.2 Debug Checklist

1. Verify config-checkpoint pair consistency.
2. Confirm class mapping identical across scripts.
3. Inspect attention maps for degenerate concentration.
4. Re-check split and balancing policy when metrics look suspiciously high.

---

## 15. Ablation Program for Publication

Minimum ablation matrix:

1. Remove CBAM.
2. Freeze vs unfreeze backbone blocks.
3. With/without split-first balancing.
4. CWT scale count sensitivity.
5. Augmentation policy sensitivity.

Report mean and variance across repeated seeds.

---

## 16. Reproducibility Checklist

1. Pin package versions and CUDA/cuDNN details.
2. Store split indices and random seeds.
3. Save exact config used for each run.
4. Archive checkpoint hashes.
5. Archive XAI outputs tied to checkpoint ID.

This ensures any metric can be traced to exact code and data state.

---

## 17. Architecture and Flow Diagrams

### 17.1 Runtime-Correct Architecture Flow

```mermaid
flowchart TD
  A[1D ECG Signal] --> B[CWT Scalogram Layer]
  B --> C[ImageNet Normalized 3x224x224 Image]
  
  C --> D[EfficientNet Backbone\nFeature Extractor]
  
  D -->|Scale 1| E1[CBAM Module 1]
  D -->|Scale 2| E2[CBAM Module 2]
  D -->|Scale 3| E3[CBAM Module 3]
  
  E1 --> F[Bilinear Upsampling & Concatenation]
  E2 --> F
  E3 --> F
  
  F --> G[Fusion Block\nConv2D + BatchNorm + ReLU]
  G --> H[Fusion CBAM Module]
  
  H --> I[Global Pooling\nGAP + GMP]
  I --> J[Classifier Head\nLinear Layers + Dropout]
  
  J --> K[5-Class Logits]
```

### 17.2 Data and Methodology Flow

```mermaid
flowchart TD
  A[Raw Segments] --> B{balance_after_split}
  B -->|true| C[Split Train Val Test First]
  B -->|false| D[Load Prebalanced Arrays]
  C --> E[Apply SMOTE or ADASYN on Train Only]
  E --> F[Build DataLoaders]
  D --> F
  F --> G[Train Paper2 Model]
  G --> H[Validate and Select Best]
  H --> I[Test Evaluation]
  I --> J[Run explain_paper2.py]
```

---

## 18. Manuscript-Ready Claim Boundaries

To avoid over-claiming:

1. If reporting Transformer behavior, clarify it as legacy design narrative unless active code path is restored.
2. If reporting current results from repository checkpoints, attribute them to AttentionEfficientNet + CBAM runtime.
3. Distinguish architectural innovation from methodology innovation (split-first balancing).

---

## 19. Equation Rendering Compatibility

For robust markdown previews, keep one expression per display block and prefer explicit LaTeX operators:

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

$$
\mathcal{L}_{\mathrm{CE}}=-\sum_{c=1}^{C}\tilde{y}_c\log\hat{y}_c
$$

$$
\hat{y}_c=\frac{e^{z_c}}{\sum_{k=1}^{C}e^{z_k}}
$$

---

## 20. Citation Pointers

Useful references for Paper 2 writeups:

- EfficientNet: Tan and Le, ICML 2019.
- CBAM: Woo et al., ECCV 2018.
- Vision Transformer tokenization concepts: Dosovitskiy et al., ICLR 2021.
- ECG benchmark framing: MIT-BIH and INCART literature.

This monograph should be treated as the authoritative deep guide for Paper 2 in this repository.








