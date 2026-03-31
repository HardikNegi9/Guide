# The Contextual Margin: From Multi-Beat Windows to Geometric Regularization Failure in High-Fidelity ECG Classification

## 1. Abstract
The AAMI standard for ECG classification presents a unique challenge due to extreme morphological variance between Normal (N) and Premature Ventricular Contraction (V) beats, alongside the historical difficulty of separating Supraventricular (S) and Fusion (F) beats. This paper presents two major findings. First, shifting from a standard 1-second segment (360 samples) to a 3-second multi-beat window (1080 samples) allows 1D Transformer and Inception models to capture the **R-R interval**, achieving a new State-of-the-Art **99.45% accuracy** and 96.39% macro F1-score on the MIT-BIH dataset. Second, we introduce the Neural Spectral-Hybrid Transformer (NSHT) to study advanced regularization, revealing a critical geometry failure mode: introducing Focal Loss, designed specifically to address hard edge-cases, fundamentally destabilizes the Euclidean margin of our prototype latent space.

## 2. The Context-Aware Temporal Breakthrough
By expanding the temporal extraction window to 1080 samples (3 seconds at 360 Hz), the \ContextAwareInceptionTime\ architecture implicitly learns multi-beat context. 
The inclusion of preceding and subsequent R-peaks provides the necessary morphological markers to dynamically compute the R-R interval. This alone pushed performance out of the 97% range into the 99% accuracy realm:
- **Supraventricular Ectopic (S):** F1 surged to 0.9458.
- **Fusion (F):** F1 surged to 0.8932 (Historically the most difficult class).

## 3. Methodology & Morphological Margins
### The Architecture: NSHT
The core architecture processes a 1D ECG signal alongside its Continuous Wavelet Transform (CWT) scalogram. The EvolutionaryFusion layer mathematically aligns spectral frequency bins with their corresponding temporal sequence utilizing a 4-head Multihead Attention gate.

### The Focal Loss Degradation
While Cross-Entropy (CE) scaled evenly, allowing the prototype_margin=2.0 constraint to establish an orthogonal latent geometry, Focal Loss dynamically scaled gradients down for high-confidence classes ('F' and 'Q'). 
- **Result 1 (Cross Entropy):** N -> V confusions: 315
- **Result 2 (Focal Loss):** N -> V confusions: 403 (+28% error rate)

The geometric analysis proves that downweighting the easy classes removed the anchor points necessary for the Prototype Loss to maintain strict boundaries, causing the Ventricular cluster to bleed back into the Normal cluster boundary.     

## 4. Visualizing the Collapse (XAI)
To prove the failure of Focal Loss geometrically, we extract the latent representations and heatmaps using T-SNE and Grad-CAM.
