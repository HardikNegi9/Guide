# Navigating the Morphological Margin: Why Classical Regularization Fails in High-Fidelity ECG Prototype Learning

## 1. Abstract
The AAMI standard for ECG classification presents a unique challenge due to extreme morphological variance between Normal (N) and Premature Ventricular Contraction (V) beats. This paper introduces the Neural Spectral-Hybrid Transformer (NSHT), a dual-stream architecture combining 1D temporal convolutions with a 2D SpecAugment Vision Transformer. Utilizing an Evolutionary Cross-Attention Fusion mechanism and Prototype Consistency Loss, the base model achieves a state-of-the-art 97.97% macro F1-score on the MIT-BIH dataset. However, an ablation study reveals a critical failure mode: introducing Focal Loss, designed specifically to address hard edge-cases like N vs V, fundamentally destabilizes the Euclidean margin of the prototype latent space, degrading accuracy to 97.61%.

## 2. Methodology & Findings
### The Architecture: NSHT
The core architecture processes a 1D ECG signal alongside its Continuous Wavelet Transform (CWT) scalogram. The EvolutionaryFusion layer mathematically aligns spectral frequency bins with their corresponding temporal sequence utilizing a 4-head Multihead Attention gate.

### The Focal Loss Degradation
While Cross-Entropy ({CE}$) scaled evenly, allowing the prototype_margin=2.0 constraint to establish an orthogonal latent geometry, Focal Loss dynamically scaled gradients down for high-confidence classes ('F' and 'Q'). 
- **Result 1 (Cross Entropy):** N $\rightarrow$ V confusions: 315
- **Result 2 (Focal Loss):** N $\rightarrow$ V confusions: 403 (+28% error rate)

The geometric analysis proves that downweighting the easy classes removed the anchor points necessary for the Prototype Loss to maintain strict boundaries, causing the Ventricular cluster to bleed back into the Normal cluster boundary.

## 3. Visualizing the Collapse (XAI)
To prove the failure of Focal Loss geometrically, we extract the latent representations and heatmaps using T-SNE and Grad-CAM.
