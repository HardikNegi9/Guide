# ECG Arrhythmia Classification - Modular Research Framework

A PyTorch-based modular codebase for ECG arrhythmia classification research, supporting three research papers plus an ensemble approach:

1. **Paper 1: InceptionTime** - Context-Aware InceptionTime for 1D ECG signal classification
2. **Paper 2: EfficientNet + Scalogram** - 2D vision approach using CWT scalograms with CBAM attention
3. **Paper 3: NSHT Dual-Evolution** - Neural Spectral-Hybrid Transformer with learnable wavelets
4. **Dual Ensemble** - Signal + Vision fusion combining InceptionTime and EfficientNet experts

## Features

- ✅ **Pre-Balanced Data**: Uses SMOTE/ADASYN pre-balanced datasets for consistent class distribution
- ✅ **Leakage-Safe Option**: Supports split-first workflow (RAW split, then SMOTE/ADASYN on train only)
- ✅ **Multi-Database Support**: MIT-BIH, INCART, and combined datasets
- ✅ **K-Fold Cross-Validation**: Stratified splits for robust evaluation
- ✅ **Mixed Precision Training**: AMP for faster training
- ✅ **torch.compile()**: Speedup for supported models
- ✅ **Paper-Ready Metrics**: Per-class precision, recall, specificity, F1, AUC-ROC
- ✅ **Comprehensive Visualizations**: Confusion matrices, ROC curves, learning curves
- ✅ **Prototype-Aware NSHT Training**: Optional learnable prototypes and prototype consistency loss for Paper 3
- ✅ **Experiment Tracking**: Automatic logging, checkpointing, and results saving

## Documentation Hub

Use these guides as the primary architecture and workflow references:

- `INCEPTIONTIME_ARCHITECTURE.md` - Paper 1 architecture, runtime pipeline, and XAI overview
- `INCEPTIONTIME_PAPER1_STUDY_GUIDE.md` - Paper 1 deep study guide (equations, flowcharts, runtime notes)
- `EFFICIENTNET_TRANSFORMER_PAPER2_STUDY_GUIDE.md` - Paper 2 deep study guide (legacy design + current runtime/XAI path)
- `NSHT_ARCHITECTURE.md` - Paper 3 architecture specification with detailed fusion/runtime diagrams
- `NSHT_PAPER3_STUDY_GUIDE.md` - Paper 3 deep study guide (components, losses, ablations, XAI)
- `NSHT_ULTIMATE_META_ARCHITECTURE.md` - Ultimate meta-learner system architecture and feature-fusion pipeline

Each guide now includes:
- end-to-end Mermaid flowcharts
- current split-first balancing behavior
- script-level XAI commands and artifact maps
- operational checklists and troubleshooting notes

## Project Structure

```
ClassficationECG/
├── src/                          # Main source package
│   ├── data/                     # Data loading and preprocessing
│   │   ├── dataset.py           # ECGDataModule with balancing handling (SMOTE/ADASYN)
│   │   └── download.py          # Download, create RAW, and balance datasets
│   ├── models/                   # Model architectures
│   │   ├── inception_time.py    # Paper 1: Context-Aware InceptionTime
│   │   ├── efficientnet_scalogram.py  # Paper 2: EfficientNet + CBAM
│   │   ├── nsht_dual_evo.py     # Paper 3: NSHT Dual-Evolution
│   │   └── dual_ensemble.py     # Dual Ensemble: Signal + Vision Fusion
│   ├── training/                 # Training utilities
│   │   └── trainer.py           # Trainer and KFoldTrainer
│   ├── evaluation/               # Metrics and evaluation
│   │   └── metrics.py           # PaperMetrics, visualizations
│   └── utils/                    # Configuration and utilities
│       └── config.py            # ExperimentConfig, tracking
├── configs/                      # YAML configuration files
│   ├── paper1_inceptiontime.yaml
│   ├── paper2_efficientnet.yaml
│   ├── paper3_nsht.yaml
│   └── dual_ensemble.yaml       # Dual Ensemble configuration
├── scripts/                      # CLI entry points
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   ├── eval_dual_ensemble.py    # Dual Ensemble evaluation script
│   ├── extract_nsht_prototypes.py # Paper 3 prototype-space extraction + t-SNE
│   └── train_all.py             # Train all models
├── balanced_data/                # Processed datasets
├── experiments/                  # Experiment outputs
├── checkpoints/                  # Model checkpoints
└── requirements.txt
```

## Installation

```bash
# Clone and navigate to project
cd ClassficationECG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Download and Prepare Data

```bash
# Download MIT-BIH and INCART databases
python -m src.data.download --download-all

# Create all RAW datasets (MIT-BIH, INCART, Combined)
python -m src.data.download --create-raw

# Create specific dataset only
python -m src.data.download --create-raw --mode mitbih
python -m src.data.download --create-raw --mode incart
python -m src.data.download --create-raw --mode combined
```

### 2. Create Balanced Datasets (SMOTE or ADASYN)

```bash
# Create balanced dataset from combined RAW data (default: 15,000 samples/class)
python -m src.data.download --create-balanced --mode combined

# Create balanced dataset from MIT-BIH only
python -m src.data.download --create-balanced --mode mitbih

# Create balanced dataset from INCART only
python -m src.data.download --create-balanced --mode incart

# Create ADASYN-balanced dataset from MIT-BIH only
python -m src.data.download --create-balanced --mode mitbih --balancing-method adasyn

# Create ADASYN-balanced dataset from combined RAW data
python -m src.data.download --create-balanced --mode combined --balancing-method adasyn

# Create all balanced datasets (MIT-BIH, INCART, Combined)
python -m src.data.download --create-balanced --mode all

# Custom samples per class
python -m src.data.download --create-balanced --mode combined --samples-per-class 20000
```

**Options:**
| Flag | Description |
|------|-------------|
| `--mode` | `mitbih`, `incart`, `combined`, or `all` (default: all) |
| `--balancing-method` | `smote` or `adasyn` (default: smote) |
| `--samples-per-class` | Target samples per class after SMOTE (default: 15000) |

This creates:
- `balanced_data/X_balanced_rpeak_smote.npy` (for combined)
- `balanced_data/y_balanced_rpeak_smote.npy`
- `balanced_data/X_mitbih_rpeak_smote.npy` (for mitbih)
- `balanced_data/X_incart_rpeak_smote.npy` (for incart)

If `--balancing-method adasyn` is used, filenames end with `_adasyn.npy`.

**Note:** Requires RAW datasets to exist first (`--create-raw`).

**Alternative:** Use standalone script for more options:
```bash
python create_balanced_rpeak_segments.py --datasets both --samples-per-class 15000 --no-visual
```

**Runtime sampling** can further limit data via `ECGDataModule`:

```python
from src.data import ECGDataModule

# Load all samples
data = ECGDataModule(mode='combined', balancing_method='smote')

# Limit to 10,000 samples per class (50k total for 5 classes)
data = ECGDataModule(mode='combined', samples_per_class=10000)
```

### 3. Train a Model

```bash
# Train InceptionTime (Paper 1)
python scripts/train.py --config configs/paper1_inceptiontime.yaml

# Train EfficientNet (Paper 2)
python scripts/train.py --config configs/paper2_efficientnet.yaml

# Train NSHT (Paper 3)
python scripts/train.py --config configs/paper3_nsht.yaml

# Evaluate Dual Ensemble (uses pre-trained Signal + Vision experts)
python scripts/eval_dual_ensemble.py --config configs/dual_ensemble.yaml
```

### 3B. Split First, Then Balance (Recommended for Strict Evaluation)

Use RAW datasets for splitting, then apply SMOTE/ADASYN only to the training split.

In your config under `data`:

```yaml
data:
    balancing_method: adasyn  # or smote
    balance_after_split: true
    samples_per_class: 20000  # optional target for train split only
```

Or from CLI:

```bash
python scripts/train.py --config configs/paper2_efficientnet.yaml --data.balance_after_split
```

Notes:
- `balance_after_split: false` keeps the original pre-balanced workflow.
- With `balance_after_split: true`, validation and test splits remain untouched.
- With `balance_after_split: true`, the method in `data.balancing_method` controls train-only balancing (`smote` or `adasyn`).
- With `balance_after_split: false`, you must pre-create matching balanced files before training.

### 3A. Paper 3 Prototype Training

`configs/paper3_nsht.yaml` now supports direct Option B prototype training for NSHT.

Key additions:
- `model.use_prototypes`: enable learnable class prototypes
- `model.prototype_dim`: latent prototype dimension
- `model.prototype_scale`: scaling for prototype similarity logits
- `model.prototype_logit_weight`: contribution of prototype logits to final class logits
- `training.prototype_loss_weight`: overall weight of prototype consistency loss
- `training.prototype_margin`: margin for hardest-negative separation
- `training.prototype_separation_weight`: weight for inter-class separation term
- `training.prototype_orthogonality_weight`: weight for prototype diversity regularization

The default Paper 3 config is now set up to retrain NSHT with prototype supervision out of the box.

### 4. K-Fold Cross-Validation (for paper results)

```bash
# Run 10-fold CV with per-fold balancing
python scripts/train.py --config configs/paper1_inceptiontime.yaml --kfold
```

K-fold config keys:

```yaml
kfold:
    enabled: true
    n_splits: 10
    apply_balancing_per_fold: true
    balancing_target: null
```

Backward compatibility:
- `apply_smote_per_fold` and `smote_target` are still accepted.
- Prefer `apply_balancing_per_fold` and `balancing_target` in new configs.

### 5. Evaluate a Model

```bash
python scripts/evaluate.py \
    --model-path checkpoints/paper1_inceptiontime/best_model.pt \
    --config configs/paper1_inceptiontime.yaml \
    --benchmark --latex
```

### 5A. Generate Paper 1 XAI Artifacts

Paper 1 now includes an evaluation-time XAI workflow tailored to the real ContextAwareInceptionTime architecture.

It produces:
- 1D Grad-CAM over the selected inception block
- Integrated Gradients over the raw ECG segment
- Branch-scale activation summaries across standard and dilated inception modules

Run it with:

```bash
python scripts/explain_paper1.py \
    --model-path checkpoints/paper1_inceptiontime/best_model.pt \
    --config configs/paper1_inceptiontime.yaml \
    --num-samples-per-class 1
```

Saved outputs under `experiments/paper1_inceptiontime/xai/` include:
- one folder per explained test sample
- `signal_attributions.png`
- `branch_summary.png`
- `attributions.npz`
- `branch_summary.json`
- top-level `summary.json`

Useful flags:
- `--target-layer-index`: choose which inception block Grad-CAM targets, with `-1` meaning the last block
- `--ig-steps`: control the Integrated Gradients approximation quality/cost tradeoff
- `--output-dir`: redirect artifacts to a custom location

### 5B. Generate Paper 2 XAI Artifacts

Paper 2 includes an evaluation-time XAI workflow for AttentionEfficientNet.  No retraining is needed.

Because the model is a CNN+CBAM pipeline (not a Transformer), the XAI methods are:
- **2D Grad-CAM** on the feature-fusion block (the `fusion` Sequential)
- **CBAM spatial attention maps** for each of the three EfficientNet feature scales and for the fusion CBAM
- **CBAM channel attention scores** (top-32 channels) per CBAM stage

Run it with:

```bash
python scripts/explain_paper2.py \
    --model-path checkpoints/paper2_efficientnet/best_model.pt \
    --config configs/paper2_efficientnet.yaml \
    --num-samples-per-class 1
```

Saved outputs under `experiments/paper2_efficientnet/xai/` include:
- one folder per explained test sample
- `scalogram_gradcam.png` – raw CWT scalogram + 2D Grad-CAM overlay
- `cbam_spatial_maps.png` – spatial attention heatmaps (3 scales + fusion)
- `cbam_channel_attention.png` – channel attention bar charts per CBAM stage
- `arrays.npz` – raw arrays for further analysis
- per-sample `summary.json`
- top-level `summary.json`

### 5C. Generate Paper 3 XAI Artifacts

Paper 3 includes an evaluation-time XAI workflow for NSHT_Dual_Evo.
Because the model has three unique components – a learnable wavelet transform, cross-modal attention between dual streams, and an optional prototype layer – the XAI methods are:

- **Learned wavelet parameters** – bar charts of `sigma`, `omega0`, and effective dilation scales across all wavelet scales (model-level, one-time)
- **Cross-modal attention map** – the L×L attention matrix produced by `EvolutionaryFusion.cross_attn`, showing how spectral stream positions attend to temporal positions (per sample)
- **Stream energy contributions** – mean absolute channel activation of the temporal and spectral streams plotted over time, revealing which regions each processing pathway focuses on (per sample)
- **Prototype-space t-SNE** – t-SNE of prototype-space embeddings with learned prototype centroids marked (optional, dataset-level; produced only when `use_prototypes=True`)

Run it with:

```bash
python scripts/explain_paper3.py \
    --model-path checkpoints/paper3_nsht/best_model.pt \
    --config configs/paper3_nsht.yaml \
    --num-samples-per-class 1
```

Saved outputs under `experiments/paper3_nsht/xai/` include:
- `wavelet_params.png` – learned sigma, omega0, and scale bars across all wavelet scales
- one folder per explained test sample
- `cross_attention.png` – cross-modal attention heatmap + raw ECG panel
- `stream_contributions.png` – temporal vs spectral stream energy over time
- `arrays.npz` – raw arrays for further analysis
- per-sample `summary.json`
- `prototype_tsne.png` – prototype-space t-SNE with class clusters and prototype centroids (when prototypes enabled)
- top-level `summary.json`

Useful flags:
- `--num-samples-per-class`: number of ECG segments to explain per AAMI class
- `--no-tsne`: skip the prototype-space t-SNE (much faster)
- `--tsne-max-samples`: cap on samples fed into t-SNE
- `--output-dir`: redirect artifacts to a custom location

### 6. Extract Paper 3 Prototypes and Prototype-Space t-SNE

After training a prototype-enabled NSHT checkpoint, you can export embeddings and learned prototypes in one run:

```bash
python scripts/extract_nsht_prototypes.py \
    --model-path checkpoints/paper3_nsht/best_model.pt \
    --config configs/paper3_nsht.yaml
```

This script saves:
- prototype-space embeddings
- predicted labels and ground-truth labels
- learned prototypes
- 2D t-SNE coordinates
- `tsne_with_prototypes.png`
- `summary.json`

## Python API Usage

```python
from src.data import ECGDataModule
from src.models import create_model
from src.training import Trainer
from src.evaluation import evaluate_model, PaperMetrics

# Load pre-balanced data (SMOTE or ADASYN)
data = ECGDataModule(mode='combined', balancing_method='smote')
loaders = data.get_signal_loaders(batch_size=2048)

# Limit samples per class for faster training (e.g., 10k per class = 50k total)
data = ECGDataModule(mode='combined', samples_per_class=10000)

# Create model
model = create_model('inception_time', num_classes=5, variant='base')
model = model.cuda()

# Train
trainer = Trainer(
    model=model,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    use_amp=True,
    use_class_weights=True,
    label_smoothing=0.05,
    augmentation_prob=0.35,
    augmentation_noise_std=0.008,
    augmentation_amplitude_jitter=0.08,
    augmentation_time_shift_pct=0.02
)
history = trainer.train(
    epochs=100,
    patience=15,
    monitor='val_acc'  # Use 'val_loss' to recover previous behavior
)

# Evaluate
results = evaluate_model(model, loaders['test'])
metrics = results['metrics']
metrics.print_full_report()
```

### Dual Ensemble Usage

```python
from src.models import DualEnsemble, create_dual_ensemble, ensemble_inference

# Option 1: Create from pre-trained experts
ensemble = DualEnsemble.from_pretrained(
    signal_path='models/signal_expert_best.pth',
    vision_path='models/vision_expert_best.pth',
    fusion='weighted',  # Options: 'weighted', 'learnable', 'attention', 'feature'
    freeze_experts=True
)

# Option 2: Use factory function
ensemble = create_dual_ensemble(
    num_classes=5,
    fusion='weighted',
    signal_weight=0.60,
    vision_weight=0.40,
    signal_path='models/signal_expert_best.pth',
    vision_path='models/vision_expert_best.pth'
)

# Inference
predictions, probabilities = ensemble.predict(X_test, batch_size=256)

# Or use standalone inference function with separate models
result = ensemble_inference(
    signal_model=signal_expert,
    vision_model=vision_expert,
    X=X_test,
    signal_weight=0.6,
    vision_weight=0.4
)
```

## Configuration

Edit YAML configs in `configs/` or override via command line:

```bash
python scripts/train.py --config configs/paper1_inceptiontime.yaml \
    --model.variant large \
    --training.epochs 150 \
    --training.lr 0.0005 \
    --data.mode mitbih
```

## Data Modes

| Mode | Description |
|------|-------------|
| `combined` | MIT-BIH + INCART combined |
| `mitbih` | MIT-BIH Arrhythmia Database only |
| `incart` | INCART Database only |
| `cross_mit_incart` | Train on MIT-BIH, test on INCART |
| `cross_incart_mit` | Train on INCART, test on MIT-BIH |

## AAMI Classes

| Class | Label | Description |
|-------|-------|-------------|
| 0 | N | Normal beats |
| 1 | S | Supraventricular ectopic beats |
| 2 | V | Ventricular ectopic beats |
| 3 | F | Fusion beats |
| 4 | Q | Unknown/Paced beats |

## Model Registry

Use the factory function to create any model by name:

```python
from src.models import create_model, list_models

# Available models
print(list_models())  # ['inception_time', 'efficientnet_scalogram', 'nsht_dual_evo', 'dual_ensemble']

# Create any model
model = create_model('inception_time', num_classes=5, variant='base')
model = create_model('efficientnet_scalogram', num_classes=5, variant='b0')
model = create_model(
    'nsht_dual_evo',
    num_classes=5,
    use_prototypes=True,
    prototype_dim=192,
)
model = create_model('dual_ensemble', num_classes=5, fusion='weighted')
```

| Model | Description | Key Args |
|-------|-------------|----------|
| `inception_time` | Context-Aware InceptionTime (1D) | `variant`: 'tiny', 'base', 'large' |
| `efficientnet_scalogram` | EfficientNet + CBAM (2D scalograms) | `variant`: 'b0', 'b1', etc. |
| `nsht_dual_evo` | NSHT Dual-Evolution Transformer | `use_prototypes`, `prototype_dim`, `prototype_scale`, `prototype_logit_weight` |
| `dual_ensemble` | Signal + Vision Fusion | `fusion`: 'weighted', 'learnable', 'attention', 'feature' |

## Fusion Strategies (Dual Ensemble)

| Strategy | Description |
|----------|-------------|
| `weighted` | Simple weighted average (default: 60% signal, 40% vision) |
| `learnable` | MLP on concatenated logits |
| `attention` | Attention network for dynamic per-sample weighting |
| `feature` | Fusion on feature level before classification heads |

## Configuration Options

Choose between SMOTE and ADASYN pre-balanced datasets:

```python
# SMOTE-balanced (default)
data = ECGDataModule(balancing_method='smote')

# ADASYN-balanced
data = ECGDataModule(balancing_method='adasyn')

# Limit samples per class
data = ECGDataModule(samples_per_class=10000)  # 10k per class = 50k total
```

Prototype-aware Paper 3 training can also be controlled in YAML:

```yaml
model:
    name: nsht_dual_evo
    use_prototypes: true
    prototype_dim: 192
    prototype_scale: 10.0
    prototype_logit_weight: 0.5

training:
    prototype_loss_weight: 0.2
    prototype_margin: 1.0
    prototype_separation_weight: 0.2
    prototype_orthogonality_weight: 0.01
```

## Citation

If you use this code, please cite:

```bibtex
@software{ecg_classification_2026,
  title={ECG Arrhythmia Classification: A Modular Research Framework},
  author={ECG Research Team},
  year={2026},
  url={https://github.com/your-repo/ecg-classification}
}
```

## License

MIT License
