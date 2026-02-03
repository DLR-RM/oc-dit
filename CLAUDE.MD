# CLAUDE.MD - OC-DiT Project Guide

## Project Overview

OC-DiT (Object-Centric Diffusion Transformer) is a conditional latent diffusion model for zero-shot instance segmentation. The model generates instance masks by conditioning the diffusion process on object templates and image features in the latent space.

**Paper**: [Conditional Latent Diffusion Models for Zero-Shot Instance Segmentation](https://arxiv.org/abs/2508.04122) (ICCV 2025)
**Authors**: Maximilian Ulmer, Wout Boerdijk, Rudolph Triebel, Maximilian Durner (DLR-RM)

## Repository Structure

```
oc-dit/
├── ocdit/
│   ├── models/
│   │   ├── ocdit.py           # Main OCDiT model (integrates all components)
│   │   ├── diffuser.py        # Diffusion model with transformer decoder
│   │   ├── vae.py             # BinaryMaskVAE for latent mask encoding
│   │   └── feature_extractor.py  # DINOv2-based spatial feature extractor
│   ├── layers/
│   │   ├── attention.py       # Modulated attention (AdaLN-Zero)
│   │   ├── decoder.py         # OCDiTBlock transformer decoder
│   │   └── embed.py           # Query/Template embedders, positional encoding
│   ├── heads/
│   │   └── linear_head.py     # Final output projection head
│   └── utils/
│       └── misc.py            # Utility functions
├── scripts/                   # [TO BE CREATED] Training/inference scripts
├── configs/                   # [TO BE CREATED] Configuration files
├── data/                      # [TO BE CREATED] Dataset utilities
├── requirements.txt
├── environment.yaml
└── setup.py
```

## Model Architecture

### Core Components

1. **Feature Extractor** (`SpatialDinov2`)
   - Uses frozen DINOv2 ViT-S/14 from Facebook Research
   - Output: spatial features at H/14 x W/14 resolution
   - Feature dimension: 384

2. **VAE** (`BinaryMaskVAE`)
   - Encodes binary masks to latent space
   - z_channels: 32
   - Encoder channels: [32, 64, 128, 256, 256]
   - Decoder channels: [256, 256, 128, 64, 32]
   - Frozen during diffusion training

3. **Diffuser** (Transformer-based)
   - Depth: 24 blocks (configurable)
   - Hidden size: 1024
   - Num heads: 16
   - MLP ratio: 4.0
   - Max classes: 8, Max templates: 12
   - Uses AdaLN-Zero modulation

### Default Configuration

```python
OCDiT(
    image_size=(480, 640),
    template_size=(128, 128),
    embed_dim=1024,
    depth=12,
    num_heads=16,
)
```

### Sampling Parameters

```python
sampling_parameters = {
    "num_steps": 9,
    "sigma_min": 0.002,
    "sigma_max": 80,
    "rho": 9,
}
```

## Datasets

The model is trained on synthetic datasets rendered from 3D meshes:

### 1. GSO Pose Estimation Dataset
- **Source**: Google Scanned Objects (GSO) meshes
- **Size**: ~1,000,000 samples
- **Purpose**: 6D pose estimation training
- **Content**: RGB images + instance masks + 6D poses

### 2. GSO Segmentation Dataset
- **Source**: GSO meshes (optimized for segmentation)
- **Size**: TBD
- **Purpose**: Instance segmentation training
- **Content**: Increased number of objects per sample

### 3. Objaverse Dataset
- **Source**: Objaverse 3D mesh repository
- **Size**: 2,600 meshes with diverse classes
- **Purpose**: Generalization to diverse object categories

### Dataset Format (Expected)

```python
# Each sample should contain:
{
    "image": torch.Tensor,        # (3, H, W) RGB image
    "templates": torch.Tensor,    # (num_classes, num_views, 3, h, w) template crops
    "masks": torch.Tensor,        # (num_classes, H, W) binary instance masks
}
```

## Training Procedure

### Loss Function

The model uses weighted MSE loss with learned variance:

```python
loss = ((weight / logvar.exp()) * loss_mask) + logvar
```

Where:
- `loss_mask = (y - D_yn) ** 2` (denoising prediction error)
- `weight` from preconditioner based on sigma
- `logvar` from learned MLP

### Preconditioner Settings

```python
precond = {
    "P_mean": -0.4,
    "P_std": 1.2,
    "sigma_data": 0.5
}
sigma_distribution = "uniform"  # or "normal", "normal_trunc"
```

### Recommended Training Hyperparameters (Based on DiT)

```python
# Optimizer
optimizer = "AdamW"
learning_rate = 1e-4
weight_decay = 0.0
betas = (0.9, 0.999)

# Schedule
warmup_steps = 1000
total_steps = 400000  # or more

# Batch size
batch_size_per_gpu = 8
gradient_accumulation = 1

# Mixed precision
use_amp = True  # FP16/BF16
```

## Development Stages

### Stage 1: CPU Testing (No GPU)

For validating the training loop without GPU:

```python
# Use small batch size and reduced model
config = {
    "batch_size": 2,
    "image_size": (224, 224),  # Smaller for testing
    "template_size": (64, 64),
    "embed_dim": 256,
    "depth": 2,
    "num_heads": 4,
    "max_steps": 10,
    "device": "cpu",
}
```

Key testing points:
1. Data loading pipeline works correctly
2. Forward pass completes without errors
3. Loss computation is valid
4. Backward pass completes
5. Optimizer step works
6. Checkpointing saves/loads correctly

### Stage 2: Cloud GPU Training

For production training on cloud GPUs:

```python
config = {
    "batch_size": 8,  # Per GPU
    "image_size": (480, 640),
    "template_size": (128, 128),
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "total_steps": 400000,
    "device": "cuda",
    "mixed_precision": True,
    "distributed": True,  # Multi-GPU
}
```

Recommended cloud setup:
- GPU: A100 40GB or better
- Multi-GPU: 4-8 GPUs with DDP
- Storage: Fast SSD for dataset I/O

## Implementation TODOs

### Priority 1: Training Infrastructure
- [ ] Create `scripts/train.py` - Main training script
- [ ] Create `configs/` - YAML configuration system
- [ ] Create `data/dataset.py` - Dataset loader base class
- [ ] Create `data/transforms.py` - Data augmentation
- [ ] Add learning rate scheduler
- [ ] Add gradient clipping
- [ ] Add EMA model

### Priority 2: Dataset Loaders
- [ ] Create `data/gso_dataset.py` - GSO dataset loader
- [ ] Create `data/objaverse_dataset.py` - Objaverse loader
- [ ] Create `data/synthetic_renderer.py` - (Optional) BlenderProc integration

### Priority 3: Evaluation & Logging
- [ ] Create `scripts/eval.py` - Evaluation script
- [ ] Add metrics (IoU, Precision, Recall, F1)
- [ ] Add TensorBoard/WandB logging
- [ ] Add visualization utilities

### Priority 4: Inference & Demo
- [ ] Create `scripts/demo.py` - Interactive demo
- [ ] Add pre-trained model loading
- [ ] Add example inference code

## Commands

```bash
# Install dependencies
pip install -e .
pip install -r requirements.txt

# Or with conda
conda env create -f environment.yaml
conda activate ocdit

# CPU test (once implemented)
python scripts/train.py --config configs/test_cpu.yaml --device cpu

# GPU training (once implemented)
python scripts/train.py --config configs/train_gso.yaml --device cuda

# Multi-GPU training
torchrun --nproc_per_node=4 scripts/train.py --config configs/train_gso.yaml
```

## Key Code Patterns

### Forward Pass (Training)

```python
model = OCDiT(image_size=(480, 640), template_size=(128, 128))

# Training forward
loss_dict = model.forward_loss(images, templates, masks_gt)
loss = loss_dict["loss"].mean()
loss.backward()
```

### Inference

```python
# Generate segmentations
pred_masks = model.generate_segmentations(
    images,
    templates,
    ensemble_size=3,
)  # Returns: (B, num_classes, H, W)
```

### Feature Extraction

```python
# Image normalization expected
images = TF.normalize(images, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

## Notes for Implementation

1. **VAE Pre-training**: The BinaryMaskVAE needs to be pre-trained separately before diffusion training. The VAE is frozen during OC-DiT training.

2. **DINOv2 Download**: First run will download DINOv2 weights from torch.hub. Ensure internet access or pre-download.

3. **Memory Optimization**:
   - Use gradient checkpointing for large models
   - Mixed precision (AMP) is highly recommended
   - Consider using `torch.compile()` for PyTorch 2.x

4. **Data Augmentation** (recommended):
   - Random horizontal flip
   - Color jitter
   - Random crop/resize
   - Random rotation (for templates)

5. **Debugging Tips**:
   - Monitor loss components separately (loss_masks vs logvar)
   - Check sigma distribution during training
   - Visualize intermediate denoising steps

## References

- [Original Paper](https://arxiv.org/abs/2508.04122)
- [DiT Paper](https://arxiv.org/abs/2212.09748) - Scalable Diffusion Models with Transformers
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research)
- [Objaverse](https://objaverse.allenai.org/)
