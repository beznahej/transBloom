# Model + Training Explainer

This is a practical walkthrough of `model.py` and `train.py` for quick orientation.

## `model.py`: what it does

- Defines two backbones for binary classification (`flower` vs `non_flower`):
  - `FlowerNetCnn`: small CNN for fast/local use.
  - `FlowerNetTransformer`: ViT-style encoder for stronger global reasoning.
- `FlowerNet` is a wrapper that picks one backbone based on `model_type`.
- Includes checkpoint helpers:
  - `make_checkpoint_payload`: stores weights + metadata.
  - `unpack_checkpoint` / `load_model_state_compat`: reads both new and legacy checkpoint formats safely.

## `model.py`: why the main numbers exist

- CNN channels `16 -> 32` and hidden dim `128`:
  - Small enough to train quickly on this dataset size.
  - Large enough to learn texture/color patterns for flower detection.
- Adaptive pool to `6x6`:
  - Keeps FC input size stable even if earlier spatial sizes shift slightly.
- Transformer defaults:
  - `img_size=32`: matches current pipeline and keeps compute low.
  - `patch_size=4`: gives `8x8 = 64` patches, a good token count for 32x32 images.
  - `embed_dim=128`, `depth=4`, `num_heads=4`:
    - Balanced capacity vs speed/memory on laptop/mobile-class hardware.
  - `mlp_ratio=4.0`:
    - Standard transformer width multiplier; good baseline for feed-forward blocks.
  - `dropout=0.1`:
    - Mild regularization without heavily slowing convergence.
- Init std `0.02`:
  - Common ViT-style small initialization for stable early training.

## `train.py`: what it does

- Parses CLI arguments for data paths, architecture, and optimization settings.
- Builds ImageFolder datasets:
  - Transformer training uses stronger augmentation.
  - Validation uses deterministic transforms.
- Builds model + loss + optimizer/scheduler.
- Runs epoch loop:
  - training pass
  - validation pass
  - save best checkpoint by validation accuracy
- Saves:
  - `best_model.pt` (best val checkpoint)
  - `last_model.pt` (final epoch checkpoint)
  - `class_to_idx.json` (label mapping)

## `train.py`: why the main numbers exist

- `epochs=20`:
  - Fast default for iteration; enough to see convergence trends.
- `batch_size=32`:
  - Safe memory default across CPU/MPS/CUDA.
- `lr=1e-3`:
  - Works well for Adam-family optimizers as a baseline.
- Transformer augmentation:
  - `RandomResizedCrop(scale=(0.6, 1.0))`: improves robustness to zoom/crop while keeping object content likely present.
  - `ColorJitter(..., hue=0.05)` with `p=0.5`: regularizes color reliance without destroying flower cues.
- Normalization `(0.5, 0.5, 0.5)` / `(0.5, 0.5, 0.5)`:
  - Maps pixel range near `[-1, 1]`; simple and consistent between train/eval.
- Auto optimizer policy:
  - CNN -> Adam, Transformer -> AdamW (better regularization behavior for transformers).
- Default Transformer AdamW weight decay `0.05`:
  - Helps generalization and prevents runaway weights.
- Cosine schedule + warmup:
  - Warmup reduces unstable early updates.
  - Cosine decay improves late-epoch refinement.
- Optional label smoothing / grad clipping:
  - Smoothing reduces overconfident logits.
  - Clipping prevents occasional gradient spikes.

## Practical read order

1. `model.py`:
   - `FlowerNetCnn.forward`
   - `FlowerNetTransformer.forward`
   - `FlowerNet.__init__`
2. `train.py`:
   - argument parser
   - transform block
   - optimizer/scheduler block
   - epoch loop
