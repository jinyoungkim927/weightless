# Copy Gate Branch Results

## Experiments

### copy_gate_5500 (0.72B tier)
- **Date**: 2026-02-27
- **Config**: 1,000 steps gate-only training on frozen `baseline_10x_less.pt`, 1 GPU, batch_size=32, d_model=768, n_layers=8, n_heads=8, d_ff=2048
- **Modification**: Learnable copy gate (769 trainable params) blending generation and copy distributions
  - Base model frozen, only gate parameters trained
  - p_copy_base converged to 0.041 (4.1%)
- **Tokens trained**: 0.72B (base) + 1000 steps gate training
- **Initial val_loss**: 3.4349 → **Final val_loss**: 3.4151 (delta: -0.0199)
- **Non-zero params**: 95,247,361
- **Story QA accuracy**: 28% (100-sample benchmark, LLM judge)
- **Checkpoint**: `checkpoints/copy_gate_5500.pt` (364 MB)
- **Note**: Rerun on correct baseline_10x_less checkpoint

### copy_gate_50M (0.05B tier)
- **Date**: 2026-02-27
- **Config**: 1,000 steps gate-only training on frozen `baseline_50M_retrain.pt`, 1 GPU, batch_size=32, d_model=768, n_layers=8, n_heads=8, d_ff=3072
- **Modification**: Same copy gate mechanism on 50M-token baseline
  - p_copy_base converged to 0.054 (5.4%)
- **Initial val_loss**: 4.3124 → **Final val_loss**: 4.2903 (delta: -0.0221)
- **Non-zero params**: 114,121,729
- **Story QA accuracy**: 1% (100-sample benchmark, LLM judge)
- **Checkpoint**: `checkpoints/copy_gate_50M.pt` (436 MB)
- **Note**: Copy gate provides minimal benefit when base model is too weak (50M tokens insufficient)

### copy_gate (0.72B tier, original) — previous run
- **Story QA accuracy**: 34%
- **WandB**: N/A (original run)

## Bug Fix
- Fixed dtype mismatch in `model.py` scatter_add_ operation (line ~400): added `.to(copy_probs.dtype)` for bf16 autocast compatibility
