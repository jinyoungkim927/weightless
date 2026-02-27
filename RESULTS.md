# Weight Sparsity Branch Results

## Experiments

### weight_sparsity_5500 (0.72B tier)
- **Date**: 2026-02-27
- **Config**: 5,500 steps, 8x H100 DDP, batch_size=32, d_model=768, n_layers=8, n_heads=8, d_ff=2048
- **Modification**: RigL-style weight sparsity (30%) with gradient-guided regrowth
  - `--weight_sparsity 0.3 --sparsity_warmup 1000 --prune_interval 100 --drop_fraction 0.3`
- **Tokens trained**: 720,896,000 (0.72B)
- **Final val_loss**: 3.5097
- **Non-zero params**: 78,325,183
- **Story QA accuracy**: 22% (100-sample benchmark, LLM judge)
- **Checkpoint**: `checkpoints/weight_sparsity_5500.pt` (364 MB)
- **WandB**: https://wandb.ai/jinyoungkim927/weightless/runs/1hyu3bd6

### weight_sparsity_50M (0.05B tier) — previous run
- **Config**: 3,050 steps, 1 GPU, batch_size=32, d_ff=2048
- **Tokens trained**: 49,152,000 (0.05B)
- **Final val_loss**: 4.5154
- **Story QA accuracy**: 2%
- **WandB**: https://wandb.ai/jinyoungkim927/weightless/runs/iaoigghe
