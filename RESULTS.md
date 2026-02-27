# Data Optimization Branch Results

## Experiments

### data_optimization_5500_rerun (0.72B tier)
- **Date**: 2026-02-27
- **Config**: 5,500 steps, 8x H100 DDP, batch_size=32, d_model=768, n_layers=8, n_heads=8, d_ff=2048
- **Modification**: QA data augmentation (4,681 QA samples mixed at 10% ratio)
  - `--qa_dir data/qa_augmented --qa_ratio 0.1`
- **Tokens trained**: 720,896,000 (0.72B)
- **Final val_loss**: 3.4568
- **Non-zero params**: 95,246,592
- **Story QA accuracy**: 51% (100-sample benchmark, LLM judge)
- **Checkpoint**: `checkpoints/data_optimization_5500_rerun.pt` (364 MB)
- **WandB**: https://wandb.ai/jinyoungkim927/weightless/runs/ader9y8r
- **Note**: Rerun of original data_optimization_5500 which was incorrectly run on 1 GPU (0.09B tokens). This rerun uses correct 8-GPU DDP for 0.72B tokens.

### data_opt_50M (0.05B tier) — previous run
- **Config**: 3,050 steps, 1 GPU, batch_size=32, d_ff=3072
- **Tokens trained**: 49,152,000 (0.05B)
- **Final val_loss**: 4.3795
- **Story QA accuracy**: 1%
- **WandB**: https://wandb.ai/jinyoungkim927/weightless/runs/o81o5dxt

## QA Data Generation
- Generated 4,681 QA-augmented passages from FineWeb-edu using Claude Haiku 4.5
- Async pipeline with 50 concurrent API requests
- Saved as parquet: `data/qa_augmented/qa_augmented-00000-of-00001.parquet`
