# Everything Optimized Branch Results

## Experiments

### everything_optimized_5500_rerun (0.72B tier)
- **Date**: 2026-02-27
- **Config**: 5,500 steps, 8x H100 DDP, batch_size=32, d_model=768, n_layers=8, n_heads=8, d_ff=3072
- **Modification**: All tricks combined
  - `--ffn_type relu2 --qk_norm --softcap 15.0 --resid_scalars`
  - `--use_muon --matrix_lr 0.005 --embedding_lr 0.05 --muon_wd 0.005`
  - `--qa_dir data/qa_augmented --qa_ratio 0.1`
- **Tokens trained**: 720,896,000 (0.72B)
- **Final val_loss**: 3.2661
- **Non-zero params**: 95,246,608
- **Story QA accuracy**: 72% (100-sample benchmark, LLM judge)
- **Checkpoint**: `checkpoints/everything_optimized_5500_rerun.pt` (364 MB)
- **WandB**: https://wandb.ai/jinyoungkim927/weightless/runs/friejl4l
- **Note**: Rerun of original everything_optimized_5500 which was incorrectly run on 1 GPU (0.09B tokens). This rerun uses correct 8-GPU DDP for 0.72B tokens. Best val_loss and Story QA accuracy in the 0.72B tier.

### everything_50M_rerun (0.05B tier)
- **Date**: 2026-02-27
- **Config**: 3,050 steps, 1 GPU, batch_size=32, d_model=768, n_layers=8, n_heads=8, d_ff=3072
- **Modification**: Same as above, at smallest scale
- **Tokens trained**: 49,152,000 (0.05B)
- **Final val_loss**: 3.8948
- **Non-zero params**: 95,246,608
- **Story QA accuracy**: 69% (100-sample benchmark, LLM judge)
- **Checkpoint**: `checkpoints/everything_50M_rerun.pt` (364 MB)
- **WandB**: https://wandb.ai/jinyoungkim927/weightless/runs/jf0s3p1l
- **Note**: Rerun for reproducibility. Original recorded 65% vs a separate report of 25%. This rerun yields 69%, confirming the original results replicate well.

### everything_optimized_5500 (original) — previous run
- **Story QA accuracy**: 63% (original) / 25% (separate report)
- **Note**: Was single-GPU (0.09B tokens incorrectly), now corrected to 0.72B

### everything_50M (original) — previous run
- **Story QA accuracy**: 65%
- **WandB**: https://wandb.ai/jinyoungkim927/weightless/runs/8gi63uty
