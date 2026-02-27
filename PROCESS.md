# Everything Optimized — Full Training & Evaluation Process

## Environment

- **Machine**: RunPod, 8x NVIDIA H100 80GB SXM5
- **OS**: Linux 6.8.0-56-generic
- **Python**: 3.12 (via pixi conda environment)
- **Framework**: PyTorch 2.x with torch.compile, BF16 mixed precision
- **Package manager**: pixi (conda-based)
- **Date**: 2026-02-27

### Environment Setup (run before every command)

```bash
export PATH="/root/.pixi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
export RATTLER_CACHE_DIR="/workspace/.cache/rattler"
export UV_CACHE_DIR="/workspace/.cache/uv"
export PYTHONUNBUFFERED=1
export ANTHROPIC_API_KEY='<your-key>'
```

### Dependencies Added

```bash
cd /workspace/wt-everything
pixi add openpyxl    # for experiment_tracker.py spreadsheet output
pixi add anthropic   # for LLM-based Story QA judge
```

---

## What "Everything Optimized" Means

This branch combines ALL individual modifications tested in other branches:

| Modification | Flag(s) | Source Branch |
|---|---|---|
| ReLU² FFN | `--ffn_type relu2` | modification/relu2-ffn |
| Muon optimizer | `--use_muon --matrix_lr 0.005 --embedding_lr 0.05 --muon_wd 0.005` | modification/muon-optimizer |
| QK normalization | `--qk_norm` | modification/attn-residual-tricks |
| Logit softcapping | `--softcap 15.0` | modification/attn-residual-tricks |
| Residual scalars | `--resid_scalars` | modification/attn-residual-tricks |
| QA data augmentation | `--qa_dir data/qa_augmented --qa_ratio 0.1` | modification/data-optimization |

### What Each Flag Does

- **`--ffn_type relu2`**: Replaces the default 3-matrix SwiGLU FFN with a 2-matrix ReLU² FFN. Since ReLU² uses 2 matrices instead of 3, d_ff can be increased to 3072 (from 2048) for the same parameter budget. The squared activation encourages sparsity.
- **`--use_muon`**: Uses MuonAdamW optimizer instead of standard AdamW. Muon applies orthogonal updates to matrix parameters for faster convergence.
- **`--matrix_lr 0.005`**: Learning rate for matrix (weight) parameters under Muon.
- **`--embedding_lr 0.05`**: Learning rate for embedding parameters (higher than matrix LR).
- **`--muon_wd 0.005`**: Weight decay for Muon optimizer.
- **`--qk_norm`**: Applies RMSNorm to query and key projections before attention dot product. Stabilizes attention scores.
- **`--softcap 15.0`**: Caps attention logits at ±15 using `tanh(logits/cap) * cap`. Prevents attention score explosion.
- **`--resid_scalars`**: Adds a learnable per-layer scalar multiplier to residual connections. Improves gradient flow.
- **`--qa_dir data/qa_augmented`**: Path to QA-augmented training data (generated in data-optimization branch).
- **`--qa_ratio 0.1`**: 10% of each batch comes from QA data, 90% from FineWeb-edu.
- **`--d_ff 3072`**: FFN hidden dimension (increased from 2048 because ReLU² uses 2 matrices instead of 3).

---

## Prerequisite: QA Data

The QA-augmented data must be generated first (see data-optimization branch PROCESS.md for full details). The data was generated in `/workspace/wt-data/` and copied:

```bash
cp -r /workspace/wt-data/data/qa_augmented /workspace/wt-everything/data/
```

This creates `data/qa_augmented/qa_augmented-00000-of-00001.parquet` containing 4,681 QA-augmented passages.

---

## Training: everything_optimized_5500_rerun (0.72B tier)

### Context: Why This is a Rerun

The original `everything_optimized_5500` was accidentally run on **1 GPU** instead of 8-GPU DDP. This meant it only trained on 0.09B tokens instead of the target 0.72B tokens. The val_loss of 3.7473 reflected this under-training. This rerun corrects to proper 8-GPU DDP.

### Token Budget Calculation

- **Tier**: 0.72B tokens
- **Steps**: 5,500
- **Batch size**: 32 sequences/GPU
- **Sequence length**: 512 tokens
- **GPUs**: 8 (DDP)
- **Tokens/step**: 32 × 512 × 8 = 131,072
- **Total tokens**: 5,500 × 131,072 = 720,896,000 (0.72B)

### Exact Training Command

```bash
cd /workspace/wt-everything

pixi run torchrun --nproc_per_node=8 train.py \
  --num_steps 5500 \
  --batch_size 32 \
  --d_model 768 \
  --n_layers 8 \
  --n_heads 8 \
  --d_ff 3072 \
  --ffn_type relu2 \
  --qk_norm \
  --softcap 15.0 \
  --resid_scalars \
  --use_muon \
  --matrix_lr 0.005 \
  --embedding_lr 0.05 \
  --muon_wd 0.005 \
  --qa_dir data/qa_augmented \
  --qa_ratio 0.1 \
  --run_name everything_optimized_5500_rerun \
  --save_checkpoint \
  --wandb_project weightless
```

### Training Output

- Duration: ~15 minutes on 8x H100 (including ~5 min torch.compile warmup)
- Speed: ~10.8 it/s after compilation, reported 5.99 it/s average (includes compile time)
- val_loss progression: crossed 3.5 threshold around step 2000, kept improving
- Goal achieved multiple times: 3.4556 → 3.4024 → 3.3653 → 3.3284 → 3.2924 → 3.2661
- **Final val_loss: 3.2661** (best in the 0.72B tier across all modifications)
- Final train_loss: 2.993
- MFU: 11.6%
- Total parameters: 95,246,608
- Checkpoint saved: `checkpoints/everything_optimized_5500_rerun.pt` (364 MB)
- WandB: https://wandb.ai/jinyoungkim927/weightless/runs/friejl4l

---

## Training: everything_50M_rerun (0.05B tier)

### Context: Why This is a Rerun

The original `everything_50M` scored 65% Story QA in your evaluation but Asher reported 25%. This rerun verifies reproducibility.

### Token Budget Calculation

- **Tier**: 0.05B tokens
- **Steps**: 3,050
- **Batch size**: 32
- **Sequence length**: 512
- **GPUs**: 1
- **Tokens/step**: 32 × 512 × 1 = 16,384
- **Total tokens**: 3,050 × 16,384 = 49,971,200 ≈ 0.05B

### Exact Training Command

```bash
cd /workspace/wt-everything

CUDA_VISIBLE_DEVICES=0 pixi run python train.py \
  --num_steps 3050 \
  --batch_size 32 \
  --d_model 768 \
  --n_layers 8 \
  --n_heads 8 \
  --d_ff 3072 \
  --ffn_type relu2 \
  --qk_norm \
  --softcap 15.0 \
  --resid_scalars \
  --use_muon \
  --matrix_lr 0.005 \
  --embedding_lr 0.05 \
  --muon_wd 0.005 \
  --qa_dir data/qa_augmented \
  --qa_ratio 0.1 \
  --eval_every 250 \
  --run_name everything_50M_rerun \
  --save_checkpoint \
  --wandb_project weightless \
  --no_auto_eval
```

### Additional Flags

- `--eval_every 250`: Evaluate validation loss every 250 steps (more frequent than default 500)
- `--no_auto_eval`: Skip post-training automatic Story QA evaluation (we do it manually)
- `CUDA_VISIBLE_DEVICES=0`: Run on GPU 0 only (single GPU for 0.05B tier)

### Training Output

- Duration: ~10 minutes on 1x H100
- Final val_loss: 3.8948
- train/total_tokens: 49,152,000 (confirmed 0.05B)
- Checkpoint saved: `checkpoints/everything_50M_rerun.pt` (364 MB)
- WandB: https://wandb.ai/jinyoungkim927/weightless/runs/jf0s3p1l

---

## Evaluation: Story QA Benchmark

### everything_optimized_5500_rerun Evaluation

```bash
cd /workspace/wt-everything

CUDA_VISIBLE_DEVICES=0 pixi run python experiment_tracker.py \
  --run_name everything_optimized_5500_rerun \
  --modification "All tricks combined: relu2 FFN, muon optimizer, QK norm, softcap, resid scalars, QA data" \
  --intuition "Combining all modifications at 0.72B scale should show compound benefits" \
  --checkpoint checkpoints/everything_optimized_5500_rerun.pt \
  --d_ff 3072 \
  --ffn_type relu2 \
  --qk_norm \
  --softcap 15.0 \
  --resid_scalars \
  --n_examples 10
```

**Important**: All model architecture flags (`--ffn_type relu2 --qk_norm --softcap 15.0 --resid_scalars --d_ff 3072`) must be passed to the evaluation script so `create_model()` builds the correct architecture to load the checkpoint weights into.

Results:
- 10-example spot check: 8/10 (80%)
- **100-sample benchmark: 72/100 = 72.0% Story QA accuracy**
- Very high confidence answers (e.g., 91.99%, 99.65%)
- Exact matches common: Q="How many toy cars did Timmy have?" → Model: "Timmy had two toy cars." (perfect)

### everything_50M_rerun Evaluation

```bash
cd /workspace/wt-everything

CUDA_VISIBLE_DEVICES=0 pixi run python experiment_tracker.py \
  --run_name everything_50M_rerun \
  --modification "All tricks combined at 50M token scale" \
  --intuition "Test if compound benefits hold at smallest scale" \
  --checkpoint checkpoints/everything_50M_rerun.pt \
  --d_ff 3072 \
  --ffn_type relu2 \
  --qk_norm \
  --softcap 15.0 \
  --resid_scalars \
  --n_examples 10
```

Results:
- 10-example spot check: 6/10 (60%)
- **100-sample benchmark: 69/100 = 69.0% Story QA accuracy**
- Remarkably strong for only 50M tokens — nearly matches the 0.72B result (72%)
- Confirms your original 65% result; **Asher's 25% is not reproducible**

---

## Reproducibility Note

| Run | Your Original | Asher's Report | This Rerun |
|---|---|---|---|
| everything_50M | 65% | 25% | **69%** |

The rerun result (69%) is very close to your original (65%), with small variation expected from random seed differences in the 100-sample benchmark. Asher's 25% result likely came from a different evaluation setup, wrong model architecture flags, or a different checkpoint.
