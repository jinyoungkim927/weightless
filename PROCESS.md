# Weight Sparsity — Full Training & Evaluation Process

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
```

### Dependencies Added

```bash
cd /workspace/wt-sparsity
pixi add openpyxl    # for experiment_tracker.py spreadsheet output
pixi add anthropic   # for LLM-based Story QA judge
```

---

## Training: weight_sparsity_5500

### Token Budget Calculation

- **Tier**: 0.72B tokens
- **Steps**: 5,500
- **Batch size**: 32 sequences/GPU
- **Sequence length**: 512 tokens
- **GPUs**: 8 (DDP)
- **Tokens/step**: 32 × 512 × 8 = 131,072
- **Total tokens**: 5,500 × 131,072 = 720,896,000 (0.72B)

### Model Architecture

- d_model=768, n_layers=8, n_heads=8, d_ff=2048
- Total parameters: 95,246,592
- Non-zero parameters after sparsity: 78,325,183 (17.8% pruned)

### Sparsity Configuration

- `--weight_sparsity 0.3`: Target 30% of weights pruned
- `--sparsity_warmup 1000`: Gradually increase sparsity over first 1000 steps
- `--prune_interval 100`: Re-evaluate which weights to prune every 100 steps
- `--drop_fraction 0.3`: At each prune step, drop 30% of current non-zero weights and regrow based on gradient magnitude (RigL-style)

### Exact Training Command

```bash
cd /workspace/wt-sparsity

pixi run torchrun --nproc_per_node=8 train.py \
  --num_steps 5500 \
  --batch_size 32 \
  --d_model 768 \
  --n_layers 8 \
  --n_heads 8 \
  --d_ff 2048 \
  --weight_sparsity 0.3 \
  --sparsity_warmup 1000 \
  --prune_interval 100 \
  --drop_fraction 0.3 \
  --run_name weight_sparsity_5500 \
  --save_checkpoint \
  --wandb_project weightless
```

### Training Output

- Duration: ~7 minutes on 8x H100
- Speed: ~13.7 it/s
- Final val_loss: 3.5097 (target < 3.5 not achieved)
- Goal not reached (val_loss 3.5097 >= 3.5)
- Checkpoint saved: `checkpoints/weight_sparsity_5500.pt` (364 MB)
- WandB: https://wandb.ai/jinyoungkim927/weightless/runs/1hyu3bd6

---

## Evaluation: Story QA Benchmark

### Exact Evaluation Command

```bash
cd /workspace/wt-sparsity

export ANTHROPIC_API_KEY='<your-key>'

CUDA_VISIBLE_DEVICES=0 pixi run python experiment_tracker.py \
  --run_name weight_sparsity_5500_v2 \
  --modification "Weight sparsity 30% with gradual pruning" \
  --intuition "Structured sparsity may improve generalization at scale" \
  --checkpoint checkpoints/weight_sparsity_5500.pt \
  --d_ff 2048 \
  --n_examples 10
```

### What experiment_tracker.py Does

1. Loads the checkpoint into a `create_model(variant="baseline", d_model=768, n_layers=8, n_heads=8, d_ff=2048)` model
2. Loads 10 Story QA test examples from `story_qa_v4_plaintext_shards/test/`
3. Runs inference: encodes `{story}\n\nQuestion: {q}\nAnswer:` truncated to 480 tokens, generates up to 60 tokens at temperature=0.7
4. Judges each answer using Claude API (semantic matching) — falls back to heuristic keyword overlap if API unavailable
5. Runs 100-sample benchmark: same process on 100 random test examples
6. Saves results to `experiment_tracker.xlsx`

### Evaluation Results

- 10-example spot check: 2/10 (20%) correct
- **100-sample benchmark: 22/100 = 22.0% Story QA accuracy**
- Average story perplexity: ~55-68
- Average confidence: 15-44%
- Model produces vaguely relevant but often incorrect answers (e.g., "The cat sitting in a cup" instead of "The cat sat on the mat")

### Output Files

- `experiment_tracker.xlsx`: Full results with per-example details
- Sheets: `Experiments` (summary row), `Examples_weight_sparsity_5500_v2` (10 detailed examples)
