# Copy Gate — Full Training & Evaluation Process

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
cd /workspace/wt-copygat
pixi add openpyxl    # for experiment_tracker.py spreadsheet output
pixi add anthropic   # for LLM-based Story QA judge
```

---

## How Copy Gate Works

The copy gate is a lightweight mechanism (769 trainable parameters) added on top of a **frozen** pretrained baseline model. It learns to blend:
- **Generation distribution**: The model's normal next-token prediction (softmax over vocab)
- **Copy distribution**: Attention-weighted probability of copying a token from the input context

A learnable gate `p_copy` (scalar per position) decides the blend ratio. The base transformer weights are completely frozen — only the gate's linear layer (d_model → 1 = 768 + 1 = 769 params) is trained.

### Architecture

```
CopyGateTransformer(SimpleTransformer):
  - All base transformer layers (FROZEN, no gradients)
  - copy_gate = CopyGate(d_model=768)
    - gate = nn.Linear(768, 1)  # 769 trainable params
    - base_p = nn.Parameter(torch.tensor(0.1))
```

The forward pass:
1. Run frozen transformer → get `gen_logits` and hidden states `x`
2. Compute copy attention: `softmax(x @ embedding.T / sqrt(d_model))` with causal mask
3. Scatter copy attention into vocab space using input token IDs
4. Blend: `output = (1 - p_copy) * softmax(gen_logits) + p_copy * copy_probs`
5. Return `log(output)` for cross-entropy loss

---

## Bug Fix: dtype mismatch in model.py

Before training, a bug had to be fixed in `model.py` line ~400. Under BF16 autocast:
- `copy_probs` (from `torch.zeros_like(gen_logits)`) was BF16
- `copy_attn` (from `F.softmax(...)`) was FP32

`scatter_add_` requires matching dtypes. Fix:

```python
# Before (broken):
copy_probs.scatter_add_(2, input_ids.unsqueeze(1).expand(-1, T, -1), copy_attn)

# After (fixed):
copy_probs.scatter_add_(2, input_ids.unsqueeze(1).expand(-1, T, -1), copy_attn.to(copy_probs.dtype))
```

---

## Training: copy_gate_5500 (0.72B tier)

### Prerequisites

Requires a pretrained frozen baseline checkpoint. For the 0.72B tier, this is `baseline_10x_less.pt` which was trained on 720M tokens (5500 steps, 8-GPU DDP).

The baseline checkpoint already existed at:
```
/workspace/weightless/checkpoints/baseline_10x_less.pt
```

### Exact Training Command

```bash
cd /workspace/wt-copygat

CUDA_VISIBLE_DEVICES=0 pixi run python train_copy_gate.py \
  --checkpoint /workspace/weightless/checkpoints/baseline_10x_less.pt \
  --d_model 768 \
  --n_layers 8 \
  --n_heads 8 \
  --d_ff 2048 \
  --batch_size 32 \
  --lr 1e-3 \
  --num_steps 1000 \
  --save_path checkpoints/copy_gate_5500.pt
```

### Flag Explanation

- `--checkpoint`: Path to the frozen pretrained baseline model
- `--d_model 768 --n_layers 8 --n_heads 8 --d_ff 2048`: Must match the baseline's architecture exactly (0.72B tier uses d_ff=2048)
- `--batch_size 32`: Sequences per step (single GPU)
- `--lr 1e-3`: Learning rate for the 769 gate parameters only
- `--num_steps 1000`: Gate training steps (short — only 769 params to learn)
- `--save_path`: Where to save the full model (base weights + trained gate)

### Training Output

- Duration: ~5 minutes on 1x H100
- Initial val_loss: 3.4349
- Final val_loss: 3.4151 (delta: -0.0199)
- p_copy_base converged to: 0.041 (4.1% copy probability)
- Eval every 100 steps showed steady but small improvement
- Checkpoint saved: `checkpoints/copy_gate_5500.pt` (364 MB)

---

## Training: copy_gate_50M (0.05B tier)

### Prerequisites

Requires a baseline trained at the 50M token scale. The existing `baseline_50M.pt` was from a wrong run (550 steps, 8-GPU = 72M tokens). A correct baseline had to be retrained first.

### Step 1: Retrain Baseline at 50M Scale

```bash
cd /workspace/weightless

CUDA_VISIBLE_DEVICES=0 pixi run python train.py \
  --num_steps 3050 \
  --batch_size 32 \
  --d_model 768 \
  --n_layers 8 \
  --n_heads 8 \
  --d_ff 3072 \
  --run_name baseline_50M_retrain \
  --save_checkpoint \
  --wandb_project weightless \
  --no_auto_eval
```

Token calculation: 3050 × 32 × 512 × 1 GPU = 49,971,200 ≈ 0.05B tokens

- Duration: ~10 minutes on 1x H100
- Final val_loss: 4.374
- Checkpoint: `/workspace/weightless/checkpoints/baseline_50M_retrain.pt` (456 MB)

Note: d_ff=3072 for the 0.05B tier (all 50M experiments used d_ff=3072 to match parameter count).

### Step 2: Train Copy Gate on 50M Baseline

```bash
cd /workspace/wt-copygat

CUDA_VISIBLE_DEVICES=0 pixi run python train_copy_gate.py \
  --checkpoint /workspace/weightless/checkpoints/baseline_50M_retrain.pt \
  --d_model 768 \
  --n_layers 8 \
  --n_heads 8 \
  --d_ff 3072 \
  --batch_size 32 \
  --lr 1e-3 \
  --num_steps 1000 \
  --save_path checkpoints/copy_gate_50M.pt
```

Note: `--d_ff 3072` to match the 50M baseline's architecture.

### Training Output

- Duration: ~5 minutes on 1x H100
- Initial val_loss: 4.3124
- Final val_loss: 4.2903 (delta: -0.0221)
- p_copy_base converged to: 0.054 (5.4% copy probability)
- Checkpoint saved: `checkpoints/copy_gate_50M.pt` (436 MB)

---

## Evaluation: Story QA Benchmark

### copy_gate_5500 Evaluation

```bash
cd /workspace/wt-copygat

CUDA_VISIBLE_DEVICES=0 pixi run python experiment_tracker.py \
  --run_name copy_gate_5500 \
  --modification "Copy gate trained on baseline_10x_less (1000 steps, frozen base)" \
  --intuition "Copy mechanism for direct token recall" \
  --checkpoint checkpoints/copy_gate_5500.pt \
  --model copy_gate \
  --d_ff 2048 \
  --n_examples 10
```

**Important**: `--model copy_gate` is required so `create_model()` instantiates `CopyGateTransformer` instead of `SimpleTransformer`. Without this, the checkpoint's gate weights would fail to load.

Results:
- 10-example spot check: 4/10 (40%)
- **100-sample benchmark: 28/100 = 28.0% Story QA accuracy**

### copy_gate_50M Evaluation

```bash
cd /workspace/wt-copygat

CUDA_VISIBLE_DEVICES=0 pixi run python experiment_tracker.py \
  --run_name copy_gate_50M \
  --modification "Copy gate trained on baseline_50M_retrain (1000 steps, frozen base)" \
  --intuition "Copy mechanism for direct token recall at smallest scale" \
  --checkpoint checkpoints/copy_gate_50M.pt \
  --model copy_gate \
  --d_ff 3072 \
  --n_examples 10
```

Results:
- 10-example spot check: 0/10 (0%)
- **100-sample benchmark: 1/100 = 1.0% Story QA accuracy**
- Model produces nonsensical outputs — the frozen 50M baseline is too weak for the copy gate to help

### Code Fix Required for Evaluation

The `experiment_tracker.py` `--model` argument originally only accepted `["baseline", "baseline_plus"]`. Added `"copy_gate"` to the choices list at line 909 to support evaluating copy gate checkpoints.
