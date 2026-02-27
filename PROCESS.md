# True Everything — Full Training & Evaluation Process

Branch: `modification/true-everything`

## Environment

- **Machine**: RunPod, 8x NVIDIA H100 80GB SXM5
- **OS**: Linux 6.8.0-56-generic
- **Python**: 3.11
- **Framework**: PyTorch 2.x with torch.compile, BF16 mixed precision
- **Date**: 2026-02-27

### Dependencies

```bash
pip install pyarrow datasets wandb transformers openpyxl anthropic
```

---

## What "True Everything" Means

This branch adds **weight sparsity** and **copy gate** on top of everything-optimized, making it the first branch to combine ALL tested modifications:

| Modification | Flag(s) | In everything-optimized? |
|---|---|---|
| ReLU² FFN | `--ffn_type relu2`, d_ff=3072 | Yes |
| Muon optimizer | `--use_muon --matrix_lr 0.005 --embedding_lr 0.05 --muon_wd 0.005` | Yes |
| QK normalization | `--qk_norm` | Yes |
| Logit softcapping | `--softcap 15.0` | Yes |
| Residual scalars | `--resid_scalars` | Yes |
| QA data augmentation | `--qa_dir data/qa_augmented --qa_ratio 0.1` | Yes (variant A only) |
| **Weight sparsity** | `--weight_sparsity 0.3 --sparsity_warmup 1000 --prune_interval 100 --drop_fraction 0.3` | **NEW** |
| **Copy gate** | Phase B on frozen checkpoint | **NEW** |

---

## Phase A: Base Model Training

### Variant 1: With QA Data

```bash
cd /workspace/wt-everything

CUDA_VISIBLE_DEVICES=0 python train.py \
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
  --weight_sparsity 0.3 \
  --sparsity_warmup 1000 \
  --prune_interval 100 \
  --drop_fraction 0.3 \
  --run_name true_everything_50M \
  --save_checkpoint \
  --wandb_project weightless
```

**Token budget**: 3,050 steps x 32 batch x 512 seq_len = 49,971,200 (~0.05B)

**Training output**:
- Final val_loss: 3.9465
- Weight sparsity: 28.9% (48 managed matrices)
- Total params: 95,246,608; non-zero: 95,233,546
- WandB: https://wandb.ai/jinyoungkim927/weightless/runs/t50rm09f

### Variant 2: Without QA Data

```bash
cd /workspace/wt-everything

CUDA_VISIBLE_DEVICES=0 python train.py \
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
  --weight_sparsity 0.3 \
  --sparsity_warmup 1000 \
  --prune_interval 100 \
  --drop_fraction 0.3 \
  --run_name true_everything_no_qa_50M \
  --save_checkpoint \
  --wandb_project weightless
```

**Training output**:
- Final val_loss: 3.9429
- Weight sparsity: 28.9% (48 managed matrices)
- WandB: https://wandb.ai/jinyoungkim927/weightless/runs/kr80al8r

---

## Phase B: Copy Gate Training

### On With-QA Checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python train_copy_gate.py \
  --checkpoint checkpoints/true_everything_50M.pt \
  --d_model 768 \
  --n_layers 8 \
  --n_heads 8 \
  --d_ff 3072 \
  --ffn_type relu2 \
  --qk_norm \
  --softcap 15.0 \
  --resid_scalars \
  --batch_size 32 \
  --lr 1e-3 \
  --num_steps 1000 \
  --save_path checkpoints/true_everything_copy_gate_50M.pt
```

Results: val_loss 3.9376 → 3.9247 (delta -0.013), p_copy_base = 3.5%

### On No-QA Checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python train_copy_gate.py \
  --checkpoint checkpoints/true_everything_no_qa_50M.pt \
  --d_model 768 \
  --n_layers 8 \
  --n_heads 8 \
  --d_ff 3072 \
  --ffn_type relu2 \
  --qk_norm \
  --softcap 15.0 \
  --resid_scalars \
  --batch_size 32 \
  --lr 1e-3 \
  --num_steps 1000 \
  --save_path checkpoints/true_everything_no_qa_copy_gate_50M.pt
```

Results: val_loss 3.9341 → 3.9211 (delta -0.013), p_copy_base = 3.5%

---

## Phase C: Story QA Evaluation

### With QA Data

```bash
CUDA_VISIBLE_DEVICES=0 python experiment_tracker.py \
  --run_name "true_everything_50M" \
  --modification "True everything (all mods + QA data + sparsity + copy gate)" \
  --checkpoint checkpoints/true_everything_copy_gate_50M.pt \
  --model copy_gate \
  --d_model 768 --n_layers 8 --n_heads 8 --d_ff 3072 \
  --ffn_type relu2 --qk_norm --softcap 15.0 --resid_scalars
```

Result: **57/100 = 57.0% Story QA accuracy**

### Without QA Data

```bash
CUDA_VISIBLE_DEVICES=1 python experiment_tracker.py \
  --run_name "true_everything_no_qa_50M" \
  --modification "True everything (all mods + sparsity + copy gate, NO QA data)" \
  --checkpoint checkpoints/true_everything_no_qa_copy_gate_50M.pt \
  --model copy_gate \
  --d_model 768 --n_layers 8 --n_heads 8 --d_ff 3072 \
  --ffn_type relu2 --qk_norm --softcap 15.0 --resid_scalars
```

Result: **15/100 = 15.0% Story QA accuracy**

---

## Code Changes from everything-optimized

### train.py
- Added `SparseTopologyManager` class (RigL-style gradual magnitude pruning with gradient-guided regrowth)
- Added CLI args: `--weight_sparsity`, `--sparsity_warmup`, `--prune_interval`, `--drop_fraction`
- Added `sparse_manager.step(step)` after optimizer.step() in training loop
- Added sparsity logging to WandB
- Copy gate params excluded from sparsity management

### train_copy_gate.py
- Updated default d_ff from 2048 to 3072
- Added `--ffn_type`, `--qk_norm`, `--softcap`, `--resid_scalars` CLI args
- Passes architecture args to `create_model()` for correct model construction
