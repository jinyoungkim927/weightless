# weightless

Train sparse/efficient language models.

## Your Task

Train a model that achieves **val_loss < 3.5** on a 1.3B slice of [FineWebEdu](https://huggingface.co/datasets/kushalt/fineweb-edu-gpt2) as efficiently as possible.

The baseline model achieves this loss in about an hour with the default config. Your job is to improve efficiency while maintaining (or improving) loss.

## Dataset

This repo uses the tokenized [FineWeb-edu-gpt2](https://huggingface.co/datasets/kushalt/fineweb-edu-gpt2) dataset (GPT-2 tokenizer, 513-token sequences, 1.31B token subset).

## Setup

Using pixi (recommended):
```bash
pixi install
```

Or with pip:
```bash
pip install -r requirements.txt
```

Log in to wandb (optional but recommended):
```bash
wandb login
```

## Files

- `data.py` - Data loading from HuggingFace streaming dataset
- `model.py` - Baseline and baseline+ transformer models (modify this!)
- `train.py` - Training loop with wandb logging
- `eval.py` - Evaluation script

## Training

```bash
# Single GPU (no torchrun needed)
python train.py

# With a specific model variant
python train.py --model baseline
python train.py --model baseline_plus

# Custom config
python train.py --batch_size 64 --max_lr 8e-4 --num_steps 30000 --d_model 768 --n_layers 8

# Without wandb
python train.py --no_wandb

# Multi-GPU with DDP
torchrun --nproc_per_node=2 train.py
```

The default config (d_model=768, n_layers=8) reaches val_loss < 3.5 in ~55K steps (~60 min on H100).

## Evaluation

```bash
python eval.py --checkpoint model.pt
```

## Baselines

Two model variants are provided:

| Variant | Description | Key changes |
|---|---|---|
| `baseline` | Dense transformer | Standard MHA, full SwiGLU FFN |
| `baseline_plus` | GQA + top-k FFN | Fewer KV heads, activation sparsity in FFN |

The `baseline` is your starting point. The `baseline_plus` is included purely as an example to show how architectural changes improve efficiency -- it is not something you need to use or build on.

## Challenge

**Goal**: `val_loss < 3.5` with maximum efficiency.
