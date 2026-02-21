# Weightless – Project Log

> Tracking all changes, experiments, and the evaluation workflow for the Weightless language model project.

---

## Project Rules

1. **Dashboards must never hold GPUs.** All Gradio apps must pre-compute every output at startup, then release the GPU (`model.cpu()` + `torch.cuda.empty_cache()`) before launching the web server. No interactive callback should require the model or GPU. Inference examples should always use a random sample from `story_qa_v4_plaintext_shards`.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Baseline Training](#baseline-training)
4. [Evaluation & Inference Workflow](#evaluation--inference-workflow)
5. [Experiment Tracking Pipeline](#experiment-tracking-pipeline)
6. [Story QA Evaluation Dataset](#story-qa-evaluation-dataset)
7. [Gradio Web App (`generate.py`)](#gradio-web-app-generatepy)
8. [LLM-as-Judge (Semantic Matching)](#llm-as-judge-semantic-matching)
9. [Files Created / Modified](#files-created--modified)
10. [Experiment Log](#experiment-log)

---

## Project Overview

**Goal:** Train a compact transformer language model and systematically track modifications aimed at reducing inference memory cost (measured by `bytes_per_token_infer`) while maintaining quality.

**Model:** `SimpleTransformer` (baseline) — 8-layer transformer with:
- `d_model = 768`, `n_heads = 8`, `n_kv_heads = 8`, `d_ff = 2048`
- Vocabulary: GPT-2 tokenizer (50,257 tokens)
- Sequence length: 512 tokens
- RoPE positional encoding
- SwiGLU FFN

**Training data:** `fineweb-edu-gpt2` (subset `sample-10BT_max_length_513`) — ~10B tokens total, streamed from HuggingFace Hub.

---

## Environment Setup

```bash
# Clone repo
git clone https://github.com/jinyoungkim927/weightless.git
cd weightless

# Install dependencies
pip install -r requirements.txt
pip install pyarrow huggingface_hub

# wandb login
python3 -c "import wandb; wandb.login(key='...')"

# Additional packages installed later
pip install gradio transformers matplotlib openpyxl anthropic
```

**Machine:** 8× H100 GPUs (pod with SSH access)

---

## Baseline Training

### Configuration
| Parameter | Value |
|-----------|-------|
| `num_steps` | 55,000 |
| `batch_size` | 32 |
| `seq_len` | 512 (513 raw, minus 1 in `collate_fn`) |
| `world_size` | 8 (DDP across 8 GPUs) |
| `learning_rate` | default (cosine schedule) |
| `save_checkpoint` | ✓ |

### Command
```bash
nohup torchrun --nproc_per_node=8 train.py \
    --run_name baseline_full_8gpu \
    --save_checkpoint > train.log 2>&1 &
```

### Results
| Metric | Value |
|--------|-------|
| Final val_loss | **3.4955** |
| Perplexity | **32.93** |
| Non-zero params | **85,326,337** |
| Total tokens trained | **7.21B** (55,000 × 32 × 512 × 8 GPUs) |
| Training status | ✅ COMPLETE, GOAL ACHIEVED |
| Checkpoint | `checkpoints/baseline_full_8gpu.pt` |
| WandB | https://wandb.ai/jinyoungkim927/weightless/runs/dzpejlpz |

### Train/Val Split
- **Train:** `get_dataloader(split="train")` — streams from `train-*.parquet` files
- **Validation:** `get_dataloader(split="test")` — streams from `test-*.parquet` files
- No explicit val split carved from train; the dataset has a pre-defined test split

### Token Accounting
- **Unique token types (vocabulary):** 50,257 (GPT-2 tokenizer)
- **Total tokens per step:** `batch_size × seq_len × world_size` = 32 × 512 × 8 = **131,072**
- **Total tokens over training:** 55,000 × 131,072 = **7,208,960,000 (~7.21B)**
- The underlying dataset has ~10B tokens; baseline sees ~72% of them in one epoch

---

## Evaluation & Inference Workflow

### Standard Workflow for Each Experiment

1. **Train** the model (or modify architecture and re-train)
2. **Run `experiment_tracker.py`** to:
   - Load the checkpoint
   - Extract training metrics from `train.log` and wandb
   - Run inference on 10 Story QA test examples (interpretable spot-checks)
   - Use Claude as LLM judge for semantic answer matching
   - Update `experiment_tracker.xlsx` with a new row + examples sheet
3. **Launch Gradio app** (`generate.py --share`) to:
   - Interactively generate text and inspect model internals
   - Run the **📖 Story QA Inference** tab for spot-checks (configurable # examples)
   - Run the **🎯 100-Sample Accuracy Benchmark** for a standardized accuracy score
4. **Record** the Gradio share URL and wandb URL in the spreadsheet

### Experiment Tracker CLI
```bash
python experiment_tracker.py \
    --run_name "baseline" \
    --modification "None (baseline SimpleTransformer)" \
    --intuition "Establish baseline performance" \
    --checkpoint checkpoints/baseline_full_8gpu.pt \
    --gradio_url "https://..." \
    --wandb_url "https://wandb.ai/jinyoungkim927/weightless/runs/dzpejlpz"
```

### Excel Spreadsheet (`experiment_tracker.xlsx`)

**Experiments sheet** columns:
| Column | Description |
|--------|-------------|
| Run Name | Short identifier |
| Modification | What changed from baseline |
| Intuition | Why this modification was made |
| Training Run | COMPLETE / IN PROGRESS / GOAL status |
| Val Loss | Final validation loss |
| Perplexity | exp(val_loss) |
| Non-zero Params | Count of non-zero parameters |
| Sparsity % | Fraction of zero parameters |
| # Tokens Trained | Total tokens seen (e.g. "7.21B") |
| Story QA Acc % | 100-sample accuracy on Story QA test set (Claude-judged) |
| Checkpoint Path | Path to `.pt` file |
| Gradio URL | Public share link |
| WandB URL | Weights & Biases run link |
| Timestamp | When the row was added |

**Per-run Examples sheet** (`Examples_<run_name>`):
| Column | Description |
|--------|-------------|
| Doc ID | Story QA document identifier |
| Story (truncated) | First 300 chars of the story |
| Words | Word count of full story |
| # Questions | Number of QA pairs in the doc |
| Test Question | The question posed to the model |
| Gold Answer | Ground-truth answer |
| Model Answer | Model's generated answer |
| Match? | ✅/❌ (Claude LLM judge) |
| Story Loss | Cross-entropy loss on the story |
| Story PPL | Perplexity on the story |
| Avg Confidence | Mean probability of generated tokens |
| Top-5 for 1st Token | Top-5 token predictions for the first generated token |

**Dataset Overview sheet:** Summary statistics of the Story QA v4 dataset.

---

## Story QA Evaluation Dataset

**Dataset:** `story_qa_v4_plaintext_shards` (stored locally in the repo)

| Stat | Value |
|------|-------|
| Train documents | 100,000 |
| Test documents | 2,048 |
| Train shards | 10 files (~16 MB each) |
| Test shards | 1 file (~3.1 MB) |
| Total train questions | ~1,038,314 |
| Total test questions | ~21,432 |
| Avg questions/doc | ~10.4 |
| Avg story length | ~80 words (~445 chars) |
| Docs with definitions | ~90.6% |

**Format:** Plaintext with `<question>`, `<answer>`, `<definition>` tags, delimited by `===== DOC START / DOC END =====`.

**Question type distribution (test set):**
- What: 62.8%, Where: 9.1%, Who: 6.8%, Did: 6.4%, How: 6.0%, Was: 5.2%, Other: 3.7%

**Usage:** The model is prompted with `{story}\n\nQuestion: {question}\nAnswer:` and generates a completion. The generated answer is compared to the gold answer using the LLM judge.

---

## Gradio Web App (`generate.py`) — Static Dashboard

**Design principle:** Dashboards should **never hold GPUs**. All inference outputs are pre-computed at startup from a random Story QA sample, then the model is unloaded and the GPU is released. The Gradio app serves only static, pre-generated content.

**Launch:**
```bash
ANTHROPIC_API_KEY="sk-ant-..." CUDA_VISIBLE_DEVICES=0 python generate.py --port 7860 --share
```

### Startup Sequence
1. Load model onto GPU
2. Pre-compute all outputs (6 stages):
   - Pick a random Story QA example as the generation prompt
   - Run generation with full diagnostics (attention, FFN, residuals)
   - Generate attention heatmaps for all 8 layers
   - Generate FFN activation / sparsity / residual norm plots
   - Run 10-example Story QA spot-check with Claude judge
   - Run 100-sample accuracy benchmark with Claude judge
3. **Release GPU** (`model.cpu()` + `torch.cuda.empty_cache()`)
4. Build and serve a fully static Gradio app — no callbacks require the GPU

### Tabs (all pre-populated, no live inference)

1. **🖊️ Generation Example** — Pre-computed text generation from a random Story QA sample. Shows confidence-colored highlighted text, per-token probability/entropy charts, and top-5 candidate table.

2. **👁️ Attention Patterns** — Pre-computed attention heatmaps for every layer (head 0) and last-token attention bar charts. One section per layer.

3. **⚡ Activations** — Pre-computed FFN gate activation norms, sparsity maps, residual stream norms, and top-firing neuron plots.

4. **📖 Story QA Inference** — Pre-computed results:
   - 10-example spot-check with match status, perplexity, confidence, top-5 predictions
   - Detailed per-example markdown with story, QA pairs, definitions
   - **🎯 100-Sample Accuracy Benchmark:** accuracy score on 100 random test examples (Claude-judged)

5. **📋 Model Info & Metrics** — Architecture details and parameter counts.

---

## LLM-as-Judge (Semantic Matching)

**API:** Anthropic Claude (`claude-sonnet-4-20250514`)

**Key:** Set via `ANTHROPIC_API_KEY` environment variable (persisted in `~/.bashrc`)

**Approach:**
- Batch questions/gold/model answer triples (chunks of 25) to Claude
- Prompt instructs **very liberal** matching:
  - Different phrasing → ✅
  - Shorter/longer answers → ✅
  - Extra (non-contradictory) info → ✅
  - Paraphrasing / pronoun substitution → ✅
  - Example: Gold="The little bird wanted to go flying" vs Model="He wanted to fly" → ✅
- Only mark ❌ if factually wrong, contradictory, off-topic, or gibberish
- Response format: JSON array of booleans
- **Fallback:** If API key is missing or call fails, falls back to heuristic (substring + keyword overlap matching)

**Used in:**
- `generate.py` — Story QA tab and 100-sample benchmark
- `experiment_tracker.py` — Excel spreadsheet Match? column

---

## Files Created / Modified

| File | Status | Purpose |
|------|--------|---------|
| `generate.py` | **Created** | Gradio web app for inference, interpretability, Story QA evaluation, 100-sample benchmark |
| `experiment_tracker.py` | **Created** | Reusable pipeline: load model → run Story QA inference → update Excel spreadsheet |
| `experiment_tracker.xlsx` | **Created** | Excel workbook tracking all experiments with metrics and example outputs |
| `metric.py` | Restored (was accidentally deleted) | `InferenceProfile` dataclass and `bytes_per_token_infer` breakdown utilities |
| `model.py` | Unmodified | Model definitions (`SimpleTransformer`, `BaselinePlusTransformer`) |
| `train.py` | Unmodified | Training script with DDP, wandb logging, checkpoint saving |
| `data.py` | Unmodified | Streaming data loader for fineweb-edu-gpt2 |
| `rope.py` | Unmodified | Rotary Positional Embedding implementation |
| `eval.py` | Unmodified | Standalone evaluation script |
| `story_qa_v4_plaintext_shards/` | Pre-existing | Story QA evaluation dataset (train + test splits) |

---

## Experiment Log

### Experiment 1: Baseline (55,000 steps, 8× H100)

| Field | Value |
|-------|-------|
| **Run name** | `baseline` |
| **Modification** | None (baseline `SimpleTransformer`) |
| **Intuition** | Establish baseline performance |
| **Val loss** | 3.4955 |
| **Perplexity** | 32.93 |
| **Non-zero params** | 85,326,337 |
| **Sparsity** | 0.00% |
| **Tokens trained** | 7.21B |
| **Story QA Acc (100)** | **40%** |
| **Training time** | ~91 min (8× H100) |
| **Checkpoint** | `checkpoints/baseline_full_8gpu.pt` |
| **WandB** | https://wandb.ai/jinyoungkim927/weightless/runs/dzpejlpz |

### Experiment 2: Baseline 10× Less Data (5,500 steps, 8× H100)

| Field | Value |
|-------|-------|
| **Run name** | `baseline_10x_less` |
| **Modification** | Same architecture, 10× less data (5,500 steps instead of 55,000) |
| **Intuition** | Measure impact of data scaling on Story QA accuracy |
| **Val loss** | 3.4181 |
| **Perplexity** | 30.51 |
| **Non-zero params** | 95,246,592 |
| **Sparsity** | 0.00% |
| **Tokens trained** | 0.72B |
| **Story QA Acc (100)** | **18%** |
| **Training time** | ~10 min (8× H100) |
| **Checkpoint** | `checkpoints/baseline_10x_less.pt` |
| **WandB** | https://wandb.ai/jinyoungkim927/weightless/runs/gyr7s6bh |

### Summary Table

| Experiment | Steps | Tokens | Val Loss | PPL | Story QA Acc | Time |
|------------|-------|--------|----------|-----|-------------|------|
| Baseline | 55,000 | 7.21B | 3.4955 | 32.93 | **40%** | ~91 min |
| 10× Less Data | 5,500 | 0.72B | 3.4181 | 30.51 | **18%** | ~10 min |

**Key takeaway:** 10× less training data cuts Story QA accuracy from 40% → 18% (more than halved), even though perplexity is similar. This suggests the model needs more data exposure to develop reading comprehension capabilities, not just next-token prediction quality.

---

*Last updated: 2026-02-21*
