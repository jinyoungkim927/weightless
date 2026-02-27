# Data Optimization — Full Training & Evaluation Process

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
cd /workspace/wt-data
pixi add anthropic   # for QA generation API calls + LLM judge
pixi add openpyxl    # for experiment_tracker.py spreadsheet output
```

---

## Phase 0: QA Data Generation

### Why This Step is Needed

The data-optimization modification mixes QA-formatted data (question-answer pairs) into the FineWeb-edu training stream at a 10% ratio. This QA data must be generated first from FineWeb-edu passages using Claude.

### Step 1: Generate QA Pairs from FineWeb-edu

The script `generate_qa_data.py` does the following:
1. Loads 5,000 tokenized passages from FineWeb-edu via `StreamingParquetDataset`
2. Decodes each passage back to text (stripping padding tokens)
3. Sends each passage (truncated to 2000 chars) to Claude Haiku 4.5 with a prompt asking for 2-3 question-answer pairs
4. Concatenates original passage + generated QA pairs
5. Tokenizes using GPT2Tokenizer, pads/truncates to 513 tokens
6. Saves as parquet with columns `input_ids` and `pad_mask`

### Exact QA Generation Command

```bash
cd /workspace/wt-data

pixi run python generate_qa_data.py \
  --num_passages 5000 \
  --output_dir data/qa_augmented \
  --model claude-haiku-4-5-20251001 \
  --concurrency 50
```

### QA Generation Details

- **Model**: claude-haiku-4-5-20251001 (fast, cheap)
- **Concurrency**: 50 simultaneous async API requests using `anthropic.AsyncAnthropic()`
- **Max tokens per response**: 512
- **Passage truncation**: 2000 characters max sent to Claude
- **Rate limit handling**: Some requests fail (~7% failure rate due to 90,000 output tokens/min limit); failures are silently skipped
- **Duration**: ~10 minutes for 5000 passages
- **Result**: 4,681 out of 5,000 passages succeeded (93.6% success rate)
- **Output**: `data/qa_augmented/qa_augmented-00000-of-00001.parquet`

### Prompt Template Used

```
Given this educational passage, generate 2-3 question-answer pairs that test understanding of the key concepts.

Format each pair exactly as:
Question: <question>
Answer: <answer>

Passage:
{passage}

Generate the question-answer pairs:
```

### Copy QA Data to Everything Branch

The everything-optimized branch also uses QA data, so copy it:

```bash
cp -r /workspace/wt-data/data/qa_augmented /workspace/wt-everything/data/
```

---

## Training: data_optimization_5500_rerun

### Context: Why This is a Rerun

The original `data_optimization_5500` was accidentally run on **1 GPU** instead of 8-GPU DDP. This meant it only trained on 0.09B tokens (5500 × 32 × 512 × 1 = 90M) instead of the target 0.72B tokens. This rerun corrects that.

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
- Variant: baseline (standard SwiGLU FFN)

### Data Mixing Configuration

- `--qa_dir data/qa_augmented`: Path to QA-augmented parquet files
- `--qa_ratio 0.1`: 10% of each training batch comes from QA data, 90% from FineWeb-edu
- The train.py script creates two data loaders and interleaves them

### Exact Training Command

```bash
cd /workspace/wt-data

pixi run torchrun --nproc_per_node=8 train.py \
  --num_steps 5500 \
  --batch_size 32 \
  --d_model 768 \
  --n_layers 8 \
  --n_heads 8 \
  --d_ff 2048 \
  --qa_dir data/qa_augmented \
  --qa_ratio 0.1 \
  --run_name data_optimization_5500_rerun \
  --save_checkpoint \
  --wandb_project weightless
```

### Training Output

- Duration: ~16 minutes on 8x H100
- Speed: ~5.69 it/s (final reported, varies during run)
- val_loss progression: crossed 3.5 threshold during training
- Final val_loss: 3.4568 (target < 3.5 ACHIEVED)
- Checkpoint saved: `checkpoints/data_optimization_5500_rerun.pt` (364 MB)
- WandB: https://wandb.ai/jinyoungkim927/weightless/runs/ader9y8r

---

## Evaluation: Story QA Benchmark

### Exact Evaluation Command

```bash
cd /workspace/wt-data

CUDA_VISIBLE_DEVICES=0 pixi run python experiment_tracker.py \
  --run_name data_optimization_5500_rerun \
  --modification "QA data augmentation (10% ratio, 4681 samples)" \
  --intuition "QA-format data may improve comprehension at scale" \
  --checkpoint checkpoints/data_optimization_5500_rerun.pt \
  --d_ff 2048 \
  --n_examples 10
```

### Evaluation Results

- 10-example spot check: 8/10 (80%) correct
- **100-sample benchmark: 51/100 = 51.0% Story QA accuracy**
- Model produces relevant, often correct answers
- Example: Q="Where did the cat sit?" → Model: "The cat sat on the mat and was ready to study." (correct)
- Some failures on specific factual recall (e.g., counting "how many")

### Output Files

- `experiment_tracker.xlsx`: Full results
- Sheets: `Experiments`, `Examples_data_optimization_5500_rerun`, `Dataset_Overview`
