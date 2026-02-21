#!/usr/bin/env python3
"""Diagnose why the model performs poorly on StoryQA despite generating sensible text.

Analyses:
1. Truncation analysis: how many prompts exceed 480/512 tokens
2. Format presence in training data: search for Q&A patterns in FineWeb-edu
3. Model generation analysis: greedy + sampled outputs on 20 random examples
4. First-token analysis after "Answer:": what does the model predict?
"""

import os
import sys
import re
import random
import numpy as np
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import create_model

# ============================================================================
# Config
# ============================================================================
CHECKPOINT = "/root/weightless/checkpoints/baseline_full_8gpu.pt"
TEST_FILE = "/root/weightless/story_qa_v4_plaintext_shards/test/test_shard_00000.txt"
MAX_SEQ_LEN = 512
TRUNCATION_LEN = 480
N_GEN_EXAMPLES = 20
N_FIRST_TOKEN_EXAMPLES = 200  # for first-token analysis
SEED = 42

# ============================================================================
# Parse StoryQA test data (reused from generate.py)
# ============================================================================

def parse_story_qa_file(filepath: str) -> list:
    """Parse a story QA plaintext shard into structured documents."""
    with open(filepath) as f:
        text = f.read()
    docs = []
    for raw in text.split("===== DOC START"):
        if not raw.strip():
            continue
        header = re.search(r'split=(\w+)\s+idx=(\d+)\s+id=(\w+)', raw)
        if not header:
            continue
        split, idx, doc_id = header.groups()
        content = raw.split("=====\n", 1)[-1].split("===== DOC END")[0].strip()
        parts = content.split("<question>")
        story = parts[0].strip()
        story_clean = re.sub(r'<definition>.*?</definition>', '', story).strip()
        story_clean = re.sub(r'\s+', ' ', story_clean)
        qa_pairs = []
        for p in parts[1:]:
            q = re.search(r'(.*?)</question>', p, re.DOTALL)
            a = re.search(r'<answer>(.*?)</answer>', p, re.DOTALL)
            if q and a:
                qa_pairs.append({"question": q.group(1).strip(),
                                 "answer": a.group(1).strip()})
        docs.append({"id": doc_id, "idx": int(idx), "split": split,
                      "story": story_clean, "qa_pairs": qa_pairs})
    return docs


# ============================================================================
# 1. Truncation Analysis
# ============================================================================

def truncation_analysis(docs, tokenizer):
    print("=" * 80)
    print("  1. TRUNCATION ANALYSIS")
    print("=" * 80)

    all_prompt_lengths = []
    exceed_480 = 0
    exceed_512 = 0
    question_cutoff_480 = 0
    question_cutoff_512 = 0
    total_prompts = 0

    for doc in docs:
        for qa in doc["qa_pairs"]:
            prompt = f'{doc["story"]}\n\nQuestion: {qa["question"]}\nAnswer:'
            tokens = tokenizer.encode(prompt)
            length = len(tokens)
            all_prompt_lengths.append(length)
            total_prompts += 1

            if length > 480:
                exceed_480 += 1
                # Check if Question/Answer: gets cut off at 480
                truncated_text = tokenizer.decode(tokens[:480])
                if "Answer:" not in truncated_text:
                    question_cutoff_480 += 1

            if length > 512:
                exceed_512 += 1
                truncated_text = tokenizer.decode(tokens[:512])
                if "Answer:" not in truncated_text:
                    question_cutoff_512 += 1

    lengths = np.array(all_prompt_lengths)

    print(f"\n  Total prompts (all QA pairs across all docs): {total_prompts}")
    print(f"  Number of docs: {len(docs)}")
    print(f"\n  Prompt length statistics:")
    print(f"    Min:    {lengths.min()}")
    print(f"    Max:    {lengths.max()}")
    print(f"    Mean:   {lengths.mean():.1f}")
    print(f"    Median: {np.median(lengths):.1f}")
    print(f"    Std:    {lengths.std():.1f}")

    print(f"\n  Truncation counts:")
    print(f"    Exceed 480 tokens: {exceed_480}/{total_prompts} ({100*exceed_480/total_prompts:.1f}%)")
    print(f"    Exceed 512 tokens: {exceed_512}/{total_prompts} ({100*exceed_512/total_prompts:.1f}%)")

    print(f"\n  Question/Answer: cutoff (prompt truncated before 'Answer:' appears):")
    print(f"    At 480 truncation: {question_cutoff_480}/{exceed_480 if exceed_480 else 1} of truncated prompts")
    print(f"    At 512 truncation: {question_cutoff_512}/{exceed_512 if exceed_512 else 1} of truncated prompts")

    # Distribution buckets
    buckets = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 480, 512, 600, 800, 1000, 2000]
    print(f"\n  Length distribution (cumulative):")
    for b in buckets:
        count = (lengths <= b).sum()
        print(f"    <= {b:4d} tokens: {count:5d} ({100*count/total_prompts:.1f}%)")

    # Also check: for first QA pair only (as used in benchmark)
    first_qa_lengths = []
    first_exceed_480 = 0
    first_qa_cutoff_480 = 0
    for doc in docs:
        if doc["qa_pairs"]:
            qa = doc["qa_pairs"][0]
            prompt = f'{doc["story"]}\n\nQuestion: {qa["question"]}\nAnswer:'
            tokens = tokenizer.encode(prompt)
            length = len(tokens)
            first_qa_lengths.append(length)
            if length > 480:
                first_exceed_480 += 1
                truncated_text = tokenizer.decode(tokens[:480])
                if "Answer:" not in truncated_text:
                    first_qa_cutoff_480 += 1

    first_lengths = np.array(first_qa_lengths)
    print(f"\n  First QA pair only (as used in benchmark):")
    print(f"    Total docs with QA: {len(first_qa_lengths)}")
    print(f"    Mean length: {first_lengths.mean():.1f}")
    print(f"    Exceed 480: {first_exceed_480}/{len(first_qa_lengths)} ({100*first_exceed_480/len(first_qa_lengths):.1f}%)")
    if first_exceed_480 > 0:
        print(f"    'Answer:' cutoff at 480: {first_qa_cutoff_480}/{first_exceed_480}")

    # Show some examples of truncated prompts
    print(f"\n  Examples of truncated prompts (first 5 exceeding 480):")
    shown = 0
    for doc in docs:
        if shown >= 5:
            break
        for qa in doc["qa_pairs"]:
            if shown >= 5:
                break
            prompt = f'{doc["story"]}\n\nQuestion: {qa["question"]}\nAnswer:'
            tokens = tokenizer.encode(prompt)
            if len(tokens) > 480:
                truncated_text = tokenizer.decode(tokens[:480])
                last_100_chars = truncated_text[-150:]
                has_answer = "Answer:" in truncated_text
                has_question = "Question:" in truncated_text
                print(f"\n    --- Doc {doc['id']}, {len(tokens)} tokens ---")
                print(f"    Has 'Question:' in truncated: {has_question}")
                print(f"    Has 'Answer:' in truncated:   {has_answer}")
                print(f"    Last 150 chars of truncated:  ...{last_100_chars}")
                shown += 1

    return lengths


# ============================================================================
# 2. Format Presence in Training Data
# ============================================================================

def format_presence_analysis(tokenizer):
    print("\n" + "=" * 80)
    print("  2. FORMAT PRESENCE IN TRAINING DATA (FineWeb-edu-gpt2)")
    print("=" * 80)

    try:
        from data import get_parquet_files
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq

        fs = HfFileSystem()
        files = get_parquet_files(split="train")
        print(f"\n  Found {len(files)} training parquet files")
        print(f"  Sampling from first file: {files[0]}")

        # Read a sample from the first file
        with fs.open(files[0], "rb") as f:
            table = pq.read_table(f)

        n_rows = len(table)
        print(f"  Rows in first file: {n_rows}")

        # Sample up to 5000 rows
        sample_size = min(5000, n_rows)
        indices = list(range(sample_size))

        # Decode token sequences and search for patterns
        patterns = {
            "Question:": 0,
            "Answer:": 0,
            "Q:": 0,
            "A:": 0,
            "question": 0,
            "answer": 0,
            "\\nQ:": 0,
            "\\nA:": 0,
            "\\nQuestion:": 0,
            "\\nAnswer:": 0,
        }

        qa_format_count = 0  # has both Question: and Answer:
        any_qa_count = 0  # has any Q/A pattern

        print(f"  Decoding and searching {sample_size} training documents...")

        for i in indices:
            row = {col: table[col][i].as_py() for col in table.column_names}
            input_ids = row["input_ids"]
            text = tokenizer.decode(input_ids)

            has_any = False
            for pat in patterns:
                actual_pat = pat.replace("\\n", "\n")
                if actual_pat in text:
                    patterns[pat] += 1
                    has_any = True

            if has_any:
                any_qa_count += 1

            if "Question:" in text and "Answer:" in text:
                qa_format_count += 1

        print(f"\n  Pattern frequencies in {sample_size} training documents:")
        for pat, count in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"    {pat:20s}: {count:5d} ({100*count/sample_size:.2f}%)")

        print(f"\n  Documents with BOTH 'Question:' AND 'Answer:': {qa_format_count} ({100*qa_format_count/sample_size:.2f}%)")
        print(f"  Documents with ANY Q/A pattern: {any_qa_count} ({100*any_qa_count/sample_size:.2f}%)")

        # Show a few examples of docs that have Q&A format
        print(f"\n  First 3 training documents containing 'Question:' + 'Answer:':")
        shown = 0
        for i in indices:
            if shown >= 3:
                break
            row = {col: table[col][i].as_py() for col in table.column_names}
            text = tokenizer.decode(row["input_ids"])
            if "Question:" in text and "Answer:" in text:
                # Find the Q&A section
                q_idx = text.index("Question:")
                excerpt = text[max(0, q_idx - 50):q_idx + 200]
                print(f"\n    --- Training doc {i} ---")
                print(f"    ...{excerpt}...")
                shown += 1

    except Exception as e:
        print(f"\n  ERROR loading training data: {e}")
        print(f"  Skipping training data analysis.")
        import traceback
        traceback.print_exc()


# ============================================================================
# 3. Model Generation Analysis
# ============================================================================

@torch.no_grad()
def greedy_generate(model, input_ids, max_new_tokens, tokenizer, device):
    """Greedy (temperature=0) generation."""
    gen_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        if gen_ids.shape[1] > MAX_SEQ_LEN:
            gen_ids = gen_ids[:, -MAX_SEQ_LEN:]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(gen_ids)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok_id = next_token[0, 0].item()
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        if tok_id == tokenizer.eos_token_id or "\n" in tokenizer.decode([tok_id]):
            break
    return gen_ids


@torch.no_grad()
def sample_generate(model, input_ids, max_new_tokens, tokenizer, device, temperature=0.7):
    """Sampling generation with temperature."""
    gen_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        if gen_ids.shape[1] > MAX_SEQ_LEN:
            gen_ids = gen_ids[:, -MAX_SEQ_LEN:]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(gen_ids)
        next_logits = logits[:, -1, :].float() / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tok_id = next_token[0, 0].item()
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        if tok_id == tokenizer.eos_token_id or "\n" in tokenizer.decode([tok_id]):
            break
    return gen_ids


@torch.no_grad()
def get_top_k_after_prompt(model, input_ids, tokenizer, k=5):
    """Get top-k token predictions after the prompt."""
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids)
    next_logits = logits[:, -1, :].float()
    probs = F.softmax(next_logits, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, k, dim=-1)
    tokens = [tokenizer.decode([tid]) for tid in topk_ids[0].tolist()]
    probs_list = topk_probs[0].tolist()
    return list(zip(tokens, probs_list))


def model_generation_analysis(model, docs, tokenizer, device):
    print("\n" + "=" * 80)
    print("  3. MODEL GENERATION ANALYSIS (20 random examples)")
    print("=" * 80)

    rng = random.Random(SEED)
    # Pick docs with QA pairs
    docs_with_qa = [d for d in docs if d["qa_pairs"]]
    samples = rng.sample(docs_with_qa, min(N_GEN_EXAMPLES, len(docs_with_qa)))

    for i, doc in enumerate(samples):
        qa = doc["qa_pairs"][0]
        prompt = f'{doc["story"]}\n\nQuestion: {qa["question"]}\nAnswer:'

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        orig_len = input_ids.shape[1]
        truncated = False
        if input_ids.shape[1] > TRUNCATION_LEN:
            input_ids = input_ids[:, :TRUNCATION_LEN]
            truncated = True
        prompt_len = input_ids.shape[1]

        # Top-5 tokens after "Answer:"
        top5 = get_top_k_after_prompt(model, input_ids, tokenizer, k=5)

        # Greedy generation
        greedy_ids = greedy_generate(model, input_ids, 60, tokenizer, device)
        greedy_answer = tokenizer.decode(greedy_ids[0, prompt_len:].tolist()).strip().split("\n")[0]

        # Sample generation
        sample_ids = sample_generate(model, input_ids, 60, tokenizer, device, temperature=0.7)
        sample_answer = tokenizer.decode(sample_ids[0, prompt_len:].tolist()).strip().split("\n")[0]

        # Print
        prompt_display = prompt[:200]
        if len(prompt) > 200:
            prompt_display += "..."

        print(f"\n  --- Example {i+1} (doc {doc['id']}) ---")
        print(f"  Prompt length: {orig_len} tokens {'(TRUNCATED to 480)' if truncated else ''}")
        print(f"  Prompt: {prompt_display}")
        print(f"  Gold answer:     {qa['answer']}")
        print(f"  Greedy (T=0):    {greedy_answer}")
        print(f"  Sampled (T=0.7): {sample_answer}")
        print(f"  Top-5 after 'Answer:': {', '.join(f'{tok!r} ({p:.1%})' for tok, p in top5)}")

        if truncated:
            # Show what the truncated prompt ends with
            trunc_text = tokenizer.decode(input_ids[0].tolist())
            print(f"  Truncated prompt ends with: ...{trunc_text[-100:]}")


# ============================================================================
# 4. First-Token Analysis after "Answer:"
# ============================================================================

@torch.no_grad()
def first_token_analysis(model, docs, tokenizer, device):
    print("\n" + "=" * 80)
    print("  4. FIRST-TOKEN ANALYSIS AFTER 'Answer:'")
    print("=" * 80)

    rng = random.Random(SEED + 1)
    docs_with_qa = [d for d in docs if d["qa_pairs"]]
    samples = rng.sample(docs_with_qa, min(N_FIRST_TOKEN_EXAMPLES, len(docs_with_qa)))

    # Collect top-1 predictions and aggregate top-20
    all_top1_tokens = Counter()
    all_token_probs = Counter()  # accumulate probabilities
    n_analyzed = 0
    n_truncated = 0

    # Also do a baseline: what does the model predict after "Answer:" without any story context?
    bare_prompt = "Answer:"
    bare_ids = tokenizer.encode(bare_prompt, return_tensors="pt").to(device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        bare_logits = model(bare_ids)
    bare_probs = F.softmax(bare_logits[:, -1, :].float(), dim=-1)
    bare_top20_probs, bare_top20_ids = torch.topk(bare_probs, 20, dim=-1)
    bare_tokens = [tokenizer.decode([tid]) for tid in bare_top20_ids[0].tolist()]

    print(f"\n  Baseline: Top-20 tokens after bare 'Answer:' (no context):")
    for tok, p in zip(bare_tokens, bare_top20_probs[0].tolist()):
        print(f"    {tok!r:20s} {p:.4f} ({p:.1%})")

    # Now analyze with story context
    print(f"\n  Analyzing {len(samples)} StoryQA examples...")

    for doc in samples:
        qa = doc["qa_pairs"][0]
        prompt = f'{doc["story"]}\n\nQuestion: {qa["question"]}\nAnswer:'
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        if input_ids.shape[1] > TRUNCATION_LEN:
            input_ids = input_ids[:, :TRUNCATION_LEN]
            n_truncated += 1

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
        next_probs = F.softmax(logits[:, -1, :].float(), dim=-1)

        # Top-1
        top1_id = next_probs.argmax(dim=-1).item()
        top1_tok = tokenizer.decode([top1_id])
        all_top1_tokens[top1_tok] += 1

        # Accumulate all token probabilities for top-50
        top50_probs, top50_ids = torch.topk(next_probs, 50, dim=-1)
        for tid, p in zip(top50_ids[0].tolist(), top50_probs[0].tolist()):
            tok = tokenizer.decode([tid])
            all_token_probs[tok] += p

        n_analyzed += 1

    print(f"\n  Analyzed: {n_analyzed} examples ({n_truncated} truncated to 480)")

    print(f"\n  Top-20 most common FIRST tokens predicted after 'Answer:' (across {n_analyzed} examples):")
    for tok, count in all_top1_tokens.most_common(20):
        print(f"    {tok!r:20s} appeared as top-1 in {count:4d}/{n_analyzed} examples ({100*count/n_analyzed:.1f}%)")

    print(f"\n  Top-20 tokens by cumulative probability mass after 'Answer:':")
    for tok, total_p in all_token_probs.most_common(20):
        avg_p = total_p / n_analyzed
        print(f"    {tok!r:20s} avg prob={avg_p:.4f} ({avg_p:.1%}), total mass={total_p:.2f}")

    # Entropy analysis
    print(f"\n  Entropy of first-token distribution (how uncertain is the model?):")
    entropies = []
    for doc in samples[:50]:  # use first 50 for detailed entropy
        qa = doc["qa_pairs"][0]
        prompt = f'{doc["story"]}\n\nQuestion: {qa["question"]}\nAnswer:'
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if input_ids.shape[1] > TRUNCATION_LEN:
            input_ids = input_ids[:, :TRUNCATION_LEN]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
        probs = F.softmax(logits[:, -1, :].float(), dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        entropies.append(entropy)

    entropies = np.array(entropies)
    print(f"    Mean entropy:   {entropies.mean():.2f} nats")
    print(f"    Median entropy: {np.median(entropies):.2f} nats")
    print(f"    Min entropy:    {entropies.min():.2f} nats")
    print(f"    Max entropy:    {entropies.max():.2f} nats")
    print(f"    (For reference, uniform over 50257 tokens = {np.log(50257):.2f} nats)")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("  STORYQA DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(f"  Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Parse test data
    print(f"  Parsing test data from {TEST_FILE}...")
    docs = parse_story_qa_file(TEST_FILE)
    print(f"  Parsed {len(docs)} documents")

    total_qa = sum(len(d["qa_pairs"]) for d in docs)
    print(f"  Total QA pairs: {total_qa}")

    # 1. Truncation analysis (no model needed)
    truncation_analysis(docs, tokenizer)

    # 2. Format presence in training data (no model needed)
    format_presence_analysis(tokenizer)

    # Load model for analyses 3 and 4
    print(f"\n  Loading model from {CHECKPOINT}...")
    model = create_model(variant="baseline", d_model=768, n_layers=8, n_heads=8, d_ff=2048)
    sd = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"  Model loaded. Parameters: {model.count_parameters(count_zeros=True):,}")

    # 3. Model generation analysis
    model_generation_analysis(model, docs, tokenizer, device)

    # 4. First-token analysis
    first_token_analysis(model, docs, tokenizer, device)

    print("\n" + "=" * 80)
    print("  DIAGNOSTIC ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
