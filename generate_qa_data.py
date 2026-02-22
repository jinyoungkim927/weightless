#!/usr/bin/env python3
"""Generate QA-formatted training data from FineWeb-edu passages using Claude.

Reads educational passages from the FineWeb-edu dataset, sends them to Claude
to generate question-answer pairs, then tokenizes and saves as parquet files
with the same schema as the original dataset (input_ids, pad_mask).

Usage:
    python generate_qa_data.py --num_passages 5000
    python generate_qa_data.py --num_passages 5 --dry_run
"""

import argparse
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import GPT2Tokenizer


MAX_SEQ_LEN = 513  # Same as FineWeb-edu tokenized format


def load_passages(num_passages: int, seed: int = 42) -> list[str]:
    """Load raw text passages from FineWeb-edu via streaming."""
    from data import StreamingParquetDataset

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = StreamingParquetDataset(split="train", shuffle=True)

    passages = []
    for row in dataset:
        input_ids = row["input_ids"]
        # Decode to text, stripping padding
        pad_mask = row["pad_mask"]
        valid_ids = [tid for tid, m in zip(input_ids, pad_mask) if m == 1]
        text = tokenizer.decode(valid_ids)
        if len(text.strip()) > 100:  # Skip very short passages
            passages.append(text.strip())
        if len(passages) >= num_passages:
            break

    return passages


def generate_qa_pairs(passages: list[str], model: str = "claude-haiku-4-5-20251001",
                      batch_size: int = 5) -> list[str]:
    """Use Claude API to generate QA pairs from passages."""
    import anthropic

    client = anthropic.Anthropic()
    qa_texts = []

    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        for passage in batch:
            # Truncate very long passages
            if len(passage) > 2000:
                passage = passage[:2000]

            prompt = f"""Given this educational passage, generate 2-3 question-answer pairs that test understanding of the key concepts.

Format each pair exactly as:
Question: <question>
Answer: <answer>

Passage:
{passage}

Generate the question-answer pairs:"""

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                qa_text = response.content[0].text.strip()

                # Combine passage + QA into training text
                combined = f"{passage}\n\n{qa_text}"
                qa_texts.append(combined)
            except Exception as e:
                print(f"  Warning: API call failed: {e}")
                continue

        print(f"  Generated QA for {min(i + batch_size, len(passages))}/{len(passages)} passages")

    return qa_texts


def tokenize_and_save(qa_texts: list[str], output_dir: str, max_seq_len: int = MAX_SEQ_LEN):
    """Tokenize QA texts and save as parquet with same schema as FineWeb."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    pad_token_id = tokenizer.eos_token_id  # 50256

    all_input_ids = []
    all_pad_masks = []

    for text in qa_texts:
        ids = tokenizer.encode(text)

        # Truncate or pad to max_seq_len
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
            mask = [1] * max_seq_len
        else:
            mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))
            ids = ids + [pad_token_id] * (max_seq_len - len(ids))

        all_input_ids.append(ids)
        all_pad_masks.append(mask)

    # Save as parquet
    os.makedirs(output_dir, exist_ok=True)
    table = pa.table({
        "input_ids": all_input_ids,
        "pad_mask": all_pad_masks,
    })

    output_path = os.path.join(output_dir, "qa_augmented-00000-of-00001.parquet")
    pq.write_table(table, output_path)
    print(f"  Saved {len(qa_texts)} QA samples to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate QA training data from FineWeb passages")
    parser.add_argument("--num_passages", type=int, default=5000,
                        help="Number of passages to process")
    parser.add_argument("--output_dir", type=str, default="data/qa_augmented",
                        help="Output directory for parquet files")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
                        help="Claude model to use for QA generation")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Number of passages per batch")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only generate a few samples and print them")
    args = parser.parse_args()

    print("  Loading passages from FineWeb-edu...")
    num = 5 if args.dry_run else args.num_passages
    passages = load_passages(num)
    print(f"  Loaded {len(passages)} passages")

    if args.dry_run:
        print("\n  [DRY RUN] First passage preview:")
        print(f"  {passages[0][:200]}...")
        print("\n  Generating QA pairs...")

    qa_texts = generate_qa_pairs(passages, model=args.model, batch_size=args.batch_size)
    print(f"  Generated {len(qa_texts)} QA-augmented texts")

    if args.dry_run:
        print("\n  [DRY RUN] Sample QA text:")
        if qa_texts:
            print(f"  {qa_texts[0][:500]}...")
        return

    tokenize_and_save(qa_texts, args.output_dir)
    print("  Done!")


if __name__ == "__main__":
    main()
