"""Analyze FineWeb-edu-gpt2 training data for content patterns."""

import re
import tiktoken
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from collections import Counter

# Same constants as data.py
DATASET_REPO = "kushalt/fineweb-edu-gpt2"
SUBSET = "sample-10BT_max_length_513"


def get_parquet_files(split: str = "train"):
    """Get list of parquet file URLs for a split."""
    fs = HfFileSystem()
    path = f"datasets/{DATASET_REPO}/{SUBSET}"
    files = fs.ls(path, detail=False)
    split_files = [f for f in files if f.endswith(".parquet") and f"/{split}-" in f]
    return sorted(split_files)


def classify_content(text):
    """Classify the type of content in decoded text."""
    text_lower = text.lower().strip()

    # Check for code
    code_indicators = [
        'def ', 'class ', 'import ', 'function ', 'var ', 'const ', 'let ',
        'public ', 'private ', 'static ', '#include', '<?php', '<html',
        'console.log', 'print(', 'return ', 'if (', 'for (', 'while (',
        '#!/', '{', '}', '//', '/*', '*/', 'int main',
    ]
    code_score = sum(1 for ind in code_indicators if ind in text_lower)

    # Check for lists (numbered or bulleted)
    list_lines = len(re.findall(r'^\s*[\d]+[\.\)]\s', text, re.MULTILINE))
    list_lines += len(re.findall(r'^\s*[-\*\u2022]\s', text, re.MULTILINE))

    # Check for Q&A patterns
    qa_patterns = [
        r'\bquestion\s*[:]\s', r'\banswer\s*[:]\s',
        r'\bq\s*[:]\s', r'\ba\s*[:]\s',
        r'\bwhat is\b', r'\bwhat are\b', r'\bhow does\b', r'\bhow do\b',
        r'\bhow can\b', r'\bhow to\b', r'\bwhy does\b', r'\bwhy do\b',
        r'\bwhy is\b', r'\bwhen does\b', r'\bwhen is\b',
        r'\bwhere is\b', r'\bwhere are\b', r'\bwhere does\b',
        r'\bwho is\b', r'\bwho are\b', r'\bwho was\b',
        r'\bwhich is\b', r'\bwhich are\b',
        r'\bexplain\b', r'\bdescribe\b', r'\bdefine\b',
    ]
    qa_score = sum(1 for pat in qa_patterns if re.search(pat, text_lower))

    # Structured Q&A (explicit question/answer format)
    has_explicit_qa = bool(
        re.search(r'(?:question|q)\s*[:]\s', text_lower) and
        re.search(r'(?:answer|a)\s*[:]\s', text_lower)
    )

    # Check for headers / structured content
    headers = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
    headers += len(re.findall(r'^[A-Z][A-Za-z\s]{3,50}$', text, re.MULTILINE))

    # Check for tables
    table_lines = len(re.findall(r'\|.*\|', text))

    # Classify
    total_lines = max(len(text.split('\n')), 1)

    if has_explicit_qa:
        return "explicit_qa"
    elif code_score >= 4:
        return "code"
    elif table_lines >= 3:
        return "table"
    elif list_lines >= 3:
        return "list_heavy"
    elif qa_score >= 3:
        return "qa_style"
    elif headers >= 2:
        return "structured_article"
    else:
        return "paragraph_prose"


def main():
    enc = tiktoken.get_encoding("gpt2")
    fs = HfFileSystem()

    # Get first 3 training parquet files
    all_files = get_parquet_files("train")
    print(f"Total training parquet files: {len(all_files)}")
    files_to_check = all_files[:3]
    print(f"Checking first {len(files_to_check)} files:")
    for f in files_to_check:
        print(f"  {f}")
    print()

    # Patterns for Q&A detection
    qa_regex = re.compile(
        r'(?:'
        r'question\s*[:\-]\s|answer\s*[:\-]\s'
        r'|^q\s*[:\-]\s|^a\s*[:\-]\s'
        r'|\bq\d*\s*[:\.\)]\s|\ba\d*\s*[:\.\)]\s'
        r'|what is\b|what are\b|what was\b|what were\b'
        r'|how does\b|how do\b|how can\b|how to\b|how is\b'
        r'|why does\b|why do\b|why is\b|why are\b'
        r'|when does\b|when is\b|when was\b'
        r'|where is\b|where are\b|where does\b'
        r'|who is\b|who are\b|who was\b'
        r'|which is\b|which are\b'
        r'|explain\s+(?:how|why|what|the)\b'
        r'|describe\s+(?:how|the|a)\b'
        r'|define\s+(?:the|a)\b'
        r')',
        re.IGNORECASE | re.MULTILINE
    )

    explicit_qa_regex = re.compile(
        r'(?:question|q)\s*[:\-]\s.*?(?:answer|a)\s*[:\-]\s',
        re.IGNORECASE | re.DOTALL
    )

    total_sequences = 0
    qa_sequences = 0
    explicit_qa_sequences = 0

    qa_examples = []       # sequences with Q&A patterns
    non_qa_examples = []   # sequences without Q&A patterns
    content_types = Counter()

    # Track specific pattern frequencies
    pattern_hits = Counter()

    for file_path in files_to_check:
        print(f"Reading {file_path.split('/')[-1]}...")
        with fs.open(file_path, "rb") as f:
            table = pq.read_table(f)

        num_rows = len(table)
        print(f"  Rows: {num_rows}")

        for i in range(num_rows):
            input_ids = table["input_ids"][i].as_py()
            pad_mask = table["pad_mask"][i].as_py()

            # Decode only non-padded tokens
            non_pad_ids = [t for t, m in zip(input_ids, pad_mask) if m == 1]
            try:
                text = enc.decode(non_pad_ids)
            except Exception:
                text = enc.decode(non_pad_ids, errors="replace")

            total_sequences += 1

            # Check for Q&A patterns
            matches = qa_regex.findall(text)
            has_qa = len(matches) > 0
            has_explicit_qa = bool(explicit_qa_regex.search(text))

            if has_qa:
                qa_sequences += 1
                for m in matches:
                    # Normalize
                    pattern_hits[m.strip().lower()[:30]] += 1

                if len(qa_examples) < 10:
                    qa_examples.append(text)
            else:
                if len(non_qa_examples) < 10:
                    non_qa_examples.append(text)

            if has_explicit_qa:
                explicit_qa_sequences += 1

            # Classify content type
            ctype = classify_content(text)
            content_types[ctype] += 1

    # === REPORT ===
    print("\n" + "=" * 80)
    print("ANALYSIS REPORT")
    print("=" * 80)

    print(f"\nTotal sequences checked: {total_sequences}")
    print(f"Sequences with Q&A-like patterns: {qa_sequences} ({100*qa_sequences/total_sequences:.1f}%)")
    print(f"Sequences with EXPLICIT Q&A format (Question:/Answer:): {explicit_qa_sequences} ({100*explicit_qa_sequences/total_sequences:.1f}%)")
    print(f"Sequences with NO Q&A patterns: {total_sequences - qa_sequences} ({100*(total_sequences-qa_sequences)/total_sequences:.1f}%)")

    print("\n" + "-" * 80)
    print("TOP 30 MOST COMMON Q&A-LIKE PATTERN MATCHES:")
    print("-" * 80)
    for pattern, count in pattern_hits.most_common(30):
        print(f"  {count:6d}x  '{pattern}'")

    print("\n" + "-" * 80)
    print("CONTENT TYPE BREAKDOWN:")
    print("-" * 80)
    for ctype, count in content_types.most_common():
        print(f"  {ctype:25s}: {count:6d} ({100*count/total_sequences:.1f}%)")

    print("\n" + "-" * 80)
    print("10 EXAMPLE SEQUENCES WITH Q&A-LIKE PATTERNS:")
    print("-" * 80)
    for idx, text in enumerate(qa_examples):
        print(f"\n--- Example {idx+1} ---")
        print(text[:500])
        if len(text) > 500:
            print(f"  ... [{len(text)} total chars]")

    print("\n" + "-" * 80)
    print("10 EXAMPLE SEQUENCES WITHOUT Q&A PATTERNS:")
    print("-" * 80)
    for idx, text in enumerate(non_qa_examples):
        print(f"\n--- Example {idx+1} ---")
        print(text[:300])
        if len(text) > 300:
            print(f"  ... [{len(text)} total chars]")


if __name__ == "__main__":
    main()
