"""FineWeb-edu-gpt2 Training Data Quality Analysis Dashboard."""

import os
import collections
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import tiktoken

DATASET_REPO = "kushalt/fineweb-edu-gpt2"
SUBSET = "sample-10BT_max_length_513"
SEQ_LEN = 513

# ── Load a sample ────────────────────────────────────────────────────────────

print("Loading FineWeb-edu-gpt2 sample from HuggingFace...")

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

fs = HfFileSystem()
base_path = f"datasets/{DATASET_REPO}/{SUBSET}"

# List all files
all_files = sorted(fs.ls(base_path, detail=False))
train_files = [f for f in all_files if f.endswith(".parquet") and "/train-" in f]
test_files = [f for f in all_files if f.endswith(".parquet") and "/test-" in f]

print(f"  Train parquet files: {len(train_files)}")
print(f"  Test parquet files: {len(test_files)}")

# Load samples from train (first file + random file + last file for diversity)
def load_parquet_sample(files, max_rows=10000, n_files=3):
    """Load sample rows from a selection of parquet files."""
    if not files:
        return pd.DataFrame()

    # Pick first, middle, and last file for representativeness
    indices = [0]
    if len(files) > 1:
        indices.append(len(files) // 2)
    if len(files) > 2:
        indices.append(len(files) - 1)

    selected = [files[i] for i in indices[:n_files]]
    rows_per_file = max_rows // len(selected)

    all_rows = []
    file_info = []
    for fpath in selected:
        fname = fpath.split("/")[-1]
        print(f"  Reading {fname}...")
        with fs.open(fpath, "rb") as f:
            table = pq.read_table(f)
            n = len(table)
            # Take evenly spaced sample
            if n > rows_per_file:
                step = n // rows_per_file
                indices_to_take = list(range(0, n, step))[:rows_per_file]
                table = table.take(indices_to_take)
            df = table.to_pandas()
            df["source_file"] = fname
            all_rows.append(df)
            file_info.append({"file": fname, "total_rows": n})

    combined = pd.concat(all_rows, ignore_index=True)
    return combined, file_info

print("Loading train sample...")
train_df, train_file_info = load_parquet_sample(train_files, max_rows=10000)
print(f"  Loaded {len(train_df)} train rows")

print("Loading test sample...")
test_df, test_file_info = load_parquet_sample(test_files, max_rows=5000)
print(f"  Loaded {len(test_df)} test rows")

# Setup tokenizer for decoding
enc = tiktoken.get_encoding("gpt2")

# ── Compute features ────────────────────────────────────────────────────────

def compute_features(df):
    """Add computed features to dataframe."""
    if "input_ids" not in df.columns:
        return df

    # Sequence stats
    df["seq_len"] = df["input_ids"].apply(len)

    # Padding stats
    if "pad_mask" in df.columns:
        df["n_real_tokens"] = df["pad_mask"].apply(lambda x: sum(x) if isinstance(x, list) else x.sum())
        df["n_pad_tokens"] = df["seq_len"] - df["n_real_tokens"]
        df["pad_ratio"] = df["n_pad_tokens"] / df["seq_len"]

    # Unique tokens per sequence
    df["n_unique_tokens"] = df["input_ids"].apply(lambda x: len(set(x)))

    return df

train_df = compute_features(train_df)
test_df = compute_features(test_df)

# ── Analysis Functions ───────────────────────────────────────────────────────

def length_analysis(split):
    df = train_df if split == "Train" else test_df
    file_info = train_file_info if split == "Train" else test_file_info
    files = train_files if split == "Train" else test_files

    stats = f"""DATASET OVERVIEW ({split})
═══════════════════════════
Repository: {DATASET_REPO}
Subset: {SUBSET}
Total parquet files: {len(files)}
Sequence length: {SEQ_LEN} tokens

Sample analyzed: {len(df):,} sequences
Files sampled from: {len(file_info)}
"""
    for fi in file_info:
        stats += f"  {fi['file']}: {fi['total_rows']:,} rows\n"

    if "n_real_tokens" in df.columns:
        stats += f"""
── Token counts per sequence ──
  Mean real tokens: {df['n_real_tokens'].mean():.1f}
  Median real tokens: {df['n_real_tokens'].median():.1f}
  Min real tokens: {int(df['n_real_tokens'].min())}
  Max real tokens: {int(df['n_real_tokens'].max())}
  Std: {df['n_real_tokens'].std():.1f}

── Padding ──
  Mean padding: {df['n_pad_tokens'].mean():.1f} tokens ({100*df['pad_ratio'].mean():.1f}%)
  Sequences with zero padding: {(df['n_pad_tokens'] == 0).sum():,} ({100*(df['n_pad_tokens'] == 0).mean():.1f}%)
  Sequences with >50% padding: {(df['pad_ratio'] > 0.5).sum():,} ({100*(df['pad_ratio'] > 0.5).mean():.1f}%)

── Unique tokens per sequence ──
  Mean unique tokens: {df['n_unique_tokens'].mean():.1f}
  Median unique tokens: {df['n_unique_tokens'].median():.1f}
"""

    # Estimated total tokens
    if file_info:
        avg_rows_per_file = np.mean([fi['total_rows'] for fi in file_info])
        est_total_seqs = avg_rows_per_file * len(files)
        est_total_tokens = est_total_seqs * SEQ_LEN
        if "n_real_tokens" in df.columns:
            est_real_tokens = est_total_seqs * df['n_real_tokens'].mean()
            stats += f"""
── Estimated totals ──
  Est. total sequences: {est_total_seqs:,.0f}
  Est. total tokens (incl. padding): {est_total_tokens:,.0f} ({est_total_tokens/1e9:.2f}B)
  Est. real tokens: {est_real_tokens:,.0f} ({est_real_tokens/1e9:.2f}B)
"""

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"FineWeb-edu-gpt2 – {split} Length Analysis", fontsize=14)

    if "n_real_tokens" in df.columns:
        axes[0, 0].hist(df["n_real_tokens"], bins=50, color="#4C72B0", edgecolor="white")
        axes[0, 0].set_title("Real Tokens per Sequence")
        axes[0, 0].set_xlabel("# Tokens")
        axes[0, 0].axvline(df["n_real_tokens"].mean(), color="red", linestyle="--", label=f"mean={df['n_real_tokens'].mean():.0f}")
        axes[0, 0].legend()

        axes[0, 1].hist(df["pad_ratio"], bins=50, color="#55A868", edgecolor="white")
        axes[0, 1].set_title("Padding Ratio")
        axes[0, 1].set_xlabel("Fraction padded")

        axes[1, 0].hist(df["n_unique_tokens"], bins=50, color="#C44E52", edgecolor="white")
        axes[1, 0].set_title("Unique Tokens per Sequence")
        axes[1, 0].set_xlabel("# Unique tokens")

    # Sequence length (should all be 513)
    axes[1, 1].hist(df["seq_len"], bins=50, color="#8172B2", edgecolor="white")
    axes[1, 1].set_title("Sequence Length Distribution")
    axes[1, 1].set_xlabel("Length")

    plt.tight_layout()
    return stats, fig


def quality_variety_analysis(split):
    df = train_df if split == "Train" else test_df

    # Decode a sample of sequences
    n_decode = min(500, len(df))
    sample_idx = np.random.choice(len(df), n_decode, replace=False)

    decoded_texts = []
    word_counter = collections.Counter()
    char_counter = collections.Counter()

    for i in sample_idx:
        row = df.iloc[i]
        tokens = row["input_ids"]
        if "pad_mask" in df.columns:
            mask = row["pad_mask"]
            real_tokens = [t for t, m in zip(tokens, mask) if m == 1]
        else:
            real_tokens = tokens

        text = enc.decode(real_tokens)
        decoded_texts.append(text)
        words = text.lower().split()
        word_counter.update(words)
        char_counter.update(text)

    # Analyze decoded texts
    text_lengths = [len(t) for t in decoded_texts]
    word_counts = [len(t.split()) for t in decoded_texts]

    # Language detection heuristic: ratio of common English words
    common_english = {"the", "a", "an", "is", "was", "are", "to", "in", "of", "and", "that", "it", "for", "on", "with"}
    english_ratios = []
    for t in decoded_texts:
        words = t.lower().split()
        if words:
            ratio = sum(1 for w in words if w in common_english) / len(words)
            english_ratios.append(ratio)

    # Topic diversity: unique starting tokens
    start_tokens = collections.Counter()
    for t in decoded_texts:
        first_word = t.split()[0] if t.split() else ""
        start_tokens[first_word] += 1

    vocab_size = len(word_counter)
    total_words = sum(word_counter.values())

    # Content word analysis
    stopwords = {"the", "a", "an", "is", "was", "are", "were", "to", "in", "on", "of",
                 "and", "it", "he", "she", "they", "his", "her", "with", "at", "for",
                 "that", "this", "had", "has", "have", "not", "but", "or", "be", "by",
                 "as", "do", "did", "from", "one", "all", "can", "will", "up",
                 "out", "about", "so", "no", "if", "we", "you", "its",
                 "their", "there", "than", "been", "would", "could",
                 "into", "more", "which", "when", "what", "who", "how"}
    content_words = {w: c for w, c in word_counter.items() if w not in stopwords and len(w) > 2 and w.isalpha()}
    top_content = sorted(content_words.items(), key=lambda x: -x[1])[:20]

    report = f"""QUALITY & VARIETY ({split}, decoded {n_decode} sequences)
══════════════════════════════════════════════
Vocabulary (in decoded sample):
  Unique words: {vocab_size:,}
  Total words: {total_words:,}
  Type-token ratio: {vocab_size/max(total_words,1):.4f}

Decoded text lengths:
  Mean chars: {np.mean(text_lengths):.1f}
  Mean words: {np.mean(word_counts):.1f}
  Median words: {np.median(word_counts):.1f}

English content ratio (common words):
  Mean: {np.mean(english_ratios):.3f}
  Min: {np.min(english_ratios):.3f}
  Texts with ratio < 0.05 (likely non-English): {sum(1 for r in english_ratios if r < 0.05)} ({100*sum(1 for r in english_ratios if r < 0.05)/len(english_ratios):.1f}%)

Unique first words: {len(start_tokens):,} / {n_decode}

Top 20 content words:
{chr(10).join(f"  {w}: {c}" for w, c in top_content)}
"""

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"FineWeb-edu-gpt2 – {split} Quality & Variety", fontsize=14)

    axes[0].hist(word_counts, bins=50, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Decoded Text Length (words)")
    axes[0].set_xlabel("Words")

    axes[1].hist(english_ratios, bins=30, color="#55A868", edgecolor="white")
    axes[1].set_title("English Content Ratio")
    axes[1].set_xlabel("Ratio of common English words")

    tc_labels = [w for w, _ in top_content[:15]]
    tc_counts = [c for _, c in top_content[:15]]
    axes[2].barh(tc_labels[::-1], tc_counts[::-1], color="#C44E52")
    axes[2].set_title("Top Content Words")
    axes[2].set_xlabel("Frequency")

    plt.tight_layout()
    return report, fig


def cleanliness_analysis(split):
    df = train_df if split == "Train" else test_df

    issues = {
        "wrong_seq_len": 0,
        "all_padding": 0,
        "mostly_padding": 0,
        "no_padding": 0,
        "has_negative_tokens": 0,
        "very_few_real_tokens": 0,
    }

    for _, row in df.iterrows():
        tokens = row["input_ids"]

        # Wrong sequence length
        if len(tokens) != SEQ_LEN:
            issues["wrong_seq_len"] += 1

        if "pad_mask" in df.columns:
            mask = row["pad_mask"]
            n_real = sum(mask) if isinstance(mask, list) else mask.sum()

            if n_real == 0:
                issues["all_padding"] += 1
            elif n_real < 10:
                issues["very_few_real_tokens"] += 1
            elif n_real / len(tokens) < 0.3:
                issues["mostly_padding"] += 1
            elif n_real == len(tokens):
                issues["no_padding"] += 1

        # Negative token IDs
        if any(t < 0 for t in tokens):
            issues["has_negative_tokens"] += 1

    n = len(df)
    report = f"""CLEANLINESS REPORT ({split}, n={n:,} sampled)
═══════════════════════════════════════════
Wrong sequence length (≠{SEQ_LEN}): {issues['wrong_seq_len']} ({100*issues['wrong_seq_len']/n:.2f}%)
All-padding sequences: {issues['all_padding']} ({100*issues['all_padding']/n:.2f}%)
Mostly padding (>70%): {issues['mostly_padding']} ({100*issues['mostly_padding']/n:.2f}%)
No padding (full sequences): {issues['no_padding']} ({100*issues['no_padding']/n:.2f}%)
Very few real tokens (<10): {issues['very_few_real_tokens']} ({100*issues['very_few_real_tokens']/n:.2f}%)
Negative token IDs: {issues['has_negative_tokens']} ({100*issues['has_negative_tokens']/n:.2f}%)

OVERALL: {'Clean ✓' if issues['wrong_seq_len'] == 0 and issues['all_padding'] == 0 and issues['has_negative_tokens'] == 0 else 'Issues found'}
"""

    # Decode some samples to check for garbled text
    n_check = min(100, len(df))
    garbled = 0
    for i in range(n_check):
        row = df.iloc[i]
        tokens = row["input_ids"]
        if "pad_mask" in df.columns:
            mask = row["pad_mask"]
            real_tokens = [t for t, m in zip(tokens, mask) if m == 1]
        else:
            real_tokens = tokens
        text = enc.decode(real_tokens)
        # Heuristic: lots of replacement chars or very high non-printable ratio
        non_printable = sum(1 for c in text if not c.isprintable() and c not in '\n\r\t')
        if non_printable > len(text) * 0.1:
            garbled += 1

    report += f"\nDecoded text quality ({n_check} checked):\n"
    report += f"  Garbled/non-printable heavy: {garbled} ({100*garbled/n_check:.1f}%)\n"

    # Token ID distribution
    all_tokens = []
    for _, row in df.head(1000).iterrows():
        all_tokens.extend(row["input_ids"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"FineWeb-edu-gpt2 – {split} Cleanliness", fontsize=14)

    # Token ID distribution
    axes[0].hist(all_tokens, bins=100, color="#4C72B0", edgecolor="white", log=True)
    axes[0].set_title("Token ID Distribution (log scale)")
    axes[0].set_xlabel("Token ID")

    # Issue counts
    issue_labels = [k.replace("_", " ").title() for k in issues]
    issue_counts = list(issues.values())
    colors = ["#C44E52" if c > 0 else "#55A868" for c in issue_counts]
    axes[1].barh(issue_labels[::-1], issue_counts[::-1], color=colors[::-1])
    axes[1].set_title("Issue Counts")

    # Padding distribution
    if "pad_ratio" in df.columns:
        axes[2].hist(df["pad_ratio"], bins=50, color="#8172B2", edgecolor="white")
        axes[2].set_title("Padding Ratio Distribution")
        axes[2].set_xlabel("Fraction padded")

    plt.tight_layout()
    return report, fig


def randomness_analysis(split):
    df = train_df if split == "Train" else test_df

    # For streaming data, "randomness" = are the sampled files representative?
    # Check token distribution across sampled files

    file_stats = []
    for fname, group in df.groupby("source_file"):
        n_real_mean = group["n_real_tokens"].mean() if "n_real_tokens" in group.columns else 0
        n_real_std = group["n_real_tokens"].std() if "n_real_tokens" in group.columns else 0
        unique_mean = group["n_unique_tokens"].mean()
        file_stats.append({
            "file": fname,
            "n_rows": len(group),
            "mean_real_tokens": n_real_mean,
            "std_real_tokens": n_real_std,
            "mean_unique_tokens": unique_mean,
        })

    # Cross-file consistency
    report = f"""RANDOMNESS / SAMPLING ANALYSIS ({split})
══════════════════════════════════════════
This dataset is streamed from HuggingFace parquet files.
We check consistency across sampled files.

Files analyzed:
"""
    for fs_info in file_stats:
        report += f"  {fs_info['file']}: {fs_info['n_rows']} rows, "
        report += f"mean real tokens={fs_info['mean_real_tokens']:.1f} (±{fs_info['std_real_tokens']:.1f}), "
        report += f"mean unique={fs_info['mean_unique_tokens']:.1f}\n"

    # Check if first tokens vary (not stuck on same starting point)
    if "input_ids" in df.columns:
        first_tokens = df["input_ids"].apply(lambda x: x[0] if len(x) > 0 else -1)
        n_unique_first = first_tokens.nunique()
        most_common_first = first_tokens.value_counts().head(5)

        report += f"""
First-token diversity:
  Unique first tokens: {n_unique_first:,} / {len(df):,}
  Most common first tokens:
"""
        for tok, cnt in most_common_first.items():
            decoded = enc.decode([tok]) if tok >= 0 else "N/A"
            report += f"    Token {tok} ({repr(decoded)}): {cnt} ({100*cnt/len(df):.1f}%)\n"

    # Statistical test: are samples from different files drawn from same distribution?
    if len(file_stats) >= 2 and "n_real_tokens" in df.columns:
        from scipy import stats as scipy_stats
        groups = [group["n_real_tokens"].values for _, group in df.groupby("source_file")]
        if len(groups) >= 2:
            # Kruskal-Wallis test
            stat, pval = scipy_stats.kruskal(*groups)
            report += f"""
Cross-file consistency (Kruskal-Wallis test on real token counts):
  H-statistic: {stat:.4f}
  p-value: {pval:.6f}
  Interpretation: {"Files are consistent ✓" if pval > 0.05 else "Files differ significantly ✗ (expected for different content)"}
"""

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"FineWeb-edu-gpt2 – {split} Sampling Analysis", fontsize=14)

    # Real tokens by file
    if "n_real_tokens" in df.columns:
        for fname, group in df.groupby("source_file"):
            axes[0].hist(group["n_real_tokens"], bins=30, alpha=0.5, label=fname.split("-")[1][:5])
        axes[0].set_title("Real Tokens by Source File")
        axes[0].set_xlabel("# Real tokens")
        axes[0].legend(fontsize=8)

    # First token distribution
    if "input_ids" in df.columns:
        first_toks = df["input_ids"].apply(lambda x: x[0] if len(x) > 0 else -1)
        axes[1].hist(first_toks, bins=50, color="#55A868", edgecolor="white")
        axes[1].set_title("First Token Distribution")
        axes[1].set_xlabel("Token ID")

    # Unique tokens by position in file
    if "n_unique_tokens" in df.columns:
        axes[2].scatter(range(len(df)), df["n_unique_tokens"], s=2, alpha=0.3, color="#C44E52")
        axes[2].set_title("Unique Tokens vs Row Position")
        axes[2].set_xlabel("Row index")
        axes[2].set_ylabel("# Unique tokens")

    plt.tight_layout()
    return report, fig


def sample_texts(split, n=5):
    """Decode and display random samples."""
    df = train_df if split == "Train" else test_df
    sample_idx = random.sample(range(len(df)), min(n, len(df)))

    output = ""
    for rank, i in enumerate(sample_idx, 1):
        row = df.iloc[i]
        tokens = row["input_ids"]
        if "pad_mask" in df.columns:
            mask = row["pad_mask"]
            real_tokens = [t for t, m in zip(tokens, mask) if m == 1]
        else:
            real_tokens = tokens

        text = enc.decode(real_tokens)
        output += f"{'='*60}\n"
        output += f"SAMPLE {rank} | Row {i} | File: {row.get('source_file', '?')}\n"
        output += f"Real tokens: {len(real_tokens)} | Total: {len(tokens)} | Unique: {row['n_unique_tokens']}\n"
        output += f"{'='*60}\n"
        output += text[:2000]  # Cap display
        if len(text) > 2000:
            output += "\n... [truncated] ..."
        output += "\n\n"
    return output


# ── Gradio App ───────────────────────────────────────────────────────────────

with gr.Blocks(title="FineWeb-edu-gpt2 Data Quality Dashboard") as app:
    gr.Markdown("# FineWeb-edu-gpt2 Training Data Quality Dashboard")
    gr.Markdown(f"**Repo:** {DATASET_REPO} | **Subset:** {SUBSET}")
    gr.Markdown(f"**Train sample:** {len(train_df):,} seqs | **Test sample:** {len(test_df):,} seqs")

    with gr.Tabs():
        with gr.Tab("1. Length Analysis"):
            split1 = gr.Radio(["Train", "Test"], value="Train", label="Split")
            btn1 = gr.Button("Analyze Lengths", variant="primary")
            stats1 = gr.Textbox(label="Statistics", lines=30)
            plot1 = gr.Plot(label="Distributions")
            btn1.click(length_analysis, inputs=split1, outputs=[stats1, plot1])

        with gr.Tab("2. Quality & Variety"):
            split2 = gr.Radio(["Train", "Test"], value="Train", label="Split")
            btn2 = gr.Button("Analyze Quality", variant="primary")
            stats2 = gr.Textbox(label="Quality Report", lines=30)
            plot2 = gr.Plot(label="Variety Plots")
            btn2.click(quality_variety_analysis, inputs=split2, outputs=[stats2, plot2])

        with gr.Tab("3. Cleanliness"):
            split3 = gr.Radio(["Train", "Test"], value="Train", label="Split")
            btn3 = gr.Button("Check Cleanliness", variant="primary")
            stats3 = gr.Textbox(label="Cleanliness Report", lines=20)
            plot3 = gr.Plot(label="Issue Visualization")
            btn3.click(cleanliness_analysis, inputs=split3, outputs=[stats3, plot3])

        with gr.Tab("4. Sampling Analysis"):
            split4 = gr.Radio(["Train", "Test"], value="Train", label="Split")
            btn4 = gr.Button("Check Sampling", variant="primary")
            stats4 = gr.Textbox(label="Sampling Report", lines=25)
            plot4 = gr.Plot(label="Sampling Plots")
            btn4.click(randomness_analysis, inputs=split4, outputs=[stats4, plot4])

        with gr.Tab("5. Sample Texts"):
            split5 = gr.Radio(["Train", "Test"], value="Train", label="Split")
            n_samples = gr.Slider(1, 20, value=5, step=1, label="Number of samples")
            btn5 = gr.Button("Show Random Samples", variant="primary")
            samples_out = gr.Textbox(label="Decoded Samples", lines=40)
            btn5.click(sample_texts, inputs=[split5, n_samples], outputs=samples_out)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7863, share=True)
