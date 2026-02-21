"""StoryQA v4 Data Quality Analysis Dashboard."""

import os
import re
import glob
import json
import collections
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

DATA_DIR = "/root/weightless/story_qa_v4_plaintext_shards"

# ── Parsing ──────────────────────────────────────────────────────────────────

DOC_RE = re.compile(
    r"===== DOC START split=(\w+) idx=(\d+) id=(story_qa_\d+) =====\n"
    r"(.*?)"
    r"===== DOC END =====",
    re.DOTALL,
)
QUESTION_RE = re.compile(r"<question>(.*?)</question>")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>")
DEFINITION_RE = re.compile(r"<definition>(.*?)</definition>")


def parse_all_docs(split="train"):
    """Parse every document from the plaintext shards."""
    if split == "train":
        shard_pattern = os.path.join(DATA_DIR, "train", "train_shard_*.txt")
    else:
        shard_pattern = os.path.join(DATA_DIR, "test", "test_shard_*.txt")

    docs = []
    for path in sorted(glob.glob(shard_pattern)):
        text = open(path, encoding="utf-8").read()
        for m in DOC_RE.finditer(text):
            sp, idx, doc_id, body = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            questions = QUESTION_RE.findall(body)
            answers = ANSWER_RE.findall(body)
            definitions = DEFINITION_RE.findall(body)

            # Extract story (text before first <question> tag)
            first_q_pos = body.find("<question>")
            story = body[:first_q_pos].strip() if first_q_pos != -1 else body.strip()
            # Remove definition tags from story for clean text
            story_clean = re.sub(r"<definition>.*?</definition>", "", story).strip()

            # Numeric id from doc_id
            numeric_id = int(doc_id.split("_")[-1])

            docs.append({
                "split": sp,
                "idx": idx,
                "doc_id": doc_id,
                "numeric_id": numeric_id,
                "body": body.strip(),
                "story": story,
                "story_clean": story_clean,
                "questions": questions,
                "answers": answers,
                "definitions": definitions,
                "n_questions": len(questions),
                "n_definitions": len(definitions),
                "body_chars": len(body.strip()),
                "story_chars": len(story_clean),
                "story_words": len(story_clean.split()),
                "story_sentences": len([s for s in re.split(r'[.!?]+', story_clean) if s.strip()]),
                "total_words": len(body.strip().split()),
            })
    return docs


print("Parsing StoryQA v4 documents...")
train_docs = parse_all_docs("train")
test_docs = parse_all_docs("test")
all_docs = train_docs + test_docs
print(f"  Train: {len(train_docs)} docs, Test: {len(test_docs)} docs")

train_df = pd.DataFrame(train_docs)
test_df = pd.DataFrame(test_docs)

# ── Analysis Functions ───────────────────────────────────────────────────────

def length_analysis(split):
    df = train_df if split == "Train" else test_df
    docs = train_docs if split == "Train" else test_docs

    stats = {
        "Total documents": len(df),
        "Total characters (all bodies)": int(df["body_chars"].sum()),
        "Total words (all bodies)": int(df["total_words"].sum()),
        "": "",
        "── Story length (chars) ──": "",
        "  Mean": f"{df['story_chars'].mean():.1f}",
        "  Median": f"{df['story_chars'].median():.1f}",
        "  Min": int(df["story_chars"].min()),
        "  Max": int(df["story_chars"].max()),
        "  Std": f"{df['story_chars'].std():.1f}",
        " ": "",
        "── Story length (words) ──": "",
        "  Mean ": f"{df['story_words'].mean():.1f}",
        "  Median ": f"{df['story_words'].median():.1f}",
        "  Min ": int(df["story_words"].min()),
        "  Max ": int(df["story_words"].max()),
        "  ": "",
        "── Questions per doc ──": "",
        "  Mean  ": f"{df['n_questions'].mean():.2f}",
        "  Median  ": f"{df['n_questions'].median():.1f}",
        "  Min  ": int(df["n_questions"].min()),
        "  Max  ": int(df["n_questions"].max()),
        "   ": "",
        "── Definitions per doc ──": "",
        "  Mean   ": f"{df['n_definitions'].mean():.2f}",
        "  Docs with definitions": f"{(df['n_definitions'] > 0).sum()} ({100*(df['n_definitions'] > 0).mean():.1f}%)",
    }
    stats_text = "\n".join(f"{k}: {v}" if v != "" else "" for k, v in stats.items())

    # Histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"StoryQA v4 – {split} Length Distributions", fontsize=14)

    axes[0, 0].hist(df["story_words"], bins=50, color="#4C72B0", edgecolor="white")
    axes[0, 0].set_title("Story Length (words)")
    axes[0, 0].set_xlabel("Words")
    axes[0, 0].axvline(df["story_words"].mean(), color="red", linestyle="--", label=f"mean={df['story_words'].mean():.1f}")
    axes[0, 0].legend()

    axes[0, 1].hist(df["story_chars"], bins=50, color="#55A868", edgecolor="white")
    axes[0, 1].set_title("Story Length (characters)")
    axes[0, 1].set_xlabel("Characters")

    axes[1, 0].hist(df["n_questions"], bins=range(0, df["n_questions"].max() + 2), color="#C44E52", edgecolor="white", align="left")
    axes[1, 0].set_title("Questions per Document")
    axes[1, 0].set_xlabel("# Questions")

    axes[1, 1].hist(df["body_chars"], bins=50, color="#8172B2", edgecolor="white")
    axes[1, 1].set_title("Total Doc Length (chars, incl. Q&A)")
    axes[1, 1].set_xlabel("Characters")

    plt.tight_layout()
    return stats_text, fig


def quality_variety_analysis(split):
    df = train_df if split == "Train" else test_df
    docs = train_docs if split == "Train" else test_docs

    # Vocabulary analysis
    all_story_words = []
    for d in docs:
        all_story_words.extend(re.findall(r'\b\w+\b', d["story_clean"].lower()))

    word_freq = collections.Counter(all_story_words)
    vocab_size = len(word_freq)
    total_tokens = len(all_story_words)

    # Question first-word distribution
    q_first_words = []
    for d in docs:
        for q in d["questions"]:
            w = q.split()[0] if q.split() else "?"
            q_first_words.append(w)
    q_first_dist = collections.Counter(q_first_words).most_common(15)

    # Topic words (nouns/subjects – rough heuristic: capitalize words or common nouns)
    # Use most common non-stopword words as proxy for topics
    stopwords = {"the", "a", "an", "is", "was", "are", "were", "to", "in", "on", "of",
                 "and", "it", "he", "she", "they", "his", "her", "with", "at", "for",
                 "that", "this", "had", "has", "have", "not", "but", "or", "be", "by",
                 "as", "do", "did", "from", "one", "two", "all", "can", "will", "up",
                 "out", "about", "so", "no", "if", "my", "your", "its", "we", "you",
                 "him", "them", "their", "what", "who", "how", "when", "where", "which",
                 "there", "then", "than", "very", "just", "like", "also", "some", "any",
                 "been", "being", "would", "could", "should", "may", "might", "shall",
                 "into", "over", "after", "before", "down", "too", "more", "much", "many",
                 "other", "each", "every", "own", "such", "only", "same", "because",
                 "went", "got", "said", "see", "saw", "go", "come", "came", "make",
                 "made", "take", "took", "get", "give", "gave", "put", "let", "say",
                 "tell", "told", "know", "think", "look", "looked", "want", "wanted",
                 "day", "time", "way"}
    content_words = {w: c for w, c in word_freq.items() if w not in stopwords and len(w) > 2}
    top_content = sorted(content_words.items(), key=lambda x: -x[1])[:30]

    # Unique story starts (first 5 words)
    story_starts = collections.Counter()
    for d in docs:
        first5 = " ".join(d["story_clean"].split()[:5])
        story_starts[first5] += 1
    duplicate_starts = sum(1 for c in story_starts.values() if c > 1)

    # Definition words
    all_defined = []
    for d in docs:
        for defn in d["definitions"]:
            # Usually "X means Y" pattern
            word = defn.split()[0] if defn.split() else ""
            all_defined.append(word.lower())
    defined_freq = collections.Counter(all_defined).most_common(20)

    stats = f"""VOCABULARY & VARIETY ({split})
═══════════════════════════════
Vocabulary size (unique words): {vocab_size:,}
Total story tokens: {total_tokens:,}
Type-token ratio: {vocab_size/max(total_tokens,1):.4f}
Hapax legomena (words appearing once): {sum(1 for c in word_freq.values() if c == 1):,}

Top 15 question starters:
{chr(10).join(f"  {w}: {c}" for w, c in q_first_dist)}

Top 20 content words (stories):
{chr(10).join(f"  {w}: {c}" for w, c in top_content[:20])}

Story start diversity:
  Unique first-5-word starts: {len(story_starts):,} / {len(docs):,}
  Starts appearing >1 time: {duplicate_starts}

Defined vocabulary words (top 20):
{chr(10).join(f"  {w}: {c}" for w, c in defined_freq)}
"""

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"StoryQA v4 – {split} Quality & Variety", fontsize=14)

    # Question type distribution
    q_labels = [w for w, _ in q_first_dist[:10]]
    q_counts = [c for _, c in q_first_dist[:10]]
    axes[0].barh(q_labels[::-1], q_counts[::-1], color="#4C72B0")
    axes[0].set_title("Question Types (first word)")
    axes[0].set_xlabel("Count")

    # Content word cloud (bar chart)
    tc_labels = [w for w, _ in top_content[:15]]
    tc_counts = [c for _, c in top_content[:15]]
    axes[1].barh(tc_labels[::-1], tc_counts[::-1], color="#55A868")
    axes[1].set_title("Top Content Words")
    axes[1].set_xlabel("Frequency")

    # Story length vs #questions scatter
    axes[2].scatter(df["story_words"], df["n_questions"], alpha=0.1, s=5, color="#C44E52")
    axes[2].set_title("Story Length vs # Questions")
    axes[2].set_xlabel("Story words")
    axes[2].set_ylabel("# Questions")

    plt.tight_layout()
    return stats, fig


def cleanliness_analysis(split):
    df = train_df if split == "Train" else test_df
    docs = train_docs if split == "Train" else test_docs

    issues = {
        "empty_story": 0,
        "empty_questions": 0,
        "mismatched_qa": 0,
        "unmatched_tags": 0,
        "non_ascii": 0,
        "very_short_story": 0,
        "very_long_story": 0,
        "duplicate_doc_ids": 0,
        "definition_outside_story": 0,
        "question_in_story": 0,
    }

    examples = {k: [] for k in issues}

    seen_ids = set()
    for d in docs:
        # Empty story
        if len(d["story_clean"].strip()) == 0:
            issues["empty_story"] += 1
            examples["empty_story"].append(d["doc_id"])

        # No questions
        if d["n_questions"] == 0:
            issues["empty_questions"] += 1
            examples["empty_questions"].append(d["doc_id"])

        # Mismatched Q&A count
        if len(d["questions"]) != len(d["answers"]):
            issues["mismatched_qa"] += 1
            examples["mismatched_qa"].append(f"{d['doc_id']}: {len(d['questions'])}Q vs {len(d['answers'])}A")

        # Unmatched/broken tags
        body = d["body"]
        open_q = body.count("<question>")
        close_q = body.count("</question>")
        open_a = body.count("<answer>")
        close_a = body.count("</answer>")
        if open_q != close_q or open_a != close_a:
            issues["unmatched_tags"] += 1
            examples["unmatched_tags"].append(d["doc_id"])

        # Non-ASCII characters
        non_ascii = [c for c in d["story_clean"] if ord(c) > 127]
        if non_ascii:
            issues["non_ascii"] += 1
            examples["non_ascii"].append(f"{d['doc_id']}: {''.join(set(non_ascii[:5]))}")

        # Very short stories (< 10 words)
        if d["story_words"] < 10:
            issues["very_short_story"] += 1
            examples["very_short_story"].append(f"{d['doc_id']}: {d['story_words']} words")

        # Very long stories (> 500 words)
        if d["story_words"] > 500:
            issues["very_long_story"] += 1
            examples["very_long_story"].append(f"{d['doc_id']}: {d['story_words']} words")

        # Duplicate IDs
        if d["doc_id"] in seen_ids:
            issues["duplicate_doc_ids"] += 1
            examples["duplicate_doc_ids"].append(d["doc_id"])
        seen_ids.add(d["doc_id"])

    n = len(docs)
    report = f"""CLEANLINESS REPORT ({split}, n={n:,})
═════════════════════════════════════
Empty stories: {issues['empty_story']} ({100*issues['empty_story']/n:.2f}%)
No questions: {issues['empty_questions']} ({100*issues['empty_questions']/n:.2f}%)
Mismatched Q/A counts: {issues['mismatched_qa']} ({100*issues['mismatched_qa']/n:.2f}%)
Broken/unmatched tags: {issues['unmatched_tags']} ({100*issues['unmatched_tags']/n:.2f}%)
Non-ASCII in stories: {issues['non_ascii']} ({100*issues['non_ascii']/n:.2f}%)
Very short stories (<10 words): {issues['very_short_story']} ({100*issues['very_short_story']/n:.2f}%)
Very long stories (>500 words): {issues['very_long_story']} ({100*issues['very_long_story']/n:.2f}%)
Duplicate doc IDs: {issues['duplicate_doc_ids']} ({100*issues['duplicate_doc_ids']/n:.2f}%)

OVERALL CLEAN: {'YES ✓' if sum(issues.values()) == 0 else f'Issues found in {sum(issues.values())} docs'}
"""

    # Add examples for any issues found
    for key, count in issues.items():
        if count > 0:
            exs = examples[key][:5]
            report += f"\n  Examples of {key}:\n"
            for ex in exs:
                report += f"    • {ex}\n"

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"StoryQA v4 – {split} Cleanliness", fontsize=14)

    # Issue counts bar chart
    issue_labels = [k.replace("_", " ").title() for k in issues]
    issue_counts = list(issues.values())
    colors = ["#C44E52" if c > 0 else "#55A868" for c in issue_counts]
    axes[0].barh(issue_labels[::-1], issue_counts[::-1], color=colors[::-1])
    axes[0].set_title("Issue Counts")
    axes[0].set_xlabel("Count")

    # Character distribution in stories (check for unusual patterns)
    all_chars = collections.Counter()
    for d in docs[:5000]:  # sample
        all_chars.update(d["story_clean"])
    # Show non-alphanumeric, non-space characters
    special = {c: n for c, n in all_chars.items() if not c.isalnum() and c != " "}
    if special:
        sp_sorted = sorted(special.items(), key=lambda x: -x[1])[:15]
        sp_labels = [repr(c) for c, _ in sp_sorted]
        sp_counts = [n for _, n in sp_sorted]
        axes[1].barh(sp_labels[::-1], sp_counts[::-1], color="#8172B2")
        axes[1].set_title("Special Characters in Stories")
        axes[1].set_xlabel("Frequency (sample)")
    else:
        axes[1].text(0.5, 0.5, "No special characters found", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Special Characters")

    plt.tight_layout()
    return report, fig


def randomness_analysis(split):
    df = train_df if split == "Train" else test_df
    docs = train_docs if split == "Train" else test_docs

    ids = sorted(df["numeric_id"].values)
    n = len(ids)

    # ID distribution analysis
    id_min = min(ids)
    id_max = max(ids)
    id_range = id_max - id_min + 1

    # Check for sequential vs random
    # If truly random sample, IDs should be spread across the range
    gaps = [ids[i+1] - ids[i] for i in range(len(ids)-1)]
    mean_gap = np.mean(gaps) if gaps else 0
    std_gap = np.std(gaps) if gaps else 0

    # Expected gap for uniform random sample from [0, id_range)
    # If n items from range R, expected gap ≈ R/n
    expected_gap = id_range / n if n > 0 else 0

    # Kolmogorov-Smirnov test for uniformity
    from scipy import stats as scipy_stats
    normalized_ids = [(i - id_min) / max(id_range - 1, 1) for i in ids]
    ks_stat, ks_pvalue = scipy_stats.kstest(normalized_ids, 'uniform')

    # Check for runs of consecutive IDs (non-random pattern)
    consecutive_runs = 0
    max_run = 1
    current_run = 1
    for i in range(1, len(ids)):
        if ids[i] == ids[i-1] + 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            if current_run > 2:
                consecutive_runs += 1
            current_run = 1

    # Duplicate IDs
    id_counts = collections.Counter(ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}

    report = f"""RANDOMNESS / SAMPLING ANALYSIS ({split}, n={n:,})
══════════════════════════════════════════════
ID Range: {id_min} to {id_max} (span = {id_range:,})
Coverage: {n:,} / {id_range:,} = {100*n/max(id_range,1):.1f}%

Gap Analysis (between sorted IDs):
  Mean gap: {mean_gap:.2f}  (expected for uniform: {expected_gap:.2f})
  Std gap: {std_gap:.2f}
  Max consecutive run: {max_run}
  Runs of >2 consecutive IDs: {consecutive_runs}

KS Test for Uniformity:
  KS statistic: {ks_stat:.6f}
  p-value: {ks_pvalue:.6f}
  Interpretation: {"UNIFORM (looks random) ✓" if ks_pvalue > 0.05 else "NOT UNIFORM (possible bias) ✗"}

Duplicate IDs: {len(duplicates)}
"""
    if duplicates:
        report += "  Duplicated IDs (first 10):\n"
        for did, cnt in list(duplicates.items())[:10]:
            report += f"    story_qa_{did:06d}: appears {cnt}x\n"

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"StoryQA v4 – {split} Randomness Analysis", fontsize=14)

    # ID distribution histogram
    axes[0, 0].hist(ids, bins=50, color="#4C72B0", edgecolor="white")
    axes[0, 0].set_title("Document ID Distribution")
    axes[0, 0].set_xlabel("Numeric ID")
    axes[0, 0].set_ylabel("Count")

    # CDF vs uniform
    sorted_norm = np.sort(normalized_ids)
    axes[0, 1].plot(sorted_norm, np.linspace(0, 1, len(sorted_norm)), label="Observed CDF", color="#4C72B0")
    axes[0, 1].plot([0, 1], [0, 1], "r--", label="Uniform CDF")
    axes[0, 1].set_title(f"CDF vs Uniform (KS p={ks_pvalue:.4f})")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Normalized ID")

    # Gap distribution
    if gaps:
        axes[1, 0].hist(gaps, bins=50, color="#55A868", edgecolor="white")
        axes[1, 0].axvline(expected_gap, color="red", linestyle="--", label=f"Expected={expected_gap:.1f}")
        axes[1, 0].set_title("Gap Distribution (sorted IDs)")
        axes[1, 0].set_xlabel("Gap size")
        axes[1, 0].legend()

    # Index vs ID scatter (check for sequential patterns)
    axes[1, 1].scatter(range(len(ids)), ids, s=1, alpha=0.3, color="#C44E52")
    axes[1, 1].set_title("Sorted Index vs ID (should be linear for uniform)")
    axes[1, 1].set_xlabel("Sorted rank")
    axes[1, 1].set_ylabel("Numeric ID")

    plt.tight_layout()
    return report, fig


def sample_docs(split, n=5):
    """Show random sample documents for manual inspection."""
    docs = train_docs if split == "Train" else test_docs
    samples = random.sample(docs, min(n, len(docs)))
    output = ""
    for i, d in enumerate(samples, 1):
        output += f"{'='*60}\n"
        output += f"SAMPLE {i} | {d['doc_id']} | idx={d['idx']}\n"
        output += f"Story words: {d['story_words']} | Questions: {d['n_questions']} | Definitions: {d['n_definitions']}\n"
        output += f"{'='*60}\n"
        output += f"STORY:\n{d['story']}\n\n"
        for q, a in zip(d['questions'], d['answers']):
            output += f"Q: {q}\nA: {a}\n\n"
        output += "\n"
    return output


# ── Gradio App ───────────────────────────────────────────────────────────────

with gr.Blocks(title="StoryQA v4 Data Quality Dashboard") as app:
    gr.Markdown("# StoryQA v4 Data Quality Dashboard")
    gr.Markdown(f"**Train:** {len(train_docs):,} docs | **Test:** {len(test_docs):,} docs")

    with gr.Tabs():
        # Tab 1: Length Analysis
        with gr.Tab("1. Length Analysis"):
            split1 = gr.Radio(["Train", "Test"], value="Train", label="Split")
            btn1 = gr.Button("Analyze Lengths", variant="primary")
            stats1 = gr.Textbox(label="Statistics", lines=25)
            plot1 = gr.Plot(label="Distributions")
            btn1.click(length_analysis, inputs=split1, outputs=[stats1, plot1])

        # Tab 2: Quality & Variety
        with gr.Tab("2. Quality & Variety"):
            split2 = gr.Radio(["Train", "Test"], value="Train", label="Split")
            btn2 = gr.Button("Analyze Quality", variant="primary")
            stats2 = gr.Textbox(label="Quality Report", lines=30)
            plot2 = gr.Plot(label="Variety Plots")
            btn2.click(quality_variety_analysis, inputs=split2, outputs=[stats2, plot2])

        # Tab 3: Cleanliness
        with gr.Tab("3. Cleanliness"):
            split3 = gr.Radio(["Train", "Test"], value="Train", label="Split")
            btn3 = gr.Button("Check Cleanliness", variant="primary")
            stats3 = gr.Textbox(label="Cleanliness Report", lines=25)
            plot3 = gr.Plot(label="Issue Visualization")
            btn3.click(cleanliness_analysis, inputs=split3, outputs=[stats3, plot3])

        # Tab 4: Randomness
        with gr.Tab("4. Random Subsample Check"):
            split4 = gr.Radio(["Train", "Test"], value="Train", label="Split")
            btn4 = gr.Button("Check Randomness", variant="primary")
            stats4 = gr.Textbox(label="Randomness Report", lines=20)
            plot4 = gr.Plot(label="Randomness Plots")
            btn4.click(randomness_analysis, inputs=split4, outputs=[stats4, plot4])

        # Tab 5: Sample Inspector
        with gr.Tab("5. Sample Documents"):
            split5 = gr.Radio(["Train", "Test"], value="Train", label="Split")
            n_samples = gr.Slider(1, 20, value=5, step=1, label="Number of samples")
            btn5 = gr.Button("Show Random Samples", variant="primary")
            samples_out = gr.Textbox(label="Sampled Documents", lines=40)
            btn5.click(sample_docs, inputs=[split5, n_samples], outputs=samples_out)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7862, share=True)
