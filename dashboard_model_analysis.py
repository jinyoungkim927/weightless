"""Model Internals Analysis Dashboard: Residuals, Attention Maps, Output Patterns.

Analyzes the baseline_full_8gpu checkpoint on StoryQA bad samples.
"""

import os
import re
import glob
import random
import collections
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
from transformers import GPT2Tokenizer

from model import create_model

# ── Setup ────────────────────────────────────────────────────────────────────

CHECKPOINT = "/root/weightless/checkpoints/baseline_full_8gpu.pt"
DATA_DIR = "/root/weightless/story_qa_v4_plaintext_shards"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print("Loading model...")
model = create_model(variant="baseline")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
state = ckpt.get("model_state_dict", ckpt)
# Strip DDP "module." prefix
cleaned = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(cleaned, strict=False)
model = model.to(DEVICE).eval()
print(f"Model loaded on {DEVICE}, params: {model.count_parameters(count_zeros=True):,}")

# ── Parse StoryQA data ──────────────────────────────────────────────────────

DOC_RE = re.compile(
    r"===== DOC START split=(\w+) idx=(\d+) id=(story_qa_\d+) =====\n"
    r"(.*?)"
    r"===== DOC END =====",
    re.DOTALL,
)
QUESTION_RE = re.compile(r"<question>(.*?)</question>")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>")

def parse_test_docs():
    path = os.path.join(DATA_DIR, "test", "test_shard_00000.txt")
    text = open(path, encoding="utf-8").read()
    docs = []
    for m in DOC_RE.finditer(text):
        sp, idx, doc_id, body = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        questions = QUESTION_RE.findall(body)
        answers = ANSWER_RE.findall(body)
        first_q_pos = body.find("<question>")
        story = body[:first_q_pos].strip() if first_q_pos != -1 else body.strip()
        story_clean = re.sub(r'<definition>.*?</definition>', '', story).strip()
        story_clean = re.sub(r'\s+', ' ', story_clean)
        if questions and answers:
            docs.append({
                "doc_id": doc_id,
                "story": story_clean,
                "questions": questions,
                "answers": answers,
            })
    return docs

print("Parsing test data...")
test_docs = parse_test_docs()
print(f"Parsed {len(test_docs)} test docs")


# ── Instrumented Forward Pass ────────────────────────────────────────────────

class InstrumentedForward:
    """Run forward pass capturing attention weights, residuals, logits."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def run(self, prompt, max_gen_tokens=60):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]

        if prompt_len > 480:
            input_ids = input_ids[:, :480]
            prompt_len = 480

        # Capture attention weights via manual computation
        attn_weights = {}
        residual_pre = {}
        residual_post = {}
        ffn_gate_norms = {}

        hooks = []

        for i, layer in enumerate(self.model.layers):
            def make_attn_hook(idx):
                def hook(module, inputs, output):
                    x = inputs[0]
                    rope_mod = inputs[3]
                    positions = inputs[4]
                    B, T, C = x.shape
                    hd = module.head_dim
                    q = module.q_proj(x).reshape(B, T, module.n_heads, hd).transpose(1, 2)
                    k = module.k_proj(x).reshape(B, T, module.n_kv_heads, hd).transpose(1, 2)
                    q = rope_mod(q, positions)
                    k = rope_mod(k, positions)
                    if module.n_rep > 1:
                        k = k.unsqueeze(2).expand(B, module.n_kv_heads, module.n_rep, T, hd)
                        k = k.reshape(B, module.n_heads, T, hd)
                    scale = hd ** -0.5
                    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                    mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
                    scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                    weights = F.softmax(scores, dim=-1)
                    attn_weights[idx] = weights.cpu().float().numpy()
                return hook
            hooks.append(layer.attn.register_forward_hook(make_attn_hook(i)))

            def make_layer_hook(idx):
                def hook(module, inputs, output):
                    residual_pre[idx] = inputs[0].detach().cpu().float().numpy()
                    residual_post[idx] = output.detach().cpu().float().numpy()
                return hook
            hooks.append(layer.register_forward_hook(make_layer_hook(i)))

            def make_ffn_hook(idx):
                def hook(module, inputs, output):
                    x = inputs[0]
                    gate = F.silu(module.w1(x)) * module.w3(x)
                    # Store per-position norm
                    ffn_gate_norms[idx] = gate.norm(dim=-1).cpu().float().numpy()
                return hook
            hooks.append(layer.ff.register_forward_hook(make_ffn_hook(i)))

        # Forward pass on prompt only (no generation yet)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input_ids)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Get predictions at each position
        probs = F.softmax(logits[0].float(), dim=-1)

        # Generate answer
        gen_ids = input_ids.clone()
        gen_tokens = []
        gen_probs = []
        gen_entropies = []
        for _ in range(max_gen_tokens):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = self.model(gen_ids)
            next_logits = out[:, -1, :].float()
            next_probs = F.softmax(next_logits, dim=-1)
            entropy = -(next_probs * (next_probs + 1e-10).log()).sum(dim=-1).item()

            # Greedy
            tok_id = next_logits.argmax(dim=-1).item()
            gen_tokens.append(self.tokenizer.decode([tok_id]))
            gen_probs.append(next_probs[0, tok_id].item())
            gen_entropies.append(entropy)

            gen_ids = torch.cat([gen_ids, torch.tensor([[tok_id]], device=self.device)], dim=1)
            if tok_id == self.tokenizer.eos_token_id or "\n" in self.tokenizer.decode([tok_id]):
                break

        return {
            "input_ids": input_ids[0].cpu().tolist(),
            "prompt_len": prompt_len,
            "prompt_tokens": [self.tokenizer.decode([t]) for t in input_ids[0].cpu().tolist()],
            "attn_weights": attn_weights,
            "residual_pre": residual_pre,
            "residual_post": residual_post,
            "ffn_gate_norms": ffn_gate_norms,
            "logits": logits[0].cpu().float(),
            "probs": probs.cpu(),
            "gen_tokens": gen_tokens,
            "gen_probs": gen_probs,
            "gen_entropies": gen_entropies,
            "generated_answer": "".join(gen_tokens).strip().split("\n")[0],
        }


instrument = InstrumentedForward(model, tokenizer, DEVICE)


# ── Analysis Functions ───────────────────────────────────────────────────────

def analyze_single_example(doc_idx, qa_idx):
    """Full analysis of a single StoryQA example."""
    doc = test_docs[doc_idx % len(test_docs)]
    qi = qa_idx % len(doc["questions"])
    question = doc["questions"][qi]
    gold = doc["answers"][qi]

    prompt = f'{doc["story"]}\n\nQuestion: {question}\nAnswer:'
    result = instrument.run(prompt)

    info = f"""Document: {doc['doc_id']}
Story: {doc['story'][:300]}...
Question: {question}
Gold answer: {gold}
Model answer (greedy): {result['generated_answer']}
Prompt tokens: {result['prompt_len']}
"""
    return result, info, doc, question, gold


def residual_analysis(doc_idx, qa_idx):
    """Analyze residual stream norms across layers for a specific example."""
    result, info, doc, question, gold = analyze_single_example(doc_idx, qa_idx)

    n_layers = len(result["residual_pre"])
    prompt_len = result["prompt_len"]

    # Compute residual norms per layer per position
    pre_norms = []
    post_norms = []
    deltas = []
    for i in range(n_layers):
        pre = result["residual_pre"][i][0]  # (T, d_model)
        post = result["residual_post"][i][0]
        pre_norm = np.linalg.norm(pre, axis=-1)  # (T,)
        post_norm = np.linalg.norm(post, axis=-1)
        delta = post - pre
        delta_norm = np.linalg.norm(delta, axis=-1)
        pre_norms.append(pre_norm)
        post_norms.append(post_norm)
        deltas.append(delta_norm)

    pre_norms = np.array(pre_norms)   # (n_layers, T)
    post_norms = np.array(post_norms)
    deltas = np.array(deltas)

    tokens = result["prompt_tokens"]

    # Figure 1: Residual norm heatmap
    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(tokens)*0.3), 12))
    fig.suptitle(f"Residual Stream Analysis – {doc['doc_id']}", fontsize=14)

    im0 = axes[0].imshow(pre_norms, aspect="auto", cmap="viridis")
    axes[0].set_title("Pre-layer Residual Norms")
    axes[0].set_ylabel("Layer")
    axes[0].set_xlabel("Token position")
    fig.colorbar(im0, ax=axes[0], fraction=0.02)

    im1 = axes[1].imshow(post_norms, aspect="auto", cmap="viridis")
    axes[1].set_title("Post-layer Residual Norms")
    axes[1].set_ylabel("Layer")
    fig.colorbar(im1, ax=axes[1], fraction=0.02)

    im2 = axes[2].imshow(deltas, aspect="auto", cmap="hot")
    axes[2].set_title("Layer Contribution (|post - pre|)")
    axes[2].set_ylabel("Layer")
    axes[2].set_xlabel("Token position")
    fig.colorbar(im2, ax=axes[2], fraction=0.02)

    # Mark the boundary between story, question, and "Answer:"
    # Find "Question" and "Answer" token positions
    prompt_text = tokenizer.decode(result["input_ids"])
    q_pos = prompt_text.find("Question:")
    a_pos = prompt_text.find("Answer:")
    if q_pos >= 0:
        q_tok_pos = len(tokenizer.encode(prompt_text[:q_pos]))
        for ax in axes:
            ax.axvline(q_tok_pos, color="cyan", linestyle="--", alpha=0.7, label="Question:")
    if a_pos >= 0:
        a_tok_pos = len(tokenizer.encode(prompt_text[:a_pos]))
        for ax in axes:
            ax.axvline(a_tok_pos, color="red", linestyle="--", alpha=0.7, label="Answer:")

    axes[0].legend(fontsize=8)
    plt.tight_layout()

    # Figure 2: Last-position residual norms (what the model "knows" at Answer:)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    last_pre = [result["residual_pre"][i][0, -1, :] for i in range(n_layers)]
    last_post = [result["residual_post"][i][0, -1, :] for i in range(n_layers)]
    last_pre_norms = [np.linalg.norm(x) for x in last_pre]
    last_post_norms = [np.linalg.norm(x) for x in last_post]
    last_deltas = [np.linalg.norm(last_post[i] - last_pre[i]) for i in range(n_layers)]

    x = range(n_layers)
    ax2.plot(x, last_pre_norms, 'b-o', label="Pre-layer norm", markersize=6)
    ax2.plot(x, last_post_norms, 'g-o', label="Post-layer norm", markersize=6)
    ax2.plot(x, last_deltas, 'r-o', label="Layer delta", markersize=6)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("L2 Norm")
    ax2.set_title(f"Last Token (Answer:) Residual Norms Per Layer")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # Stats
    stats = info
    stats += f"\n── Residual Analysis ──\n"
    for i in range(n_layers):
        stats += f"Layer {i}: pre={last_pre_norms[i]:.2f}, post={last_post_norms[i]:.2f}, delta={last_deltas[i]:.2f}\n"

    # Check for anomalies
    max_delta_layer = np.argmax(last_deltas)
    stats += f"\nLargest contribution at Answer: position: Layer {max_delta_layer} (delta={last_deltas[max_delta_layer]:.2f})\n"

    # FFN gate norm analysis
    stats += f"\n── FFN Gate Norms at Answer: position ──\n"
    for i in range(n_layers):
        gate_norm = result["ffn_gate_norms"][i][0, -1]
        stats += f"Layer {i}: FFN gate norm = {gate_norm:.2f}\n"

    return stats, fig, fig2


def attention_analysis(doc_idx, qa_idx, layer):
    """Attention map analysis for a specific example and layer."""
    result, info, doc, question, gold = analyze_single_example(doc_idx, qa_idx)

    tokens = result["prompt_tokens"]
    T = len(tokens)
    layer = min(layer, len(result["attn_weights"]) - 1)
    attn = result["attn_weights"][layer]  # (1, n_heads, T, T)

    n_heads = attn.shape[1]

    # Figure 1: Mean attention heatmap
    fig1, ax = plt.subplots(figsize=(min(16, T * 0.35 + 2), min(14, T * 0.3 + 2)))
    mean_attn = attn[0].mean(axis=0)  # (T, T)
    im = ax.imshow(mean_attn, cmap="Blues", aspect="auto")
    short = [t.replace("\n", "\\n")[:8] for t in tokens]
    if T <= 60:
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        fs = max(4, min(8, 100 // T))
        ax.set_xticklabels(short, rotation=90, fontsize=fs)
        ax.set_yticklabels(short, fontsize=fs)
    ax.set_xlabel("Key (attending to)")
    ax.set_ylabel("Query (from)")
    ax.set_title(f"Layer {layer} – Mean Attention (all heads)")
    fig1.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()

    # Figure 2: Per-head attention at the LAST position (what does Answer: attend to?)
    fig2, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig2.suptitle(f"Layer {layer} – Last Token (Answer:) Attention Pattern", fontsize=14)
    axes = axes.flatten()
    for h in range(min(n_heads, 8)):
        ax = axes[h]
        vals = attn[0, h, -1, :]  # What the last token attends to
        ax.bar(range(T), vals, width=1.0, color=plt.cm.Blues(vals / max(vals.max(), 1e-6)))
        ax.set_title(f"Head {h}", fontsize=10)
        if T <= 40:
            ax.set_xticks(range(0, T, max(1, T//10)))
        ax.set_ylim(0, max(vals.max() * 1.1, 0.01))

        # Highlight top-3 attended positions
        top3 = np.argsort(vals)[-3:]
        for pos in top3:
            tok = tokens[pos].replace("\n", "\\n")[:12]
            ax.annotate(tok, (pos, vals[pos]), fontsize=6, ha='center', va='bottom', rotation=45)

    plt.tight_layout()

    # Figure 3: Attention entropy per head per position
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    entropies = np.zeros((n_heads, T))
    for h in range(n_heads):
        for t in range(T):
            dist = attn[0, h, t, :t+1]
            dist = dist + 1e-10
            entropies[h, t] = -(dist * np.log(dist)).sum()
    im3 = ax3.imshow(entropies, aspect="auto", cmap="plasma")
    ax3.set_xlabel("Token position")
    ax3.set_ylabel("Head")
    ax3.set_title(f"Layer {layer} – Attention Entropy (higher = more diffuse)")
    fig3.colorbar(im3, ax=ax3, fraction=0.04)
    plt.tight_layout()

    stats = info
    stats += f"\n── Attention Analysis (Layer {layer}) ──\n"
    stats += f"Num heads: {n_heads}\n"
    stats += f"Sequence length: {T}\n"

    # What does the last token attend to most?
    stats += f"\nTop attended positions from Answer: token (per head):\n"
    for h in range(n_heads):
        vals = attn[0, h, -1, :]
        top5_idx = np.argsort(vals)[-5:][::-1]
        top5_strs = [(tokens[p].replace("\n", "\\n")[:15], f"{vals[p]:.3f}") for p in top5_idx]
        stats += f"  Head {h}: {', '.join(f'{t}({v})' for t, v in top5_strs)}\n"

    return stats, fig1, fig2, fig3


def output_pattern_analysis(n_examples):
    """Analyze output patterns across many examples to find strange behaviors."""
    n = min(int(n_examples), 100)
    random.seed(42)
    sample = random.sample(test_docs, n)

    results_data = []
    for doc in sample:
        q = doc["questions"][0]
        gold = doc["answers"][0]
        prompt = f'{doc["story"]}\n\nQuestion: {q}\nAnswer:'

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = input_ids.shape[1]

        # Greedy generation
        gen_ids = input_ids.clone()
        gen_text_tokens = []
        for _ in range(60):
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(gen_ids)
            tok_id = logits[:, -1, :].float().argmax(dim=-1).item()
            gen_text_tokens.append(tok_id)
            gen_ids = torch.cat([gen_ids, torch.tensor([[tok_id]], device=DEVICE)], dim=1)
            if tok_id == tokenizer.eos_token_id or "\n" in tokenizer.decode([tok_id]):
                break

        model_answer = tokenizer.decode(gen_text_tokens).strip().split("\n")[0]

        # Check patterns
        is_repetitive = len(set(gen_text_tokens)) < len(gen_text_tokens) * 0.3
        is_empty = len(model_answer.strip()) < 3
        starts_with_the = model_answer.lower().startswith("the ")
        has_hallucination_markers = any(w in model_answer.lower() for w in [
            "wikipedia", "according to", "the united states", "http", "www",
            "university", "century", "population"
        ])
        is_continuation = model_answer.count(".") > 3

        results_data.append({
            "doc_id": doc["doc_id"],
            "question": q,
            "gold": gold,
            "model": model_answer,
            "n_gen_tokens": len(gen_text_tokens),
            "n_unique_tokens": len(set(gen_text_tokens)),
            "is_repetitive": is_repetitive,
            "is_empty": is_empty,
            "starts_with_the": starts_with_the,
            "has_hallucination": has_hallucination_markers,
            "is_continuation": is_continuation,
        })

    # Aggregate patterns
    total = len(results_data)
    n_repetitive = sum(r["is_repetitive"] for r in results_data)
    n_empty = sum(r["is_empty"] for r in results_data)
    n_starts_the = sum(r["starts_with_the"] for r in results_data)
    n_halluc = sum(r["has_hallucination"] for r in results_data)
    n_continuation = sum(r["is_continuation"] for r in results_data)

    # Simple correctness heuristic
    n_correct_heuristic = 0
    for r in results_data:
        gold_words = set(r["gold"].lower().split())
        model_words = set(r["model"].lower().split())
        overlap = gold_words & model_words
        stopwords = {"the", "a", "an", "is", "was", "are", "to", "in", "of", "and", "it"}
        content_gold = gold_words - stopwords
        content_overlap = overlap - stopwords
        if content_gold and len(content_overlap) / len(content_gold) >= 0.5:
            n_correct_heuristic += 1

    report = f"""OUTPUT PATTERN ANALYSIS ({total} examples, greedy decoding)
══════════════════════════════════════════════════════════
Heuristic correctness (50%+ keyword overlap): {n_correct_heuristic}/{total} ({100*n_correct_heuristic/total:.1f}%)

Pattern breakdown:
  Repetitive outputs (>70% repeated tokens): {n_repetitive} ({100*n_repetitive/total:.1f}%)
  Empty/near-empty answers (<3 chars): {n_empty} ({100*n_empty/total:.1f}%)
  Starts with "The ": {n_starts_the} ({100*n_starts_the/total:.1f}%)
  Contains hallucination markers: {n_halluc} ({100*n_halluc/total:.1f}%)
  Long continuation (>3 sentences): {n_continuation} ({100*n_continuation/total:.1f}%)

Generation length stats:
  Mean tokens generated: {np.mean([r['n_gen_tokens'] for r in results_data]):.1f}
  Mean unique tokens: {np.mean([r['n_unique_tokens'] for r in results_data]):.1f}
  Uniqueness ratio: {np.mean([r['n_unique_tokens']/max(r['n_gen_tokens'],1) for r in results_data]):.2f}
"""

    # Show examples by category
    report += "\n── CORRECT examples ──\n"
    correct = [r for r in results_data if r["model"].lower().strip()[:20] == r["gold"].lower().strip()[:20]]
    for r in correct[:5]:
        report += f"  Q: {r['question'][:60]}  Gold: {r['gold'][:40]}  Model: {r['model'][:40]}\n"

    report += "\n── REPETITIVE examples ──\n"
    reps = [r for r in results_data if r["is_repetitive"]]
    for r in reps[:5]:
        report += f"  Q: {r['question'][:60]}  Model: {r['model'][:80]}\n"

    report += "\n── HALLUCINATED examples ──\n"
    halls = [r for r in results_data if r["has_hallucination"]]
    for r in halls[:5]:
        report += f"  Q: {r['question'][:60]}  Model: {r['model'][:80]}\n"

    report += "\n── ALL RESULTS ──\n"
    for r in results_data:
        marker = ""
        if r["is_repetitive"]: marker += "[REP]"
        if r["has_hallucination"]: marker += "[HAL]"
        if r["is_continuation"]: marker += "[CONT]"
        report += f"  {r['doc_id']} {marker}\n    Q: {r['question'][:60]}\n    Gold: {r['gold'][:50]}\n    Model: {r['model'][:80]}\n\n"

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Output Patterns ({total} examples)", fontsize=14)

    # Token count distribution
    axes[0].hist([r["n_gen_tokens"] for r in results_data], bins=20, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Generated Token Count")
    axes[0].set_xlabel("# Tokens")

    # Uniqueness ratio
    axes[1].hist([r["n_unique_tokens"]/max(r["n_gen_tokens"],1) for r in results_data],
                 bins=20, color="#55A868", edgecolor="white")
    axes[1].set_title("Token Uniqueness Ratio")
    axes[1].set_xlabel("Unique/Total")

    # Pattern counts
    labels = ["Correct\n(heuristic)", "Repetitive", "Hallucination", "Continuation", "Starts 'The'"]
    counts = [n_correct_heuristic, n_repetitive, n_halluc, n_continuation, n_starts_the]
    colors = ["#55A868", "#C44E52", "#DD8452", "#8172B2", "#4C72B0"]
    axes[2].bar(labels, counts, color=colors)
    axes[2].set_title("Output Pattern Counts")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    return report, fig


def batch_residual_comparison(n_examples):
    """Compare residual patterns between correct and incorrect outputs."""
    n = min(int(n_examples), 50)
    random.seed(123)
    sample = random.sample(test_docs, n)

    correct_last_norms = []
    incorrect_last_norms = []

    for doc in sample:
        q = doc["questions"][0]
        gold = doc["answers"][0]
        prompt = f'{doc["story"]}\n\nQuestion: {q}\nAnswer:'

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

        # Capture residuals
        layer_norms = []
        hooks = []
        for i, layer in enumerate(model.layers):
            def make_hook(idx):
                def hook(module, inputs, output):
                    pre_norm = inputs[0][0, -1].float().norm().item()
                    post_norm = output[0, -1].float().norm().item()
                    layer_norms.append((pre_norm, post_norm))
                return hook
            hooks.append(layer.register_forward_hook(make_hook(i)))

        # Forward pass + greedy generation
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
        for h in hooks:
            h.remove()

        # Quick greedy answer
        gen_ids = input_ids.clone()
        for _ in range(30):
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(gen_ids)
            tok_id = out[:, -1, :].float().argmax(dim=-1).item()
            gen_ids = torch.cat([gen_ids, torch.tensor([[tok_id]], device=DEVICE)], dim=1)
            if tok_id == tokenizer.eos_token_id or "\n" in tokenizer.decode([tok_id]):
                break
        answer = tokenizer.decode(gen_ids[0, input_ids.shape[1]:].tolist()).strip().split("\n")[0]

        # Heuristic correctness
        gold_words = set(gold.lower().split()) - {"the", "a", "an", "is", "was", "to", "in", "of", "and"}
        model_words = set(answer.lower().split()) - {"the", "a", "an", "is", "was", "to", "in", "of", "and"}
        overlap = gold_words & model_words
        correct = len(gold_words) > 0 and len(overlap) / len(gold_words) >= 0.5

        norms = [n[1] for n in layer_norms]  # post norms
        if correct:
            correct_last_norms.append(norms)
        else:
            incorrect_last_norms.append(norms)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Residual Norms: Correct vs Incorrect ({n} examples)", fontsize=14)

    n_layers = model.n_layers
    if correct_last_norms:
        correct_arr = np.array(correct_last_norms)
        mean_c = correct_arr.mean(axis=0)
        std_c = correct_arr.std(axis=0)
        axes[0].plot(range(n_layers), mean_c, 'g-o', label=f"Correct (n={len(correct_last_norms)})")
        axes[0].fill_between(range(n_layers), mean_c - std_c, mean_c + std_c, alpha=0.2, color='green')

    if incorrect_last_norms:
        incorrect_arr = np.array(incorrect_last_norms)
        mean_i = incorrect_arr.mean(axis=0)
        std_i = incorrect_arr.std(axis=0)
        axes[0].plot(range(n_layers), mean_i, 'r-o', label=f"Incorrect (n={len(incorrect_last_norms)})")
        axes[0].fill_between(range(n_layers), mean_i - std_i, mean_i + std_i, alpha=0.2, color='red')

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Post-layer Residual Norm (last position)")
    axes[0].set_title("Residual Norms at Answer: Position")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot per layer
    if correct_last_norms and incorrect_last_norms:
        for layer_i in range(n_layers):
            c_vals = [x[layer_i] for x in correct_last_norms]
            i_vals = [x[layer_i] for x in incorrect_last_norms]
            pos_c = layer_i * 3
            pos_i = layer_i * 3 + 1
            bp1 = axes[1].boxplot([c_vals], positions=[pos_c], widths=0.6,
                                   patch_artist=True, boxprops=dict(facecolor='lightgreen'))
            bp2 = axes[1].boxplot([i_vals], positions=[pos_i], widths=0.6,
                                   patch_artist=True, boxprops=dict(facecolor='lightcoral'))
        axes[1].set_xticks([i * 3 + 0.5 for i in range(n_layers)])
        axes[1].set_xticklabels([f"L{i}" for i in range(n_layers)])
        axes[1].set_title("Distribution Comparison Per Layer")
        axes[1].set_ylabel("Residual Norm")

    plt.tight_layout()

    report = f"""BATCH RESIDUAL COMPARISON
══════════════════════════
Total examples: {n}
Correct (heuristic): {len(correct_last_norms)}
Incorrect: {len(incorrect_last_norms)}
"""
    if correct_last_norms and incorrect_last_norms:
        correct_arr = np.array(correct_last_norms)
        incorrect_arr = np.array(incorrect_last_norms)
        report += "\nPer-layer norm comparison (mean ± std):\n"
        for i in range(n_layers):
            c_mean, c_std = correct_arr[:, i].mean(), correct_arr[:, i].std()
            i_mean, i_std = incorrect_arr[:, i].mean(), incorrect_arr[:, i].std()
            diff_pct = 100 * (i_mean - c_mean) / max(c_mean, 1e-6)
            report += f"  Layer {i}: Correct={c_mean:.2f}±{c_std:.2f}, Incorrect={i_mean:.2f}±{i_std:.2f} ({diff_pct:+.1f}%)\n"

    return report, fig


# ── Gradio App ───────────────────────────────────────────────────────────────

with gr.Blocks(title="Model Internals Analysis") as app:
    gr.Markdown("# Model Internals Analysis Dashboard")
    gr.Markdown(f"**Checkpoint:** baseline_full_8gpu | **Model:** 95M params, 8 layers, 8 heads")

    with gr.Tabs():
        with gr.Tab("1. Residual Analysis (Single Example)"):
            gr.Markdown("Analyze residual stream norms and layer contributions for a specific StoryQA example.")
            with gr.Row():
                doc_idx1 = gr.Slider(0, len(test_docs)-1, value=0, step=1, label="Document index")
                qa_idx1 = gr.Slider(0, 15, value=0, step=1, label="Question index")
            btn1 = gr.Button("Analyze Residuals", variant="primary")
            stats1 = gr.Textbox(label="Analysis", lines=20)
            plot1a = gr.Plot(label="Residual Heatmaps")
            plot1b = gr.Plot(label="Last Position Norms")
            btn1.click(residual_analysis, inputs=[doc_idx1, qa_idx1], outputs=[stats1, plot1a, plot1b])

        with gr.Tab("2. Attention Maps"):
            gr.Markdown("Attention patterns: what does the model attend to when generating answers?")
            with gr.Row():
                doc_idx2 = gr.Slider(0, len(test_docs)-1, value=0, step=1, label="Document index")
                qa_idx2 = gr.Slider(0, 15, value=0, step=1, label="Question index")
                layer2 = gr.Slider(0, 7, value=7, step=1, label="Layer")
            btn2 = gr.Button("Analyze Attention", variant="primary")
            stats2 = gr.Textbox(label="Attention Analysis", lines=20)
            plot2a = gr.Plot(label="Mean Attention Heatmap")
            plot2b = gr.Plot(label="Per-Head Last-Token Attention")
            plot2c = gr.Plot(label="Attention Entropy")
            btn2.click(attention_analysis, inputs=[doc_idx2, qa_idx2, layer2],
                      outputs=[stats2, plot2a, plot2b, plot2c])

        with gr.Tab("3. Output Patterns (Batch)"):
            gr.Markdown("Analyze output patterns across many examples to find systematic issues.")
            n_ex3 = gr.Slider(10, 100, value=50, step=10, label="Number of examples")
            btn3 = gr.Button("Analyze Patterns", variant="primary")
            stats3 = gr.Textbox(label="Pattern Analysis", lines=40)
            plot3 = gr.Plot(label="Pattern Distribution")
            btn3.click(output_pattern_analysis, inputs=n_ex3, outputs=[stats3, plot3])

        with gr.Tab("4. Correct vs Incorrect Residuals"):
            gr.Markdown("Compare residual stream patterns between correct and incorrect answers.")
            n_ex4 = gr.Slider(10, 50, value=30, step=5, label="Number of examples")
            btn4 = gr.Button("Compare Residuals", variant="primary")
            stats4 = gr.Textbox(label="Comparison", lines=20)
            plot4 = gr.Plot(label="Norm Comparison")
            btn4.click(batch_residual_comparison, inputs=n_ex4, outputs=[stats4, plot4])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7864, share=True)
