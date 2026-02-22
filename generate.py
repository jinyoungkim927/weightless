#!/usr/bin/env python3
"""Interactive inference & interpretability web app for Weightless models.

Usage:
    python generate.py                                                    # random-init model
    python generate.py --checkpoint checkpoints/baseline_full_8gpu.pt     # trained model
    python generate.py --port 7860 --share                                # share publicly
"""

import argparse
import io
import os

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
from transformers import GPT2Tokenizer

import re
import json
import random

from model import create_model


# ---------------------------------------------------------------------------
# LLM-based semantic answer matching
# ---------------------------------------------------------------------------

def llm_judge_matches(qa_triples: list[dict], api_key: str = None) -> list[bool]:
    """Use Claude to judge whether model answers are semantically correct.

    Args:
        qa_triples: list of {"question": ..., "gold": ..., "model": ...}
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var)

    Returns:
        list of bools, True = correct / acceptable match
    """
    if not qa_triples:
        return []

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return _heuristic_matches(qa_triples)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)

        items = []
        for i, t in enumerate(qa_triples):
            items.append(
                f"{i+1}. Question: {t['question']}\n"
                f"   Gold answer: {t['gold']}\n"
                f"   Model answer: {t['model']}"
            )

        prompt = (
            "You are judging whether a model's answer to a reading comprehension "
            "question is semantically correct compared to the gold answer.\n\n"
            "Be VERY liberal — the model answer is correct if it conveys the same "
            "meaning as the gold answer, even if:\n"
            "- It uses different words or phrasing\n"
            "- It's shorter or longer\n"
            "- It includes extra (but not contradictory) information\n"
            "- It paraphrases or rephrases the answer\n"
            "- It uses pronouns instead of names (or vice versa)\n\n"
            "Only mark as incorrect if the model answer is factually wrong, "
            "contradicts the gold, is completely off-topic, or is gibberish.\n\n"
            "Here are the items to judge:\n\n"
            + "\n\n".join(items) +
            "\n\nRespond with ONLY a JSON array of booleans, one per item. "
            "Example: [true, false, true]\n"
            "No explanation needed."
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        results = json.loads(text)

        if isinstance(results, list) and len(results) == len(qa_triples):
            return [bool(r) for r in results]
        else:
            print(f"  ⚠ LLM returned unexpected format, falling back to heuristic")
            return _heuristic_matches(qa_triples)

    except Exception as e:
        print(f"  ⚠ LLM judge failed ({e}), falling back to heuristic")
        return _heuristic_matches(qa_triples)


def _heuristic_matches(qa_triples: list[dict]) -> list[bool]:
    """Simple heuristic fallback: substring + keyword overlap."""
    results = []
    for t in qa_triples:
        gold = t["gold"].lower().strip().rstrip(".")
        model = t["model"].lower().strip().rstrip(".")
        # Substring match
        if gold in model or model in gold or gold == model:
            results.append(True)
            continue
        # Keyword overlap: if 60%+ of gold content words appear in model
        stop = {"a", "an", "the", "is", "was", "were", "are", "he", "she",
                "it", "they", "his", "her", "its", "to", "of", "in", "and",
                "that", "this", "yes", "no", "did", "do", "does"}
        gold_words = {w for w in gold.split() if w not in stop and len(w) > 1}
        model_words = set(model.split())
        if gold_words and len(gold_words & model_words) / len(gold_words) >= 0.6:
            results.append(True)
        else:
            results.append(False)
    return results

# ---------------------------------------------------------------------------
# Story QA data loading
# ---------------------------------------------------------------------------

STORY_QA_DIR = os.path.join(os.path.dirname(__file__), "story_qa_v4_plaintext_shards")


def _parse_story_qa_file(filepath: str) -> list[dict]:
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
        definitions = re.findall(r'<definition>(.*?)</definition>', content)
        docs.append({"id": doc_id, "idx": int(idx), "split": split,
                      "story": story_clean, "story_raw": story,
                      "qa_pairs": qa_pairs, "definitions": definitions})
    return docs


def _load_story_qa_pool(split="test", max_docs=200) -> list[dict]:
    """Load a pool of story QA docs for the UI."""
    split_dir = os.path.join(STORY_QA_DIR, split)
    if not os.path.isdir(split_dir):
        return []
    all_docs = []
    for fname in sorted(os.listdir(split_dir)):
        if fname.endswith(".txt"):
            all_docs.extend(_parse_story_qa_file(os.path.join(split_dir, fname)))
        if len(all_docs) >= max_docs:
            break
    return all_docs[:max_docs]


# ---------------------------------------------------------------------------
# Hook-based instrumentation
# ---------------------------------------------------------------------------

class ModelInstrument:
    """Registers forward hooks on a SimpleTransformer to capture attention
    weights, FFN gate activations, and residual-stream norms."""

    def __init__(self, model):
        self.model = model
        self.attn_weights: dict[int, np.ndarray] = {}   # layer -> (B,H,T,T)
        self.ffn_gate: dict[int, np.ndarray] = {}       # layer -> (B,T,d_ff)
        self.residual_pre: dict[int, np.ndarray] = {}   # layer -> (B,T,d)
        self.residual_post: dict[int, np.ndarray] = {}  # layer -> (B,T,d)
        self._hooks: list = []
        self._install()

    # -- hook installation ---------------------------------------------------

    def _install(self):
        raw = self.model
        for i, layer in enumerate(raw.layers):
            self._hooks.append(
                layer.attn.register_forward_hook(self._attn_hook(i))
            )
            self._hooks.append(
                layer.ff.register_forward_hook(self._ffn_hook(i))
            )
            self._hooks.append(
                layer.register_forward_hook(self._layer_hook(i))
            )

    def _attn_hook(self, layer_idx):
        def fn(module, inputs, output):
            x = inputs[0]
            rope_mod = inputs[3]
            positions = inputs[4]
            B, T, C = x.shape
            hd = module.head_dim
            with torch.no_grad():
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
            self.attn_weights[layer_idx] = weights.cpu().float().numpy()
        return fn

    def _ffn_hook(self, layer_idx):
        def fn(module, inputs, output):
            x = inputs[0]
            with torch.no_grad():
                gate = F.silu(module.w1(x)) * module.w3(x)
            self.ffn_gate[layer_idx] = gate.cpu().float().numpy()
        return fn

    def _layer_hook(self, layer_idx):
        def fn(module, inputs, output):
            self.residual_pre[layer_idx] = inputs[0].detach().cpu().float().numpy()
            self.residual_post[layer_idx] = output.detach().cpu().float().numpy()
        return fn

    # -- helpers -------------------------------------------------------------

    def clear(self):
        self.attn_weights.clear()
        self.ffn_gate.clear()
        self.residual_pre.clear()
        self.residual_post.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Generation with per-token diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_diagnostics(
    model, tokenizer, device, instrument,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """Autoregressive generation; hooks capture data on the *last* forward pass."""
    instrument.clear()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    all_token_info: list[dict] = []

    model.eval()
    for _ in range(max_tokens):
        logits = model(input_ids)                       # hooks fire here
        next_logits = logits[:, -1, :].clone()

        # raw probabilities (before sampling transforms)
        raw_probs = F.softmax(next_logits, dim=-1)
        entropy = -(raw_probs * (raw_probs + 1e-10).log()).sum(dim=-1).item()

        # top-10 candidates
        top10_probs, top10_ids = torch.topk(raw_probs, 10, dim=-1)
        top10_tokens = [tokenizer.decode([tid]) for tid in top10_ids[0].tolist()]

        # temperature
        next_logits = next_logits / max(temperature, 1e-8)

        # top-k filter
        if top_k > 0:
            kth = torch.topk(next_logits, top_k).values[:, -1:]
            next_logits[next_logits < kth] = float("-inf")

        # top-p (nucleus) filter
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            indices_to_remove = remove.scatter(1, sorted_idx, remove)
            next_logits[indices_to_remove] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tok_id = next_token[0, 0].item()

        all_token_info.append({
            "token": tokenizer.decode([tok_id]),
            "token_id": tok_id,
            "probability": raw_probs[0, tok_id].item(),
            "entropy": entropy,
            "top_tokens": top10_tokens,
            "top_probs": top10_probs[0].tolist(),
        })

        input_ids = torch.cat([input_ids, next_token], dim=1)
        if tok_id == tokenizer.eos_token_id:
            break

    return {
        "prompt": prompt,
        "prompt_len": prompt_len,
        "generated_text": tokenizer.decode(input_ids[0].tolist()),
        "tokens": all_token_info,
        "input_ids": input_ids[0].tolist(),
        "attention": dict(instrument.attn_weights),
        "ffn_gate": dict(instrument.ffn_gate),
        "residual_pre": dict(instrument.residual_pre),
        "residual_post": dict(instrument.residual_post),
    }


# ---------------------------------------------------------------------------
# Matplotlib → PIL helpers
# ---------------------------------------------------------------------------

def _fig_to_pil(fig):
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

BG       = "#0f1117"
BG_INNER = "#161b22"
TEXT_CLR  = "#c9d1d9"
GRID_CLR  = "#21262d"


def _style_ax(ax):
    ax.set_facecolor(BG_INNER)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def plot_attention_heatmap(attn, tokens, layer, head):
    T = len(tokens)
    mat = attn[layer][0, head, :T, :T]
    fig, ax = plt.subplots(figsize=(min(14, T * 0.45 + 2), min(12, T * 0.38 + 2)))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=max(mat.max(), 0.01))
    short = [t.replace("\n", "\\n")[:10] for t in tokens]
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    fs = max(5, min(9, 110 // T))
    ax.set_xticklabels(short, rotation=90, fontsize=fs, color=TEXT_CLR)
    ax.set_yticklabels(short, fontsize=fs, color=TEXT_CLR)
    ax.set_xlabel("Key (attending to)", color=TEXT_CLR, fontsize=10)
    ax.set_ylabel("Query (from)", color=TEXT_CLR, fontsize=10)
    ax.set_title(f"Layer {layer} · Head {head}", color=TEXT_CLR, fontsize=12, fontweight="bold")
    _style_ax(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_pil(fig)


def plot_attention_mean(attn, tokens, layer):
    T = len(tokens)
    mat = attn[layer][0, :, :T, :T].mean(axis=0)
    n_h = attn[layer].shape[1]
    fig, ax = plt.subplots(figsize=(min(14, T * 0.45 + 2), min(12, T * 0.38 + 2)))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(mat, cmap="Purples", aspect="auto", vmin=0)
    short = [t.replace("\n", "\\n")[:10] for t in tokens]
    fs = max(5, min(9, 110 // T))
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(short, rotation=90, fontsize=fs, color=TEXT_CLR)
    ax.set_yticklabels(short, fontsize=fs, color=TEXT_CLR)
    ax.set_xlabel("Key", color=TEXT_CLR, fontsize=10)
    ax.set_ylabel("Query", color=TEXT_CLR, fontsize=10)
    ax.set_title(f"Layer {layer} · Mean over {n_h} heads", color=TEXT_CLR, fontsize=12, fontweight="bold")
    _style_ax(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_pil(fig)


def plot_last_token_attention(attn, tokens, layer):
    """Bar chart: what the *last* generated token attends to, per head."""
    T = len(tokens)
    n_heads = attn[layer].shape[1]
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows))
    fig.patch.set_facecolor(BG)
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    short = [t.replace("\n", "\\n")[:10] for t in tokens]
    for h in range(n_heads):
        ax = axes[h]
        vals = attn[layer][0, h, T - 1, :T]
        colors = plt.cm.Blues(vals / max(vals.max(), 1e-6))
        ax.barh(range(T), vals, color=colors)
        ax.set_yticks(range(T))
        ax.set_yticklabels(short, fontsize=max(5, min(8, 80 // T)), color=TEXT_CLR)
        ax.set_title(f"Head {h}", color=TEXT_CLR, fontsize=9)
        ax.invert_yaxis()
        _style_ax(ax)
    for h in range(n_heads, len(axes)):
        axes[h].set_visible(False)
    fig.suptitle(f"Layer {layer} – Last token attends to …", color=TEXT_CLR, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _fig_to_pil(fig)


def plot_token_confidence(token_info):
    if not token_info:
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor(BG)
        ax.text(0.5, 0.5, "No tokens generated", ha="center", va="center", color=TEXT_CLR)
        _style_ax(ax)
        return _fig_to_pil(fig)

    tokens = [t["token"] for t in token_info]
    probs = [t["probability"] for t in token_info]
    entropies = [t["entropy"] for t in token_info]
    n = len(tokens)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, n * 0.35 + 1), 5.5), sharex=True)
    fig.patch.set_facecolor(BG)

    # confidence bars
    cmap = plt.cm.RdYlGn
    colors = [cmap(p) for p in probs]
    ax1.bar(range(n), probs, color=colors, edgecolor="none", width=0.8)
    ax1.set_ylabel("P(token)", color=TEXT_CLR, fontsize=9)
    ax1.set_title("Per-Token Confidence", color=TEXT_CLR, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1)
    _style_ax(ax1)
    ax1.grid(axis="y", color=GRID_CLR, linewidth=0.5)

    # entropy bars
    ax2.bar(range(n), entropies, color="#58a6ff", edgecolor="none", width=0.8)
    ax2.set_ylabel("Entropy (nats)", color=TEXT_CLR, fontsize=9)
    ax2.set_title("Distribution Entropy (higher = more uncertain)", color=TEXT_CLR, fontsize=11, fontweight="bold")
    _style_ax(ax2)
    ax2.grid(axis="y", color=GRID_CLR, linewidth=0.5)

    short = [t.replace("\n", "\\n")[:8] for t in tokens]
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(short, rotation=90, fontsize=max(5, min(8, 90 // n)), color=TEXT_CLR)

    fig.tight_layout()
    return _fig_to_pil(fig)


def plot_ffn_norms(ffn_gate, tokens, n_layers):
    T = len(tokens)
    norms = np.zeros((n_layers, T))
    for li in range(n_layers):
        if li in ffn_gate:
            norms[li] = np.linalg.norm(ffn_gate[li][0, :T, :], axis=-1)

    fig, ax = plt.subplots(figsize=(min(14, T * 0.45 + 2), max(3.5, n_layers * 0.55 + 1)))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(norms, aspect="auto", cmap="inferno", interpolation="nearest")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)], color=TEXT_CLR, fontsize=9)
    short = [t.replace("\n", "\\n")[:10] for t in tokens]
    ax.set_xticks(range(T))
    ax.set_xticklabels(short, rotation=90, fontsize=max(5, min(8, 100 // T)), color=TEXT_CLR)
    ax.set_xlabel("Token", color=TEXT_CLR, fontsize=10)
    ax.set_ylabel("Layer", color=TEXT_CLR, fontsize=10)
    ax.set_title("FFN Gate Activation Norms", color=TEXT_CLR, fontsize=12, fontweight="bold")
    _style_ax(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_pil(fig)


def plot_residual_norms(res_pre, res_post, tokens, n_layers):
    T = len(tokens)
    pre, post = [], []
    for i in range(n_layers):
        if i in res_pre:
            pre.append(np.linalg.norm(res_pre[i][0, :T, :], axis=-1).mean())
            post.append(np.linalg.norm(res_post[i][0, :T, :], axis=-1).mean())

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(BG)
    x = range(len(pre))
    ax.plot(x, pre, "o-", color="#58a6ff", label="Pre-block", linewidth=2, markersize=5)
    ax.plot(x, post, "s-", color="#f85149", label="Post-block", linewidth=2, markersize=5)
    ax.set_xlabel("Layer", color=TEXT_CLR, fontsize=10)
    ax.set_ylabel("Mean Residual ‖x‖", color=TEXT_CLR, fontsize=10)
    ax.set_title("Residual Stream Norms Across Layers", color=TEXT_CLR, fontsize=12, fontweight="bold")
    ax.legend(facecolor=BG_INNER, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _style_ax(ax)
    ax.grid(True, alpha=0.15, color=TEXT_CLR)
    return _fig_to_pil(fig)


def plot_top_neurons(ffn_gate, layer, tokens, top_n=25):
    T = len(tokens)
    if layer not in ffn_gate:
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor(BG)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=TEXT_CLR)
        return _fig_to_pil(fig)

    gate = ffn_gate[layer][0, :T, :]
    mean_act = np.abs(gate).mean(axis=0)
    top_idx = np.argsort(mean_act)[-top_n:][::-1]

    data = gate[:, top_idx].T
    vmax = max(abs(data.max()), abs(data.min()), 0.01)

    fig, ax = plt.subplots(figsize=(min(14, T * 0.45 + 2), max(4, top_n * 0.28 + 1)))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", interpolation="nearest", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"n{idx}" for idx in top_idx], fontsize=7, color=TEXT_CLR)
    short = [t.replace("\n", "\\n")[:10] for t in tokens]
    ax.set_xticks(range(T))
    ax.set_xticklabels(short, rotation=90, fontsize=max(5, min(8, 100 // T)), color=TEXT_CLR)
    ax.set_xlabel("Token", color=TEXT_CLR, fontsize=10)
    ax.set_ylabel("Neuron index", color=TEXT_CLR, fontsize=10)
    ax.set_title(f"Layer {layer} · Top {top_n} FFN Neurons (by mean |activation|)",
                 color=TEXT_CLR, fontsize=11, fontweight="bold")
    _style_ax(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_pil(fig)


def plot_ffn_sparsity(ffn_gate, tokens, n_layers, threshold=0.01):
    """Fraction of near-zero FFN neurons per layer per token."""
    T = len(tokens)
    sparsity = np.zeros((n_layers, T))
    for li in range(n_layers):
        if li in ffn_gate:
            g = ffn_gate[li][0, :T, :]
            sparsity[li] = (np.abs(g) < threshold).mean(axis=-1)

    fig, ax = plt.subplots(figsize=(min(14, T * 0.45 + 2), max(3.5, n_layers * 0.55 + 1)))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(sparsity, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)], color=TEXT_CLR, fontsize=9)
    short = [t.replace("\n", "\\n")[:10] for t in tokens]
    ax.set_xticks(range(T))
    ax.set_xticklabels(short, rotation=90, fontsize=max(5, min(8, 100 // T)), color=TEXT_CLR)
    ax.set_xlabel("Token", color=TEXT_CLR, fontsize=10)
    ax.set_ylabel("Layer", color=TEXT_CLR, fontsize=10)
    ax.set_title(f"FFN Activation Sparsity (fraction |gate| < {threshold})",
                 color=TEXT_CLR, fontsize=11, fontweight="bold")
    _style_ax(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Sparsity")
    return _fig_to_pil(fig)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 100-sample accuracy benchmark (runs at startup, reusable)
# ---------------------------------------------------------------------------

def run_benchmark(model, tokenizer, device, seed=0, max_tokens=60,
                  temperature=0.7, n_samples=100):
    """Run n_samples Story QA inferences and return (summary_md, rows_list)."""
    N = n_samples
    pool = _load_story_qa_pool(split="test", max_docs=2100)
    if not pool:
        return "❌ story_qa_v4_plaintext_shards not found.", []
    pool = [d for d in pool if d["qa_pairs"]]
    if len(pool) < N:
        return (f"⚠ Only {len(pool)} docs with QA pairs available "
                f"(need {N}).", [])

    rng = random.Random(int(seed))
    samples = rng.sample(pool, N)

    rows = []
    qa_triples = []
    for i, doc in enumerate(samples):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Benchmark: {i+1}/{N} …")
        qa = doc["qa_pairs"][0]
        prompt = f'{doc["story"]}\n\nQuestion: {qa["question"]}\nAnswer:'
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if input_ids.shape[1] > 480:
            input_ids = input_ids[:, :480]
        prompt_len = input_ids.shape[1]

        gen_ids = input_ids.clone()
        for _ in range(int(max_tokens)):
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(gen_ids)
            next_logits = logits[:, -1, :].float()
            next_logits = next_logits / max(float(temperature), 1e-8)
            sampled = torch.multinomial(
                F.softmax(next_logits, dim=-1), num_samples=1)
            tok_id = sampled[0, 0].item()
            gen_ids = torch.cat([gen_ids, sampled], dim=1)
            if (tok_id == tokenizer.eos_token_id
                    or "\n" in tokenizer.decode([tok_id])):
                break

        model_answer = tokenizer.decode(
            gen_ids[0, prompt_len:].tolist()).strip().split("\n")[0]

        qa_triples.append({
            "question": qa["question"],
            "gold": qa["answer"],
            "model": model_answer,
        })
        rows.append([
            i + 1, doc["id"],
            qa["question"][:120],
            qa["answer"][:120],
            model_answer[:120],
            "",  # placeholder
        ])

    # Batch LLM judge in chunks
    print(f"  Benchmark: judging {N} answers with Claude …")
    CHUNK = 25
    all_matches = []
    for start in range(0, len(qa_triples), CHUNK):
        chunk = qa_triples[start:start + CHUNK]
        all_matches.extend(llm_judge_matches(chunk))

    n_correct = 0
    for idx, m in enumerate(all_matches):
        rows[idx][5] = "✅" if m else "❌"
        n_correct += int(m)

    accuracy = n_correct / N
    judge = "Claude" if os.environ.get("ANTHROPIC_API_KEY") else "heuristic"
    summary = (
        f"## 🎯 Benchmark Complete\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Samples | **{N}** |\n"
        f"| Correct ({judge} judge) | **{n_correct}/{N}** |\n"
        f"| **Accuracy** | **{accuracy:.1%}** |\n"
    )
    print(f"  Benchmark done: {n_correct}/{N} = {accuracy:.1%}")
    return summary, rows


# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------

def build_static_app(precomputed: dict):
    """Build a fully static Gradio dashboard from pre-computed results.

    No model is held in memory — every output was generated at startup and the
    GPU was released before this function is called.
    """
    pc = precomputed  # shorthand

    color_map = {
        "prompt": "#6e7681",
        "high": "#2ea043",
        "mid": "#d29922",
        "low": "#f85149",
    }

    with gr.Blocks(title="Weightless – Model Inspector") as app:
        gr.Markdown(
            "# 🔬 Weightless – Static Inference Dashboard\n"
            "All outputs were **pre-computed at startup** from a random "
            "Story QA sample. No GPU is held by this dashboard.\n"
        )

        # ==== Tab 1: Generation Example ====
        with gr.Tab("🖊️ Generation Example"):
            gr.Markdown(
                f"**Prompt (from Story QA):**\n\n"
                f"> {pc['gen_prompt'][:500]}{'…' if len(pc['gen_prompt']) > 500 else ''}\n\n"
                f"**Generation parameters:** temp=0.8, top-k=50, top-p=0.9, max_tokens=80"
            )

            gr.HighlightedText(
                value=pc["gen_highlighted"],
                label="Generated text (green = high conf, yellow = medium, red = low)",
                color_map=color_map,
                show_legend=True,
                interactive=False,
            )
            gr.Textbox(value=pc["gen_plain_text"], label="Plain text",
                       lines=3, interactive=False)
            gr.Image(value=pc["gen_conf_img"],
                     label="Token Confidence & Entropy", type="pil")
            gr.Dataframe(
                headers=["#", "Token", "Prob", "Entropy", "Top-5 Candidates"],
                value=pc["gen_token_rows"],
                label="Per-Token Breakdown", wrap=True,
            )

        # ==== Tab 2: Attention ====
        with gr.Tab("👁️ Attention Patterns"):
            gr.Markdown(
                "Attention patterns from the pre-computed generation example."
            )
            for layer_idx, imgs in pc["attn_images"].items():
                gr.Markdown(f"### Layer {layer_idx}")
                with gr.Row():
                    gr.Image(value=imgs["heatmap"],
                             label=f"L{layer_idx} Head 0 Heatmap", type="pil")
                    gr.Image(value=imgs["last_tok"],
                             label=f"L{layer_idx} Last Token → Context", type="pil")

        # ==== Tab 3: Activations ====
        with gr.Tab("⚡ Activations"):
            gr.Markdown(
                "FFN gate activations and residual-stream dynamics from "
                "the pre-computed generation example."
            )
            with gr.Row():
                gr.Image(value=pc["ffn_norms_img"],
                         label="FFN Gate Norms (all layers × tokens)", type="pil")
                gr.Image(value=pc["ffn_sparsity_img"],
                         label="FFN Sparsity (fraction ≈ 0)", type="pil")
            with gr.Row():
                gr.Image(value=pc["residual_img"],
                         label="Residual Stream Norms", type="pil")
                gr.Image(value=pc["top_neurons_img"],
                         label="Top FFN Neurons (layer 0)", type="pil")

        # ==== Tab 4: Story QA Inference ====
        with gr.Tab("📖 Story QA Inference"):
            gr.Markdown(
                "# Story QA – Pre-computed Inference Results\n"
                "10 random examples from `story_qa_v4_plaintext_shards` test set.\n"
                "Correctness judged by Claude."
            )
            gr.Markdown(value=pc["sqa_summary"])
            gr.Dataframe(
                headers=["#", "Doc ID", "Story (excerpt)", "Question",
                         "Gold Answer", "Model Answer", "Match?",
                         "Story PPL", "Avg Conf", "Top-5 First Token"],
                value=pc["sqa_rows"],
                label="Story QA Results", wrap=True,
            )
            gr.Markdown(value=pc["sqa_details"])

            # ----- 100-sample benchmark -----
            gr.Markdown("---\n## 🎯 100-Sample Accuracy Benchmark\n"
                        "Evaluated at startup on 100 random test examples "
                        "(seed 0, temp 0.7). Judged by Claude.")
            gr.Markdown(value=pc["bench_summary"])
            gr.Dataframe(
                headers=["#", "Doc ID", "Question", "Gold", "Model", "Match?"],
                value=pc["bench_rows"] if pc["bench_rows"] else None,
                label="Benchmark Results (100 samples)", wrap=True,
            )

        # ==== Tab 5: Model Info ====
        with gr.Tab("📋 Model Info & Metrics"):
            gr.Markdown(value=pc["model_info"])

    return app


# ---------------------------------------------------------------------------
# Pre-computation: generate everything, then release GPU
# ---------------------------------------------------------------------------

def precompute_all(model, tokenizer, device, n_layers, n_heads):
    """Run all inference & visualization at startup, return a dict of outputs."""
    instrument = ModelInstrument(model)
    MAX_DISPLAY_TOKENS = 64
    results = {}

    def _tok_labels(ids):
        return [tokenizer.decode([tid]).replace("\n", "\\n") for tid in ids]

    # --- 1. Pick a random Story QA example as the generation prompt ---
    print("  [1/6] Picking random Story QA example for generation …")
    pool = _load_story_qa_pool(split="test", max_docs=2100)
    pool = [d for d in pool if d["qa_pairs"]]
    rng = random.Random(99)
    sample_doc = rng.choice(pool)
    qa = sample_doc["qa_pairs"][0]
    gen_prompt = f'{sample_doc["story"]}\n\nQuestion: {qa["question"]}\nAnswer:'
    results["gen_prompt"] = gen_prompt

    # --- 2. Run generation with diagnostics ---
    print("  [2/6] Running generation with diagnostics …")
    gen_result = generate_with_diagnostics(
        model, tokenizer, device, instrument,
        prompt=gen_prompt[:1500],  # ensure reasonable length
        max_tokens=80,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    # Build highlighted text
    highlighted = [(gen_prompt[:300] + ("…" if len(gen_prompt) > 300 else ""), "prompt")]
    for t in gen_result["tokens"]:
        p = t["probability"]
        label = "high" if p > 0.7 else ("mid" if p > 0.3 else "low")
        highlighted.append((t["token"], label))
    results["gen_highlighted"] = highlighted
    results["gen_plain_text"] = gen_result["generated_text"]

    # Confidence plot
    results["gen_conf_img"] = plot_token_confidence(gen_result["tokens"])

    # Token table
    gen_rows = []
    for i, t in enumerate(gen_result["tokens"]):
        top5 = " | ".join(f"{tok}({p:.0%})" for tok, p in
                          zip(t["top_tokens"][:5], t["top_probs"][:5]))
        gen_rows.append([
            i + 1, repr(t["token"]),
            f"{t['probability']:.2%}", f"{t['entropy']:.2f}", top5,
        ])
    results["gen_token_rows"] = gen_rows

    # --- 3. Attention visualizations ---
    print("  [3/6] Generating attention visualizations …")
    ids = gen_result["input_ids"][:MAX_DISPLAY_TOKENS]
    tokens = _tok_labels(ids)
    attn_images = {}
    for layer in range(n_layers):
        heatmap = plot_attention_heatmap(gen_result["attention"], tokens, layer, 0)
        last_tok = plot_last_token_attention(gen_result["attention"], tokens, layer)
        attn_images[layer] = {"heatmap": heatmap, "last_tok": last_tok}
    results["attn_images"] = attn_images

    # --- 4. Activation visualizations ---
    print("  [4/6] Generating activation visualizations …")
    results["ffn_norms_img"] = plot_ffn_norms(gen_result["ffn_gate"], tokens, n_layers)
    results["residual_img"] = plot_residual_norms(
        gen_result["residual_pre"], gen_result["residual_post"], tokens, n_layers)
    results["top_neurons_img"] = plot_top_neurons(gen_result["ffn_gate"], 0, tokens)
    results["ffn_sparsity_img"] = plot_ffn_sparsity(gen_result["ffn_gate"], tokens, n_layers)

    # --- 5. Story QA spot-check (10 examples) ---
    print("  [5/6] Running Story QA spot-check (10 examples) …")
    sqa_results = _run_story_qa_spotcheck(model, tokenizer, device, pool,
                                           n_examples=10, seed=42)
    results["sqa_summary"] = sqa_results["summary"]
    results["sqa_rows"] = sqa_results["rows"]
    results["sqa_details"] = sqa_results["details"]

    # --- 6. 100-sample benchmark ---
    print("  [6/6] Running 100-sample benchmark …")
    bench_summary, bench_rows = run_benchmark(
        model, tokenizer, device, seed=0, max_tokens=60, temperature=0.7)
    results["bench_summary"] = bench_summary
    results["bench_rows"] = bench_rows

    # --- Model info (static) ---
    total_params = model.count_parameters(count_zeros=True)
    nonzero_params = model.count_parameters(count_zeros=False)
    sparsity = 1 - nonzero_params / total_params if total_params > 0 else 0
    results["model_info"] = f"""## Architecture
| Property | Value |
|----------|-------|
| d_model | {model.d_model} |
| n_layers | {model.n_layers} |
| n_heads | {model.n_heads} |
| n_kv_heads | {model.n_kv_heads} |
| d_ff | {model.d_ff} |
| vocab_size | {model.vocab_size:,} |

## Parameters
| Metric | Value |
|--------|-------|
| Total parameters | **{total_params:,}** |
| Non-zero parameters | **{nonzero_params:,}** |
| Sparsity | {sparsity:.2%} |"""

    # Clean up hooks
    instrument.remove_hooks()

    return results


def _run_story_qa_spotcheck(model, tokenizer, device, pool,
                             n_examples=10, seed=42):
    """Run a small Story QA spot-check and return formatted outputs."""
    rng = random.Random(seed)
    samples = rng.sample(pool, min(n_examples, len(pool)))

    inference_data = []
    for i, doc in enumerate(samples):
        if not doc["qa_pairs"]:
            continue
        qa = doc["qa_pairs"][0]
        prompt = f'{doc["story"]}\n\nQuestion: {qa["question"]}\nAnswer:'
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if input_ids.shape[1] > 480:
            input_ids = input_ids[:, :480]
        prompt_len = input_ids.shape[1]

        # Story perplexity
        story_ids = tokenizer.encode(doc["story"], return_tensors="pt").to(device)
        if story_ids.shape[1] > 1:
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                story_logits = model(story_ids)
            sl = story_logits[:, :-1, :].contiguous()
            st = story_ids[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1)).item()
            ppl = float(np.exp(loss))
        else:
            loss, ppl = float('nan'), float('nan')

        # Generate answer
        gen_ids = input_ids.clone()
        gen_tokens_info = []
        for _ in range(60):
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(gen_ids)
            next_logits = logits[:, -1, :].float()
            probs = F.softmax(next_logits, dim=-1)
            top5p, top5i = torch.topk(probs, 5, dim=-1)
            top5_toks = [tokenizer.decode([t]) for t in top5i[0].tolist()]
            next_logits = next_logits / 0.7
            sampled = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            tok_id = sampled[0, 0].item()
            gen_tokens_info.append({
                "prob": probs[0, tok_id].item(),
                "top5": list(zip(top5_toks, top5p[0].tolist())),
            })
            gen_ids = torch.cat([gen_ids, sampled], dim=1)
            if tok_id == tokenizer.eos_token_id or "\n" in tokenizer.decode([tok_id]):
                break

        model_answer = tokenizer.decode(gen_ids[0, prompt_len:].tolist()).strip().split("\n")[0]
        avg_conf = np.mean([t["prob"] for t in gen_tokens_info]) if gen_tokens_info else 0
        top5_str = ""
        if gen_tokens_info:
            top5_str = " | ".join(f"{t}({p:.0%})" for t, p in gen_tokens_info[0]["top5"])

        inference_data.append({
            "idx": i, "doc": doc, "qa": qa, "model_answer": model_answer,
            "ppl": ppl, "avg_conf": avg_conf, "top5_str": top5_str,
        })

    # LLM judge
    qa_triples = [{"question": d["qa"]["question"], "gold": d["qa"]["answer"],
                    "model": d["model_answer"]} for d in inference_data]
    match_results = llm_judge_matches(qa_triples)

    # Format outputs
    rows = []
    details_parts = []
    total_ppl = total_conf = n_close = 0
    judge_method = "Claude" if os.environ.get("ANTHROPIC_API_KEY") else "heuristic"

    for d, is_match in zip(inference_data, match_results):
        doc, qa, i = d["doc"], d["qa"], d["idx"]
        rows.append([
            i + 1, doc["id"],
            doc["story"][:100] + ("…" if len(doc["story"]) > 100 else ""),
            qa["question"], qa["answer"], d["model_answer"][:150],
            "✅" if is_match else "❌",
            f"{d['ppl']:.1f}" if not np.isnan(d['ppl']) else "N/A",
            f"{d['avg_conf']:.0%}", d["top5_str"],
        ])
        total_ppl += d["ppl"] if not np.isnan(d["ppl"]) else 0
        total_conf += d["avg_conf"]
        n_close += int(is_match)

        details_parts.append(
            f"### Example {i+1}: `{doc['id']}`\n\n"
            f"**Story:** {doc['story'][:400]}{'…' if len(doc['story']) > 400 else ''}\n\n")
        for qi, qapair in enumerate(doc["qa_pairs"]):
            marker = "👉 " if qi == 0 else ""
            details_parts.append(f"{marker}**Q{qi+1}:** {qapair['question']}  \n**Gold:** {qapair['answer']}")
            if qi == 0:
                details_parts.append(f"  \n**Model:** {d['model_answer']}  {'✅' if is_match else '❌'}")
            details_parts.append("\n")
        if doc.get("definitions"):
            details_parts.append(f"**Definitions:** " + "; ".join(doc["definitions"]) + "\n")
        details_parts.append("---\n")

    n_done = len(inference_data)
    avg_ppl = f"{total_ppl / n_done:.1f}" if n_done else "N/A"
    avg_conf = f"{total_conf / n_done:.0%}" if n_done else "N/A"
    match_pct = f"{n_close/n_done:.0%}" if n_done > 0 else "N/A"
    summary = (
        f"### Results Summary\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Examples run | **{n_done}** |\n"
        f"| Semantic match ({judge_method} judge) | **{n_close}/{n_done}** ({match_pct}) |\n"
        f"| Average story perplexity | **{avg_ppl}** |\n"
        f"| Average answer confidence | **{avg_conf}** |\n"
    )

    return {"summary": summary, "rows": rows, "details": "\n".join(details_parts)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Static inference dashboard (no GPU held)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint")
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "baseline_plus", "copy_gate"])
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # ----- Load model onto GPU -----
    print("  Loading model …")
    model = create_model(
        variant=args.model,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )

    ckpt = args.checkpoint
    if ckpt is None:
        for auto in [
            "checkpoints/baseline_full_8gpu.pt",
            "checkpoints/baseline_full.pt",
        ]:
            if os.path.exists(auto):
                ckpt = auto
                break

    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        strict = (args.model != "copy_gate")
        model.load_state_dict(sd, strict=strict)
        print(f"  ✓ Loaded checkpoint: {ckpt} (strict={strict})")
    else:
        print("  ⚠ No checkpoint found — using random-init model.")

    model.to(device)
    model.eval()

    # ----- Pre-compute everything -----
    print("  Pre-computing all outputs (GPU will be released after) …")
    precomputed = precompute_all(model, tokenizer, device,
                                 args.n_layers, args.n_heads)

    # ----- Release GPU -----
    print("  Releasing GPU …")
    model.cpu()
    del model
    del sd
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print("  ✓ GPU released — dashboard is fully static")

    # Check GPU is free
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e6
        print(f"  GPU memory still allocated: {mem:.0f} MB")

    # ----- Build & launch static Gradio -----
    print(f"  Building static UI (port {args.port}) …")
    app = build_static_app(precomputed)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
    )


if __name__ == "__main__":
    main()
