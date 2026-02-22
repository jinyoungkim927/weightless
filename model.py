"""Starter model for the FineWeb challenge — everything-optimized variant.

Combines all modifications:
  - ReLU² FFN (2 matrices instead of SwiGLU's 3)
  - QK normalization in attention
  - Logit softcapping
  - Per-layer residual scalars
  - Copy gate mechanism

Variants:
  - baseline:       dense transformer (supports all feature flags)
  - baseline_plus:  GQA + top-k FFN (no feature flags)
  - copy_gate:      baseline + learnable copy gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RotaryPositionalEmbedding

# GPT-2 tokenizer vocab size
VOCAB_SIZE = 50257
SEQ_LEN = 512  # 513 - 1 for causal LM


# ============================================================================
# Baseline: dense transformer
# ============================================================================

class SimpleTransformer(nn.Module):
    """A minimal transformer for language modeling."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        ffn_type: str = "swiglu",
        qk_norm: bool = False,
        softcap: float = 0.0,
        use_resid_scalars: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads  # MHA: all heads are KV heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        self.weight_tied = True
        self.ffn_type = ffn_type
        self.softcap = softcap
        self.use_resid_scalars = use_resid_scalars

        # Token embeddings (no learned positional embedding - using RoPE)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # RoPE for positional encoding (applied in attention)
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.head_dim,
            max_seq_len=max_seq_len,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_heads, d_ff, dropout,
                             ffn_type=ffn_type, qk_norm=qk_norm)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Per-layer residual scalars
        if use_resid_scalars:
            self.resid_lambdas = nn.Parameter(torch.ones(n_layers))
            self.x0_lambdas = nn.Parameter(torch.zeros(n_layers) + 0.1)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights(n_layers)

    def _init_weights(self, n_layers):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                std = fan_in ** -0.5
                if "proj" in name or ".w2" in name:
                    std = std / (2 * n_layers) ** 0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=module.weight.shape[1] ** -0.5)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids)
        x = self.dropout(x)

        positions = torch.arange(0, T, dtype=torch.long, device=device)

        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        # Save initial embedding for residual scalar mixing
        if self.use_resid_scalars:
            x0 = x

        for i, layer in enumerate(self.layers):
            if self.use_resid_scalars:
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = layer(x, causal_mask, attention_mask, self.rope, positions)

        x = self.ln_f(x)
        logits = self.head(x)

        # Logit softcapping
        if self.softcap > 0:
            logits = self.softcap * torch.tanh(logits.float() / self.softcap)

        return logits

    def count_parameters(self, count_zeros: bool = False):
        if count_zeros:
            return sum(p.numel() for p in self.parameters())
        else:
            return sum((p != 0).sum().item() for p in self.parameters())


# ============================================================================
# Attention
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional GQA, QK normalization, and RoPE + Flash Attention."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float,
                 qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qk_norm = qk_norm
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // n_kv_heads  # how many Q heads per KV head

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, causal_mask, attention_mask, rope, positions):
        B, T, C = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = rope(q, positions)
        k = rope(k, positions)

        # QK normalization (after RoPE, no learnable params)
        if self.qk_norm:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))

        # Expand KV heads for GQA: (B, n_kv_heads, T, hd) -> (B, n_heads, T, hd)
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim)
            k = k.reshape(B, self.n_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim)
            v = v.reshape(B, self.n_heads, T, self.head_dim)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


# ============================================================================
# FFN
# ============================================================================

class SwiGLUFF(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, device=device, dtype=dtype, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, device=device, dtype=dtype, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ReLU2FF(nn.Module):
    """ReLU squared feed-forward network (2 matrices instead of SwiGLU's 3)."""
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.c_fc = nn.Linear(d_model, d_ff, bias=bias)
        self.c_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class TopKSwiGLUFF(nn.Module):
    """SwiGLU FFN with top-k activation sparsity."""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        top_k: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k if top_k is not None else d_ff // 4
        self.w1 = nn.Linear(d_model, d_ff, device=device, dtype=dtype, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, device=device, dtype=dtype, bias=False)

    def forward(self, x):
        gate = F.silu(self.w1(x)) * self.w3(x)  # (B, T, d_ff)
        # Top-k: zero out all but the top-k activations
        if self.top_k < self.d_ff:
            topk_vals, topk_idx = torch.topk(gate.abs(), self.top_k, dim=-1)
            mask = torch.zeros_like(gate)
            mask.scatter_(-1, topk_idx, 1.0)
            gate = gate * mask
        return self.w2(gate)


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, dropout: float, ffn_top_k: int | None = None,
                 ffn_type: str = "swiglu", qk_norm: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, n_kv_heads, dropout,
                                       qk_norm=qk_norm)
        self.ln2 = nn.LayerNorm(d_model)
        if ffn_type == "relu2":
            self.ff = ReLU2FF(d_model, d_ff)
        elif ffn_top_k is not None:
            self.ff = TopKSwiGLUFF(d_model, d_ff, top_k=ffn_top_k)
        else:
            self.ff = SwiGLUFF(d_model, d_ff)

    def forward(self, x, causal_mask, attention_mask, rope, positions):
        x = x + self.attn(self.ln1(x), causal_mask, attention_mask, rope, positions)
        x = x + self.ff(self.ln2(x))
        return x


# ============================================================================
# Baseline+ : GQA + top-k FFN
# ============================================================================

class BaselinePlusTransformer(SimpleTransformer):
    """Baseline with GQA + top-k FFN activation sparsity."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_layers: int = 8,
        d_ff: int = 2048,
        ffn_top_k: int | None = None,  # defaults to d_ff // 4
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
    ):
        # Bypass SimpleTransformer.__init__ -- we rebuild with GQA + top-k
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        self.ffn_top_k = ffn_top_k if ffn_top_k is not None else d_ff // 4
        self.weight_tied = True

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.head_dim,
            max_seq_len=max_seq_len,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ff, dropout,
                             ffn_top_k=self.ffn_top_k)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self._init_weights(n_layers)


# ============================================================================
# Copy Gate
# ============================================================================

class CopyGate(nn.Module):
    """Learnable gate that decides per-position how much to copy from input.

    Parameters: d_model weights + 1 bias = 769 for d_model=768.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        # Initialize bias negative so p_copy starts near 0 (generation-dominant)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -3.0)

    def forward(self, h):
        """Returns p_copy in [0, 1] for each position. Shape: (B, T, 1)."""
        return torch.sigmoid(self.linear(h))


class CopyGateTransformer(SimpleTransformer):
    """SimpleTransformer with a learnable copy gate mechanism.

    Adds a small gate (769 params for d_model=768) that blends between
    the LM head's generation distribution and a copy distribution formed
    by attending over input token positions and scattering into vocab space.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.copy_gate = CopyGate(self.d_model)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids)
        x = self.dropout(x)

        positions = torch.arange(0, T, dtype=torch.long, device=device)
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        # Residual scalar mixing
        if self.use_resid_scalars:
            x0 = x

        for i, layer in enumerate(self.layers):
            if self.use_resid_scalars:
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = layer(x, causal_mask, attention_mask, self.rope, positions)

        x = self.ln_f(x)

        # Generation logits from LM head
        gen_logits = self.head(x)  # (B, T, vocab_size)

        # Logit softcapping
        if self.softcap > 0:
            gen_logits = self.softcap * torch.tanh(gen_logits.float() / self.softcap)

        # Copy gate probability
        p_copy = self.copy_gate(x)  # (B, T, 1)

        # Copy distribution: dot-product attention over input embeddings
        emb = self.token_emb(input_ids)  # (B, T, d_model)
        copy_scores = torch.bmm(x, emb.transpose(1, 2))  # (B, T, T)
        copy_scores = copy_scores / (self.d_model ** 0.5)
        copy_scores.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
        copy_attn = F.softmax(copy_scores, dim=-1)  # (B, T, T)

        # Scatter copy attention into vocab space
        copy_probs = torch.zeros_like(gen_logits)
        copy_probs.scatter_add_(
            2,
            input_ids.unsqueeze(1).expand(-1, T, -1),
            copy_attn,
        )

        # Blend generation and copy distributions
        gen_probs = F.softmax(gen_logits, dim=-1)
        blended = (1 - p_copy) * gen_probs + p_copy * copy_probs

        # Return log-probs (compatible with cross_entropy: CE(log(p), y) = -log(p_y))
        return torch.log(blended + 1e-10)


# ============================================================================
# Factory
# ============================================================================

# Params unsupported by BaselinePlusTransformer
_BP_UNSUPPORTED = {"qk_norm", "softcap", "use_resid_scalars", "ffn_type"}


def create_model(variant: str = "baseline", **kwargs):
    """Factory function to create a model.

    Args:
        variant: "baseline", "baseline_plus", or "copy_gate"
        **kwargs: passed to the model constructor
    """
    if variant == "baseline_plus":
        bp_kwargs = {k: v for k, v in kwargs.items() if k not in _BP_UNSUPPORTED}
        return BaselinePlusTransformer(**bp_kwargs)
    elif variant == "copy_gate":
        return CopyGateTransformer(**kwargs)
    else:
        return SimpleTransformer(**kwargs)


def get_inference_profile(model):
    """Compute inference profile (bytes per token) for the model."""
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    d = raw.d_model
    n_layers = raw.n_layers
    n_heads = raw.n_heads
    n_kv_heads = getattr(raw, "n_kv_heads", n_heads)
    d_ff = raw.d_ff
    head_dim = d // n_heads
    vocab_size = raw.vocab_size

    # Embedding + LM head (weight-tied = counted once)
    emb_bytes = vocab_size * d * 2  # bf16

    # Attention: Q, K, V, O projections per layer
    attn_bytes = n_layers * (
        d * n_heads * head_dim       # Q
        + d * n_kv_heads * head_dim  # K
        + d * n_kv_heads * head_dim  # V
        + n_heads * head_dim * d     # O
    ) * 2  # bf16

    # FFN per layer
    ffn_type = getattr(raw, "ffn_type", "swiglu")
    if ffn_type == "relu2":
        ffn_bytes = n_layers * (d * d_ff + d_ff * d) * 2  # 2 matrices
    else:
        ffn_bytes = n_layers * (d * d_ff * 3) * 2  # 3 matrices (SwiGLU)

    # LayerNorm
    ln_bytes = n_layers * 2 * d * 2 + d * 2  # per-layer + final

    total = emb_bytes + attn_bytes + ffn_bytes + ln_bytes
    return total


if __name__ == "__main__":
    for variant in ["baseline", "baseline_plus"]:
        print(f"\n{'='*60}")
        print(f"  {variant}")
        print(f"{'='*60}")
        model = create_model(variant=variant)
        total_params = model.count_parameters(count_zeros=True)
        nonzero_params = model.count_parameters(count_zeros=False)
        print(f"  Total parameters:    {total_params:,}")
        print(f"  Non-zero parameters: {nonzero_params:,}")
