"""Starter model for the FineWeb challenge.

Your goal: achieve val loss < 3.3 with the most efficient model possible.
Modify this model architecture to be as sparse/efficient as possible.

Two variants are provided:
  - baseline:       dense transformer (the starting point)
  - baseline_plus:  GQA + top-k FFN activation sparsity (shows clear improvement)
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
    """A minimal transformer for language modeling.
    
    This is a basic starter -- you should modify/replace this
    to maximize efficiency while achieving val loss < 3.3.
    """
    
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
            TransformerBlock(d_model, n_heads, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
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
        
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask, self.rope, positions)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def count_parameters(self, count_zeros: bool = False):
        """Count model parameters.
        
        Args:
            count_zeros: If False, only count non-zero parameters
        
        Returns:
            Total parameter count
        """
        if count_zeros:
            return sum(p.numel() for p in self.parameters())
        else:
            return sum((p != 0).sum().item() for p in self.parameters())



# ============================================================================
# Attention
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional GQA and RoPE + Flash Attention.
    
    When n_kv_heads < n_heads, uses Grouped Query Attention:
    Q has n_heads, K/V have n_kv_heads, heads are repeated for the
    dot product.
    """
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
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


class TopKSwiGLUFF(nn.Module):
    """SwiGLU FFN with top-k activation sparsity.

    After computing the gate activations (w1, w3), only the top-k
    neurons are kept.  During inference this means only k rows of w2
    need to be read from memory (instead of all d_ff rows).

    Training uses the full d_ff (via straight-through or just dense)
    to keep gradients flowing; the top-k mask is applied to the
    activation values so the model learns which neurons matter.
    """
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
                 d_ff: int, dropout: float, ffn_top_k: int | None = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, n_kv_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        if ffn_top_k is not None:
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
    """Baseline with two clear optimizations for efficiency:

    1. Grouped Query Attention (GQA):  n_kv_heads < n_heads
       -> fewer KV projection weights, smaller KV cache
    2. Top-k FFN activation sparsity:  only top-k neurons of w1/w3 gate
       -> only k rows of w2 read during inference
    """

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
# Factory
# ============================================================================

def create_model(variant: str = "baseline", **kwargs):
    """Factory function to create a model.

    Args:
        variant: "baseline" or "baseline_plus"
        **kwargs: passed to the model constructor
    """
    if variant == "baseline_plus":
        return BaselinePlusTransformer(**kwargs)
    else:
        return SimpleTransformer(**kwargs)


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
