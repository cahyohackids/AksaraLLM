"""
AksaraLLM Model Architecture — Production-Ready
LLaMA-3 style: GQA + RoPE + SwiGLU + RMSNorm + KV Cache

Features:
- Grouped Query Attention (GQA) for memory efficiency
- Rotary Position Embeddings (RoPE) for length generalization
- SwiGLU activation for better training dynamics
- RMSNorm for faster normalization
- KV Cache for O(1) autoregressive decoding
- Optional gradient checkpointing for memory-efficient training
- Weight tying between embedding and output head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .config import ModelConfig

__all__ = ["AksaraLLM", "RMSNorm", "RoPE", "Attention", "FeedForward", "TransformerBlock"]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm — no mean subtraction, just variance scaling.
    Used in LLaMA, Gemma, and Mistral architectures.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class RoPE(nn.Module):
    """Rotary Position Embedding.

    Encodes position information directly into attention queries and keys
    via rotation, enabling better length generalization than absolute embeddings.

    Both ``max_len`` and ``theta`` are required: the 20B plan uses
    ``theta=1_000_000`` with ``max_len`` up to 32K, while the 1.5B smoke test
    uses ``theta=1_000_000`` with 32K, and the 200M/500M/7B smoke configs use
    ``theta=10_000`` with 2–8K. Forgetting to pass these caused a latent bug
    where the 20B config silently ran with ``theta=10_000`` in any path that
    instantiated ``RoPE`` directly.
    """
    def __init__(self, dim: int, max_len: int, theta: float):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        angles = torch.outer(torch.arange(max_len).float(), freqs)
        self.register_buffer("cos_c", angles.cos(), persistent=False)
        self.register_buffer("sin_c", angles.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) buffers for the given sequence length."""
        return self.cos_c[:seq_len], self.sin_c[:seq_len]
    
    def get_offset(self, offset: int, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get position embeddings at a specific offset (for KV cache)."""
        return self.cos_c[offset:offset + length], self.sin_c[offset:offset + length]


def apply_rope(q: torch.Tensor, k: torch.Tensor, 
               cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys.
    
    Args:
        q: Queries (B, T, n_heads, head_dim)
        k: Keys (B, T, n_kv_heads, head_dim)
        cos_sin: Tuple of (cos, sin) tensors from RoPE
    """
    cos, sin = cos_sin
    cos = cos[None, :, None, :]  # (1, T, 1, head_dim/2)
    sin = sin[None, :, None, :]
    
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    
    q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
    return q_rot, k_rot


class Attention(nn.Module):
    """Grouped Query Attention (GQA).
    
    Uses fewer KV heads than query heads to reduce memory while
    maintaining most of the quality of Multi-Head Attention.
    Supports KV caching for efficient autoregressive generation.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
    
    def forward(self, x: torch.Tensor, cos_sin: Tuple, 
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor (B, T, dim)
            cos_sin: RoPE (cos, sin) for current positions
            kv_cache: Previous (K, V) cache or None
            
        Returns:
            output: Attention output (B, T, dim)
            new_cache: Updated (K, V) cache for next step
        """
        B, T, _ = x.shape
        
        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        q, k = apply_rope(q, k, cos_sin)
        
        # Reshape for attention: (B, n_heads, T, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Append to KV cache
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        
        # Store cache BEFORE GQA expansion (n_kv_heads, not n_heads — saves memory!)
        new_cache = (k, v)
        
        # GQA: repeat KV heads to match query heads
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1)
            k = k.reshape(B, self.n_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1)
            v = v.reshape(B, self.n_heads, -1, self.head_dim)
        
        # ⚡ OPTIMIZATION: FLASH ATTENTION 2 (SDPA)
        S = k.shape[2]  # Total sequence length including cached tokens
        
        if T == 1:
            # 1-token decode: No causal mask needed (attending to past context)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            if S == T:
                # Full prefill / training: utilize heavily-optimized native causal FlashAttention
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                # Chunked prefill (S > T): need custom causal mask
                causal = torch.triu(
                    torch.ones(T, S, dtype=torch.bool, device=x.device), 
                    diagonal=S - T + 1
                )
                # SDPA expects True for valid positions, False for masked.
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=~causal)
        
        # Reshape to final output
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.wo(out), new_cache


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network.
    
    Uses gated linear units with SiLU activation, shown to outperform
    standard ReLU/GELU FFNs in transformer language models.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up   = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm residual connections."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.dim, config.norm_eps)
        self.attn = Attention(config)
        self.ln2 = RMSNorm(config.dim, config.norm_eps)
        self.ffn = FeedForward(config)
    
    def forward(self, x: torch.Tensor, cos_sin: Tuple,
                kv_cache: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        h, cache = self.attn(self.ln1(x), cos_sin, kv_cache)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x, cache


class AksaraLLM(nn.Module):
    """AksaraLLM: Open-Source Indonesian Language Model.
    
    A decoder-only transformer using the LLaMA-3 architecture with:
    - Grouped Query Attention (GQA) for efficient KV caching
    - Rotary Position Embeddings (RoPE) for position encoding
    - SwiGLU activation in feed-forward layers
    - RMSNorm for layer normalization
    - Weight tying between input embeddings and output head
    
    Supports:
    - Standard forward pass for training
    - KV-cached forward pass for efficient generation
    - Gradient checkpointing for memory-efficient training
    
    Example:
        >>> config = ModelConfig(dim=1024, n_layers=16, n_heads=16)
        >>> model = AksaraLLM(config)
        >>> logits, loss, caches = model(input_ids, targets=target_ids)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._gradient_checkpointing = False
        
        # Token embedding (shared with output head via weight tying)
        self.emb = nn.Embedding(config.vocab_size, config.dim)
        
        # Positional encoding
        self.rope = RoPE(config.dim // config.n_heads, config.max_seq_len * 2, config.rope_theta)
        
        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying: embedding and output head share weights
        self.emb.weight = self.head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights using scaled normal distribution."""
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scale output projection by depth for better training stability
            if hasattr(module, '_is_residual'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory during training."""
        self._gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
    
    def forward(
        self, 
        x: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input token IDs (B, T)
            targets: Target token IDs for loss computation (B, T). 
                     Use -100 for masked positions.
            kv_caches: List of per-layer KV caches for incremental decoding.
                       Pass [None]*n_layers for first step, then reuse returned caches.
            
        Returns:
            logits: Output logits (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
            new_caches: Updated KV caches for next decoding step
        """
        B, T = x.shape
        h = self.emb(x)
        
        # Compute position embeddings
        if kv_caches is not None and kv_caches[0] is not None:
            # Cached: offset positions by past sequence length
            past_len = kv_caches[0][0].shape[2]
            cos_sin = self.rope.get_offset(past_len, T)
        else:
            cos_sin = self.rope(T)
        
        # Process through transformer layers
        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            
            if self._gradient_checkpointing and self.training:
                # Save memory during training
                h, new_cache = torch.utils.checkpoint.checkpoint(
                    block, h, cos_sin, cache_i, use_reentrant=False
                )
            else:
                h, new_cache = block(h, cos_sin, cache_i)
            
            new_caches.append(new_cache)
        
        # Output projection
        logits = self.head(self.norm(h))
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-100,
            )
        
        return logits, loss, new_caches
    
    @property
    def num_parameters(self) -> int:
        """Total number of parameters (including tied weights only once)."""
        return sum(p.numel() for p in self.parameters())
    
    @property 
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (
            f"AksaraLLM(\n"
            f"  params={self.num_parameters/1e6:.1f}M,\n"
            f"  layers={self.config.n_layers}, dim={self.config.dim},\n"
            f"  heads={self.config.n_heads}, kv_heads={self.config.n_kv_heads},\n"
            f"  ffn={self.config.ffn_dim}, max_seq={self.config.max_seq_len},\n"
            f"  grad_ckpt={self._gradient_checkpointing}\n"
            f")"
        )
