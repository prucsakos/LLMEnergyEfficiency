from __future__ import annotations


def flops_dense(num_params: float, num_tokens: int, multiplier: float = 2.0) -> float:
    """
    Dense-only (attention-length agnostic) FLOPs estimate.

    FLOPs ≈ multiplier * num_params * num_tokens

    Args:
        num_params: Number of model parameters.
        num_tokens: Total number of tokens processed (prompt + generated).
        multiplier: Empirical constant (default ≈ 2).

    Returns:
        Estimated FLOPs.
    """
    return multiplier * num_params * num_tokens


def flops_prefill(
    num_prompt_tokens: int, num_layers: int, hidden_dim: int
) -> float:
    """
    FLOPs for prefill stage (processing prompt tokens) with KV cache.

    F_prefill ≈ num_layers * (8 * P * d^2 + 2 * P^2 * d)

    Args:
        num_prompt_tokens: Number of prompt (input) tokens (P).
        num_layers: Number of transformer layers (L).
        hidden_dim: Hidden dimension size (d).

    Returns:
        Estimated FLOPs for the prefill stage.
    """
    P, L, d = num_prompt_tokens, num_layers, hidden_dim
    return L * (8 * P * d**2 + 2 * P**2 * d)


def flops_decode(
    num_generated_tokens: int, num_layers: int, hidden_dim: int, num_prompt_tokens: int
) -> float:
    """
    FLOPs for decode stage (generating new tokens) with KV cache.

    F_decode ≈ num_layers * (8 * G * d^2 + 4 * d * (G * P + G(G-1)/2))

    Args:
        num_generated_tokens: Number of generated tokens (G).
        num_layers: Number of transformer layers (L).
        hidden_dim: Hidden dimension size (d).
        num_prompt_tokens: Number of prompt tokens (P).

    Returns:
        Estimated FLOPs for the decode stage.
    """
    G, L, d, P = num_generated_tokens, num_layers, hidden_dim, num_prompt_tokens
    return L * (8 * G * d**2 + 4 * d * (G * P + G * (G - 1) / 2))


def flops_attention_kv(
    num_layers: int,
    hidden_dim: int,
    num_prompt_tokens: int,
    num_generated_tokens: int,
    num_params: float = 0.0,
    include_dense_anchor: bool = True,
) -> float:
    """
    Total FLOPs estimate using attention-aware formula with KV cache.

    Includes:
    - Prefill: L * (8 * P * d^2 + 2 * P^2 * d)
    - Decode:  L * (8 * G * d^2 + 4 * d * (G * P + G(G-1)/2))
    - (Optional) Dense anchor: num_params * (P + G)

    Args:
        num_layers: Number of transformer layers (L).
        hidden_dim: Hidden dimension size (d).
        num_prompt_tokens: Number of prompt (input) tokens (P).
        num_generated_tokens: Number of generated tokens (G).
        num_params: Number of model parameters (used for dense anchor).
        include_dense_anchor: Whether to include the small dense anchor term.

    Returns:
        Total estimated FLOPs.
    """
    prefill_cost = flops_prefill(num_prompt_tokens, num_layers, hidden_dim)
    decode_cost = flops_decode(num_generated_tokens, num_layers, hidden_dim, num_prompt_tokens)

    total = prefill_cost + decode_cost
    if include_dense_anchor and num_params:
        total += num_params * (num_prompt_tokens + num_generated_tokens)
    return total


# --- Convenience formatting helpers (optional) ---

def to_tflops(flops: float) -> float:
    """Convert FLOPs to tera-FLOPs (1e12)."""
    return flops / 1e12


def to_pflops(flops: float) -> float:
    """Convert FLOPs to peta-FLOPs (1e15)."""
    return flops / 1e15
