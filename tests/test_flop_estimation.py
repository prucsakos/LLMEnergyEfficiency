import math

# If your package is literally the folder "src" (you have src/__init__.py),
# this import is correct:
from src.utils.flop_estimation import (
    flops_dense,
    flops_prefill,
    flops_decode,
    flops_attention_kv,
)


def test_dense_exact():
    # FLOPs ≈ c * params * tokens
    params = 1.5e9
    tokens = 1000
    c = 2.0
    assert flops_dense(params, tokens) == c * params * tokens
    assert flops_dense(params, tokens, multiplier=1.7) == 1.7 * params * tokens


def test_prefill_matches_formula():
    # F_prefill ≈ L * (8 * P * d^2 + 2 * P^2 * d)
    P, L, d = 1024, 24, 4096
    expected = L * (8 * P * d**2 + 2 * P**2 * d)
    assert flops_prefill(P, L, d) == expected


def test_decode_matches_formula():
    # F_decode ≈ L * (8 * G * d^2 + 4 * d * (G*P + G(G-1)/2))
    G, L, d, P = 256, 24, 4096, 1024
    expected = L * (8 * G * d**2 + 4 * d * (G * P + G * (G - 1) / 2))
    assert flops_decode(G, L, d, P) == expected


def test_attention_total_includes_anchor():
    L, d, P, G = 32, 4096, 1024, 128
    num_params = 7e9

    prefill = flops_prefill(P, L, d)
    decode = flops_decode(G, L, d, P)
    anchor = num_params * (P + G)

    assert flops_attention_kv(L, d, P, G, num_params, include_dense_anchor=True) == (
        prefill + decode + anchor
    )
    assert flops_attention_kv(L, d, P, G, num_params, include_dense_anchor=False) == (
        prefill + decode
    )


def test_zero_tokens_edge_cases():
    L, d, P, G = 32, 4096, 0, 0
    num_params = 7e9

    # With no tokens, all costs should be zero regardless of params
    assert flops_prefill(P, L, d) == 0
    assert flops_decode(G, L, d, P) == 0
    assert flops_attention_kv(L, d, P, G, num_params) == 0


def test_monotonicity_simple():
    # More generated tokens should not reduce the attention-aware estimate
    L, d, P = 24, 4096, 512
    base = flops_attention_kv(L, d, P, 100, num_params=0)
    bigger = flops_attention_kv(L, d, P, 200, num_params=0)
    assert bigger > base
