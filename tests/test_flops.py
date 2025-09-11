from src.metrics.flop_estimation import flops_dense, flops_prefill, flops_decode, flops_attention_kv

def test_flops_dense_simple():
    assert flops_dense(1e9, 10) == 2.0 * 1e9 * 10

def test_flops_prefill_decode_additive():
    p = flops_prefill(5, 2, 64)
    d = flops_decode(7, 2, 64, 5)
    t = flops_attention_kv(2, 64, 5, 7, num_params=0, include_dense_anchor=False)
    assert abs((p + d) - t) < 1e-6
