import types
from src.data.adapters import exact_match

def test_exact_match_norm():
    assert exact_match(" 42.", "42")
    assert exact_match("New-York", "new york") is False  # punctuation change

# For HF datasets we avoid network in unit tests, so we simulate one sample
def test_sample_shape_like():
    from src.data.adapters import Sample
    s = Sample(id="x", question="Q", gold="a", choices=["a","b"])
    assert s.id and s.question and s.gold
