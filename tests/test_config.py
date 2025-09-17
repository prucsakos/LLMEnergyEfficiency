import io, yaml
from src.config.bench_config import load_bench_config, expand_runs

YAML = """
datasets: [gsm8k, csqa]
models:
  - name: m1
    hf_repo: repo/m1
    card: { params_B: 7, layers: 32, hidden_dim: 4096, heads: 32 }
    think_budgets: [0, 64]
"""

def test_expand_runs_grid(tmp_path):
    p = tmp_path/"cfg.yaml"
    p.write_text(YAML, encoding="utf-8")
    cfg = load_bench_config(p)
    runs = list(expand_runs(cfg))
    # 2 budgets x 2 datasets = 4
    assert len(runs) == 4
    assert runs[0].think_budget in (0, 64)
    assert runs[0].dataset in ("gsm8k", "csqa")
