import types
from src.logging.wandb_logger import WandbRunLogger

class DummyRun:
    def __init__(self): self.logged=[]; self.summary={}
    def log(self, row): self.logged.append(row)
    def finish(self): self.finished=True

def test_one_run_one_row(monkeypatch):
    dummy = DummyRun()
    def fake_init(*a, **k): return dummy
    import src.logging.wandb_logger as m
    monkeypatch.setattr(m.wandb, "init", fake_init)
    w = WandbRunLogger(project="p", run_name="r")
    w.log_row({"accuracy":0.5, "notes":"x"})
    w.finish()
    assert dummy.logged[0]["accuracy"] == 0.5
    assert dummy.summary["accuracy"] == 0.5
