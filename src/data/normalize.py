from __future__ import annotations
import re

def normalize_freeform(s: str) -> str:
    """Light normalization for exact-match on short answers."""
    s = s.strip()
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;")
    return s.lower()

def extract_gsm8k_final(answer_field: str) -> str:
    """GSM8K gold answers are often '... #### 42' per dataset card. :contentReference[oaicite:13]{index=13}"""
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip()
    return answer_field.strip()
