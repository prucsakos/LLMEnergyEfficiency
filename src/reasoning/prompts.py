from __future__ import annotations

def build_prompt_cot(question: str) -> str:
    """CoT thinking prompt with explicit tags for later parsing."""
    return (
        "Solve the problem. First write your reasoning inside <scratchpad>...</scratchpad>.\n"
        "Do not write the final answer outside of <final>...</final>.\n\n"
        f"<question>\n{question}\n</question>\n\n<scratchpad>"
    )

def build_prompt_plan(question: str) -> str:
    """Plan-and-Solve: pass-1 asks for a short plan. :contentReference[oaicite:10]{index=10}"""
    return (
        "Devise a brief step-by-step plan to solve the task. "
        "Write the plan only inside <plan>...</plan>.\n\n"
        f"<question>\n{question}\n</question>\n\n<plan>"
    )

def build_prompt_answer(question: str, scratch_or_plan: str) -> str:
    """Answer-only prompt that consumes the earlier scratch/plan."""
    return (
        "Using the information below, produce ONLY the final answer inside <final>...</final>.\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"{scratch_or_plan}\n\n<final>"
    )

def render(template: str, **kw) -> str:
    return template.format(**kw)

def closing_tag(style: str) -> str:
    return "</scratchpad>" if style == "cot" else "</plan>"
