"""
relief.pricing
==============
Cost estimates, and workload estimates before a manifest is run.

Two functions with different roles; do not mix them up in the paper:

    estimate_cost    cost of one call, computed from tokens that ACTUALLY
                     occurred. This number may be reported.
    estimate_effort  forecast of a manifest's size before anything runs,
                     from pilot averages. For deciding whether a plan is
                     affordable, not for reporting.

Rates can change. Check the "Pricing" tab on the model page before using
these numbers in a publication. Unlisted models still run; their cost is
None rather than zero, so it does not pass as free when summed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .models import base_name

# USD per 1 million tokens.
#
# FILL IN the gemini, qwen and granite rates yourself from the "Pricing" tab
# of each model page on replicate.com, then uncomment them. They are left
# unfilled on purpose: a wrong rate is worse than a missing one, because a
# missing rate shows up visibly as "—" while a wrong one slips into the
# paper's tables unnoticed.
PRICE_PER_MILLION: Dict[str, Dict[str, float]] = {
    "openai/gpt-4o":       {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini":  {"input": 0.15, "output": 0.60},
    "openai/gpt-4.1":      {"input": 2.00, "output": 8.00},
    "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    # "google/gemini-3-flash":             {"input": 0.00, "output": 0.00},
    # "qwen/qwen3-7-plus":                 {"input": 0.00, "output": 0.00},
    # "ibm-granite/granite-vision-3.3-2b": {"input": 0.00, "output": 0.00},
}


def estimate_cost(model: str, usage: Any) -> Optional[float]:
    """
    Cost of one call from measured tokens.

    Matched on the exact model name, not a prefix: with prefix matching,
    "openai/gpt-4.1-mini" would be billed at the "openai/gpt-4.1" rate.
    """
    rate = PRICE_PER_MILLION.get(base_name(model))
    if rate is None:
        return None
    tokens_in = getattr(usage, "input_tokens", 0) or 0
    tokens_out = getattr(usage, "output_tokens", 0) or 0
    return round(
        (tokens_in / 1e6) * rate["input"] + (tokens_out / 1e6) * rate["output"], 6
    )


# Mean input tokens per condition, from the pilot.
PILOT_TOKENS_IN = {
    "RGB_SAJA": 5380,
    "RGB_DEPTH": 6624,
    "RGB_FUSION_CAVITY": 5799,
    "RGB_X3": 5799,
    "RGB_FUSION": 5600,
    "RGB_CAVITY": 5600,
}


def estimate_effort(
    cells: Sequence[Any],
    tokens_in: Optional[Dict[str, int]] = None,
    tokens_out: int = 4300,
    seconds_per_call: float = 36.0,
    sleep_s: float = 3.0,
    repair_overhead: float = 0.16,
) -> dict:
    """
    Forecast of a manifest's size, tokens, cost, and time.

    Defaults come from the pilot: about 4.3k output tokens, about 36 seconds
    per call, and 16% overhead from format-repair attempts.
    """
    tokens_in = tokens_in or PILOT_TOKENS_IN
    per_model: Dict[str, dict] = {}

    for c in cells:
        cond = c.condition if hasattr(c, "condition") else c["condition"]
        model = c.model if hasattr(c, "model") else c["model"]
        max_side = c.max_side if hasattr(c, "max_side") else c["max_side"]

        ti = tokens_in.get(cond, 5800)
        if max_side != 1536:
            # image tokens grow roughly with area; rough approximation
            ti = int(ti * (max_side / 1536) ** 2)

        d = per_model.setdefault(model, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
        d["calls"] += 1
        d["tokens_in"] += ti
        d["tokens_out"] += tokens_out

    total_calls = sum(d["calls"] for d in per_model.values())
    for model, d in per_model.items():
        d["tokens_in"] = int(d["tokens_in"] * (1 + repair_overhead))
        d["tokens_out"] = int(d["tokens_out"] * (1 + repair_overhead))
        rate = PRICE_PER_MILLION.get(base_name(model))
        d["cost_usd"] = (
            round((d["tokens_in"] / 1e6) * rate["input"]
                  + (d["tokens_out"] / 1e6) * rate["output"], 2)
            if rate else None
        )

    hours = total_calls * (seconds_per_call * (1 + repair_overhead) + sleep_s) / 3600
    known = [d["cost_usd"] for d in per_model.values() if d["cost_usd"] is not None]

    return {
        "calls": total_calls,
        "per_model": per_model,
        "tokens_in_total": sum(d["tokens_in"] for d in per_model.values()),
        "tokens_out_total": sum(d["tokens_out"] for d in per_model.values()),
        "cost_usd_known_models": round(sum(known), 2) if known else None,
        "models_without_price": sorted(
            m for m, d in per_model.items() if d["cost_usd"] is None
        ),
        "hours_estimate": round(hours, 1),
        "note": (
            f"assumes {seconds_per_call:.0f}s per call, {sleep_s:.0f}s pause, "
            f"and {repair_overhead:.0%} overhead from repair attempts"
        ),
    }
