"""
relief.usage
============
Normalise token usage from Replicate Prediction objects.

Each Replicate endpoint names its metrics differently, so a fixed list of
key names inevitably misses some, and a miss silently records usage as 0.

Approach
--------
1. The raw `metrics` are ALWAYS kept as-is (see `UsageTokens.metrics_raw`
   and the `*.prediction.json` artifacts). Anything that fails to normalise
   can be recomputed later without calling the API again.
2. Normalisation uses pattern matching rather than exact names: a key
   counts as input tokens when it contains a direction marker
   ("input" / "prompt") AND a token marker, and none of the excluded
   markers (cache, reasoning, etc.). So `input_token_count`,
   `input_tokens`, `prompt_tokens`, `num_input_tokens` and `tokens_input`
   are all caught without listing each one.
3. When `metrics` is empty (some models only write to stdout), `logs` is
   parsed as a fallback.
4. `source` records where the numbers came from ("metrics" / "logs" /
   "none") and `unmatched_metric_keys` records unrecognised numeric keys,
   so the patterns can be fixed when a new model shows up.

Explicit keys in `PREFERRED_*` are still tried first, so models with known
key names do not depend on the heuristic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Known key names, tried first
# --------------------------------------------------------------------------

PREFERRED_INPUT = (
    "input_token_count",      # openai/* on Replicate
    "input_tokens",           # some generic endpoints
    "prompt_tokens",          # native OpenAI style
    "token_input_count",
)
PREFERRED_OUTPUT = (
    "output_token_count",
    "output_tokens",
    "completion_tokens",
    "token_output_count",
)
PREFERRED_TOTAL = ("total_token_count", "total_tokens", "token_count")
PREFERRED_CACHE = ("cache_read_input_tokens", "cached_tokens",
                   "cache_read_tokens", "input_token_cache_read_count")
PREFERRED_REASONING = ("reasoning_tokens", "reasoning_token_count",
                       "thinking_tokens", "output_token_reasoning_count")
PREFERRED_PREDICT_TIME = ("predict_time", "predict_time_s", "duration")

# --------------------------------------------------------------------------
# Fallback patterns: match on word composition, not exact names
# --------------------------------------------------------------------------

_TOKEN_MARK = re.compile(r"tok(en)?s?", re.I)
_INPUT_MARK = re.compile(r"\b(input|prompt|in)\b|input|prompt", re.I)
_OUTPUT_MARK = re.compile(r"\b(output|completion|generated|out)\b|output|completion", re.I)
_TOTAL_MARK = re.compile(r"total", re.I)
_CACHE_MARK = re.compile(r"cach", re.I)
_REASON_MARK = re.compile(r"reason|think", re.I)
_TIME_MARK = re.compile(r"time|latency|duration|second|_s$", re.I)


def _norm(key: str) -> str:
    return key.replace("-", "_").lower()


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _flatten(obj: Any, prefix: str = "") -> Dict[str, float]:
    """Flatten a nested dict into {key_path: number}."""
    flat: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            num = _numeric(v)
            if num is not None:
                flat[path] = num
            elif isinstance(v, (dict, list, tuple)):
                flat.update(_flatten(v, path))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flat.update(_flatten(v, f"{prefix}[{i}]"))
    return flat


def _pick(flat: Dict[str, float], preferred: Tuple[str, ...],
          want: re.Pattern, exclude: Tuple[re.Pattern, ...] = (),
          require_token: bool = True) -> Tuple[Optional[float], Optional[str]]:
    """Take the first match; return (value, key_path_used)."""
    # 1. exact name (including inside nested dicts)
    for name in preferred:
        for path, val in flat.items():
            if _norm(path.rsplit(".", 1)[-1]) == name:
                return val, path
    # 2. pattern match
    for path, val in flat.items():
        leaf = _norm(path.rsplit(".", 1)[-1])
        if require_token and not _TOKEN_MARK.search(leaf):
            continue
        if not want.search(leaf):
            continue
        if any(ex.search(leaf) for ex in exclude):
            continue
        return val, path
    return None, None


# --------------------------------------------------------------------------
# Fallback: read from logs
# --------------------------------------------------------------------------

_LOG_PATTERNS = {
    "input": re.compile(r"input\s+token\s+count[:=]\s*(\d+)", re.I),
    "output": re.compile(r"output\s+token\s+count[:=]\s*(\d+)", re.I),
    "total": re.compile(r"total\s+token\s+count[:=]\s*(\d+)", re.I),
    "input_alt": re.compile(r"prompt[_ ]tokens[:=]\s*(\d+)", re.I),
    "output_alt": re.compile(r"completion[_ ]tokens[:=]\s*(\d+)", re.I),
}


def _from_logs(logs: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, pattern in _LOG_PATTERNS.items():
        found = pattern.findall(logs or "")
        if found:
            # take the last one: logs may cover several steps
            out[name.replace("_alt", "")] = int(found[-1])
    return out


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class UsageTokens:
    """Normalised usage of one or more predictions."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    predict_seconds: float = 0.0
    n_predictions: int = 0
    cost_usd: Optional[float] = None

    # audit trail
    source: str = "none"                       # metrics | logs | metrics+logs | none
    key_map: Dict[str, str] = field(default_factory=dict)
    unmatched_metric_keys: List[str] = field(default_factory=list)
    metrics_raw: Dict[str, Any] = field(default_factory=dict)

    _ACCUMULATE = ("input_tokens", "output_tokens", "total_tokens",
                   "cached_tokens", "reasoning_tokens", "predict_seconds",
                   "n_predictions")

    @property
    def complete(self) -> bool:
        """True when token counts were actually read, not just the default 0."""
        return self.source != "none" and (self.input_tokens or self.output_tokens)

    def add(self, other: "UsageTokens") -> "UsageTokens":
        """Accumulate another prediction's usage (e.g. a format-repair attempt)."""
        for name in self._ACCUMULATE:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        if other.source != "none":
            self.source = other.source if self.source == "none" else self.source
        self.key_map.update(other.key_map)
        for k in other.unmatched_metric_keys:
            if k not in self.unmatched_metric_keys:
                self.unmatched_metric_keys.append(k)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def extract_usage(prediction: Any, logger=None) -> UsageTokens:
    """
    Read token usage from a Replicate Prediction object.

    Accepts a `replicate.prediction.Prediction` object or a plain dict
    (useful when recomputing from `*.prediction.json` artifacts).
    """
    usage = UsageTokens()
    if prediction is None:
        return usage

    if isinstance(prediction, dict):
        metrics = prediction.get("metrics") or {}
        logs = prediction.get("logs") or ""
    else:
        metrics = getattr(prediction, "metrics", None) or {}
        logs = getattr(prediction, "logs", None) or ""

    usage.metrics_raw = dict(metrics) if isinstance(metrics, dict) else {}
    usage.n_predictions = 1

    flat = _flatten(metrics)
    used_paths: List[str] = []

    def take(preferred, want, exclude=(), require_token=True):
        value, path = _pick(flat, preferred, want, exclude, require_token)
        if path is not None:
            used_paths.append(path)
        return value, path

    inp, p_in = take(PREFERRED_INPUT, _INPUT_MARK,
                     exclude=(_CACHE_MARK, _REASON_MARK, _OUTPUT_MARK, _TOTAL_MARK))
    out, p_out = take(PREFERRED_OUTPUT, _OUTPUT_MARK,
                      exclude=(_CACHE_MARK, _REASON_MARK, _INPUT_MARK, _TOTAL_MARK))
    tot, p_tot = take(PREFERRED_TOTAL, _TOTAL_MARK,
                      exclude=(_CACHE_MARK, _REASON_MARK))
    cache, p_cache = take(PREFERRED_CACHE, _CACHE_MARK)
    reason, p_reason = take(PREFERRED_REASONING, _REASON_MARK)
    ptime, p_time = take(PREFERRED_PREDICT_TIME, _TIME_MARK, require_token=False)

    source = "metrics" if (inp is not None or out is not None) else "none"

    # Fallback: logs
    if inp is None or out is None:
        from_logs = _from_logs(logs)
        if from_logs:
            if inp is None and "input" in from_logs:
                inp, p_in = from_logs["input"], "logs:input token count"
            if out is None and "output" in from_logs:
                out, p_out = from_logs["output"], "logs:output token count"
            if tot is None and "total" in from_logs:
                tot, p_tot = from_logs["total"], "logs:total token count"
            source = "logs" if source == "none" else "metrics+logs"

    usage.input_tokens = int(inp or 0)
    usage.output_tokens = int(out or 0)
    usage.total_tokens = int(tot) if tot is not None else usage.input_tokens + usage.output_tokens
    usage.cached_tokens = int(cache or 0)
    usage.reasoning_tokens = int(reason or 0)
    usage.predict_seconds = round(float(ptime or 0.0), 3)
    usage.source = source

    usage.key_map = {
        k: v for k, v in {
            "input_tokens": p_in,
            "output_tokens": p_out,
            "total_tokens": p_tot,
            "cached_tokens": p_cache,
            "reasoning_tokens": p_reason,
            "predict_seconds": p_time,
        }.items() if v
    }
    usage.unmatched_metric_keys = sorted(
        k for k in flat if k not in used_paths
    )

    if logger is not None:
        if source == "none":
            logger.warning(
                "Prediction %s reported no tokens at all. Available metrics "
                "keys: %s. If this is a new model, add the key names to "
                "PREFERRED_* in relief/usage.py.",
                getattr(prediction, "id", "?"), sorted(flat) or "(empty)",
            )
        else:
            logger.info(
                "Tokens: in=%d out=%d total=%d (source=%s, key=%s), predict=%.2fs",
                usage.input_tokens, usage.output_tokens, usage.total_tokens,
                source, usage.key_map.get("input_tokens", "?"),
                usage.predict_seconds,
            )
    return usage


def usage_from_record(record: Dict[str, Any]) -> UsageTokens:
    """Rebuild UsageTokens from a saved record.json."""
    data = record.get("usage") or {}
    usage = UsageTokens()
    for f in fields(usage):
        if f.name in data:
            setattr(usage, f.name, data[f.name])
    return usage
