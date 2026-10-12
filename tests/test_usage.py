"""
Tests for token-usage normalisation.

What is checked is not just "the numbers are right", but that the key names,
which differ between Replicate endpoints, are all picked up, and that a
failure to pick them up is flagged explicitly (`source == "none"`) instead of
silently becoming zero.
"""

from relief.usage import UsageTokens, extract_usage


def pred(metrics=None, logs="", pid="abc123"):
    """Just enough of a Prediction-like object for extract_usage."""
    return {"id": pid, "metrics": metrics or {}, "logs": logs}


def test_openai_style_keys():
    u = extract_usage(pred({"input_token_count": 5380,
                            "output_token_count": 4310,
                            "predict_time": 36.2}))
    assert (u.input_tokens, u.output_tokens) == (5380, 4310)
    assert u.total_tokens == 9690          # summed when not reported
    assert u.predict_seconds == 36.2
    assert u.source == "metrics"
    assert u.complete


def test_generic_plural_keys():
    u = extract_usage(pred({"input_tokens": 100, "output_tokens": 20}))
    assert (u.input_tokens, u.output_tokens) == (100, 20)


def test_openai_native_keys():
    u = extract_usage(pred({"prompt_tokens": 7, "completion_tokens": 9,
                            "total_tokens": 16}))
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (7, 9, 16)


def test_unseen_spelling_still_matched_by_pattern():
    """Names not listed in PREFERRED_* are still picked up by pattern."""
    u = extract_usage(pred({"num_input_tokens": 11, "num_output_tokens": 3}))
    assert (u.input_tokens, u.output_tokens) == (11, 3)
    assert u.key_map["input_tokens"] == "num_input_tokens"


def test_nested_usage_block():
    u = extract_usage(pred({"usage": {"input_tokens": 42, "output_tokens": 8},
                            "predict_time": 1.5}))
    assert (u.input_tokens, u.output_tokens) == (42, 8)
    assert u.key_map["input_tokens"] == "usage.input_tokens"


def test_cache_and_reasoning_not_mistaken_for_input_output():
    u = extract_usage(pred({
        "input_token_count": 100,
        "output_token_count": 50,
        "cache_read_input_tokens": 80,
        "reasoning_tokens": 30,
    }))
    assert u.input_tokens == 100          # not 80
    assert u.output_tokens == 50          # not 30
    assert u.cached_tokens == 80
    assert u.reasoning_tokens == 30


def test_logs_fallback_when_metrics_empty():
    logs = "Input token count: 29\nOutput token count: 512\nTotal token count: 541\n"
    u = extract_usage(pred({}, logs=logs))
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (29, 512, 541)
    assert u.source == "logs"


def test_logs_takes_last_occurrence():
    """Logs may contain several steps; the last occurrence wins."""
    logs = "Input token count: 10\nInput token count: 900\n"
    assert extract_usage(pred({}, logs=logs)).input_tokens == 900


def test_missing_usage_is_flagged_not_zeroed_silently():
    u = extract_usage(pred({"predict_time": 2.0}))
    assert u.input_tokens == 0
    assert u.source == "none"
    assert not u.complete          # this is what tells "zero" apart from "unreadable"


def test_metrics_raw_is_preserved_for_recomputation():
    metrics = {"weird_name_for_tokens_in": 5, "predict_time": 1.0}
    u = extract_usage(pred(metrics))
    assert u.metrics_raw == metrics
    assert u.unmatched_metric_keys      # recorded so the patterns can be fixed


def test_add_accumulates_repair_attempts():
    a = extract_usage(pred({"input_token_count": 100, "output_token_count": 40}))
    b = extract_usage(pred({"input_token_count": 120, "output_token_count": 50}))
    a.add(b)
    assert (a.input_tokens, a.output_tokens) == (220, 90)
    assert a.n_predictions == 2          # two predictions billed, one logical call


def test_string_numbers_are_coerced():
    u = extract_usage(pred({"input_token_count": "31", "output_token_count": "7"}))
    assert (u.input_tokens, u.output_tokens) == (31, 7)


def test_none_prediction_is_harmless():
    u = extract_usage(None)
    assert isinstance(u, UsageTokens)
    assert u.source == "none"
