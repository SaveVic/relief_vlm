"""
relief.analyze
==============
One logical call: send images + prompt, receive text, then derive three
output layers and save all of them.

    raw        text as returned       -> always saved, even on success
    parsed     json.loads(raw)        -> saved if the text parses
    validated  AnalisisRelief(...)    -> saved if it passes the schema

All three are written to disk for every attempt so that, after a run, one can
check whether schema-valid output really matches the original JSON (Pydantic
may coerce types) and inspect well-formed JSON that violated an enum, without
spending credits to re-fetch them.

One logical call can include several billed predictions: the first attempt
plus format-repair attempts when the output is invalid. Every prediction is
recorded, tokens are summed across attempts, and each attempt gets its own
files.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import ValidationError

from .artifacts import CallArtifacts
from .client import ReplicateClient
from .imaging import encode_image, load_condition_images
from .prompts import build_repair_prompt, build_system_prompt, build_user_prompt, get_condition
from .pricing import estimate_cost
from .schema import AnalisisRelief
from .usage import UsageTokens

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Any:
    """
    Extract the JSON object from the model's output text.

    Tolerates code fences and preamble text: without structured output, both
    still show up occasionally even though the prompt forbids them.

    The ValueError messages below are left untranslated on purpose: they reach
    the model as repair feedback via `summarise_error`.
    """
    if not text or not text.strip():
        raise ValueError("keluaran model kosong")
    t = text.strip()
    fence = _FENCE.search(t)
    if fence:
        t = fence.group(1)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("tidak ditemukan objek JSON pada keluaran model")
    return json.loads(t[start:end + 1])


def summarise_error(exc: BaseException, limit: int = 15) -> str:
    """
    Error summary clear enough to feed back to the model.

    The text is part of the repair prompt; do not change its wording.
    """
    if isinstance(exc, ValidationError):
        lines = []
        for e in exc.errors()[:limit]:
            loc = ".".join(str(x) for x in e.get("loc", ())) or "(akar)"
            lines.append(f"- {loc}: {e.get('msg')}")
        rest = exc.error_count() - limit
        if rest > 0:
            lines.append(f"- ... dan {rest} galat lain")
        return f"Validasi skema gagal ({exc.error_count()} galat):\n" + "\n".join(lines)
    if isinstance(exc, json.JSONDecodeError):
        return f"JSON tidak dapat diparse: {exc.msg} (baris {exc.lineno}, kolom {exc.colno})"
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

# possible statuses, from best to worst
STATUS_OK = "ok"                    # JSON valid against the schema
STATUS_SCHEMA_FAILED = "schema_failed"  # JSON parsed but violates the schema
STATUS_JSON_FAILED = "json_failed"      # text present but not JSON
STATUS_ERROR = "error"                  # the API call itself failed


@dataclass
class CallResult:
    """Result of one logical call, including the trail of all attempts."""

    condition: str
    model: str
    status: str
    validated: Optional[dict]          # schema-valid result, or None
    parsed: Optional[dict]             # last successful json.loads result
    error: Optional[str]
    usage: UsageTokens
    seconds: float
    prediction_ids: List[str] = field(default_factory=list)
    n_attempts: int = 0
    rep: int = 1
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    artifact_dir: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_record(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "model": self.model,
            "status": self.status,
            "ok": self.ok,
            "validated": self.validated,
            "parsed": self.parsed,
            "error": self.error,
            "usage": self.usage.to_dict(),
            "seconds": self.seconds,
            "prediction_ids": self.prediction_ids,
            "n_attempts": self.n_attempts,
            "rep": self.rep,
            "timestamp": self.timestamp,
        }


# --------------------------------------------------------------------------
# Calls
# --------------------------------------------------------------------------

def analyze_panel(
    images: Sequence[Tuple[str, Union[str, Path, Any]]],
    condition: str,
    client: ReplicateClient,
    artifacts: Optional[CallArtifacts] = None,
    max_side: int = 1536,
    jpeg_quality: int = 90,
    max_repairs: int = 1,
    rep: int = 1,
    schema_model: Any = AnalisisRelief,
) -> CallResult:
    """
    Run one inference cycle for one panel under one condition.

    images      : [(label, path_or_array)] in the order the condition expects
    artifacts   : if given, every attempt is written to disk
    max_repairs : how many times the model is re-called when its output is
                  invalid. Every attempt is billed and its tokens are summed.
                  Keep it the same for all models: giving weak models more
                  repair chances would skew the comparison.
    """
    cond = get_condition(condition)
    expected = list(cond.images)
    got = [label for label, _ in images]
    if got != expected:
        raise ValueError(
            f"condition {condition!r} expects images {expected}, got {got}"
        )

    system_prompt = build_system_prompt(schema_model)
    base_prompt = build_user_prompt(condition)

    usage = UsageTokens()
    prediction_ids: List[str] = []
    parsed: Optional[Any] = None
    validated: Optional[Any] = None
    error: Optional[str] = None
    feedback: Optional[str] = None
    n_attempts = 0
    started = time.perf_counter()

    try:
        encoded = [
            (label, encode_image(src, max_side=max_side, jpeg_quality=jpeg_quality))
            for label, src in images
        ]
        image_urls, image_meta = client.upload_images(encoded)

        for attempt in range(1, max_repairs + 2):
            n_attempts = attempt
            prompt = base_prompt if feedback is None else build_repair_prompt(base_prompt, feedback)

            response = client.invoke(system_prompt, prompt, image_urls)
            usage.add(response.usage)
            prediction_ids.append(getattr(response.prediction, "id", None))

            if artifacts is not None and attempt == 1:
                artifacts.save_request(system_prompt, prompt, response.model_input, image_meta)

            if response.usage.output_tokens >= 0.98 * client.max_tokens:
                logger.warning(
                    "output tokens (%d) hit max_tokens (%d); JSON is "
                    "probably truncated. Raise max_tokens for this model.",
                    response.usage.output_tokens, client.max_tokens,
                )

            # --- layer 2: parse JSON ---------------------------------------
            attempt_parsed = None
            parse_error = None
            try:
                attempt_parsed = extract_json(response.text)
                parsed = attempt_parsed
            except (ValueError, json.JSONDecodeError) as exc:
                parse_error = summarise_error(exc)

            # --- layer 3: schema validation --------------------------------
            attempt_validated = None
            schema_error = None
            if attempt_parsed is not None:
                try:
                    model_obj = schema_model.model_validate(attempt_parsed)
                    attempt_validated = model_obj.model_dump()
                    validated = attempt_validated
                except ValidationError as exc:
                    schema_error = summarise_error(exc)

            if artifacts is not None:
                artifacts.save_attempt(
                    attempt,
                    raw_text=response.text,
                    prediction=response.prediction,
                    parsed=attempt_parsed,
                    validated=attempt_validated,
                    parse_error=parse_error,
                    schema_error=schema_error,
                    usage=response.usage.to_dict(),
                )

            if attempt_validated is not None:
                error = None
                break

            error = parse_error or schema_error
            logger.warning("attempt %d failed: %s", attempt, (error or "")[:200])
            feedback = error

    except Exception as exc:  # noqa: BLE001 - API failures are recorded, not allowed to stop the manifest
        seconds = round(time.perf_counter() - started, 2)
        message = f"{type(exc).__name__}: {exc}"
        logger.error("call failed after %.2fs: %s", seconds, message)
        usage.cost_usd = estimate_cost(client.model, usage)
        return CallResult(
            condition=condition, model=client.model, status=STATUS_ERROR,
            validated=None, parsed=parsed, error=message, usage=usage,
            seconds=seconds, prediction_ids=[p for p in prediction_ids if p],
            n_attempts=n_attempts, rep=rep,
            artifact_dir=str(artifacts.dir) if artifacts else None,
        )

    seconds = round(time.perf_counter() - started, 2)
    usage.cost_usd = estimate_cost(client.model, usage)

    if validated is not None:
        status = STATUS_OK
    elif parsed is not None:
        status = STATUS_SCHEMA_FAILED
    else:
        status = STATUS_JSON_FAILED

    return CallResult(
        condition=condition, model=client.model, status=status,
        validated=validated, parsed=parsed, error=error, usage=usage,
        seconds=seconds, prediction_ids=[p for p in prediction_ids if p],
        n_attempts=n_attempts, rep=rep,
        artifact_dir=str(artifacts.dir) if artifacts else None,
    )


def analyze_panel_from_root(
    root: Union[str, Path],
    panel_id: str,
    condition: str,
    client: ReplicateClient,
    **kwargs,
) -> CallResult:
    """Convenience wrapper: images are looked up in the panel folder."""
    cond = get_condition(condition)
    images = load_condition_images(root, panel_id, cond.images)
    return analyze_panel(images, condition, client, **kwargs)
