"""
relief.artifacts
================
File storage for EVERY API call.

Three output layers are stored separately, each self-contained:

    raw        raw text exactly as returned by the model
    parsed     json.loads of that text (if it parses)
    validated  AnalisisRelief.model_validate result (if it passes the schema)

All three are written per ATTEMPT, not per call. One logical call can
contain several billed predictions: the first attempt plus format-repair
attempts. Keeping only the final result would lose the per-attempt parse
failure rate, which is precisely the interesting number for small models.

Layout of one manifest cell:

    runs/openai-gpt-41/<cell_id>/
        record.json                 summary: status, usage, attempt trail
        request.json                system prompt, user prompt, model input
        attempt-01.raw.txt          raw text
        attempt-01.prediction.json  full Prediction object (metrics, logs, status)
        attempt-01.parsed.json      json.loads result          (if any)
        attempt-01.validated.json   schema validation result   (if any)
        attempt-01.error.txt        parse/validation error     (if any)
        attempt-02.*                repair attempt, if one happened

`record.json` is the only file the evaluation stage reads; the rest is for
auditing, recomputation and the paper appendix. The runner also uses the
existence of `record.json` to skip finished cells, so it is written last: a
cell that dies midway is never treated as finished.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

RECORD_NAME = "record.json"
REQUEST_NAME = "request.json"


def _dump(obj: Any) -> Any:
    """Make anything JSON-serialisable without raising."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {str(k): _dump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dump(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_dump(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prediction_to_dict(prediction: Any) -> Dict[str, Any]:
    """
    Replicate Prediction object -> storable dict.

    Some SDK versions provide `.dict()`; otherwise the relevant fields are
    read one by one. `metrics` and `logs` matter most: from them usage can
    be recomputed at any time without calling the API again.
    """
    if prediction is None:
        return {}
    for method in ("model_dump", "dict"):
        fn = getattr(prediction, method, None)
        if callable(fn):
            try:
                return _dump(fn())
            except Exception:  # noqa: BLE001 - fall back to manual extraction
                pass
    keys = ("id", "model", "version", "status", "error", "logs", "metrics",
            "created_at", "started_at", "completed_at", "urls")
    return {k: _dump(getattr(prediction, k, None)) for k in keys}


class CallArtifacts:
    """Artifact writer for one manifest cell / one logical call."""

    def __init__(self, base_dir: Path, save_prediction: bool = True):
        self.dir = Path(base_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_prediction = save_prediction
        self.attempts: List[Dict[str, Any]] = []

    # ---- request ---------------------------------------------------------

    def save_request(
        self,
        system_prompt: str,
        user_prompt: str,
        model_input: Dict[str, Any],
        image_refs: List[Dict[str, Any]],
    ) -> None:
        """
        Save what was sent. `model_input` is written without image data:
        data URIs can be megabytes and add nothing beyond the hashes already
        in `image_refs`.
        """
        redacted = {
            k: ("<<%d images>>" % len(v)
                if isinstance(v, list) and v and isinstance(v[0], str)
                and v[0].startswith("data:")
                else v)
            for k, v in model_input.items()
        }
        write_json(self.dir / REQUEST_NAME, {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model_input": redacted,
            "images": image_refs,
        })

    # ---- per attempt -----------------------------------------------------

    def save_attempt(
        self,
        n: int,
        raw_text: Optional[str],
        prediction: Any = None,
        parsed: Any = None,
        validated: Any = None,
        parse_error: Optional[str] = None,
        schema_error: Optional[str] = None,
        usage: Any = None,
    ) -> Dict[str, Any]:
        """
        Write one attempt and record its status.

        The three layers are written independently: JSON that parses but
        fails schema validation is still kept in `*.parsed.json`, so schema
        violations can be analysed (and partly still scored) without calling
        the model again.
        """
        tag = f"attempt-{n:02d}"
        written: Dict[str, str] = {}

        if raw_text is not None:
            path = self.dir / f"{tag}.raw.txt"
            path.write_text(raw_text, encoding="utf-8")
            written["raw"] = path.name

        if self.save_prediction and prediction is not None:
            write_json(self.dir / f"{tag}.prediction.json",
                       prediction_to_dict(prediction))
            written["prediction"] = f"{tag}.prediction.json"

        if parsed is not None:
            write_json(self.dir / f"{tag}.parsed.json", parsed)
            written["parsed"] = f"{tag}.parsed.json"

        if validated is not None:
            write_json(self.dir / f"{tag}.validated.json", validated)
            written["validated"] = f"{tag}.validated.json"

        errors = "\n\n".join(e for e in (parse_error, schema_error) if e)
        if errors:
            path = self.dir / f"{tag}.error.txt"
            path.write_text(errors, encoding="utf-8")
            written["error"] = path.name

        info = {
            "attempt": n,
            "prediction_id": (prediction_to_dict(prediction) or {}).get("id"),
            "raw_chars": len(raw_text) if raw_text else 0,
            "json_ok": parsed is not None,
            "schema_ok": validated is not None,
            "parse_error": parse_error,
            "schema_error": schema_error,
            "usage": _dump(usage) if usage is not None else None,
            "files": written,
        }
        self.attempts.append(info)
        return info

    # ---- record ----------------------------------------------------------

    def save_record(self, record: Dict[str, Any]) -> Path:
        """Write record.json. Called last; marks the cell as finished."""
        record = dict(record)
        record["attempts"] = self.attempts
        record["artifact_dir"] = str(self.dir)
        path = self.dir / RECORD_NAME
        write_json(path, record)
        return path

    @staticmethod
    def is_done(base_dir: Path) -> bool:
        return (Path(base_dir) / RECORD_NAME).exists()


def load_records(runs_dir: str | Path) -> List[Dict[str, Any]]:
    """Read every record.json under `runs_dir`."""
    out: List[Dict[str, Any]] = []
    for path in sorted(Path(runs_dir).rglob(RECORD_NAME)):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - skip corrupt files rather than fail
            continue
        rec.setdefault("artifact_dir", str(path.parent))
        if "cell" in rec:
            out.append(rec)
    return out


def load_attempt_file(record: Dict[str, Any], attempt: int, kind: str) -> Any:
    """
    Load the contents of one attempt artifact from a record.

    kind: 'raw' | 'parsed' | 'validated' | 'prediction' | 'error'
    """
    for info in record.get("attempts", []):
        if info.get("attempt") != attempt:
            continue
        name = (info.get("files") or {}).get(kind)
        if not name:
            return None
        path = Path(record["artifact_dir"]) / name
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if name.endswith(".json") else text
    return None
