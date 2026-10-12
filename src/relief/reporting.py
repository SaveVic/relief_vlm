"""
relief.reporting
================
MEASURED token and time usage, read from existing records.

These are the figures that go into the paper. `predict_s` is the inference
time reported by the provider; `wall_s` is measured around the call, so it
includes upload and queueing but not the pause between calls.

`usage_missing` counts records whose tokens could not be read at all. If it
is non-zero, the raw `metrics` are still in `attempt-*.prediction.json` and
can be recomputed after fixing the patterns in `relief/usage.py`, without
calling the API again.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .artifacts import load_records


def usage_table(records: Sequence[dict]) -> dict:
    """Aggregate per (model, condition)."""
    agg: Dict[str, dict] = {}

    for rec in records:
        cell = rec.get("cell") or {}
        usage = rec.get("usage") or {}
        key = f"{cell.get('model')}|{cell.get('condition')}"
        d = agg.setdefault(key, {
            "model": cell.get("model"),
            "condition": cell.get("condition"),
            "calls": 0, "ok": 0, "schema_failed": 0, "json_failed": 0, "error": 0,
            "attempts": 0, "predictions": 0,
            "tokens_in": 0, "tokens_out": 0, "tokens_total": 0,
            "predict_s": 0.0, "wall_s": 0.0, "cost_usd": 0.0,
            "usage_missing": 0, "cost_missing": 0,
        })
        d["calls"] += 1
        status = rec.get("status", "error")
        d[status] = d.get(status, 0) + 1
        d["attempts"] += rec.get("n_attempts", 0) or 0
        d["predictions"] += usage.get("n_predictions", 0) or 0
        d["tokens_in"] += usage.get("input_tokens", 0) or 0
        d["tokens_out"] += usage.get("output_tokens", 0) or 0
        d["tokens_total"] += usage.get("total_tokens", 0) or 0
        d["predict_s"] += usage.get("predict_seconds", 0.0) or 0.0
        d["wall_s"] += rec.get("wall_s", 0.0) or 0.0
        d["cost_usd"] += usage.get("cost_usd") or 0.0
        # A record with status 'error' never produced a prediction, so there
        # are no tokens to read. Counting it as "missing tokens" would raise
        # a warning pointing at usage.py when the problem is the network.
        if usage.get("source", "none") == "none" and status != "error":
            d["usage_missing"] += 1
        if usage.get("cost_usd") is None:
            d["cost_missing"] += 1

    rows: List[dict] = []
    total = {k: 0 for k in ("calls", "ok", "tokens_in", "tokens_out",
                            "usage_missing", "cost_missing", "attempts",
                            "predictions")}
    total.update({"predict_s": 0.0, "wall_s": 0.0, "cost_usd": 0.0})

    for d in sorted(agg.values(), key=lambda r: (r["model"] or "", r["condition"] or "")):
        n = max(1, d["calls"])
        rows.append({
            **d,
            "tokens_in_mean": round(d["tokens_in"] / n, 1),
            "tokens_out_mean": round(d["tokens_out"] / n, 1),
            "predict_s_mean": round(d["predict_s"] / n, 1),
            "wall_s_mean": round(d["wall_s"] / n, 1),
            "repair_rate": round((d["attempts"] - d["calls"]) / n, 3),
            "success_rate": round(d["ok"] / n, 3),
            "cost_usd": round(d["cost_usd"], 4),
        })
        for k in total:
            total[k] += d.get(k, 0)

    return {
        "rows": rows,
        "total": {
            **total,
            "cost_usd": round(total["cost_usd"], 4),
            "predict_h": round(total["predict_s"] / 3600, 2),
            "wall_h": round(total["wall_s"] / 3600, 2),
        },
        "warning": (
            f"{total['usage_missing']} records reported no tokens at all; "
            f"check the raw metrics in attempt-*.prediction.json and add "
            f"the key names to PREFERRED_* in relief/usage.py"
            if total["usage_missing"] else None
        ),
    }


def usage_text(table: dict) -> str:
    """Printable table."""
    head = (
        f"{'model':<26}{'condition':<20}{'n':>5}{'ok%':>6}{'tok_in':>12}"
        f"{'tok_out':>12}{'repairs':>11}{'predict_s':>11}{'wall_s':>9}{'USD':>10}"
    )
    lines = [head, "-" * len(head)]
    for r in table["rows"]:
        lines.append(
            f"{(r['model'] or '')[:25]:<26}{(r['condition'] or '')[:19]:<20}"
            f"{r['calls']:>5}{100 * r['success_rate']:>6.0f}"
            f"{r['tokens_in_mean']:>12.0f}{r['tokens_out_mean']:>12.0f}"
            f"{r['repair_rate']:>11.2f}{r['predict_s_mean']:>11.1f}"
            f"{r['wall_s_mean']:>9.1f}"
            # Models with no registered price show an em dash, not 0.0000:
            # a silent zero would pass for a free call when summed.
            f"{('—' if r['cost_missing'] == r['calls'] else format(r['cost_usd'], '.4f')):>10}"
        )
    t = table["total"]
    lines += [
        "-" * len(head),
        f"{'TOTAL':<46}{t['calls']:>5}{'':>6}{t['tokens_in']:>12,}{t['tokens_out']:>12,}"
        f"{'':>11}{t['predict_h']:>10.2f}h{t['wall_h']:>8.2f}h{t['cost_usd']:>10.4f}",
    ]
    if t["cost_missing"]:
        lines.append(
            f"NOTE: the cost above covers only models with a registered price. "
            f"{t['cost_missing']} of {t['calls']} calls are not included; "
            f"complete PRICE_PER_MILLION in relief/pricing.py."
        )
    if table.get("warning"):
        lines += ["", "WARNING: " + table["warning"]]
    return "\n".join(lines)


USAGE_CSV_COLUMNS = [
    "timestamp", "cell_id", "model", "condition", "panel_id", "site", "group",
    "split", "rep", "max_side", "status", "ok", "n_attempts",
    "n_predictions", "input_tokens", "output_tokens", "total_tokens",
    "cached_tokens", "reasoning_tokens", "predict_seconds", "wall_s",
    "cost_usd", "usage_source", "prediction_ids", "error",
]


def write_usage_csv(records: Sequence[dict], path: Union[str, Path]) -> Path:
    """One row per call, for the appendix or further analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=USAGE_CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            cell = rec.get("cell") or {}
            usage = rec.get("usage") or {}
            row = {
                "timestamp": rec.get("timestamp", ""),
                "cell_id": cell.get("cell_id", ""),
                "model": cell.get("model", rec.get("model", "")),
                "condition": cell.get("condition", rec.get("condition", "")),
                "panel_id": cell.get("panel_id", ""),
                "site": cell.get("site", ""),
                "group": cell.get("group", ""),
                "split": cell.get("split", ""),
                "rep": cell.get("rep", rec.get("rep", "")),
                "max_side": cell.get("max_side", ""),
                "status": rec.get("status", ""),
                "ok": rec.get("ok", ""),
                "n_attempts": rec.get("n_attempts", ""),
                "n_predictions": usage.get("n_predictions", ""),
                "input_tokens": usage.get("input_tokens", ""),
                "output_tokens": usage.get("output_tokens", ""),
                "total_tokens": usage.get("total_tokens", ""),
                "cached_tokens": usage.get("cached_tokens", ""),
                "reasoning_tokens": usage.get("reasoning_tokens", ""),
                "predict_seconds": usage.get("predict_seconds", ""),
                "wall_s": rec.get("wall_s", ""),
                "cost_usd": usage.get("cost_usd", ""),
                "usage_source": usage.get("source", ""),
                "prediction_ids": ";".join(rec.get("prediction_ids") or []),
                "error": (rec.get("error") or "").replace("\n", " ")[:300],
            }
            writer.writerow(row)
    return path


def status_breakdown(records: Sequence[dict]) -> Dict[str, Dict[str, int]]:
    """Status breakdown per model. The parse failure rate is itself a result."""
    out: Dict[str, Dict[str, int]] = {}
    for rec in records:
        model = (rec.get("cell") or {}).get("model", rec.get("model", "?"))
        d = out.setdefault(model, {"ok": 0, "schema_failed": 0,
                                   "json_failed": 0, "error": 0})
        d[rec.get("status", "error")] = d.get(rec.get("status", "error"), 0) + 1
    return out


def report_usage(runs_dir: Union[str, Path],
                 csv_path: Optional[Union[str, Path]] = None) -> dict:
    """Read all records, aggregate, and write the CSV if requested."""
    records = load_records(runs_dir)
    table = usage_table(records)
    table["n_records"] = len(records)
    table["by_status"] = status_breakdown(records)
    if csv_path:
        write_usage_csv(records, csv_path)
    return table
