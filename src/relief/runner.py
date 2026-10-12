"""
relief.runner
=============
Manifest execution: one model at a time, skipping cells that already have a
`record.json`.

Each cell writes its own artifacts, so an interrupted session loses at most
one call, and rerunning the same manifest resumes where it stopped. `sleep_s`
spaces calls out to avoid throttling, and `max_tokens_total` stops the run
once a token budget is spent, as a safeguard against a misconfigured loop.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from tqdm.auto import tqdm

from .analyze import analyze_panel
from .artifacts import CallArtifacts
from .client import ReplicateClient
from .imaging import load_condition_images
from .manifest import Cell, safe_name
from .prompts import get_condition

logger = logging.getLogger(__name__)


def cell_dir(runs_dir: Union[str, Path], cell: Cell) -> Path:
    return Path(runs_dir) / safe_name(cell.model) / cell.cell_id


def run_manifest(
    cells: Sequence[Cell],
    data_root: Union[str, Path],
    runs_dir: Union[str, Path],
    temperature: float = 0.0,
    max_repairs: int = 1,
    sleep_s: float = 3.0,
    jpeg_quality: int = 90,
    upload_mode: str = "files",
    timeout: int = 300,
    max_retries: int = 2,
    client_options: Optional[Dict[str, Any]] = None,
    save_predictions: bool = True,
    stop_after: Optional[int] = None,
    max_tokens_total: Optional[int] = None,
    cleanup_uploads: bool = True,
) -> dict:
    """
    Run the manifest until it finishes or a safeguard triggers.

    `max_repairs` must be the same for every model. Giving weak models more
    repair attempts than strong ones would confound the comparison.
    """
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "manifest.json").write_text(
        json.dumps([asdict(c) for c in cells], indent=2), encoding="utf-8"
    )

    by_model: Dict[str, List[Cell]] = {}
    for c in cells:
        by_model.setdefault(c.model, []).append(c)

    stats: Dict[str, Any] = {
        "done": 0, "skipped": 0, "ok": 0,
        "schema_failed": 0, "json_failed": 0, "error": 0,
        "tokens_in": 0, "tokens_out": 0, "predict_s": 0.0, "wall_s": 0.0,
        "cost_usd": 0.0, "usage_missing": 0,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "errors": [],
    }

    for model, model_cells in by_model.items():
        client = ReplicateClient(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            upload_mode=upload_mode,
            **(client_options or {}),
        )
        bar = tqdm(model_cells, desc=model.split("/")[-1], unit="call")
        try:
            for cell in bar:
                target = cell_dir(runs, cell)
                if CallArtifacts.is_done(target):
                    stats["skipped"] += 1
                    bar.set_postfix_str("skip")
                    continue

                if (max_tokens_total is not None
                        and stats["tokens_in"] + stats["tokens_out"] >= max_tokens_total):
                    bar.close()
                    logger.warning("token budget reached, stopping run")
                    stats["stopped_on_budget"] = True
                    return _finish(stats, runs)

                try:
                    cond = get_condition(cell.condition)
                    images = load_condition_images(data_root, cell.panel_id, cond.images)
                    artifacts = CallArtifacts(target, save_prediction=save_predictions)

                    t0 = time.time()
                    result = analyze_panel(
                        images, cell.condition, client,
                        artifacts=artifacts,
                        max_side=cell.max_side,
                        jpeg_quality=jpeg_quality,
                        max_repairs=max_repairs,
                        rep=cell.rep,
                    )
                    record = result.to_record()
                    record["cell"] = asdict(cell)
                    record["wall_s"] = round(time.time() - t0, 2)
                    artifacts.save_record(record)

                    stats["done"] += 1
                    stats[result.status] = stats.get(result.status, 0) + 1
                    stats["wall_s"] += record["wall_s"]
                    stats["tokens_in"] += result.usage.input_tokens
                    stats["tokens_out"] += result.usage.output_tokens
                    stats["predict_s"] += result.usage.predict_seconds
                    stats["cost_usd"] += result.usage.cost_usd or 0.0
                    if not result.usage.complete:
                        stats["usage_missing"] += 1

                    bar.set_postfix_str(
                        f"{cell.condition[:12]} {cell.panel_id[:12]} "
                        f"{result.status} tok={stats['tokens_in'] + stats['tokens_out']:,}"
                    )
                except Exception as exc:  # noqa: BLE001 - one failed cell must not stop the manifest
                    stats["error"] += 1
                    stats["errors"].append({"cell_id": cell.cell_id, "error": repr(exc)})
                    bar.write(f"ERROR {cell.cell_id}: {exc!r}")

                if stop_after and stats["done"] >= stop_after:
                    bar.close()
                    logger.info("stop_after reached")
                    return _finish(stats, runs)

                time.sleep(sleep_s)
        finally:
            bar.close()
            if cleanup_uploads:
                client.cleanup_uploads()

    return _finish(stats, runs)


def _finish(stats: dict, runs: Path) -> dict:
    stats["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["cost_usd"] = round(stats["cost_usd"], 4)
    (runs / "run_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def progress(cells: Sequence[Cell], runs_dir: Union[str, Path]) -> dict:
    """How much of the manifest is already done."""
    done = sum(1 for c in cells if CallArtifacts.is_done(cell_dir(runs_dir, c)))
    return {"total": len(cells), "done": done, "remaining": len(cells) - done}
