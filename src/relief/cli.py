"""
relief.cli
==========
Command-line interface, one subcommand per pipeline stage.

    python -m relief fetch --zip-url "<link Drive>"
    python -m relief verify
    python -m relief gt
    python -m relief split
    python -m relief models
    python -m relief plan --plan dev
    python -m relief smoke --model qwen/qwen3-7-plus
    python -m relief run --plan dev
    python -m relief usage
    python -m relief evaluate

Every stage reads its inputs from disk and writes its outputs to disk, so the
sequence can be run partially, repeated, or resumed after an interruption. No
state lives only in memory between stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config, load_config

logger = logging.getLogger("relief")


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

def cmd_fetch(args, cfg: Config) -> int:
    from .fetch_data import fetch_folder, fetch_rclone, fetch_zip, verify_dataset, verify_text

    zip_url = args.zip_url or cfg.drive_zip_url
    folder_url = args.folder_url or cfg.drive_folder_url

    if args.rclone_remote:
        dest = fetch_rclone(args.rclone_remote, cfg.data_root)
    elif zip_url:
        dest = fetch_zip(zip_url, cfg.data_root, keep_archive=args.keep_archive)
    elif folder_url:
        dest = fetch_folder(folder_url, cfg.data_root)
    else:
        print(
            "Specify one source:\n"
            "  --zip-url        Drive link to a zip (recommended)\n"
            "  --folder-url     Drive folder link (at most 50 files)\n"
            "  --rclone-remote  'remote:path' if you already use rclone\n"
            "Or set drive_zip_url in configs/default.yaml.",
            file=sys.stderr,
        )
        return 2

    print(verify_text(verify_dataset(dest, cfg.mask_name)))
    return 0


def cmd_verify(args, cfg: Config) -> int:
    from .fetch_data import verify_dataset, verify_text

    required = args.images or ["RGB", "DEPTH", "FUSION", "CAVITY"]
    report = verify_dataset(cfg.data_root, cfg.mask_name, required)
    print(verify_text(report))
    return 0 if not report["missing"] else 1


def cmd_gt(args, cfg: Config) -> int:
    from . import gt as G

    cfg.ensure_dirs()

    audit = G.audit_dataset(str(cfg.data_root), cfg.mask_name)
    print(f"panels: {audit['n_panels']} | clean masks: {audit['clean']}")
    for d in audit["dirty"]:
        print(f"  DIRTY {d['panel_id']}: {d['n_unknown']} unknown colours, "
              f"{d['unknown_pixel_frac']:.4%} of pixels")
    if audit["dirty"] and not args.force:
        print(
            "\nStopping: extracting instances from unclean masks would produce\n"
            "objects that were never annotated. Clean the masks first, or\n"
            "rerun with --force if this is intentional.",
            file=sys.stderr,
        )
        return 1

    dataset = G.extract_dataset(
        str(cfg.data_root),
        out_json=str(cfg.gt_path),
        mask_name=cfg.mask_name,
        salience_name=cfg.salience_name,
        min_area_px=cfg.min_area_px,
        merge_rule="occlusion",
        occ_threshold=0.5,
        rel_gap_max=1.0,
    )
    print(G.summarise(dataset))
    print(f"\nwritten to {cfg.gt_path}")
    return 0


def cmd_split(args, cfg: Config) -> int:
    from .manifest import make_split

    cfg.ensure_dirs()
    split = make_split(cfg.data_root, n_dev=cfg.n_dev, seed=cfg.seed,
                       mask_name=cfg.mask_name, out_json=cfg.split_path)
    print(f"seed  : {split['seed']}")
    print(f"dev   : {len(split['dev'])} panels -> {split['dev']}")
    print(f"test  : {len(split['test'])} panels")
    print(f"groups: {split['by_group']}")
    print(f"\nwritten to {cfg.split_path}")
    return 0


def cmd_models(args, cfg: Config) -> int:
    from .models import spec_table

    print(spec_table())
    return 0


def cmd_plan(args, cfg: Config) -> int:
    from .manifest import PLANS, build_manifest
    from .pricing import estimate_effort

    cfg.ensure_dirs()
    if not cfg.split_path.exists():
        print("no split yet. Run: python -m relief split", file=sys.stderr)
        return 1

    split = json.loads(cfg.split_path.read_text(encoding="utf-8"))
    models = args.models or cfg.models
    cells = build_manifest(split, models, plan=args.plan,
                           out_json=cfg.manifest_path(args.plan))

    cfg_plan = PLANS[args.plan]
    print(f"plan       : {args.plan} ({cfg_plan['note']})")
    print(f"conditions : {', '.join(cfg_plan['conditions'])}")
    print(f"split      : {cfg_plan['split']}, {cfg_plan['reps']} rep(s)")
    print(f"models     : {', '.join(models)}")
    print(f"cells      : {len(cells)} calls")
    print(f"manifest   : {cfg.manifest_path(args.plan)}\n")
    print(json.dumps(estimate_effort(cells), indent=2, ensure_ascii=False))
    return 0


def cmd_smoke(args, cfg: Config) -> int:
    """One cheap call to confirm the API token and input names are correct."""
    from .analyze import analyze_panel_from_root
    from .artifacts import CallArtifacts
    from .client import ReplicateClient
    from .manifest import list_panels

    panel = args.panel
    if panel is None:
        if cfg.split_path.exists():
            panel = json.loads(cfg.split_path.read_text(encoding="utf-8"))["dev"][0]
        else:
            panels = list_panels(cfg.data_root, cfg.mask_name)
            if not panels:
                print("no panels in the dataset", file=sys.stderr)
                return 1
            panel = panels[0]

    out_dir = cfg.output_root / "smoke" / f"{args.model.replace('/', '-')}__{panel}"
    artifacts = CallArtifacts(out_dir, save_prediction=True)
    client = ReplicateClient(model=args.model, temperature=cfg.temperature,
                             timeout=cfg.timeout, upload_mode=cfg.upload_mode)
    try:
        result = analyze_panel_from_root(
            cfg.data_root, panel, args.condition, client,
            artifacts=artifacts, max_side=cfg.max_side,
            max_repairs=cfg.max_repairs,
        )
        artifacts.save_record({**result.to_record(),
                               "cell": {"panel_id": panel, "condition": args.condition,
                                        "model": args.model, "cell_id": "smoke"}})
    finally:
        client.cleanup_uploads()

    print(f"panel     : {panel}")
    print(f"condition : {args.condition}")
    print(f"status    : {result.status}")
    print(f"attempts  : {result.n_attempts}, predictions: {result.prediction_ids}")
    print(f"tokens    : in={result.usage.input_tokens} out={result.usage.output_tokens} "
          f"(source={result.usage.source})")
    if result.usage.key_map:
        print(f"keys      : {result.usage.key_map}")
    if result.usage.unmatched_metric_keys:
        print(f"unknown metrics: {result.usage.unmatched_metric_keys}")
    if result.error:
        print(f"error     : {result.error[:500]}")
    print(f"artifacts : {out_dir}")
    return 0 if result.ok else 1


def cmd_run(args, cfg: Config) -> int:
    from .manifest import load_manifest
    from .runner import progress, run_manifest

    path = cfg.manifest_path(args.plan)
    if not path.exists():
        print(f"no manifest yet. Run: python -m relief plan --plan {args.plan}",
              file=sys.stderr)
        return 1

    cells = load_manifest(path)
    before = progress(cells, cfg.runs_dir)
    print(f"manifest {args.plan}: {before['done']}/{before['total']} done, "
          f"{before['remaining']} remaining")
    if args.dry_run:
        return 0

    stats = run_manifest(
        cells, cfg.data_root, cfg.runs_dir,
        temperature=cfg.temperature,
        max_repairs=cfg.max_repairs,
        sleep_s=cfg.sleep_s,
        jpeg_quality=cfg.jpeg_quality,
        upload_mode=cfg.upload_mode,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        save_predictions=cfg.save_predictions,
        stop_after=args.stop_after,
        max_tokens_total=args.max_tokens_total or cfg.max_tokens_total,
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "errors"},
                     indent=2, ensure_ascii=False))
    if stats["errors"]:
        print("\nfirst errors:")
        for e in stats["errors"][:5]:
            print(" ", e["cell_id"], e["error"][:140])
    print("\nremaining:", progress(cells, cfg.runs_dir))
    return 0


def cmd_usage(args, cfg: Config) -> int:
    from .reporting import report_usage, usage_text

    table = report_usage(cfg.runs_dir, csv_path=cfg.output_root / "usage.csv")
    if not table["rows"]:
        print("no records yet in", cfg.runs_dir)
        return 1
    print(usage_text(table))
    print("\nstatus by model:")
    for model, counts in table["by_status"].items():
        print(f"  {model:<34}{counts}")
    print(f"\nper-call CSV: {cfg.output_root / 'usage.csv'}")
    return 0


def cmd_evaluate(args, cfg: Config) -> int:
    from . import evaluate as E
    from .artifacts import load_records

    if not cfg.gt_path.exists():
        print("no ground truth yet. Run: python -m relief gt", file=sys.stderr)
        return 1

    gt = E.load_gt(str(cfg.gt_path))
    records = load_records(cfg.runs_dir)
    if not records:
        print(f"no records yet in {cfg.runs_dir}", file=sys.stderr)
        return 1
    print(f"{len(records)} records, {len(gt)} annotated panels")

    report = E.full_report(
        records, gt,
        baseline=args.baseline,
        metric=args.metric,
        out_dir=str(cfg.eval_dir),
    )
    print(E.summary_text(report))
    print(f"\ntables and report: {cfg.eval_dir}")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="relief",
        description="Relief analysis experiment: ground truth, LLM calls, evaluation.",
    )
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config file (default: configs/default.yaml)")
    p.add_argument("--data-root", type=Path, default=None, help="local dataset folder")
    p.add_argument("--output-root", type=Path, default=None, help="output folder")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG log level")

    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("fetch", help="download the dataset from Google Drive")
    s.add_argument("--zip-url", help="Drive link to a zip (recommended)")
    s.add_argument("--folder-url", help="Drive folder link (max. 50 files)")
    s.add_argument("--rclone-remote", help="'remote:path' for rclone")
    s.add_argument("--keep-archive", action="store_true", help="do not delete the zip")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("verify", help="check that dataset files are complete")
    s.add_argument("--images", nargs="*", help="required images, e.g. RGB FUSION CAVITY")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("gt", help="extract ground truth from segmentation masks")
    s.add_argument("--force", action="store_true", help="continue even if some masks are dirty")
    s.set_defaults(func=cmd_gt)

    s = sub.add_parser("split", help="create the dev/test split")
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("models", help="show each endpoint's input names")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("plan", help="build the manifest and estimate cost")
    s.add_argument("--plan", default="main",
                   choices=["main", "dev", "abl_imagecount", "abl_components",
                            "abl_resolution"])
    s.add_argument("--models", nargs="*", help="override the model list from the config")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("smoke", help="a single test call")
    s.add_argument("--model", default="qwen/qwen3-7-plus")
    s.add_argument("--condition", default="RGB_SAJA")
    s.add_argument("--panel", default=None, help="default: first dev panel")
    s.set_defaults(func=cmd_smoke)

    s = sub.add_parser("run", help="run a manifest")
    s.add_argument("--plan", default="main")
    s.add_argument("--stop-after", type=int, default=None,
                   help="stop after this many completed calls")
    s.add_argument("--max-tokens-total", type=int, default=None,
                   help="safety limit: stop after this many tokens")
    s.add_argument("--dry-run", action="store_true", help="only show progress")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("usage", help="measured tokens and time")
    s.set_defaults(func=cmd_usage)

    s = sub.add_parser("evaluate", help="scores, paired tests, LaTeX tables")
    s.add_argument("--baseline", default="RGB_SAJA")
    s.add_argument("--metric", default="recall_grid")
    s.set_defaults(func=cmd_evaluate)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config, overrides={
        "data_root": args.data_root,
        "output_root": args.output_root,
    })
    cfg.ensure_dirs()

    from .logging_setup import setup_logging
    setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_dir=cfg.output_root / "logs",
    )
    logger.debug("config: %s", cfg.to_dict())

    return args.func(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
