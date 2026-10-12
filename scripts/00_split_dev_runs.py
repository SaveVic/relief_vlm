"""
Move records of the `dev` plan out of the runs folder.

Use this when `main` started before the dev records were moved. Nothing gets
overwritten in that case: cell_id contains the panel name, and dev panels do
not overlap with test panels. The only problem is that `evaluate` and `usage`
read everything under the runs folder without separating splits, so tuning
panels would leak into the reported numbers.

Records are identified by `cell.plan` in record.json, not by folder name, so
this is safe to run at any time, including while `main` is half done.

Paths come from the project config (`relief.config.load_config`): records are
read from `<output_root>/runs` and moved to `<output_root>/runs_dev`. Set
RELIEF_OUTPUT_ROOT (or a `.env` file / RELIEF_CONFIG) to point elsewhere.

    python scripts/00_split_dev_runs.py          # dry run
    python scripts/00_split_dev_runs.py --move   # actually move
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relief.config import load_config

PLAN_TO_MOVE = "dev"


def main() -> int:
    move = "--move" in sys.argv

    cfg = load_config()
    runs = cfg.runs_dir
    dest = cfg.dev_runs_dir
    print(f"runs dir     : {runs}")
    print(f"dev runs dir : {dest}")

    if not runs.exists():
        print(f"{runs} does not exist.")
        return 1

    per_plan = Counter()
    targets = []
    empty = []

    for folder in sorted(p for p in runs.glob("*/*") if p.is_dir()):
        record = folder / "record.json"
        if not record.exists():
            # cell that died mid-run: no record.json yet, so the runner will
            # redo it on its own
            empty.append(folder)
            continue
        try:
            rec = json.loads(record.read_text(encoding="utf-8"))
        except Exception:
            continue
        plan = (rec.get("cell") or {}).get("plan", "?")
        per_plan[plan] += 1
        if plan == PLAN_TO_MOVE:
            targets.append(folder)

    print(f"\ncurrent contents of {runs}:")
    for plan, n in sorted(per_plan.items()):
        mark = "  <- to be moved" if plan == PLAN_TO_MOVE else ""
        print(f"  plan {plan:<8}{n:>5} records{mark}")
    if empty:
        print(f"  no record.json    {len(empty):>3} folders (interrupted cells, "
              f"will be redone)")

    if not targets:
        print(f"\nNo records with plan '{PLAN_TO_MOVE}'. Nothing to move.")
        return 0

    print(f"\n{len(targets)} '{PLAN_TO_MOVE}' records will be moved to {dest}")

    if not move:
        print("\n(dry run. Add --move to move them)")
        return 0

    for folder in targets:
        target = dest / folder.parent.name / folder.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"  SKIP, already at destination: {folder.name}")
            continue
        shutil.move(str(folder), str(target))

    # empty folders left by interrupted cells: removed to avoid confusion
    for folder in empty:
        if not any(folder.iterdir()):
            folder.rmdir()

    # manifest.json and run_stats.json in the runs folder were overwritten by
    # `main`. The original dev manifest is still intact at
    # <output_root>/manifest_dev.json.
    for name in ("manifest.json", "run_stats.json"):
        src = runs / name
        if src.exists():
            shutil.copy2(src, dest / name)

    # remove model folders that are now empty
    for folder in sorted(p for p in runs.glob("*") if p.is_dir()):
        if not any(folder.iterdir()):
            folder.rmdir()

    remaining = len(list(runs.rglob("record.json")))
    print(f"\nDone. {len(targets)} records moved to {dest}")
    print(f"{runs} now holds {remaining} records (main plan only).")
    print("\nContinue with: python -m relief run --plan main --max-tokens-total 14500000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
