"""
Step 5 — remove cells that failed because of network errors, so they get re-run.

The runner skips any cell that already has a record.json. A record with status
`error` also has a record.json, so that cell is never retried unless its
folder is removed first.

Only status `error` is touched. `schema_failed` and `json_failed` are NOT
removed: both are legitimate model answers and count as results.

    python scripts/05_reset_errors.py           # dry run, only lists
    python scripts/05_reset_errors.py --delete  # actually delete
    python -m relief run --plan dev             # then re-run

Paths come from relief.config.load_config() (RELIEF_OUTPUT_ROOT, .env,
configs/default.yaml or $RELIEF_CONFIG).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relief.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delete", action="store_true",
                    help="delete the folders of cells with status 'error' (default: only list them)")
    ap.add_argument("--output-root", help="override the output root (otherwise from config)")
    args = ap.parse_args()

    cfg = load_config(overrides={
        "output_root": Path(args.output_root) if args.output_root else None,
    })
    runs = cfg.runs_dir
    print(f"runs   : {runs}\n")

    targets = []
    for path in sorted(runs.rglob("record.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("status") == "error":
            targets.append((path.parent, (rec.get("error") or "")[:80]))

    if not targets:
        print("no cells with status 'error'.")
        return 0

    print(f"{len(targets)} cells with status 'error':\n")
    for folder, error in targets:
        print(f"  {folder.name}")
        print(f"    {error}")

    if not args.delete:
        print("\n(dry run only. Add --delete to remove these folders)")
        return 0

    for folder, _ in targets:
        shutil.rmtree(folder)
    print(f"\n{len(targets)} folders deleted.")
    print("Re-run with: python -m relief run --plan dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
