"""
Step 2: recover records that failed schema validation, without calling the API.

The model's answer is already stored in `record.json` under `parsed`. Now that
the lenient validators were added to relief/schema.py, records that were
rejected before can be revalidated straight from disk.

Changes made to each recovered record (the key names are part of the on-disk
format and are kept as they are):
    status            schema_failed -> ok
    validated         filled with the validation output
    revalidated       True
    status_asal       schema_failed (original status)
    galat_skema_asal  the old schema error message (audit trail)

The original values are not lost: `parsed` and attempt-*.parsed.json are left
untouched.

Records are read from `<output_root>/runs` as given by the project config
(`relief.config.load_config`); set RELIEF_OUTPUT_ROOT (or a `.env` file /
RELIEF_CONFIG) to point elsewhere.

    python scripts/02_revalidate.py            # dry run
    python scripts/02_revalidate.py --write    # actually save
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import get_args

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import ValidationError

from relief.config import load_config
from relief.schema import AnalisisRelief, ArahHadap, Postur


def out_of_vocabulary(parsed: dict) -> Counter:
    """Count unknown values, to show what the model actually wrote."""
    c = Counter()
    for o in parsed.get("objek") or []:
        if not isinstance(o, dict):
            continue
        a = o.get("atribut") or {}
        if a.get("postur") not in get_args(Postur):
            c[f"postur={a.get('postur')!r}"] += 1
        if a.get("arah_hadap") not in get_args(ArahHadap):
            c[f"arah_hadap={a.get('arah_hadap')!r}"] += 1
    for u in (parsed.get("interpretasi_historis") or {}).get("unsur_ikonografi") or []:
        if isinstance(u, dict) and u.get("tradisi") not in ("Hindu", "Buddha", "lokal", "tidak_jelas"):
            c[f"tradisi={u.get('tradisi')!r}"] += 1
    return c


def main() -> int:
    write = "--write" in sys.argv

    cfg = load_config()
    runs = cfg.runs_dir
    print(f"runs dir : {runs}")

    paths = sorted(runs.rglob("record.json"))
    if not paths:
        print(f"no records in {runs} yet")
        return 1

    recovered = still_failing = 0
    unknown = Counter()

    for path in paths:
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "schema_failed" or not rec.get("parsed"):
            continue

        unknown.update(out_of_vocabulary(rec["parsed"]))

        try:
            validated = AnalisisRelief.model_validate(rec["parsed"]).model_dump()
        except ValidationError as exc:
            still_failing += 1
            print(f"  still failing : {path.parent.name}")
            print(f"                  {str(exc).splitlines()[1][:100]}")
            continue

        recovered += 1
        if write:
            rec["status_asal"] = rec["status"]
            rec["galat_skema_asal"] = rec.get("error")
            rec["status"] = "ok"
            rec["ok"] = True
            rec["error"] = None
            rec["validated"] = validated
            rec["revalidated"] = True
            path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            (path.parent / "revalidated.json").write_text(
                json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nrecords recovered     : {recovered}")
    print(f"records still failing : {still_failing}")

    if unknown:
        print("\nmost frequent out-of-vocabulary values written by the model:")
        for value, n in unknown.most_common(15):
            print(f"  {n:>4}x  {value}")
        print("\nIf a value is actually valid iconographically, consider adding it")
        print("to the enum in relief/schema.py AFTER `main` has finished: adding an")
        print("enum value changes the JSON Schema, so the prompt changes too and")
        print("results before and after are no longer comparable.")

    if not write:
        print("\n(dry run. Add --write to save the changes)")
    else:
        print("\nDone. Now rerun: python -m relief evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
