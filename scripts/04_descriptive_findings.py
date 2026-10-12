"""
Confirm the descriptive findings on the test panels.

These three findings were observed on the development panels and recorded in
PRAREGISTRASI.md as things to report without hypothesis testing. This script
confirms them on the 46 test panels, the only basis allowed into the paper.

  1. Inventory completeness: what share of annotated objects the model lists,
     and whether that share shrinks as panels get denser
  2. Recall decomposition: how much of the recall shortfall comes from objects
     never mentioned, and how much from objects mentioned but not matched
  3. Calibration: whether `keyakinan` (confidence) separates matched objects
     from unmatched ones

Plus one supporting check: whether the model actually claims to use the
derived images under the RGB_FUSION_CAVITY condition.

Paths come from the project config (`relief.config.load_config`); set
RELIEF_OUTPUT_ROOT (or a `.env` file / RELIEF_CONFIG) to point elsewhere.

    python scripts/04_descriptive_findings.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relief.artifacts import load_records
from relief.config import load_config
from relief.evaluate import load_gt, score_all


def correlation(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    cfg = load_config()
    runs = cfg.runs_dir
    gt_path = cfg.gt_path
    report = cfg.eval_dir / "report.json"
    print(f"runs dir     : {runs}")
    print(f"ground truth : {gt_path}")
    print(f"report       : {report}\n")

    records = load_records(str(runs))
    if not records:
        print(f"no records in {runs} yet")
        return 1
    rows = score_all(records, load_gt(str(gt_path)), show_progress=False)

    # `score_panel` does not carry the `split` field into the score rows, so it
    # is read straight from the records.
    split = Counter((rec.get("cell") or {}).get("split", "?") for rec in records)
    print(f"{len(rows)} score rows from {len(records)} records, "
          f"split counts: {dict(split)}")
    if len(split) > 1:
        print(f"\nWARNING: more than one split in {runs}.")
        print("Run scripts/00_split_dev_runs.py --move first, otherwise tuning")
        print("panels will leak into the reported numbers.\n")

    # ================================================================
    # 1. Inventory completeness
    # ================================================================
    print("\n" + "=" * 78)
    print("1. INVENTORY COMPLETENESS: share of annotated objects that get listed")
    print("=" * 78)

    # average over replicates first, so the unit of analysis is the panel
    per_panel = defaultdict(lambda: {"pred": [], "gt": None})
    for r in rows:
        k = (r["model"], r["condition"], r["panel_id"])
        per_panel[k]["pred"].append(r["n_pred"])
        per_panel[k]["gt"] = r["n_gt"]

    agg = defaultdict(list)
    for (model, cond, panel), v in per_panel.items():
        n_gt = v["gt"] or 0
        if n_gt == 0:
            continue
        agg[(model, cond)].append((n_gt, float(np.mean(v["pred"]))))

    head = (f"{'model':<26}{'condition':<20}{'panels':>6}{'ratio':>8}"
            f"{'recall bound':>14}{'density corr':>16}")
    print(head)
    print("-" * len(head))
    upper_bound = {}
    for (model, cond), pairs in sorted(agg.items()):
        gt = np.array([g for g, _ in pairs], float)
        pr = np.array([p for _, p in pairs], float)
        ratio = pr / gt
        r_corr = correlation(gt, ratio)
        upper_bound[(model, cond)] = float(np.mean(ratio))
        print(f"{model[:25]:<26}{cond[:19]:<20}{len(gt):>6}"
              f"{np.mean(ratio):>8.3f}{np.mean(ratio):>14.3f}{r_corr:>+16.2f}")

    print("\n`ratio`        = listed objects / annotated objects, mean per panel")
    print("`recall bound` = highest possible recall, if EVERY object the model")
    print("                 mentioned matched an annotation")
    print("`density corr` = correlation between annotated object count and ratio.")
    print("                 Negative means dense panels are handled worse.")

    # ================================================================
    # 2. Recall decomposition
    # ================================================================
    print("\n" + "=" * 78)
    print("2. DECOMPOSITION OF THE RECALL SHORTFALL")
    print("=" * 78)

    actual_recall = defaultdict(list)
    for r in rows:
        v = r.get("recall_grid")
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            actual_recall[(r["model"], r["condition"])].append(v)

    head = (f"{'model':<26}{'condition':<20}{'recall':>8}{'bound':>8}"
            f"{'not listed':>14}{'no match':>12}")
    print(head)
    print("-" * len(head))
    for k in sorted(upper_bound):
        if k not in actual_recall:
            continue
        bound = upper_bound[k]
        recall = float(np.mean(actual_recall[k]))
        lost_listing = 1.0 - bound           # never mentioned
        lost_matching = bound - recall       # mentioned but not matched
        print(f"{k[0][:25]:<26}{k[1][:19]:<20}{recall:>8.3f}{bound:>8.3f}"
              f"{lost_listing:>14.3f}{lost_matching:>12.3f}")

    print("\nThe last two columns sum to (1 - recall). Whichever is larger sets the")
    print("paper's narrative: the bottleneck is in enumeration, or in perception.")

    # ================================================================
    # 3. Calibration and image use
    # ================================================================
    if report.exists():
        rep = json.loads(report.read_text(encoding="utf-8"))

        print("\n" + "=" * 78)
        print("3. CALIBRATION: does `keyakinan` separate right from wrong")
        print("=" * 78)
        head = f"{'model':<26}{'condition':<20}{'AUROC':>8}{'ECE':>8}   reading"
        print(head)
        print("-" * (len(head) + 20))
        for a in rep["aggregate"]:
            auroc = a.get("auroc")
            if auroc is None or (isinstance(auroc, float) and np.isnan(auroc)):
                reading, auroc_s = "n/a", "   --"
            else:
                auroc_s = f"{auroc:>8.3f}"
                if auroc < 0.45:
                    reading = "INVERTED: confident objects are the wrong ones"
                elif auroc < 0.55:
                    reading = "uninformative"
                elif auroc < 0.65:
                    reading = "slightly informative"
                else:
                    reading = "informative"
            print(f"{a['model'][:25]:<26}{a['condition'][:19]:<20}{auroc_s}"
                  f"{a.get('ece', float('nan')):>8.3f}   {reading}")

        print("\n" + "=" * 78)
        print("4. IMAGE USE: which images the model cites as evidence")
        print("=" * 78)
        for a in rep["aggregate"]:
            if a["condition"] != "RGB_FUSION_CAVITY":
                continue
            share = a.get("source_share") or {}
            rgb = share.get("RGB", 0.0)
            derived = share.get("FUSION", 0.0) + share.get("CAVITY", 0.0)
            note = ("derived images practically ignored" if derived < 0.10
                    else "derived images used")
            print(f"{a['model'][:25]:<26}RGB={rgb:.2f}  "
                  f"FUSION+CAVITY={derived:.2f}   {note}")
        print("\nIf the RGB_FUSION_CAVITY rows cite almost only RGB, the derived")
        print("images are indeed ignored. That is a negative result worth reporting,")
        print("and it explains why the between-condition effects are small.")
    else:
        print(f"\n{report} not found. Run first: python -m relief evaluate")

    print("\n" + "=" * 78)
    print("The three findings above are descriptive and reported without")
    print("hypothesis testing, as set out in PRAREGISTRASI.md. The only")
    print("hypothesis tests are those in scripts/03_primary_tests.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
