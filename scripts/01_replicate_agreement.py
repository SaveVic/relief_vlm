"""
Step 1: noise between replicates, and what it means for the precision of the
results.

REVISED VERSION. The previous version compared the spread between replicates
directly with the effect size and concluded "noise is as large as the effect,
more replicates needed". That conclusion was wrong because it missed one thing:

    The confidence intervals of the primary tests ALREADY account for
    replicate noise.

Each panel's value is the mean of its replicates, so the noise feeds into the
spread of the between-panel differences, and the paired bootstrap pays for it
with a wider CI. An effect that stays clear of zero holds DESPITE that noise,
not because we ignored it.

The real question is not "is the noise large" but "what share of the
uncertainty comes from noise, and how much would the CI shrink with more
replicates". That is what this script computes.

Method: for two measurements from the same distribution,
E|r1 - r2| = 2*sigma/sqrt(pi), so sigma can be estimated from the mean spread.
The noise share of the variance of between-panel differences is then compared
with the total variance implied by the CI width.

Paths come from the project config (`relief.config.load_config`); set
RELIEF_OUTPUT_ROOT (or a `.env` file / RELIEF_CONFIG) to point elsewhere.

    python scripts/01_replicate_agreement.py
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relief.artifacts import load_records
from relief.config import load_config
from relief.evaluate import load_gt, score_all

METRIC = "recall_grid"
BASELINE = "RGB_SAJA"
MAIN_CONDITION = "RGB_FUSION_CAVITY"

# E|r1 - r2| = K * sigma for two iid normal observations
K = 2 / math.sqrt(math.pi)


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

    # ---- spread between replicates per (model, condition) ----------------
    groups = defaultdict(list)
    for r in rows:
        v = r.get(METRIC)
        if v is not None and not np.isnan(v):
            groups[(r["model"], r["condition"], r["panel_id"])].append(v)

    spread = defaultdict(list)
    n_rep = defaultdict(set)
    for (model, cond, _panel), vals in groups.items():
        n_rep[(model, cond)].add(len(vals))
        if len(vals) > 1:
            spread[(model, cond)].append(max(vals) - min(vals))

    if not spread:
        print("no panel has more than one replicate.")
        return 1

    print("SPREAD BETWEEN REPLICATES")
    head = (f"{'model':<26}{'condition':<20}{'panels':>6}{'identical':>10}"
            f"{'mean |r1-r2|':>15}{'max':>8}{'sigma':>8}")
    print(head)
    print("-" * len(head))
    sigma = {}
    for (model, cond), s in sorted(spread.items()):
        a = np.array(s)
        sigma[(model, cond)] = float(np.mean(a)) / K
        print(f"{model[:25]:<26}{cond[:19]:<20}{len(a):>6}{np.mean(a == 0):>9.0%}"
              f"{np.mean(a):>15.3f}{np.max(a):>8.3f}{sigma[(model, cond)]:>8.3f}")
    print("\n`sigma` = estimated standard deviation of one measurement, from |r1-r2|.")

    # ---- variance decomposition, if primary-test CIs are available -------
    if not report.exists():
        print(f"\n{report} not found; skipping variance decomposition.")
        print("Run `python -m relief evaluate` first.")
        return 0

    rep = json.loads(report.read_text(encoding="utf-8"))
    tests = [t for t in rep.get("tests", [])
             if "error" not in t and t["cond_a"] == MAIN_CONDITION
             and t["cond_b"] == BASELINE and t.get("metric") == METRIC]
    if not tests:
        print(f"\nno {MAIN_CONDITION} vs {BASELINE} test in report.json")
        return 0

    print("\n" + "=" * 78)
    print(f"UNCERTAINTY DECOMPOSITION: {MAIN_CONDITION} vs {BASELINE}")
    print("=" * 78)
    head = (f"{'model':<26}{'effect':>8}{'CI +/-':>9}{'ratio':>7}"
            f"{'noise share':>13}{'R=4':>10}{'R=8':>10}")
    print(head)
    print("-" * len(head))

    suggestions = []
    for t in sorted(tests, key=lambda x: -abs(x["mean_diff"])):
        model = t["model"]
        lo, hi = t["ci95"]
        half = (hi - lo) / 2
        effect = t["mean_diff"]
        n = t["n_panels"]

        s_a = sigma.get((model, MAIN_CONDITION))
        s_b = sigma.get((model, BASELINE))
        if s_a is None or s_b is None or half <= 0 or n < 3:
            print(f"{model[:25]:<26}{effect:>+8.3f}{half:>9.4f}"
                  f"{'':>7}{'n/a':>13}")
            continue

        R = min(min(n_rep.get((model, MAIN_CONDITION), {2})),
                min(n_rep.get((model, BASELINE), {2})))
        R = max(R, 2)

        var_noise = s_a ** 2 / R + s_b ** 2 / R
        sd_total = half / 1.96 * math.sqrt(n)
        var_total = sd_total ** 2
        share = var_noise / var_total if var_total > 0 else float("nan")
        var_rest = max(var_total - var_noise, 0.0)

        def ci_for(r):
            return 1.96 * math.sqrt(var_rest + var_noise * R / r) / math.sqrt(n)

        print(f"{model[:25]:<26}{effect:>+8.3f}{half:>9.4f}"
              f"{abs(effect) / half:>7.1f}{share:>12.0%}"
              f"{ci_for(2 * R):>10.4f}{ci_for(4 * R):>10.4f}")
        suggestions.append((model, half, ci_for(2 * R), abs(effect) > half))

    print("\n`ratio`       = effect divided by the CI half-width. Above 1 means the")
    print("                CI excludes zero.")
    print("`noise share` = part of the variance of between-panel differences that")
    print("                comes from replicate noise. The rest is effect")
    print("                heterogeneity across panels, which does NOT shrink with")
    print("                more replicates.")
    print("`R=4`, `R=8`  = CI half-width if replicates were doubled and quadrupled.")

    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    if suggestions:
        gains = [1 - b / a for _, a, b, _ in suggestions]
        mean_gain = sum(gains) / len(gains)
        n_hold = sum(1 for *_, holds in suggestions if holds)
        print(f"{n_hold} of {len(suggestions)} models have a CI that excludes zero, "
              f"and that already\naccounts for replicate noise.")
        print(f"\nDoubling the replicates would narrow the CI by only "
              f"{mean_gain:.0%} on average.")
        if mean_gain < 0.10:
            print("That is not worth the cost and time. The current number of")
            print("replicates is enough; the remaining uncertainty comes from")
            print("differences between panels, not from repetition noise.")
        else:
            print("Consider adding replicates if you want a narrower CI.")
    print("\nFor the paper: report sigma as the measurement precision limit, and")
    print("state that the primary-test CIs already include that noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
