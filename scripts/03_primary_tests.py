"""
The pre-specified primary tests, with Holm correction.

Run AFTER `python -m relief evaluate`; reads `<output_root>/eval/report.json`
as given by the project config (`relief.config.load_config`). Set
RELIEF_OUTPUT_ROOT (or a `.env` file / RELIEF_CONFIG) to point elsewhere.

Only four comparisons are tested: `RGB_FUSION_CAVITY - RGB_SAJA` for each
model. Every other comparison (RGB_DEPTH, ablations, per-class, calibration)
is exploratory and reported without correction, clearly labelled.

Four tests at once give a 1 - 0.95^4 = 19% chance that one "passes" by luck.
The Holm correction keeps that chance at 5% for the whole family of tests, and
has more power than plain Bonferroni because its threshold relaxes after the
first test is rejected.

Effect sizes and confidence intervals still come from the paired bootstrap;
only the p-values are corrected. The paper reports the effect sizes, not the
p-values.

    pip install scipy          # if wilcoxon_p is still None
    python scripts/03_primary_tests.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relief.config import load_config

MAIN_CONDITION = "RGB_FUSION_CAVITY"
BASELINE = "RGB_SAJA"
ALPHA = 0.05


def holm(p_values, alpha=ALPHA):
    """
    Holm-Bonferroni correction.

    Returns a list of (original_index, p, p_adjusted, rejected), sorted by
    ascending p.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    out = []
    prev_p_adj = 0.0
    still_rejecting = True
    for pos, i in enumerate(order):
        p = p_values[i]
        # Holm threshold for position (pos+1)
        threshold = alpha / (m - pos)
        if still_rejecting and p <= threshold:
            rejected = True
        else:
            still_rejecting = False
            rejected = False
        # adjusted p, kept monotonically increasing
        p_adj = min(1.0, max(prev_p_adj, (m - pos) * p))
        prev_p_adj = p_adj
        out.append((i, p, p_adj, rejected))
    return out


def main() -> int:
    cfg = load_config()
    report = cfg.eval_dir / "report.json"
    print(f"report : {report}\n")

    if not report.exists():
        print(f"{report} not found. Run first: python -m relief evaluate")
        return 1

    rep = json.loads(report.read_text(encoding="utf-8"))
    primary = [t for t in rep.get("tests", [])
               if "error" not in t
               and t["cond_a"] == MAIN_CONDITION and t["cond_b"] == BASELINE]

    if not primary:
        print(f"no {MAIN_CONDITION} vs {BASELINE} test in report.json")
        return 1

    primary.sort(key=lambda t: t["model"])

    use_wilcoxon = all(t.get("wilcoxon_p") is not None for t in primary)
    p_source = "wilcoxon_p" if use_wilcoxon else "boot_p_two_sided"
    p_values = [t[p_source] for t in primary]

    print(f"PRIMARY TESTS (pre-specified): {MAIN_CONDITION} - {BASELINE}")
    print(f"metric     : {primary[0]['metric']}")
    print(f"p-value    : {p_source}")
    print(f"correction : Holm-Bonferroni, {len(primary)} tests, family alpha {ALPHA}\n")

    if not use_wilcoxon:
        print("NOTE: wilcoxon_p is not available (scipy not installed), so the")
        print("bootstrap p-values are used. For the paper, install scipy and rerun")
        print("`relief evaluate` so the signed-rank test is computed too.\n")

    head = (f"{'model':<26}{'n':>5}{'delta':>9}{'95% CI':>20}"
            f"{'p':>10}{'Holm p':>10}{'verdict':>14}")
    print(head)
    print("-" * len(head))

    for i, p, p_adj, rejected in holm(p_values):
        t = primary[i]
        lo, hi = t["ci95"]
        print(f"{t['model'][:25]:<26}{t['n_panels']:>5}{t['mean_diff']:>+9.3f}"
              f"{f'[{lo:+.3f},{hi:+.3f}]':>20}{p:>10.4f}{p_adj:>10.4f}"
              f"{('significant' if rejected else 'n.s.'):>14}")

    n_rejected = sum(1 for _, _, _, d in holm(p_values) if d)
    print(f"\n{n_rejected} of {len(primary)} primary tests survive the Holm correction.")

    if n_rejected == 0:
        print("\nNo effect survives. That is a valid, reportable result: on the test")
        print("panels, the derived images do not measurably raise recall. The")
        print("paper's main finding should then rest on characterising the failures")
        print("(under-listing of objects and inverted calibration).")
    elif n_rejected < len(primary):
        print("\nThe effect depends on the model rather than holding in general.")
        print("Report that pattern as is; do not average it into a single number.")
    else:
        print("\nEvery model shows an effect. Also check whether its size is")
        print("meaningful, not just past the statistical threshold.")

    print("\nAll other comparisons in report.json are EXPLORATORY.")
    print("Report them without correction, and label them exploratory explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
