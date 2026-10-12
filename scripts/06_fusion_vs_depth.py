"""
RGB_FUSION_CAVITY vs RGB_DEPTH — a direct comparison between the two
conditions that both supply geometric information.

Why this is needed. The primary tests compare each condition against
RGB_SAJA, and both win for some models. But the question the paper actually
asks is not "does extra information help" but "does this enhancement pipeline
help MORE than a plain depth map". A monocular depth map is far cheaper and
already a standard tool; if the two are on par, the contribution of the
fusion/cavity pipeline is not established.

This comparison was NOT pre-registered, so the results are exploratory and
must be reported with that label. Holm correction is still applied so it is
not looser than the primary tests.

    python scripts/06_fusion_vs_depth.py

Paths come from relief.config.load_config() (RELIEF_OUTPUT_ROOT, .env,
configs/default.yaml or $RELIEF_CONFIG).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relief.artifacts import load_records
from relief.config import load_config
from relief.evaluate import load_gt, paired_test, score_all

METRIC = "recall_grid"
ALPHA = 0.05


def holm(p_values, alpha=ALPHA):
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    out, p_prev, still_rejecting = [], 0.0, True
    for pos, i in enumerate(order):
        p = p_values[i]
        if still_rejecting and p <= alpha / (m - pos):
            rejected = True
        else:
            still_rejecting, rejected = False, False
        p_adj = min(1.0, max(p_prev, (m - pos) * p))
        p_prev = p_adj
        out.append((i, p, p_adj, rejected))
    return out


def main() -> int:
    cfg = load_config()
    runs, gt_path = cfg.runs_dir, cfg.gt_path
    print(f"runs   : {runs}")
    print(f"gt     : {gt_path}\n")

    records = load_records(runs)
    if not records:
        print(f"no records in {runs} yet")
        return 1
    rows = score_all(records, load_gt(str(gt_path)), show_progress=False)
    models = sorted({r["model"] for r in rows})

    tests = []
    for model in models:
        t = paired_test(rows, model, "RGB_FUSION_CAVITY", "RGB_DEPTH", metric=METRIC)
        if "error" not in t:
            tests.append(t)

    if not tests:
        print("not enough paired panels to test")
        return 1

    use_wilcoxon = all(t.get("wilcoxon_p") is not None for t in tests)
    source = "wilcoxon_p" if use_wilcoxon else "boot_p_two_sided"
    p_values = [t[source] for t in tests]

    print("EXPLORATORY (not pre-registered)")
    print("RGB_FUSION_CAVITY - RGB_DEPTH")
    print(f"metric     : {METRIC}")
    print(f"p-value    : {source}")
    print(f"correction : Holm-Bonferroni, {len(tests)} tests\n")

    head = (f"{'model':<26}{'n':>5}{'delta':>9}{'CI 95%':>20}"
            f"{'p':>10}{'p Holm':>10}{'verdict':>12}")
    print(head)
    print("-" * len(head))
    for i, p, p_adj, rejected in holm(p_values):
        t = tests[i]
        lo, hi = t["ci95"]
        print(f"{t['model'][:25]:<26}{t['n_panels']:>5}{t['mean_diff']:>+9.3f}"
              f"{f'[{lo:+.3f},{hi:+.3f}]':>20}{p:>10.4f}{p_adj:>10.4f}"
              f"{('different' if rejected else 'on par'):>12}")

    n_different = sum(1 for _, _, _, d in holm(p_values) if d)
    n_favour_fc = sum(1 for t in tests if t["mean_diff"] > 0)

    print(f"\n{n_different} of {len(tests)} models show a difference between the two conditions.")
    print(f"Direction: {n_favour_fc} models lean towards FUSION_CAVITY, "
          f"{len(tests) - n_favour_fc} lean towards DEPTH.")

    if n_different == 0:
        print("\n=> The enhancement pipeline is NOT shown to beat a plain depth")
        print("   map. This limits the paper's contribution claim: what is shown")
        print("   is 'extra geometric information helps', not 'the fusion/cavity")
        print("   pipeline helps'. State that limit explicitly, and consider")
        print("   comparing the compute cost of the two as a separate argument.")
    else:
        print("\n=> Some models distinguish the two. Check the direction: if it is")
        print("   not consistent, report it as a model-by-condition interaction,")
        print("   not as one condition being superior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
