"""
relief.evaluate
===============
Scoring for the relief analysis study: match model output against the
Layer-1 ground truth, compute the reported metrics, test the differences
between input conditions, and emit LaTeX tables.

Inputs
------
    gt      : gt_layer1.json written by relief.gt.extract_dataset
    runs    : the record.json files written by relief.runner, read with
              relief.artifacts.load_records

A record carries the model answer in two forms: `validated` (passed the
Pydantic schema) and `parsed` (json.loads succeeded, schema may not have).
Scoring prefers the first and falls back to the second, flagging the row as
`partial`, so a schema violation is not scored the same as no answer at all.

Design notes
------------
Matching is done at three levels of strictness because bounding boxes
from current models are coarse and any single criterion misrepresents
performance. A parse failure is scored as zero recall on that panel
rather than dropped, since dropping it would flatter models that fail on
exactly the panels they find hard; the parsed-only figures are reported
alongside so the selection effect stays visible.

An object entry may stand for several identical objects through its
`jumlah` field, so counts sum that field rather than counting entries.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Vocabulary mapping
# ---------------------------------------------------------------------

# The schema offers a finer vocabulary than the annotation. Mythological
# figures are folded into `human` because this version of the dataset
# does not encode them separately; without this fold, a model that
# correctly reports an apsara would be counted as hallucinating.
CATEGORY_MAP: Dict[str, Optional[str]] = {
    "figur_manusia": "human",
    "figur_dewa": "human",
    "figur_pendeta": "human",
    "figur_bangsawan": "human",
    "figur_mitologis": "human",
    "figur_hewan": "animal",
    "benda_bawaan": "artefak",
    "wadah": "artefak",
    "senjata": "artefak",
    "perabot": "artefak",
    "arsitektur": "building",
    "tumbuhan": "vegetation",
    "ornamen": "decoration",
    "teks_aksara": "decoration",
    "elemen_batu": None,             # uncarved stone: background, excluded
    "tidak_teridentifikasi": "*",    # wildcard: asserts existence, not class
}

CLASSES = ["human", "building", "animal", "vegetation", "artefak", "decoration"]


def map_category(kat: str) -> Optional[str]:
    return CATEGORY_MAP.get(kat, None)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_gt(path: str) -> Dict[str, dict]:
    """Ground truth keyed by panel id."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {p["panel_id"]: p for p in data["panels"]}


def result_payload(rec: dict, allow_unvalidated: bool = True) -> Optional[dict]:
    """
    The model answer carried by one record.

    Prefers `validated` (passed the schema). When only `parsed` exists
    (JSON that parsed but broke the schema, e.g. a value outside an enum),
    it is still returned as long as `allow_unvalidated` is True, and the
    record is flagged `partial` in its score row. Discarding it would treat
    "answered with one wrong enum value" the same as "did not answer",
    which are very different things for small models.
    """
    if rec.get("validated"):
        return rec["validated"]
    if allow_unvalidated and rec.get("parsed"):
        return rec["parsed"]
    return None


def _objects(rec: dict, allow_unvalidated: bool = True) -> List[dict]:
    h = result_payload(rec, allow_unvalidated) or {}
    lay = h.get("objek") or h.get("lapis1") or h.get("inventaris_objek") or []
    if isinstance(lay, dict):
        lay = lay.get("objek", [])
    return lay if isinstance(lay, list) else []


def parse_predictions(rec: dict, allow_unvalidated: bool = True) -> List[dict]:
    """
    Normalise one run record into a flat list of predicted objects.

    Keeps the mapped class, the box, the grid cells, the confidence, the
    evidence sources, and the multiplicity.
    """
    out = []
    for o in _objects(rec, allow_unvalidated):
        if not isinstance(o, dict):
            continue
        cls = map_category(o.get("kategori", ""))
        if cls is None:
            continue
        bbox = o.get("bbox_ternormalisasi") or []
        if len(bbox) != 4:
            bbox = None
        else:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        out.append(
            {
                "id": o.get("id"),
                "cls": cls,
                "raw_category": o.get("kategori"),
                "bbox": bbox,
                "cells": [str(c).upper() for c in (o.get("sel_grid") or [])],
                "conf": float(o.get("keyakinan", 0.0) or 0.0),
                "sources": list(o.get("sumber_bukti") or []),
                "count": max(1, int(o.get("jumlah", 1) or 1)),
            }
        )
    return out


# ---------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------

def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _class_ok(pred_cls: str, gt_cls: str) -> bool:
    return pred_cls == "*" or pred_cls == gt_cls


def match(
    preds: List[dict],
    gts: List[dict],
    level: str = "grid",
    iou_thr: float = 0.3,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Greedily pair predictions with annotated instances.

    level="grid" pairs a prediction with an instance when the classes
    agree and at least one named grid cell is among the cells the
    instance occupies; ties are broken by box overlap. level="bbox"
    requires the overlap itself to reach `iou_thr`.

    Returns the matched pairs with their score, plus the indices of the
    unmatched predictions and unmatched instances.
    """
    cand = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if not _class_ok(p["cls"], g["cls"]):
                continue
            ov = iou(p["bbox"], g["bbox"])
            if level == "bbox":
                if ov >= iou_thr:
                    cand.append((ov, pi, gi))
            else:
                shared = len(set(p["cells"]) & set(g["cells"]))
                if shared:
                    cand.append((shared + ov, pi, gi))
    cand.sort(reverse=True)

    used_p, used_g, pairs = set(), set(), []
    for score, pi, gi in cand:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi, score))
    return (
        pairs,
        [i for i in range(len(preds)) if i not in used_p],
        [i for i in range(len(gts)) if i not in used_g],
    )


def gt_objects(panel: dict) -> List[dict]:
    """Annotated instances in the shape the matcher expects."""
    return [
        {
            "cls": i["cls"],
            "bbox": tuple(i["bbox_norm"]),
            "cells": [c.upper() for c in i["grid_cells"]],
            "salience": i.get("salience"),
            "area_frac": i["area_frac"],
            "instance_id": i["instance_id"],
        }
        for i in panel["instances"]
    ]


# ---------------------------------------------------------------------
# Per-panel scoring
# ---------------------------------------------------------------------

def score_panel(rec: dict, panel: dict, allow_unvalidated: bool = True) -> dict:
    """Score one run record against one annotated panel."""
    cell = rec["cell"]
    status = rec.get("status", "error")
    ok = status == "ok"
    usable = result_payload(rec, allow_unvalidated) is not None
    gts = gt_objects(panel)
    preds = parse_predictions(rec, allow_unvalidated) if usable else []

    gt_counts = Counter(g["cls"] for g in gts)
    pred_counts: Counter = Counter()
    for p in preds:
        if p["cls"] != "*":
            pred_counts[p["cls"]] += p["count"]

    row = {
        "panel_id": cell["panel_id"],
        "site": cell["site"],
        "group": cell["group"],
        "condition": cell["condition"],
        "model": cell["model"],
        "rep": cell["rep"],
        "max_side": cell["max_side"],
        "status": status,
        "parsed": ok,
        "usable": usable,
        "partial": usable and not ok,
        "n_attempts": rec.get("n_attempts", 1),
        "n_gt": len(gts),
        "n_pred": sum(p["count"] for p in preds),
        "count_abs_err": abs(sum(p["count"] for p in preds) - len(gts)),
        "count_signed_err": sum(p["count"] for p in preds) - len(gts),
        "per_class_gt": dict(gt_counts),
        "per_class_pred": dict(pred_counts),
        "source_usage": Counter(
            s for p in preds for s in p["sources"]
        ),
    }

    for level, thr, tag in (
        ("grid", 0.0, "grid"),
        ("bbox", 0.3, "iou30"),
        ("bbox", 0.5, "iou50"),
    ):
        pairs, fp, fn = match(preds, gts, level=level, iou_thr=thr)
        tp = len(pairs)
        row[f"tp_{tag}"] = tp
        row[f"fp_{tag}"] = len(fp)
        row[f"fn_{tag}"] = len(fn)
        row[f"recall_{tag}"] = tp / len(gts) if gts else float("nan")
        row[f"precision_{tag}"] = tp / len(preds) if preds else float("nan")
        if tag == "grid":
            matched_pi = {pi for pi, _, _ in pairs}
            row["hallucination_rate"] = (
                len(fp) / len(preds) if preds else float("nan")
            )
            row["conf_labels"] = [
                (p["conf"], 1 if i in matched_pi else 0)
                for i, p in enumerate(preds)
            ]
            row["recall_by_instance"] = [
                {
                    "instance_id": gts[gi]["instance_id"],
                    "cls": gts[gi]["cls"],
                    "salience": gts[gi]["salience"],
                    "area_frac": gts[gi]["area_frac"],
                    "found": 1,
                }
                for _, gi, _ in pairs
            ] + [
                {
                    "instance_id": gts[gi]["instance_id"],
                    "cls": gts[gi]["cls"],
                    "salience": gts[gi]["salience"],
                    "area_frac": gts[gi]["area_frac"],
                    "found": 0,
                }
                for gi in fn
            ]
    return row


def score_all(
    runs: Sequence[dict],
    gt: Dict[str, dict],
    show_progress: bool = True,
    allow_unvalidated: bool = True,
) -> List[dict]:
    rows = []
    for rec in tqdm(runs, desc="scoring runs", unit="run",
                    disable=not show_progress):
        pid = rec["cell"]["panel_id"]
        if pid not in gt:
            continue
        rows.append(score_panel(rec, gt[pid], allow_unvalidated))
    return rows


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def aggregate(
    rows: Sequence[dict],
    by: Sequence[str] = ("model", "condition"),
    parsed_only: bool = False,
) -> List[dict]:
    """
    Summarise panel-level rows into one line per group.

    With `parsed_only`, panels whose output failed the schema are dropped
    instead of scored as zero recall. Report both: the difference is the
    selection effect that parse failures introduce.

    Three failure columns, deliberately kept apart:
      parse_fail_rate  share that did not pass the schema
      unusable_rate    share with no JSON at all, not even an invalid one
      partial_rate     share scored from `parsed` because `validated` was
                       missing, i.e. valid JSON that broke the schema
    """
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in by)].append(r)

    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        used = [r for r in rs if r["parsed"]] if parsed_only else rs
        if not used:
            continue
        conf = [cl for r in used for cl in r.get("conf_labels", [])]
        agg = {
            **dict(zip(by, key)),
            "n_panels": len(rs),
            "parse_fail_rate": 1 - _mean([r["parsed"] for r in rs]),
            "unusable_rate": 1 - _mean([r.get("usable", r["parsed"]) for r in rs]),
            "partial_rate": _mean([r.get("partial", False) for r in rs]),
            "repair_rate": _mean([max(0, r.get("n_attempts", 1) - 1) for r in rs]),
            "count_mae": _mean([r["count_abs_err"] for r in used]),
            "count_bias": _mean([r["count_signed_err"] for r in used]),
            "recall_grid": _mean([r["recall_grid"] for r in used]),
            "precision_grid": _mean([r["precision_grid"] for r in used]),
            "recall_iou30": _mean([r["recall_iou30"] for r in used]),
            "recall_iou50": _mean([r["recall_iou50"] for r in used]),
            "hallucination_rate": _mean([r["hallucination_rate"] for r in used]),
            "ece": ece([c for c, _ in conf], [l for _, l in conf]),
            "auroc": auroc([c for c, _ in conf], [l for _, l in conf]),
        }
        agg["f1_grid"] = _f1(agg["precision_grid"], agg["recall_grid"])
        src = Counter()
        for r in used:
            src.update(r["source_usage"])
        tot = sum(src.values()) or 1
        agg["source_share"] = {k: round(v / tot, 3) for k, v in src.most_common()}
        out.append(agg)
    return out


def _f1(p, r):
    if not p or not r or math.isnan(p) or math.isnan(r) or (p + r) == 0:
        return float("nan")
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

def ece(scores: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """
    Expected calibration error with equal-mass bins.

    Equal-mass rather than equal-width because confidence values from
    these models pile up near the top, which would leave most equal-width
    bins empty and make the number unstable.
    """
    if len(scores) < n_bins:
        return float("nan")
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    order = np.argsort(s)
    s, y = s[order], y[order]
    bins = np.array_split(np.arange(len(s)), n_bins)
    total = 0.0
    for b in bins:
        if b.size == 0:
            continue
        total += (b.size / len(s)) * abs(y[b].mean() - s[b].mean())
    return float(total)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve, by rank statistic. Ties handled."""
    y = np.asarray(labels)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    s = np.asarray(scores, dtype=float)
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(scores, labels, n_bins: int = 10) -> List[dict]:
    """Points for a reliability diagram."""
    if not scores:
        return []
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    order = np.argsort(s)
    s, y = s[order], y[order]
    out = []
    for b in np.array_split(np.arange(len(s)), n_bins):
        if b.size:
            out.append(
                {"confidence": float(s[b].mean()),
                 "accuracy": float(y[b].mean()),
                 "n": int(b.size)}
            )
    return out


# ---------------------------------------------------------------------
# Recall against relief salience
# ---------------------------------------------------------------------

def recall_vs_salience(
    rows: Sequence[dict],
    model: str,
    conditions: Sequence[str],
    n_bins: int = 5,
    key: str = "salience",
) -> Dict[str, List[dict]]:
    """
    Recall as a function of how legible an instance is.

    Bin edges come from the pooled distribution so that every condition
    is scored on the same bins; otherwise the curves are not comparable.
    Instances with no score, which happens when no salience image was
    supplied, are skipped.
    """
    pool = [
        d[key]
        for r in rows
        if r["model"] == model
        for d in r.get("recall_by_instance", [])
        if d.get(key) is not None
    ]
    if not pool:
        return {}
    edges = np.quantile(np.asarray(pool, float), np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9

    out: Dict[str, List[dict]] = {}
    for cond in conditions:
        items = [
            d
            for r in rows
            if r["model"] == model and r["condition"] == cond
            for d in r.get("recall_by_instance", [])
            if d.get(key) is not None
        ]
        if not items:
            continue
        v = np.asarray([d[key] for d in items], float)
        f = np.asarray([d["found"] for d in items], float)
        idx = np.clip(np.digitize(v, edges) - 1, 0, n_bins - 1)
        out[cond] = [
            {
                "bin": int(b),
                "range": (float(edges[b]), float(edges[b + 1])),
                "n": int((idx == b).sum()),
                "recall": float(f[idx == b].mean()) if (idx == b).any() else float("nan"),
            }
            for b in range(n_bins)
        ]
    return out


def recall_by_class(rows, model: str, condition: str) -> Dict[str, dict]:
    """Per-class recall, which is what difficulty should be derived from."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["model"] != model or r["condition"] != condition:
            continue
        for d in r.get("recall_by_instance", []):
            acc[d["cls"]].append(d["found"])
    return {
        c: {"n": len(v), "recall": float(np.mean(v))}
        for c, v in sorted(acc.items(), key=lambda kv: -len(kv[1]))
    }


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def paired_panels(
    rows: Sequence[dict], model: str, cond_a: str, cond_b: str, metric: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Values for the panels where both conditions produced a score."""
    def grab(cond):
        d = {}
        for r in rows:
            if r["model"] == model and r["condition"] == cond:
                v = r.get(metric)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    d.setdefault(r["panel_id"], []).append(v)
        return {k: float(np.mean(v)) for k, v in d.items()}

    A, B = grab(cond_a), grab(cond_b)
    keys = sorted(set(A) & set(B))
    return np.array([A[k] for k in keys]), np.array([B[k] for k in keys]), keys


def paired_test(
    rows: Sequence[dict],
    model: str,
    cond_a: str,
    cond_b: str,
    metric: str = "recall_grid",
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """
    Paired bootstrap over panels, with a Wilcoxon signed-rank test.

    Panels are the unit of analysis, so resampling is over panels and the
    two conditions move together. The effect size and its interval matter
    more than the p-value with 46 panels.
    """
    a, b, keys = paired_panels(rows, model, cond_a, cond_b, metric)
    if len(keys) < 3:
        return {"n_panels": len(keys), "error": "too few paired panels"}

    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    res = {
        "model": model,
        "metric": metric,
        "cond_a": cond_a,
        "cond_b": cond_b,
        "n_panels": len(keys),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci95": (float(lo), float(hi)),
        "boot_p_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
    }
    try:
        from scipy.stats import wilcoxon

        if np.any(diff != 0):
            stat, p = wilcoxon(a, b)
            res["wilcoxon_stat"] = float(stat)
            res["wilcoxon_p"] = float(p)
    except Exception:  # noqa: BLE001
        res["wilcoxon_p"] = None
    return res


# ---------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------

def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def latex_main_table(
    agg: Sequence[dict],
    caption: str = "Layer-1 performance by input condition.",
    label: str = "tab:main",
) -> str:
    """Main results table, one row per model and condition."""
    head = (
        "\\begin{table}[t]\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\centering\\small\n"
        "  \\begin{tabular}{@{}llrrrrr@{}}\n"
        "    \\toprule\n"
        "    Model & Condition & MAE & $R_{\\text{grid}}$ & $F_1$ & Halluc. & ECE \\\\\n"
        "    \\midrule\n"
    )
    body = ""
    last = None
    for a in agg:
        m = a.get("model", "")
        show = "" if m == last else m.replace("_", "\\_")
        last = m
        body += (
            f"    {show} & {a.get('condition','').replace('_','\\_')} & "
            f"{_fmt(a['count_mae'],2)} & {_fmt(a['recall_grid'])} & "
            f"{_fmt(a['f1_grid'])} & {_fmt(a['hallucination_rate'])} & "
            f"{_fmt(a['ece'])} \\\\\n"
        )
    return head + body + "    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"


def latex_test_table(
    tests: Sequence[dict],
    caption: str = "Paired comparison against the RGB-only condition.",
    label: str = "tab:tests",
) -> str:
    head = (
        "\\begin{table}[t]\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\centering\\small\n"
        "  \\begin{tabular}{@{}llrrl@{}}\n"
        "    \\toprule\n"
        "    Model & Comparison & $n$ & $\\Delta$ & 95\\% CI \\\\\n"
        "    \\midrule\n"
    )
    body = ""
    for t in tests:
        if "error" in t:
            continue
        comp = f"{t['cond_a']} vs {t['cond_b']}".replace("_", "\\_")
        lo, hi = t["ci95"]
        body += (
            f"    {t['model'].replace('_','\\_')} & {comp} & {t['n_panels']} & "
            f"{t['mean_diff']:+.3f} & [{lo:+.3f}, {hi:+.3f}] \\\\\n"
        )
    return head + body + "    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"


# ---------------------------------------------------------------------
# One-call report
# ---------------------------------------------------------------------

def full_report(
    runs: Sequence[dict],
    gt: Dict[str, dict],
    baseline: str = "RGB_SAJA",
    metric: str = "recall_grid",
    out_dir: Optional[str] = None,
) -> dict:
    """Score everything, run the paired tests, and write the tables."""
    rows = score_all(runs, gt)
    agg = aggregate(rows)
    agg_parsed = aggregate(rows, parsed_only=True)

    models = sorted({r["model"] for r in rows})
    conds = sorted({r["condition"] for r in rows})
    combos = [(m, c) for m in models for c in conds if c != baseline]
    tests = [
        paired_test(rows, m, c, baseline, metric=metric)
        for m, c in tqdm(combos, desc="paired tests", unit="test")
    ]

    salience = {
        m: recall_vs_salience(rows, m, conds) for m in models
    }
    per_class = {
        f"{m}|{c}": recall_by_class(rows, m, c) for m in models for c in conds
    }

    report = {
        "n_rows": len(rows),
        "models": models,
        "conditions": conds,
        "aggregate": agg,
        "aggregate_parsed_only": agg_parsed,
        "tests": tests,
        "recall_vs_salience": salience,
        "recall_by_class": per_class,
    }

    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "scores_panel.json").write_text(
            json.dumps(rows, indent=2, default=str), encoding="utf-8"
        )
        (d / "report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        (d / "table_main.tex").write_text(latex_main_table(agg), encoding="utf-8")
        (d / "table_tests.tex").write_text(latex_test_table(tests), encoding="utf-8")
    return report


def summary_text(report: dict) -> str:
    """Readable summary of the aggregate table."""
    lines = [
        f"{'model':<26}{'condition':<20}{'MAE':>7}{'R_grid':>8}"
        f"{'F1':>7}{'halluc':>8}{'ECE':>7}{'AUROC':>7}{'parsefail':>11}"
    ]
    lines.append("-" * len(lines[0]))
    for a in report["aggregate"]:
        lines.append(
            f"{a.get('model','')[:25]:<26}{a.get('condition','')[:19]:<20}"
            f"{_fmt(a['count_mae'],2):>7}{_fmt(a['recall_grid']):>8}"
            f"{_fmt(a['f1_grid']):>7}{_fmt(a['hallucination_rate']):>8}"
            f"{_fmt(a['ece']):>7}{_fmt(a['auroc']):>7}"
            f"{_fmt(a['parse_fail_rate'],2):>11}"
        )
    lines.append("")
    for t in report["tests"]:
        if "error" in t:
            continue
        lo, hi = t["ci95"]
        lines.append(
            f"{t['model'][:25]:<26}{t['cond_a']} - {t['cond_b']}: "
            f"delta={t['mean_diff']:+.3f} CI=[{lo:+.3f},{hi:+.3f}] "
            f"n={t['n_panels']}"
        )
    return "\n".join(lines)
