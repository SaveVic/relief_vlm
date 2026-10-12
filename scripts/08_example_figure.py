"""
Example figure: one best and one worst result, side by side.

Matching uses the SAME functions as the evaluation (`parse_predictions`,
`gt_objects`, `match(level="grid")`), so what the figure shows is exactly
what the tables count.

Visual encoding (two colours for meaning, shape for source):

    blue,   mask area             annotated object the model found
    orange, mask area + hatching  annotated object the model did NOT mention
    blue,   dashed box            object the model listed that matched
    orange, dashed box            object the model listed that did not match

Annotations are drawn as areas from the segmentation mask (the object's
actual shape), not boxes, so dense panels are not buried under dozens of
overlapping boxes. Use `--gt-style box` to go back to boxes.

The two colours pass the colour-blind palette check (all pairs, ΔE 24.7).
Hatching and line style keep all four categories distinguishable in
black-and-white print.

Usage:

    # 1. inspect the candidates first, without drawing
    python scripts/08_example_figure.py --list

    # 2. draw with automatic selection (highest and lowest F1)
    python scripts/08_example_figure.py

    # 3. recommended: fix model and condition so the contrast is purely about the panel
    python scripts/08_example_figure.py --model openai/gpt-4.1 \\
        --condition RGB_FUSION_CAVITY

    # 4. or pick them yourself from the candidate list
    python scripts/08_example_figure.py --best <cell_id> --worst <cell_id>

Paths come from relief.config.load_config() (RELIEF_OUTPUT_ROOT,
RELIEF_DATA_ROOT, RELIEF_FIGURES_DIR, .env, configs/default.yaml or
$RELIEF_CONFIG); --data-root, --output-root and --save-dir override them.

Output: <figure_dir>/fig_examples.pdf, .png and .json. The .json file holds
every number for both examples — use it when writing the caption.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from relief.artifacts import load_records
from relief.config import load_config
from relief.evaluate import (gt_objects, load_gt, match, parse_predictions,
                             result_payload, score_panel)

Image.MAX_IMAGE_PIXELS = None

# ---- visual encoding -----------------------------------------------------
COLOR_HIT = "#2a78d6"     # categorical slot 1, validated
COLOR_MISS = "#eb6834"    # categorical slot 2, validated
SOLID = "-"
DASHED = (0, (3.0, 1.8))
LW = 1.0

CONDITION_LABELS = {"RGB_SAJA": "RGB", "RGB_DEPTH": "RGB+D",
                    "RGB_FUSION_CAVITY": "RGB+F+C"}
MODEL_LABELS = {"openai/gpt-4.1": "GPT-4.1", "openai/gpt-4.1-mini": "GPT-4.1-mini",
                "google/gemini-3-flash": "Gemini-3-Flash",
                "qwen/qwen3-7-plus": "Qwen3-7-Plus"}

TEXTWIDTH_IN = 7.16    # IEEE two-column \textwidth


# ==========================================================================
# Selecting examples
# ==========================================================================

def score_candidates(records, gt, args):
    """Score every record that passes the filters, one row per record."""
    out = []
    for rec in records:
        cell = rec.get("cell") or {}
        if args.split and cell.get("split") != args.split:
            continue
        if args.model and cell.get("model") != args.model:
            continue
        if args.condition and cell.get("condition") != args.condition:
            continue
        if not args.allow_partial and rec.get("status") != "ok":
            continue
        if result_payload(rec, allow_unvalidated=True) is None:
            continue
        panel = gt.get(cell.get("panel_id"))
        if panel is None:
            continue
        row = score_panel(rec, panel)
        if row["n_gt"] < args.min_gt:
            continue
        r = row["recall_grid"]
        p = row["precision_grid"]
        if p != p:      # NaN: the model listed no objects at all
            p = 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out.append({"rec": rec, "panel": panel, "row": row,
                    "recall": r, "precision": p, "f1": f1})
    return out


def sort_key(x, by):
    return x["f1"] if by == "f1" else x["recall"]


def print_candidates(candidates, by, n):
    ranked = sorted(candidates, key=lambda x: sort_key(x, by), reverse=True)

    def line(x):
        c = x["rec"]["cell"]
        return (f"  {c['cell_id'][:66]:<67}"
                f"{x['row']['n_gt']:>4}{x['row']['n_pred']:>5}"
                f"{x['recall']:>7.3f}{x['precision']:>7.3f}{x['f1']:>7.3f}")

    head = f"  {'cell_id':<67}{'GT':>4}{'list':>5}{'R':>7}{'P':>7}{'F1':>7}"
    print(f"\n{len(candidates)} candidates pass the filters. Ranked by {by}.\n")
    print(f"BEST {n}"); print(head)
    for x in ranked[:n]:
        print(line(x))
    print(f"\nWORST {n}"); print(head)
    for x in ranked[::-1][:n]:
        print(line(x))
    return ranked


def find_candidate(candidates, cell_id):
    for x in candidates:
        if x["rec"]["cell"]["cell_id"] == cell_id:
            return x
    raise SystemExit(f"cell_id not found among the candidates: {cell_id}\n"
                     f"(check the --model/--condition/--min-gt/--split filters)")


# ==========================================================================
# Drawing
# ==========================================================================

def load_photo(data_root, panel_id, max_width=1800):
    path = Path(data_root) / panel_id / "rgb.png"
    if not path.exists():
        raise SystemExit(f"photo not found: {path}")
    img = Image.open(path).convert("L")
    if img.width > max_width:
        h = round(img.height * max_width / img.width)
        img = img.resize((max_width, h), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    # Lightened and greyscaled so the coloured overlays stand out against
    # the stone texture without hiding the carving.
    return 0.30 + 0.70 * a


def find_mask(data_root, panel, name="seg2.png"):
    """Path to the segmentation mask; mask_path in the GT may be absolute from another machine."""
    candidates = []
    if panel.get("mask_path"):
        mp = Path(panel["mask_path"])
        candidates += [mp, Path(data_root) / mp, Path(data_root) / panel["panel_id"] / mp.name]
    candidates.append(Path(data_root) / panel["panel_id"] / name)
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"mask not found for {panel['panel_id']}; tried: "
                     + ", ".join(str(c) for c in candidates))


def instance_labels(data_root, panel, W, H):
    """
    One label array (H, W) the size of the displayed photo: value i+1 for
    pixels of instance i (same order as gt_objects), 0 for background.

    Instance pixels = the instance's colour in the mask, restricted to its
    bbox_px, so two instances with the same colour (reused shade) stay apart.
    """
    m = np.asarray(Image.open(find_mask(data_root, panel)).convert("RGB"))
    mh, mw = m.shape[:2]
    label = np.zeros((mh, mw), dtype=np.int32)
    for i, inst in enumerate(panel["instances"]):
        rgb = inst.get("rgb")
        if rgb is None:
            continue
        x0, y0, x1, y1 = inst["bbox_px"]
        block = m[y0:y1, x0:x1]
        hit = np.all(block == np.array(rgb, dtype=np.uint8), axis=-1)
        sub = label[y0:y1, x0:x1]
        sub[hit & (sub == 0)] = i + 1
    if (mw, mh) != (W, H):
        label = np.asarray(Image.fromarray(label).resize((W, H), Image.NEAREST))
    return label


def draw_mask(ax, label, idx, color, hatch, alpha=0.42):
    """Fill the area of instances `idx` with a translucent colour, contour and hatching."""
    if not idx:
        return
    sel = np.isin(label, [i + 1 for i in idx])
    if not sel.any():
        return
    rgba = np.zeros(sel.shape + (4,), dtype=np.float32)
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = sel * alpha
    ax.imshow(rgba, interpolation="nearest", zorder=3, rasterized=True)
    # contour per instance so touching objects stay separate
    for i in idx:
        b = (label == i + 1).astype(float)
        if b.any():
            ax.contour(b, levels=[0.5], colors=[color], linewidths=0.6, zorder=3.2)
    if hatch:
        ax.contourf(sel.astype(float), levels=[0.5, 1.5], colors="none",
                    hatches=[hatch], zorder=3.1)
        # the hatch colour follows the collection's edgecolor
        for c in ax.collections[-1:]:
            c.set_edgecolor(color); c.set_linewidth(0)


def cell_centre(cell):
    try:
        col = "ABCDEFGH".index(cell[0].upper())
        row = int(cell[1:]) - 1
        return (col + 0.5) / 8, (row + 0.5) / 5
    except (ValueError, IndexError):
        return None


def halo(lw):
    return [patheffects.withStroke(linewidth=lw + 1.4, foreground="white", alpha=0.85)]


def draw_box(ax, bbox, W, H, color, style, hatch=None, z=3):
    x0, y0, x1, y1 = bbox
    r = Rectangle((x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                  fill=False, edgecolor=color, linestyle=style, linewidth=LW,
                  hatch=hatch, zorder=z)
    r.set_path_effects(halo(LW))
    ax.add_patch(r)


def draw_panel(ax, x, data_root, label, gt_style="mask"):
    rec, panel = x["rec"], x["panel"]
    cell = rec["cell"]
    photo = load_photo(data_root, cell["panel_id"])
    H, W = photo.shape
    ax.imshow(photo, cmap="gray", vmin=0, vmax=1, interpolation="lanczos",
              rasterized=True)

    # the 8x5 grid used for matching
    for i in range(1, 8):
        ax.axvline(i * W / 8, color="white", lw=0.4, alpha=0.55, zorder=2)
    for j in range(1, 5):
        ax.axhline(j * H / 5, color="white", lw=0.4, alpha=0.55, zorder=2)

    preds = parse_predictions(rec, allow_unvalidated=True)
    gts = gt_objects(panel)
    pairs, fp, fn = match(preds, gts, level="grid", iou_thr=0.0)
    matched_p = {pi for pi, _, _ in pairs}
    matched_g = {gi for _, gi, _ in pairs}

    # annotations underneath, predictions on top
    if gt_style == "mask":
        lab = instance_labels(data_root, panel, W, H)
        draw_mask(ax, lab, sorted(matched_g), COLOR_HIT, None)
        draw_mask(ax, lab, [i for i in range(len(gts)) if i not in matched_g],
                  COLOR_MISS, "////")
    else:
        for gi, g in enumerate(gts):
            if gi in matched_g:
                draw_box(ax, g["bbox"], W, H, COLOR_HIT, SOLID, z=3)
            else:
                draw_box(ax, g["bbox"], W, H, COLOR_MISS, SOLID, hatch="/////", z=3)

    no_bbox = 0
    for pi, p in enumerate(preds):
        color = COLOR_HIT if pi in matched_p else COLOR_MISS
        if p["bbox"] is not None:
            draw_box(ax, p["bbox"], W, H, color, DASHED, z=4)
        else:
            no_bbox += 1
            c = cell_centre(p["cells"][0]) if p["cells"] else None
            if c:
                ax.plot(c[0] * W, c[1] * H, marker="+", ms=6, mew=1.2,
                        color=color, zorder=4, path_effects=halo(1.2))

    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_anchor("N")      # top-align both panels even if the photo aspect ratios differ
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)

    row = x["row"]
    title = (f"({label}) {MODEL_LABELS.get(cell['model'], cell['model'])}, "
             f"{CONDITION_LABELS.get(cell['condition'], cell['condition'])}, "
             f"{cell['panel_id']}\n"
             f"{row['n_gt']} annotated, {row['n_pred']} listed, "
             f"R = {x['recall']:.2f}, P = {x['precision']:.2f}")
    ax.set_title(title, fontsize=7.5, loc="left", pad=3, linespacing=1.25)

    return {"preds": preds, "gts": gts, "pairs": pairs, "fp": fp, "fn": fn,
            "no_bbox": no_bbox, "size": (W, H)}


def add_legend(fig, gt_style="mask"):
    if gt_style == "mask":
        a = [Patch(facecolor=(*to_rgb(COLOR_HIT), 0.42), edgecolor=COLOR_HIT, lw=0.6,
                   label="Annotated, found"),
             Patch(facecolor=(*to_rgb(COLOR_MISS), 0.42), edgecolor=COLOR_MISS, lw=0.6,
                   hatch="////", label="Annotated, missed")]
    else:
        a = [Patch(facecolor="none", edgecolor=COLOR_HIT, linestyle=SOLID, lw=LW,
                   label="Annotated, found"),
             Patch(facecolor="none", edgecolor=COLOR_MISS, linestyle=SOLID, lw=LW,
                   hatch="/////", label="Annotated, missed")]
    h = a + [
        Line2D([], [], color=COLOR_HIT, linestyle=DASHED, lw=LW + 0.2,
               label="Listed, matched"),
        Line2D([], [], color=COLOR_MISS, linestyle=DASHED, lw=LW + 0.2,
               label="Listed, unmatched"),
    ]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, handlelength=2.4, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.005))


# ==========================================================================
# Numeric summary for the caption
# ==========================================================================

def summarise(role, x, g):
    rec, cell, row = x["rec"], x["rec"]["cell"], x["row"]
    payload = result_payload(rec, allow_unvalidated=True) or {}
    # keys below ("objek", "adegan", ...) are the model's output schema
    objects = {o.get("id"): o for o in payload.get("objek") or [] if isinstance(o, dict)}

    preds, gts = g["preds"], g["gts"]
    matched_p = {pi for pi, _, _ in g["pairs"]}

    def conf(idx):
        v = [preds[i]["conf"] for i in idx]
        return round(float(np.mean(v)), 3) if v else None

    unmatched_objects = []
    for pi in g["fp"]:
        p = preds[pi]
        o = objects.get(p["id"], {})
        unmatched_objects.append({"name": o.get("nama"), "category": p["raw_category"],
                                  "cells": p["cells"], "confidence": p["conf"]})

    scene = payload.get("adegan") or {}
    hyp = (payload.get("interpretasi_historis") or {}).get("hipotesis_naratif") or []

    return {
        "role": role,
        "cell_id": cell["cell_id"],
        "model": cell["model"],
        "condition": cell["condition"],
        "panel_id": cell["panel_id"],
        "site": cell.get("site"),
        "rep": cell.get("rep"),
        "status": rec.get("status"),
        "n_annotated": row["n_gt"],
        "n_listed_entries": len(preds),
        "n_listed_with_counts": row["n_pred"],
        "list_ratio": round(row["n_pred"] / row["n_gt"], 3) if row["n_gt"] else None,
        "found": len(g["pairs"]),
        "missed": len(g["fn"]),
        "unmatched": len(g["fp"]),
        "recall_grid": round(x["recall"], 3),
        "precision_grid": round(x["precision"], 3),
        "f1_grid": round(x["f1"], 3),
        "recall_iou30": round(row.get("recall_iou30", float("nan")), 3),
        "missed_per_class": dict(Counter(gts[i]["cls"] for i in g["fn"])),
        "found_per_class": dict(Counter(gts[gi]["cls"] for _, gi, _ in g["pairs"])),
        "mean_confidence_matched": conf(sorted(matched_p)),
        "mean_confidence_unmatched": conf(g["fp"]),
        "unmatched_objects": unmatched_objects,
        "predictions_without_bbox": g["no_bbox"],
        "observed_activity": scene.get("aktivitas_teramati"),
        "estimated_situation": scene.get("situasi_diperkirakan"),
        "top_hypothesis": (hyp[0].get("judul") if hyp and isinstance(hyp[0], dict) else None),
        "model_reported_legibility": (payload.get("meta") or {}).get("keterbacaan_keseluruhan"),
    }


# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="e.g. openai/gpt-4.1")
    ap.add_argument("--condition", help="RGB_SAJA | RGB_DEPTH | RGB_FUSION_CAVITY")
    ap.add_argument("--split", default="test", help="default test; '' for all")
    ap.add_argument("--min-gt", type=int, default=10,
                    help="minimum annotated objects; keeps 'best' from landing on a sparse panel")
    ap.add_argument("--rank-by", choices=["f1", "recall"], default="f1")
    ap.add_argument("--allow-partial", action="store_true",
                    help="include schema_failed records, scored from the parsed JSON")
    ap.add_argument("--best", help="cell_id of the best example (overrides automatic selection)")
    ap.add_argument("--worst", help="cell_id of the worst example (overrides automatic selection)")
    ap.add_argument("--gt-style", choices=["mask", "box"], default="mask",
                    help="draw annotations as segmentation-mask areas (default) or boxes")
    ap.add_argument("--top", type=int, default=5, help="number of candidates to print")
    ap.add_argument("--list", action="store_true", help="only print candidates, do not draw")
    ap.add_argument("--data-root", help="override the dataset root (otherwise from config)")
    ap.add_argument("--output-root", help="override the output root (otherwise from config)")
    ap.add_argument("--save-dir", help="where to write the figure (default: config figure_dir)")
    ap.add_argument("--name", default="fig_examples")
    args = ap.parse_args()

    cfg = load_config(overrides={
        "data_root": Path(args.data_root) if args.data_root else None,
        "output_root": Path(args.output_root) if args.output_root else None,
    })
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else cfg.figure_dir
    print(f"runs   : {cfg.runs_dir}")
    print(f"gt     : {cfg.gt_path}")
    print(f"photos : {cfg.data_root}")
    print(f"save   : {save_dir}")

    gt = load_gt(str(cfg.gt_path))
    records = load_records(cfg.runs_dir)
    candidates = score_candidates(records, gt, args)
    if len(candidates) < 2:
        raise SystemExit("fewer than two candidates; relax --min-gt or the other filters")

    ranked = print_candidates(candidates, args.rank_by, args.top)
    if args.list:
        return 0

    best = find_candidate(candidates, args.best) if args.best else ranked[0]
    worst = find_candidate(candidates, args.worst) if args.worst else ranked[-1]
    if best is worst:
        raise SystemExit("best and worst examples are the same; pick them with --best/--worst")

    # --- figure size from the photos' aspect ratios ------------------------
    plt.rcParams.update({"font.size": 7.5, "hatch.linewidth": 0.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    ratios = []
    for x in (best, worst):
        with Image.open(Path(cfg.data_root) / x["rec"]["cell"]["panel_id"] / "rgb.png") as im:
            ratios.append(im.height / im.width)
    panel_width = (TEXTWIDTH_IN - 0.15) / 2
    height = max(ratios) * panel_width + 0.62

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, height),
                             gridspec_kw={"wspace": 0.04})
    g1 = draw_panel(axes[0], best, cfg.data_root, "a", args.gt_style)
    g2 = draw_panel(axes[1], worst, cfg.data_root, "b", args.gt_style)
    add_legend(fig, args.gt_style)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.11)

    save_dir.mkdir(parents=True, exist_ok=True)
    pdf = save_dir / f"{args.name}.pdf"
    png = save_dir / f"{args.name}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    data = {
        "criteria": {"rank_by": args.rank_by, "min_annotated": args.min_gt,
                     "split": args.split, "model": args.model,
                     "condition": args.condition, "n_candidates": len(candidates),
                     "matching": "8x5 grid, class must match (evaluate.match)"},
        "best": summarise("best", best, g1),
        "worst": summarise("worst", worst, g2),
    }
    js = save_dir / f"{args.name}.json"
    js.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nfigure : {pdf}")
    print(f"         {png}")
    print(f"data   : {js}")
    for k in ("best", "worst"):
        d = data[k]
        print(f"\n[{k}] {d['cell_id']}")
        print(f"  annotated {d['n_annotated']}, listed {d['n_listed_with_counts']}, "
              f"found {d['found']}, missed {d['missed']}, unmatched {d['unmatched']}")
        print(f"  R {d['recall_grid']}  P {d['precision_grid']}  F1 {d['f1_grid']}  "
              f"list ratio {d['list_ratio']}")
        print(f"  mean confidence: matched {d['mean_confidence_matched']}, "
              f"unmatched {d['mean_confidence_unmatched']}")
    print("\nUse the contents of the .json file above when writing the caption and text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
