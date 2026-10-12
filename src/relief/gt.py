"""
relief.gt
=========
Extract Layer-1 ground truth from colour-coded segmentation masks of
relief panels.

Encoding
--------
Each object instance is painted with one palette colour. The colour
family encodes the class; the shade within that family separates
instances of the same class. Background is black.

This version assumes the masks have been cleaned: colours match the
palette exactly and there is no anti-aliasing. `colour_audit` is kept so
that assumption can be re-checked, and `on_unknown="error"` makes a
violation loud instead of silent.

Occluded objects
----------------
One object can be painted as several disjoint areas, for example an axe
split in two by the hand holding it. Components of the same colour whose
gap is at most `merge_gap` pixels are therefore merged into a single
instance. Components of the same colour further apart than that stay
separate, which catches the case where a shade was reused for two
different objects because the class ran out of colours.

Speed
-----
Colour lookup uses a 24-bit table, so mapping a 4608x3456 mask is a
single fancy-index. Geometry is computed from foreground coordinates
rather than full-size boolean masks, and connected components run inside
each colour's bounding box only. `extract_dataset` parallelises panels.

Entry points
------------
    extract_panel(mask_path, ...)   -> dict, one panel
    extract_dataset(mask_dir, ...)  -> dict, a folder
    colour_audit(mask_path)         -> check the clean-mask assumption
    palette_report()                -> palette separability
    shade_usage_report(dataset)     -> does shade index encode a subtype
    summarise(dataset)              -> text summary for the paper

Requires numpy, pillow, opencv-python.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("relief_gt requires opencv-python") from exc

Image.MAX_IMAGE_PIXELS = None
RGB = Tuple[int, int, int]

# ---------------------------------------------------------------------
# Palette (Layer 1, update 12-Sept-2021)
# ---------------------------------------------------------------------

BACKGROUND: RGB = (0, 0, 0)

PALETTE: Dict[str, List[RGB]] = {
    "human": [
        (242, 111, 112), (239, 67, 68), (222, 38, 38),
        (187, 32, 38), (154, 29, 31), (128, 28, 31),
        (246, 163, 176), (242, 112, 134), (239, 62, 95),
        (227, 30, 73), (190, 31, 63), (159, 28, 60),
        (249, 207, 226), (244, 169, 203), (238, 115, 172),
        (235, 72, 153), (221, 36, 121), (192, 31, 96),
    ],
    "building": [
        (204, 180, 214), (166, 135, 190), (133, 96, 169),
        (122, 84, 163), (113, 73, 157), (87, 45, 133),
    ],
    "animal": [
        (129, 210, 246), (72, 188, 236), (41, 163, 220),
        (25, 132, 199), (19, 106, 163), (13, 74, 112),
        (123, 204, 197), (12, 184, 167), (25, 149, 138),
        (16, 118, 110), (19, 94, 90), (21, 78, 74),
    ],
    "vegetation": [
        (191, 218, 101), (167, 206, 60), (134, 196, 64),
        (99, 165, 68), (77, 126, 58), (57, 84, 39),
    ],
    "artefak": [
        (250, 231, 137), (250, 211, 76), (249, 191, 31),
        (246, 158, 28), (216, 121, 40), (181, 85, 39),
        (254, 216, 171), (252, 186, 116), (247, 145, 63),
        (244, 115, 33), (234, 89, 36), (196, 67, 39),
    ],
    "decoration": [
        (187, 226, 221), (110, 201, 194), (19, 179, 161),
        (17, 134, 119), (17, 79, 72), (6, 51, 46),
    ],
}

CLASSES: List[str] = list(PALETTE)


def _build_lookup() -> Dict[RGB, Tuple[str, int]]:
    lut: Dict[RGB, Tuple[str, int]] = {}
    for cls, colours in PALETTE.items():
        for i, c in enumerate(colours):
            if c in lut:
                raise ValueError(f"Duplicate palette colour {c}")
            lut[c] = (cls, i)
    return lut


COLOUR_LUT: Dict[RGB, Tuple[str, int]] = _build_lookup()
PALETTE_KEYS: List[RGB] = list(COLOUR_LUT.keys())
PALETTE_ARRAY = np.array(PALETTE_KEYS, dtype=np.int16)

BG_CODE = -1
UNKNOWN_CODE = -2


# ---------------------------------------------------------------------
# Fast colour -> palette index mapping
# ---------------------------------------------------------------------

def _pack(rgb: np.ndarray) -> np.ndarray:
    """Pack an (H, W, 3) uint8 image into (H, W) uint32 keys."""
    a = rgb.astype(np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


_LUT_CACHE: Dict[float, np.ndarray] = {}


def _get_lut(bg_radius: float) -> np.ndarray:
    """
    Build (once) a 2^24 table mapping a packed RGB key to a palette index,
    BG_CODE for background, or UNKNOWN_CODE.

    Costs 32 MB and about a second on first call, then every mask is a
    single fancy-index. Background is expanded over the small cube around
    black rather than the whole space, so building stays cheap.
    """
    key = float(bg_radius)
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]

    lut = np.full(1 << 24, UNKNOWN_CODE, dtype=np.int16)

    r = int(np.floor(bg_radius))
    if r >= 0:
        g = np.arange(r + 1)
        bb, gg, rr = np.meshgrid(g, g, g, indexing="ij")
        d2 = rr.ravel() ** 2 + gg.ravel() ** 2 + bb.ravel() ** 2
        keep = d2 <= bg_radius ** 2
        keys = (
            (rr.ravel()[keep].astype(np.uint32) << 16)
            | (gg.ravel()[keep].astype(np.uint32) << 8)
            | bb.ravel()[keep].astype(np.uint32)
        )
        lut[keys] = BG_CODE

    for i, (cr, cg, cb) in enumerate(PALETTE_KEYS):
        lut[(cr << 16) | (cg << 8) | cb] = i

    _LUT_CACHE[key] = lut
    return lut


def _index_image(rgb: np.ndarray, bg_radius: float) -> np.ndarray:
    return _get_lut(bg_radius)[_pack(rgb)]


# ---------------------------------------------------------------------
# Palette diagnostics
# ---------------------------------------------------------------------

def palette_report() -> dict:
    """
    Separability of the palette.

    `max_safe_tolerance` is the largest colour-matching tolerance that
    cannot cross a class boundary; `max_safe_bg_radius` the largest
    background radius that cannot swallow the darkest label.
    """
    arr = PALETTE_ARRAY.astype(np.float64)
    d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    norms = np.linalg.norm(arr, axis=1)
    k = int(np.argmin(norms))
    return {
        "n_colours": len(PALETTE_KEYS),
        "capacity_per_class": {c: len(v) for c, v in PALETTE.items()},
        "min_pair_distance": float(d[i, j]),
        "min_pair": (
            {"rgb": PALETTE_KEYS[i], "label": COLOUR_LUT[PALETTE_KEYS[i]]},
            {"rgb": PALETTE_KEYS[j], "label": COLOUR_LUT[PALETTE_KEYS[j]]},
        ),
        "max_safe_tolerance": float(d[i, j] / 2.0),
        "darkest_palette_colour": {
            "rgb": PALETTE_KEYS[k],
            "distance_from_black": float(norms[k]),
        },
        "max_safe_bg_radius": float(norms[k] / 2.0),
    }


def colour_audit(mask_path: str, bg_radius: float = 25.0) -> dict:
    """
    Verify that a mask really is palette-exact.

    Reports every distinct colour, whether it is background, an exact
    palette hit, or unknown. With cleaned masks `n_unknown` should be 0.
    A non-zero count means the cleaning step missed this file.
    """
    rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    packed = _pack(rgb)
    uniq, counts = np.unique(packed, return_counts=True)
    lut = _get_lut(bg_radius)
    codes = lut[uniq]

    rows = []
    for u, c, code in zip(uniq, counts, codes):
        col = (int(u >> 16) & 255, int(u >> 8) & 255, int(u) & 255)
        rows.append(
            {
                "rgb": col,
                "pixels": int(c),
                "pixel_frac": float(c / packed.size),
                "kind": (
                    "background" if code == BG_CODE
                    else "unknown" if code == UNKNOWN_CODE
                    else "exact"
                ),
                "label": (
                    None if code < 0 else COLOUR_LUT[PALETTE_KEYS[int(code)]]
                ),
            }
        )
    n_unknown = sum(1 for r in rows if r["kind"] == "unknown")
    return {
        "path": mask_path,
        "n_distinct_colours": len(rows),
        "n_unknown": n_unknown,
        "unknown_pixel_frac": float(
            sum(r["pixel_frac"] for r in rows if r["kind"] == "unknown")
        ),
        "clean": n_unknown == 0,
        "colours": sorted(rows, key=lambda r: -r["pixels"]),
    }



# ---------------------------------------------------------------------
# Panel discovery
# ---------------------------------------------------------------------

def find_panels(
    root: str,
    mask_name: str = "seg2.png",
    recursive: bool = True,
) -> List[Tuple[str, str]]:
    """
    Locate panels laid out as one subfolder per panel.

        root/
          Boro_1/    seg2.png  [photo.jpg, cavity.png, ...]
          Brahma 10/ seg2.png
          ...

    The panel id is the subfolder name, so ids keep whatever the folders
    are called, spaces included. Returns (panel_id, mask_path) sorted by
    id. With `recursive`, nested layouts such as root/site/panel/ are
    also found and the panel id is the immediate parent folder.
    """
    found: List[Tuple[str, str]] = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            if mask_name in filenames:
                pid = os.path.basename(dirpath.rstrip(os.sep))
                found.append((pid, os.path.join(dirpath, mask_name)))
    else:
        for entry in os.scandir(root):
            if entry.is_dir():
                cand = os.path.join(entry.path, mask_name)
                if os.path.exists(cand):
                    found.append((entry.name, cand))

    if not found:
        raise FileNotFoundError(
            f"No '{mask_name}' found under {root}. Check mask_name and the "
            f"folder layout."
        )
    dupes = [k for k, v in Counter(p for p, _ in found).items() if v > 1]
    if dupes:
        raise ValueError(f"Duplicate panel ids: {dupes[:10]}")
    return sorted(found)


def sibling_file(mask_path: str, name: Optional[str]) -> Optional[str]:
    """Path to a file sitting next to the mask, or None if absent."""
    if not name:
        return None
    cand = os.path.join(os.path.dirname(mask_path), name)
    return cand if os.path.exists(cand) else None


# ---------------------------------------------------------------------
# Instance extraction
# ---------------------------------------------------------------------

@dataclass
class Instance:
    instance_id: str
    cls: str
    shade_index: int
    rgb: RGB
    area_px: int
    area_frac: float
    bbox_px: Tuple[int, int, int, int]
    bbox_norm: Tuple[float, float, float, float]
    centroid_norm: Tuple[float, float]
    grid_cells: List[str]
    grid_cell_main: str
    touches_border: bool
    n_parts: int
    part_areas: List[int]
    salience: Optional[float] = None
    extras: dict = field(default_factory=dict)


def _grid_label(col: int, row: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def _grid_from_coords(
    xs: np.ndarray, ys: np.ndarray, w: int, h: int,
    cols: int, rows: int, min_frac: float,
) -> Tuple[List[str], str]:
    ci = np.minimum((xs.astype(np.int64) * cols) // w, cols - 1)
    ri = np.minimum((ys.astype(np.int64) * rows) // h, rows - 1)
    counts = np.bincount(ri * cols + ci, minlength=cols * rows)
    order = np.argsort(counts)[::-1]
    total = xs.size
    cells = [
        _grid_label(int(k % cols), int(k // cols))
        for k in order
        if counts[k] / total >= min_frac
    ]
    main = _grid_label(int(order[0] % cols), int(order[0] // cols))
    return cells, main


def _bridge_evidence(
    comp_i: np.ndarray,
    comp_j: np.ndarray,
    local_index: np.ndarray,
    pal_i: int,
    gap: float,
    rel_gap: float,
    a_i: int,
    a_j: int,
) -> dict:
    """
    Corridor-occupancy evidence for one pair, given a gap already computed.

    Distance alone cannot tell an occluded object from two objects that
    happen to share a shade, because both can be near or far. What does
    separate them is what sits in the space between: an occluder fills
    the corridor between the two parts of one object, while two separate
    objects have uncarved background between them.

    Only the dilate-and-intersect step happens here. The gap is passed in
    so a caller can skip this entirely when the gap alone already rules
    out a merge.
    """
    r = max(1, int(np.ceil(gap / 2)) + 2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    bridge = (
        cv2.dilate(comp_i.astype(np.uint8), k).astype(bool)
        & cv2.dilate(comp_j.astype(np.uint8), k).astype(bool)
        & ~comp_i
        & ~comp_j
    )
    n = int(bridge.sum())
    occ = 0.0
    if n:
        vals = local_index[bridge]
        occ = float(np.count_nonzero((vals >= 0) & (vals != pal_i)) / n)
    return {
        "gap_px": gap, "rel_gap": rel_gap, "bridge_px": n,
        "bridge_occupancy": occ, "area_i": a_i, "area_j": a_j,
    }


def _merge_parts(
    local: np.ndarray,
    local_index: np.ndarray,
    pal_i: int,
    gap_px: int,
    rule: str = "occlusion",
    occ_threshold: float = 0.5,
    rel_gap_max: float = 1.0,
    show_progress: bool = False,
) -> Tuple[np.ndarray, int, List[dict]]:
    """
    Group same-colour components into objects.

    rule="gap"
        Merge whenever the pixel gap is at most `gap_px`. Kept only as an
        ablation: on this dataset the gap distribution has no valley, so
        the threshold is arbitrary.

    rule="occlusion"
        Merge when the corridor between two components is occupied by
        other instances (`bridge_occupancy` at least `occ_threshold`) and
        the gap is not large relative to the smaller component
        (`rel_gap` at most `rel_gap_max`). This asks why the parts are
        separate rather than only how far apart they are.

    Returns the label image, the number of groups, and the per-pair
    evidence so the decisions can be audited.

    Performance
    -----------
    Pair comparison is O(n_comp^2), so a colour that fragments into many
    small blobs could stall a panel. Two things keep that in check: each
    component's distance transform is computed once and reused across
    every pair it appears in, and the dilate step is skipped whenever the
    gap alone already exceeds `rel_gap_max`, since the merge is False
    regardless of occupancy.
    """
    n, lab = cv2.connectedComponents(local.astype(np.uint8), connectivity=8)
    n_comp = n - 1
    if n_comp <= 1:
        return lab, n_comp, []

    comps = [lab == i for i in range(1, n)]
    areas = [int(c.sum()) for c in comps]
    dts = [
        cv2.distanceTransform(1 - c.astype(np.uint8), cv2.DIST_L2, 5)
        for c in comps
    ]

    parent = list(range(n_comp))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pairs = [(i, j) for i in range(n_comp) for j in range(i + 1, n_comp)]
    evidence = []
    it = tqdm(
        pairs, desc="pair evidence", unit="pair", leave=False,
        disable=not show_progress or len(pairs) < 200,
    )
    for i, j in it:
        gap = float(dts[i][comps[j]].min())
        diam = float(np.sqrt(4.0 * min(areas[i], areas[j]) / np.pi))
        rel_gap = gap / max(diam, 1.0)

        if rule == "gap":
            e = _bridge_evidence(comps[i], comps[j], local_index, pal_i,
                                 gap, rel_gap, areas[i], areas[j])
            merge = gap <= gap_px
        elif rel_gap > rel_gap_max:
            # Cannot merge whatever the occupancy is: skip the dilate.
            e = {"gap_px": gap, "rel_gap": rel_gap, "bridge_px": 0,
                 "bridge_occupancy": 0.0, "area_i": areas[i], "area_j": areas[j]}
            merge = False
        else:
            e = _bridge_evidence(comps[i], comps[j], local_index, pal_i,
                                 gap, rel_gap, areas[i], areas[j])
            merge = e["bridge_occupancy"] >= occ_threshold

        e["merged"] = bool(merge)
        evidence.append(e)
        if merge:
            union(i, j)

    remap = {}
    out = np.zeros_like(lab)
    for i in range(n_comp):
        root = find(i)
        if root not in remap:
            remap[root] = len(remap) + 1
        out[comps[i]] = remap[root]
    return out, len(remap), evidence


def extract_panel(
    mask_path: str,
    panel_id: Optional[str] = None,
    min_area_px: int = 64,
    grid_cols: int = 8,
    grid_rows: int = 5,
    grid_min_frac: float = 0.10,
    merge_gap_px: Optional[int] = None,
    merge_gap_frac: float = 0.02,
    merge_rule: str = "occlusion",
    occ_threshold: float = 0.5,
    rel_gap_max: float = 1.0,
    bg_radius: float = 25.0,
    salience_path: Optional[str] = None,
    on_unknown: str = "error",
    show_progress: bool = False,
) -> dict:
    """
    Extract Layer-1 ground truth from one segmentation mask.

    Parameters
    ----------
    min_area_px : instances below this size are dropped and counted.
        Set it from the annotation guideline and report it in the paper.
    merge_gap_px : same-colour components separated by at most this many
        pixels are treated as one object, which is what recovers an axe
        split in two by the hand holding it. When None, it is taken as
        `merge_gap_frac` of the shorter image side, so the rule means the
        same thing at both capture resolutions. Set 0 to disable merging
        and treat every component as its own instance.
    on_unknown : what to do when a colour is neither background nor a
        palette entry: "error", "warn", or "ignore". Cleaned masks should
        never trigger it.

    Returns
    -------
    dict with panel_id, size, counts, instances, quality.
    """
    rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    h, w = rgb.shape[:2]
    index = _index_image(rgb, bg_radius)

    n_unknown = int(np.count_nonzero(index == UNKNOWN_CODE))
    if n_unknown:
        msg = (
            f"{mask_path}: {n_unknown} pixels ({n_unknown / index.size:.4%}) "
            f"match no palette colour. Run colour_audit for details."
        )
        if on_unknown == "error":
            raise ValueError(msg)
        if on_unknown == "warn":
            import warnings

            warnings.warn(msg, stacklevel=2)

    gap = (
        int(round(merge_gap_frac * min(h, w)))
        if merge_gap_px is None
        else int(merge_gap_px)
    )

    salience_img = panel_mean = panel_std = None
    if salience_path is not None:
        s = np.asarray(Image.open(salience_path).convert("L"), dtype=np.float32)
        if s.shape != (h, w):
            s = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
        salience_img = s
        panel_mean = float(s.mean())
        panel_std = float(s.std()) or 1.0

    # Foreground coordinates once, then grouped by palette index.
    ys, xs = np.nonzero(index >= 0)
    instances: List[Instance] = []
    pair_evidence: List[dict] = []
    dropped_small = 0
    merged_objects = 0

    if ys.size:
        labels = index[ys, xs]
        order = np.argsort(labels, kind="stable")
        ys, xs, labels = ys[order], xs[order], labels[order]
        bounds = np.searchsorted(labels, np.unique(labels))
        bounds = list(bounds) + [labels.size]

        spans = list(zip(bounds[:-1], bounds[1:]))
        for b0, b1 in tqdm(
            spans, desc=f"{os.path.basename(mask_path)}: colours",
            unit="colour", leave=False, disable=not show_progress,
        ):
            pal_i = int(labels[b0])
            cls, shade = COLOUR_LUT[PALETTE_KEYS[pal_i]]
            cy, cx = ys[b0:b1], xs[b0:b1]

            # Work inside this colour's bounding box only.
            y0, y1 = int(cy.min()), int(cy.max()) + 1
            x0, x1 = int(cx.min()), int(cx.max()) + 1
            local = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            local[cy - y0, cx - x0] = True

            local_index = index[y0:y1, x0:x1]
            lab, n_groups, ev = _merge_parts(
                local, local_index, pal_i, gap,
                rule=merge_rule, occ_threshold=occ_threshold,
                rel_gap_max=rel_gap_max, show_progress=show_progress,
            )
            for e in ev:
                e.update({"cls": cls, "shade": shade})
            pair_evidence.extend(ev)

            for g in range(1, n_groups + 1):
                gy, gx = np.nonzero(lab == g)
                area = gy.size
                if area == 0:
                    continue
                if area < min_area_px:
                    dropped_small += 1
                    continue
                gy = gy + y0
                gx = gx + x0

                parts = cv2.connectedComponents(
                    (lab == g).astype(np.uint8), connectivity=8
                )
                n_parts = int(parts[0]) - 1
                if n_parts > 1:
                    merged_objects += 1
                part_areas = sorted(
                    (
                        int(np.count_nonzero(parts[1] == p))
                        for p in range(1, n_parts + 1)
                    ),
                    reverse=True,
                )

                gx0, gx1 = int(gx.min()), int(gx.max()) + 1
                gy0, gy1 = int(gy.min()), int(gy.max()) + 1
                cells, main = _grid_from_coords(
                    gx, gy, w, h, grid_cols, grid_rows, grid_min_frac
                )
                sal = None
                if salience_img is not None:
                    sal = float(
                        np.abs(salience_img[gy, gx] - panel_mean).mean()
                        / panel_std
                    )

                suffix = f"_{g}" if n_groups > 1 else ""
                instances.append(
                    Instance(
                        instance_id=f"{cls}_{shade:02d}{suffix}",
                        cls=cls,
                        shade_index=shade,
                        rgb=PALETTE_KEYS[pal_i],
                        area_px=area,
                        area_frac=area / (h * w),
                        bbox_px=(gx0, gy0, gx1, gy1),
                        bbox_norm=(gx0 / w, gy0 / h, gx1 / w, gy1 / h),
                        centroid_norm=(
                            float(gx.mean()) / w, float(gy.mean()) / h
                        ),
                        grid_cells=cells,
                        grid_cell_main=main,
                        touches_border=bool(
                            gx0 == 0 or gy0 == 0 or gx1 == w or gy1 == h
                        ),
                        n_parts=n_parts,
                        part_areas=part_areas,
                        salience=sal,
                    )
                )

    counts = Counter(i.cls for i in instances)
    split_colours = [
        i.instance_id for i in instances if i.instance_id.count("_") > 1
    ]

    return {
        "panel_id": panel_id or os.path.splitext(os.path.basename(mask_path))[0],
        "mask_path": mask_path,
        "size": {"width": w, "height": h},
        "grid": {"cols": grid_cols, "rows": grid_rows},
        "counts": {
            **{c: int(counts.get(c, 0)) for c in CLASSES},
            "total": len(instances),
        },
        "instances": [asdict(i) for i in instances],
        "quality": {
            "unknown_pixels": n_unknown,
            "merge_gap_px": gap,
            "merge_rule": merge_rule,
            "pair_evidence": pair_evidence,
            "objects_recovered_from_parts": merged_objects,
            "colours_split_into_several_objects": split_colours,
            "dropped_small": dropped_small,
            "palette_capacity_hit": [
                c for c in CLASSES if counts.get(c, 0) >= len(PALETTE[c])
            ],
        },
    }


# ---------------------------------------------------------------------
# Dataset level
# ---------------------------------------------------------------------

def _worker(args):
    path, panel_id, sal, kwargs = args
    try:
        return extract_panel(path, panel_id=panel_id, salience_path=sal, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"panel_id": panel_id, "error": repr(exc)}


def extract_dataset(
    root: str,
    out_json: Optional[str] = None,
    mask_name: str = "seg2.png",
    salience_name: Optional[str] = None,
    n_jobs: int = 1,
    show_progress: bool = True,
    **kwargs,
) -> dict:
    """
    Run `extract_panel` over a folder-per-panel dataset.

    `salience_name` names a file inside each panel folder, e.g.
    "cavity.png"; panels missing it simply get no salience score.

    Progress is a single bar over panels, with the current panel shown as
    a suffix. The per-colour and per-pair bars inside `extract_panel` are
    switched off here on purpose: nested bars interleave badly in a
    notebook and the inner loops are short, so they produce noise rather
    than information. Call `extract_panel(..., show_progress=True)`
    directly when one slow panel needs to be diagnosed.

    `n_jobs` > 1 processes panels in parallel, but the 32 MB lookup table
    is rebuilt per worker, so it only pays off for large sets.
    """
    panels_found = find_panels(root, mask_name)
    kwargs.setdefault("show_progress", False)
    jobs = [
        (path, pid, sibling_file(path, salience_name), kwargs)
        for pid, path in panels_found
    ]

    results = []
    bar = tqdm(
        total=len(jobs), desc="extracting panels", unit="panel",
        disable=not show_progress,
    )
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for res in ex.map(_worker, jobs):
                results.append(res)
                bar.set_postfix_str(str(res.get("panel_id", ""))[:24])
                bar.update(1)
    else:
        for job in jobs:
            bar.set_postfix_str(str(job[1])[:24])
            res = _worker(job)
            results.append(res)
            n = res.get("counts", {}).get("total")
            bar.set_postfix_str(f"{str(job[1])[:20]} n={n}" if n is not None else str(job[1])[:24])
            bar.update(1)
    bar.close()

    panels = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    totals: Counter = Counter()
    for p in panels:
        for k, v in p["counts"].items():
            totals[k] += v

    result = {
        "root": root,
        "mask_name": mask_name,
        "n_panels": len(panels),
        "class_totals": dict(totals),
        "panels": panels,
        "errors": errors,
        "palette_report": palette_report(),
    }
    if out_json:
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
    return result


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def shade_usage_report(dataset: dict, cls: str = "human") -> dict:
    """
    Test whether shade index within a class carries meaning.

    Sequential use (0,1,2,3 ...) means shade is just an instance counter.
    Gaps, especially a cluster in one row of the palette, suggest a
    sub-range was reserved for a sub-type, for example the pink row of
    `human` for mythological figures.
    """
    n_shades = len(PALETTE[cls])
    row_size = 6 if n_shades % 6 == 0 else n_shades
    per_panel, gap_panels = [], 0
    row_counts: Counter = Counter()

    for p in dataset["panels"]:
        shades = sorted(
            {i["shade_index"] for i in p["instances"] if i["cls"] == cls}
        )
        if not shades:
            continue
        contiguous = shades == list(range(len(shades)))
        if not contiguous:
            gap_panels += 1
        for s in shades:
            row_counts[s // row_size] += 1
        per_panel.append(
            {
                "panel_id": p["panel_id"],
                "n": len(shades),
                "shades": shades,
                "contiguous_from_zero": contiguous,
            }
        )

    n = len(per_panel) or 1
    return {
        "class": cls,
        "n_panels_with_class": len(per_panel),
        "gap_panel_fraction": gap_panels / n,
        "instances_per_row": dict(sorted(row_counts.items())),
        "row_size": row_size,
        "interpretation": (
            "shade index looks like a plain instance counter"
            if gap_panels / n < 0.15
            else "shade index may encode a sub-type; inspect the flagged panels"
        ),
        "panels_with_gaps": [
            r for r in per_panel if not r["contiguous_from_zero"]
        ][:20],
    }


def summarise(dataset: dict) -> str:
    """Text summary; the class totals feed the dataset table in the paper."""
    lines = [f"panels: {dataset['n_panels']}"]
    tot = dataset["class_totals"]
    for k in CLASSES + ["total"]:
        if k in tot:
            lines.append(f"  {k:<12} {tot[k]:>6}")

    per_panel = [p["counts"]["total"] for p in dataset["panels"]]
    if per_panel:
        arr = np.array(per_panel)
        lines.append(
            f"instances per panel: mean {arr.mean():.1f}, "
            f"median {np.median(arr):.0f}, min {arr.min()}, max {arr.max()}"
        )

    merged = sum(
        p["quality"]["objects_recovered_from_parts"] for p in dataset["panels"]
    )
    lines.append(f"objects recovered from separated parts: {merged}")

    split = [
        p["panel_id"]
        for p in dataset["panels"]
        if p["quality"]["colours_split_into_several_objects"]
    ]
    if split:
        lines.append(f"panels where one colour became several objects: {split}")

    cap = [p["panel_id"] for p in dataset["panels"] if p["quality"]["palette_capacity_hit"]]
    if cap:
        lines.append(f"panels at palette capacity: {cap}")

    if dataset["errors"]:
        lines.append(f"errors: {[e['panel_id'] for e in dataset['errors']]}")
    return "\n".join(lines)


if __name__ == "__main__":
    import pprint

    pprint.pprint(palette_report())


# ---------------------------------------------------------------------
# Dataset-level audit and diagnostics
# ---------------------------------------------------------------------

def audit_dataset(
    root: str, mask_name: str = "seg2.png", bg_radius: float = 25.0
) -> dict:
    """
    Run `colour_audit` over every panel, with one progress bar.

    Returns only what matters: which panels are not palette-exact. Run
    this before extraction; there is no point extracting instances from
    masks the cleaning step missed.
    """
    panels = find_panels(root, mask_name)
    dirty = []
    for pid, path in tqdm(panels, desc="auditing masks", unit="panel"):
        a = colour_audit(path, bg_radius=bg_radius)
        if not a["clean"]:
            dirty.append({"panel_id": pid, **a})
    return {"n_panels": len(panels), "n_dirty": len(dirty),
            "clean": not dirty, "dirty": dirty}


def export_instance_crops(
    panel: dict,
    out_dir: str,
    photo_path: Optional[str] = None,
    instance_ids: Optional[Sequence[str]] = None,
    pad_frac: float = 0.15,
    min_side: int = 256,
) -> List[str]:
    """
    Save cropped images of chosen instances for visual audit.

    Use it to settle questions that only an eye can settle: whether the
    pink-row instances really are ordinary human figures, or whether a
    colour that split into two objects is in fact one occluded object.
    Crops come from `photo_path` when given, otherwise from the mask.
    """
    os.makedirs(out_dir, exist_ok=True)
    src_path = photo_path or panel["mask_path"]
    img = np.asarray(Image.open(src_path).convert("RGB"))
    H, W = img.shape[:2]
    w, h = panel["size"]["width"], panel["size"]["height"]
    sx, sy = W / w, H / h

    chosen = [
        i for i in panel["instances"]
        if instance_ids is None or i["instance_id"] in instance_ids
    ]
    written = []
    for inst in tqdm(chosen, desc="crops", unit="crop", leave=False,
                     disable=len(chosen) < 20):
        x0, y0, x1, y1 = inst["bbox_px"]
        x0, x1 = int(x0 * sx), int(x1 * sx)
        y0, y1 = int(y0 * sy), int(y1 * sy)
        pw, ph = int((x1 - x0) * pad_frac), int((y1 - y0) * pad_frac)
        x0, y0 = max(0, x0 - pw), max(0, y0 - ph)
        x1, y1 = min(W, x1 + pw), min(H, y1 + ph)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        if min(crop.shape[:2]) < min_side:
            f = min_side / min(crop.shape[:2])
            crop = cv2.resize(
                crop, (int(crop.shape[1] * f), int(crop.shape[0] * f)),
                interpolation=cv2.INTER_CUBIC,
            )
        name = f"{panel['panel_id']}__{inst['instance_id']}__parts{inst['n_parts']}.png"
        out = os.path.join(out_dir, name)
        Image.fromarray(crop).save(out)
        written.append(out)
    return written


# ---------------------------------------------------------------------
# Measuring the two gap populations
# ---------------------------------------------------------------------

def _pair_gaps(local: np.ndarray) -> List[float]:
    """Minimum pixel gap between every pair of components in a mask."""
    n, lab = cv2.connectedComponents(local.astype(np.uint8), connectivity=8)
    if n <= 2:
        return []
    gaps = []
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        # distance from every pixel to component i
        dt = cv2.distanceTransform(1 - comp, cv2.DIST_L2, 5)
        for j in range(i + 1, n):
            other = lab == j
            gaps.append(float(dt[other].min()))
    return gaps


def component_gap_report(
    root: str,
    mask_name: str = "seg2.png",
    min_area_px: int = 64,
    bg_radius: float = 25.0,
) -> dict:
    """
    Measure the distance between same-colour components across the set.

    When a shade is deliberately reused for a second object, annotators
    keep the two far apart; when one object is split by an occluder, the
    two parts are close. So the gaps should form two populations, and the
    merge threshold belongs in the valley between them.

    Returns every measured gap, both in pixels and as a fraction of the
    panel's shorter side, plus an Otsu split of the fractional gaps as a
    suggested threshold. A clean bimodal histogram means the threshold is
    well determined; a single broad mode means the two cases overlap and
    some panels will be wrong whatever value is chosen, which is a
    dataset limitation to state rather than a parameter to tune.
    """
    records = []
    for panel_id, path in tqdm(
        find_panels(root, mask_name), desc="measuring gaps", unit="panel"
    ):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        short = min(h, w)
        index = _index_image(rgb, bg_radius)
        ys, xs = np.nonzero(index >= 0)
        if ys.size == 0:
            continue
        labels = index[ys, xs]
        order = np.argsort(labels, kind="stable")
        ys, xs, labels = ys[order], xs[order], labels[order]
        bounds = list(np.searchsorted(labels, np.unique(labels))) + [labels.size]

        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            if b1 - b0 < min_area_px:
                continue
            pal_i = int(labels[b0])
            cls, shade = COLOUR_LUT[PALETTE_KEYS[pal_i]]
            cy, cx = ys[b0:b1], xs[b0:b1]
            y0, y1 = int(cy.min()), int(cy.max()) + 1
            x0, x1 = int(cx.min()), int(cx.max()) + 1
            local = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            local[cy - y0, cx - x0] = True
            for g in _pair_gaps(local):
                records.append(
                    {
                        "panel_id": panel_id,
                        "cls": cls,
                        "shade": shade,
                        "gap_px": g,
                        "gap_frac": g / short,
                    }
                )

    if not records:
        return {"n_pairs": 0, "records": []}

    fracs = np.array([r["gap_frac"] for r in records])

    # Otsu split on the fractional gaps.
    hist, edges = np.histogram(fracs, bins=64, range=(0, float(fracs.max())))
    p = hist / hist.sum()
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * centres)
    mt = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (mt * w0 - m0) ** 2 / (w0 * (1 - w0))
    k = int(np.nanargmax(between))
    threshold = float(centres[k])

    # More robust for clearly bimodal data: place the threshold in the
    # widest empty span of the sorted gaps, on a log scale.
    sf = np.sort(fracs[fracs > 0])
    valley = None
    if sf.size >= 3:
        ratios = sf[1:] / sf[:-1]
        v = int(np.argmax(ratios))
        if ratios[v] > 2.0:
            valley = float(np.sqrt(sf[v] * sf[v + 1]))

    threshold = valley if valley is not None else threshold
    below = fracs <= threshold
    return {
        "valley_threshold_frac": valley,
        "n_pairs": len(records),
        "n_panels_with_pairs": len({r["panel_id"] for r in records}),
        "gap_frac_percentiles": {
            str(q): float(np.percentile(fracs, q))
            for q in (5, 10, 25, 50, 75, 90, 95)
        },
        "otsu_threshold_frac": threshold,
        "n_below_threshold": int(below.sum()),
        "n_above_threshold": int((~below).sum()),
        "separation_ratio": float(
            np.median(fracs[~below]) / max(np.median(fracs[below]), 1e-9)
        ) if below.any() and (~below).any() else None,
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": edges.tolist(),
        },
        "records": records,
    }


def gap_histogram_text(report: dict, width: int = 46) -> str:
    """ASCII histogram of the gap distribution, for a quick visual check."""
    if not report.get("n_pairs"):
        return "no same-colour component pairs found"
    counts = np.array(report["histogram"]["counts"])
    edges = np.array(report["histogram"]["bin_edges"])
    top = counts.max() or 1
    thr = report["otsu_threshold_frac"]
    lines = [
        f"{report['n_pairs']} component pairs in "
        f"{report['n_panels_with_pairs']} panels",
        (
            f"suggested threshold: merge_gap_frac = {thr:.4f}"
            + (
                "  (valley between two modes)"
                if report.get("valley_threshold_frac") is not None
                else "  (Otsu; no clear valley found)"
            )
        ),
        f"  below (treated as one object): {report['n_below_threshold']}",
        f"  above (treated as separate)  : {report['n_above_threshold']}",
        "",
    ]
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * c / top))
        mark = " <-- threshold" if lo <= thr < hi else ""
        if c or mark:
            lines.append(f"{lo:6.3f}-{hi:6.3f} {c:5d} {bar}{mark}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Auditing the merge decisions
# ---------------------------------------------------------------------

def evidence_report(dataset: dict) -> dict:
    """
    Aggregate the per-pair evidence behind every merge decision.

    Where the raw gap distribution has no valley, corridor occupancy
    usually does: an occluded object has other instances filling the
    space between its parts, so occupancy sits near one, while two
    separate objects sharing a shade have background between them, so
    occupancy sits near zero. A bimodal histogram here means the rule is
    well determined; a flat one means the cases really are ambiguous.

    `borderline` lists the pairs closest to the threshold, which are the
    ones worth looking at by eye.
    """
    rows = []
    for p in dataset["panels"]:
        for e in p["quality"].get("pair_evidence", []):
            rows.append({**e, "panel_id": p["panel_id"]})
    if not rows:
        return {"n_pairs": 0}

    occ = np.array([r["bridge_occupancy"] for r in rows])
    hist, edges = np.histogram(occ, bins=20, range=(0.0, 1.0))
    merged = np.array([r["merged"] for r in rows])

    return {
        "n_pairs": len(rows),
        "n_merged": int(merged.sum()),
        "n_kept_separate": int((~merged).sum()),
        "occupancy_histogram": {"counts": hist.tolist(), "bin_edges": edges.tolist()},
        "frac_near_zero": float((occ < 0.10).mean()),
        "frac_near_one": float((occ > 0.90).mean()),
        "frac_ambiguous": float(((occ >= 0.30) & (occ <= 0.70)).mean()),
        "borderline": sorted(
            rows, key=lambda r: abs(r["bridge_occupancy"] - 0.5)
        )[:25],
        "rows": rows,
    }


def evidence_text(rep: dict, width: int = 46) -> str:
    """ASCII histogram of corridor occupancy."""
    if not rep.get("n_pairs"):
        return "no same-colour component pairs"
    counts = np.array(rep["occupancy_histogram"]["counts"])
    edges = np.array(rep["occupancy_histogram"]["bin_edges"])
    top = counts.max() or 1
    lines = [
        f"{rep['n_pairs']} pairs: {rep['n_merged']} merged, "
        f"{rep['n_kept_separate']} kept separate",
        f"occupancy near 0: {rep['frac_near_zero']:.1%}   "
        f"near 1: {rep['frac_near_one']:.1%}   "
        f"ambiguous (0.3-0.7): {rep['frac_ambiguous']:.1%}",
        "",
        "corridor occupancy (share of the gap filled by other instances):",
    ]
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        lines.append(f"{lo:5.2f}-{hi:5.2f} {c:5d} " + "#" * int(round(width * c / top)))
    return "\n".join(lines)
