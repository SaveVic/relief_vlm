"""
relief.manifest
===============
Enumerates every API call a plan needs before any of them is run.

Each call is one `Cell`: one panel, one input condition, one model, one
replicate, one resolution. The manifest is written to disk first, so the
experiment is reproducible and a dead session can be resumed by skipping
cells that already have a `record.json`.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .prompts import ABLATION_CONDITIONS, MAIN_CONDITIONS

SITE_OF_PREFIX = {
    "Boro": "Borobudur",
    "Brahma": "Prambanan",
    "Shiva": "Prambanan",
    "Vishnu": "Prambanan",
}


# ---------------------------------------------------------------------
# Panels and split
# ---------------------------------------------------------------------

def panel_group(panel_id: str) -> str:
    """Boro / Brahma / Shiva / Vishnu, taken from the panel name."""
    return panel_id.replace("_", " ").split()[0]


def panel_site(panel_id: str) -> str:
    return SITE_OF_PREFIX.get(panel_group(panel_id), "Unknown")


def list_panels(root: Union[str, Path], mask_name: str = "seg2.png") -> List[str]:
    """Panel ids are subfolder names, in natural sort order."""
    out = []
    for entry in os.scandir(str(root)):
        if entry.is_dir() and os.path.exists(os.path.join(entry.path, mask_name)):
            out.append(entry.name)

    def key(p):
        g = panel_group(p)
        tail = p[len(g):].strip(" _")
        num = "".join(ch for ch in tail if ch.isdigit())
        return (g, int(num) if num else 0, tail)

    return sorted(out, key=key)


def make_split(
    root: Union[str, Path],
    n_dev: int = 10,
    seed: int = 20260913,
    mask_name: str = "seg2.png",
    out_json: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Stratified development / test split, fixed by the seed.

    Development panels are used to tune prompts, the repair budget, and any
    thresholds; every reported number comes from the test panels. Freeze this
    split before the first API call and record the seed in the paper,
    otherwise the split cannot be claimed to be independent of the results.
    """
    panels = list_panels(root, mask_name)
    if not panels:
        raise FileNotFoundError(
            f"no panel folders containing {mask_name} under {root}"
        )

    by_group: Dict[str, List[str]] = {}
    for p in panels:
        by_group.setdefault(panel_group(p), []).append(p)

    rng = random.Random(seed)
    total = len(panels)
    dev: List[str] = []
    for g, members in sorted(by_group.items()):
        k = max(1, round(n_dev * len(members) / total))
        dev += rng.sample(sorted(members), min(k, len(members)))

    # Trim or pad to exactly n_dev, deterministically.
    dev = sorted(dev)
    while len(dev) > n_dev:
        dev.pop(rng.randrange(len(dev)))
    pool = [p for p in panels if p not in dev]
    while len(dev) < n_dev and pool:
        dev.append(pool.pop(rng.randrange(len(pool))))
    dev = sorted(dev)

    split = {
        "seed": seed,
        "n_panels": total,
        "dev": dev,
        "test": [p for p in panels if p not in dev],
        "by_group": {g: len(v) for g, v in sorted(by_group.items())},
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


# ---------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------

PLANS: Dict[str, dict] = {
    # main results
    "main": {
        "conditions": MAIN_CONDITIONS,
        "split": "test",
        "reps": 2,  # as run for the paper
        "max_side": [1536],
        "note": "main comparison on the test panels",
    },
    # nondeterminism at temperature 0, and prompt tuning
    "dev": {
        "conditions": MAIN_CONDITIONS,
        "split": "dev",
        "reps": 3,
        "max_side": [1536],
        "note": "tuning and measuring variation across replicates",
    },
    # does the effect survive controlling for the number of images?
    "abl_imagecount": {
        "conditions": ("RGB_X3",),
        "split": "test",
        "reps": 1,
        "max_side": [1536],
        "note": "image-count control",
    },
    # which derived image carries the effect?
    "abl_components": {
        "conditions": ("RGB_FUSION", "RGB_CAVITY"),
        "split": "test",
        "reps": 1,
        "max_side": [1536],
        "note": "contribution of each derived image",
    },
    # is the bottleneck the encoder's input resolution?
    "abl_resolution": {
        "conditions": ("RGB_SAJA", "RGB_FUSION_CAVITY"),
        "split": "test",
        "reps": 1,
        "max_side": [768, 3072],
        "note": "sensitivity to resolution",
    },
}


@dataclass
class Cell:
    """One planned API call."""

    cell_id: str
    panel_id: str
    site: str
    group: str
    split: str
    condition: str
    model: str
    rep: int
    max_side: int
    plan: str
    extra: dict = field(default_factory=dict)


def safe_name(s: str) -> str:
    return s.replace("/", "-").replace(" ", "-").replace(".", "")


def build_manifest(
    split: dict,
    models: Sequence[str],
    plan: str = "main",
    panels: Optional[Sequence[str]] = None,
    out_json: Optional[Union[str, Path]] = None,
) -> List[Cell]:
    """Enumerate every call the plan needs, in a fixed order."""
    if plan not in PLANS:
        raise KeyError(f"unknown plan {plan!r}; choices: {sorted(PLANS)}")
    cfg = PLANS[plan]
    chosen = list(panels) if panels is not None else list(split[cfg["split"]])

    cells: List[Cell] = []
    for model in models:
        for ms in cfg["max_side"]:
            for cond in cfg["conditions"]:
                for panel in chosen:
                    for rep in range(1, cfg["reps"] + 1):
                        cells.append(Cell(
                            cell_id=(f"{safe_name(model)}__{cond}__ms{ms}"
                                     f"__{safe_name(panel)}__r{rep}"),
                            panel_id=panel,
                            site=panel_site(panel),
                            group=panel_group(panel),
                            split=cfg["split"],
                            condition=cond,
                            model=model,
                            rep=rep,
                            max_side=ms,
                            plan=plan,
                        ))

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(
            json.dumps([asdict(c) for c in cells], indent=2), encoding="utf-8"
        )
    return cells


def load_manifest(path: Union[str, Path]) -> List[Cell]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Cell(**c) for c in data]
