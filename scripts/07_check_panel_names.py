"""
Step 7 — check the naming of panel folders. Do NOT rename them right away.

Naming is currently mixed: `Boro_13` uses an underscore, `Brahma 3` a space.

WARNING, and this is why the script only checks:

`make_split` picks dev panels via `rng.sample(sorted(members), k)`. The
`sorted()` there is plain string order, not natural order. Space (0x20) and
underscore (0x5F) sort differently, and "Brahma 3" vs "Brahma_03" also differ
in length, so the order changes. A changed order makes `rng.sample` pick
different panels even with the same seed.

That means renaming the folders now would CHANGE the dev/test split, and every
existing dev run would no longer be comparable. On top of that, gt_layer1.json,
split.json and every record.json still store the old names, so renamed panels
would fail to match during evaluation and silently drop out of the tables.

The benefit is purely cosmetic: `panel_group()` already normalises spaces and
underscores, so the pipeline is correct with mixed names.

ADVICE: leave it as is until `main` is finished. If you do want to tidy it up
later, do it as a separate project: rename the folders, rebuild gt and split
from scratch, and accept that the old runs must be discarded.

    python scripts/07_check_panel_names.py

The dataset location comes from relief.config.load_config() (RELIEF_DATA_ROOT,
.env, configs/default.yaml or $RELIEF_CONFIG).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relief.config import load_config

STANDARD_PATTERN = re.compile(r"^[A-Za-z]+_\d{2}$")   # e.g. Boro_13, Brahma_03


def main() -> int:
    cfg = load_config()
    root = Path(cfg.data_root)
    print(f"dataset: {root}\n")
    if not root.exists():
        print(f"dataset folder does not exist: {root}")
        return 1

    folders = sorted(p.name for p in root.iterdir() if p.is_dir())
    standard = [f for f in folders if STANDARD_PATTERN.match(f)]
    nonstandard = [f for f in folders if not STANDARD_PATTERN.match(f)]

    print(f"total panel folders    : {len(folders)}")
    print(f"match pattern Name_NN  : {len(standard)}")
    print(f"do not match           : {len(nonstandard)}")

    if nonstandard:
        print("\nnon-matching names, with the proposed standard name:")
        for name in nonstandard[:30]:
            m = re.match(r"^([A-Za-z]+)[\s_]*(\d+)$", name)
            proposal = f"{m.group(1)}_{int(m.group(2)):02d}" if m else "(unrecognised pattern)"
            print(f"  {name:<20} -> {proposal}")
        if len(nonstandard) > 30:
            print(f"  ... and {len(nonstandard) - 30} more")

        print("\nDo NOT rename now. See the explanation at the top of this file:")
        print("the dev/test split would change and existing runs would no longer")
        print("be comparable. Wait until `main` is finished.")
    else:
        print("\nnaming is already consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
