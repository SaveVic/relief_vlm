"""
relief.fetch_data
=================
Fetch the dataset from Google Drive to local disk.

Three methods, most reliable first:

1. `--zip-url`  (recommended)
   Compress the dataset folder on Drive into one zip, share it with
   "anyone with the link", and download it once. One large file is much
   faster and more robust than thousands of small ones, and avoids the
   50-file limit of `gdown --folder`.

2. `--folder-url`
   Download the folder as-is via `gdown --folder`. Convenient, but Drive
   limits this to 50 files per folder. A folder-per-panel dataset with four
   images per panel exceeds that after 12 panels, so this only suits small
   subsets.

3. `--rclone-remote`
   For large datasets or Drive folders that are not publicly shared.
   Requires rclone configured once (`rclone config`); after that the sync
   can be repeated and copies only what changed.

After downloading, the layout is verified: every panel folder must contain
the mask and the images the conditions to be run need. Better to find out
now than when the 40th API call fails.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .imaging import IMAGE_FILES

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _require(module: str, pip_name: Optional[str] = None):
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError(
            f"package '{pip_name or module}' is required. Install it with: "
            f"pip install {pip_name or module}"
        ) from exc


def drive_file_id(url: str) -> Optional[str]:
    """Extract the file or folder id from the various Google Drive URL forms."""
    patterns = (
        r"/file/d/([A-Za-z0-9_-]{10,})",
        r"/folders/([A-Za-z0-9_-]{10,})",
        r"[?&]id=([A-Za-z0-9_-]{10,})",
    )
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", url.strip()):
        return url.strip()
    return None


# --------------------------------------------------------------------------
# Download methods
# --------------------------------------------------------------------------

def fetch_zip(url: str, dest: Union[str, Path], keep_archive: bool = False) -> Path:
    """Download one zip from Drive and extract it into `dest`."""
    gdown = _require("gdown")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / "dataset_download.zip"

    file_id = drive_file_id(url)
    if file_id is None:
        raise ValueError(f"could not recognise a Drive id in URL: {url}")

    logger.info("downloading zip from Drive (id=%s)", file_id)
    gdown.download(id=file_id, output=str(archive), quiet=False)

    if not archive.exists() or archive.stat().st_size == 0:
        raise RuntimeError(
            "empty download. Check that the file is shared with 'anyone with "
            "the link'; Drive returns an HTML page for private files."
        )

    logger.info("extracting %s (%.1f MB)", archive.name, archive.stat().st_size / 1e6)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)

    if not keep_archive:
        archive.unlink()

    return _flatten_single_root(dest)


def fetch_folder(url: str, dest: Union[str, Path]) -> Path:
    """
    Download a Drive folder as-is.

    Drive limits this to 50 files per folder. For a larger dataset some
    panels go missing without any error, so always check the result with
    `verify_dataset` afterwards.
    """
    gdown = _require("gdown")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    folder_id = drive_file_id(url)
    if folder_id is None:
        raise ValueError(f"could not recognise a folder id in URL: {url}")

    logger.warning(
        "Google Drive limits gdown --folder to 50 files per folder. "
        "For the full dataset, use --zip-url."
    )
    gdown.download_folder(id=folder_id, output=str(dest), quiet=False,
                          use_cookies=False)
    return _flatten_single_root(dest)


def fetch_rclone(remote: str, dest: Union[str, Path],
                 extra_args: Sequence[str] = ()) -> Path:
    """
    Sync a Drive folder via rclone.

    remote : 'remotename:path/to/folder', as set up in `rclone config`.
    """
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone not found. Install it from https://rclone.org/downloads/ "
            "and run `rclone config` once to connect Drive."
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["rclone", "copy", remote, str(dest), "--progress", *extra_args]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return _flatten_single_root(dest)


def _flatten_single_root(dest: Path) -> Path:
    """
    Drive zips often wrap their contents in one extra folder.

    If `dest` holds a single subfolder and nothing else, its contents are
    moved up one level so panel paths stay
    `<data_root>/<panel_id>/rgb.png` as the rest of the code expects.
    """
    entries = [p for p in dest.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        if any(p.is_dir() for p in inner.iterdir()):
            logger.info("moving contents of '%s' up one level", inner.name)
            for item in list(inner.iterdir()):
                shutil.move(str(item), str(dest / item.name))
            inner.rmdir()
    return dest


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def verify_dataset(
    root: Union[str, Path],
    mask_name: str = "seg2.png",
    required_images: Sequence[str] = ("RGB", "DEPTH", "FUSION", "CAVITY"),
) -> dict:
    """
    Check that every panel folder contains the required files.

    Run before the first API call: a panel missing `fusion.png` fails the
    whole RGB_FUSION_CAVITY condition for that panel, and that is cheaper
    to find out now.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset folder does not exist: {root}")

    panels: List[str] = []
    missing: Dict[str, List[str]] = {}
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        needed = [mask_name] + [IMAGE_FILES[k] for k in required_images]
        absent = [n for n in needed if not (entry / n).exists()]
        if absent == needed:
            continue  # not a panel folder at all
        panels.append(entry.name)
        if absent:
            missing[entry.name] = absent

    report = {
        "root": str(root),
        "n_panels": len(panels),
        "n_complete": len(panels) - len(missing),
        "missing": missing,
        "panels": panels,
    }
    return report


def verify_text(report: dict) -> str:
    lines = [
        f"dataset : {report['root']}",
        f"panels  : {report['n_panels']} folders, {report['n_complete']} complete",
    ]
    if report["missing"]:
        lines.append(f"missing : {len(report['missing'])} panels")
        for panel, files in list(report["missing"].items())[:10]:
            lines.append(f"  {panel}: no {', '.join(files)}")
        if len(report["missing"]) > 10:
            lines.append(f"  ... and {len(report['missing']) - 10} more panels")
    else:
        lines.append("missing : none")
    return "\n".join(lines)
