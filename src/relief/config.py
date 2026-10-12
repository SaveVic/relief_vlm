"""
relief.config
=============
Single place for every path and setting.

Values come from `configs/default.yaml` (or the file named by
`RELIEF_CONFIG`), can be overridden through environment variables (a `.env`
file at the project root is read automatically, see `.env.example`), and every
CLI command and script receives an explicit `Config` object.

Priority: CLI argument > environment variable > YAML file > defaults below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # --- locations -------------------------------------------------------
    data_root: Path = PROJECT_ROOT / "data" / "dataset"
    output_root: Path = PROJECT_ROOT / "outputs"
    # where scripts write paper figures; defaults to <output_root>/figures
    figures_dir: Optional[Path] = None

    # --- file names inside each panel folder ----------------------------
    mask_name: str = "seg2.png"
    salience_name: str = "cavity.png"
    photo_name: str = "rgb.png"
    min_area_px: int = 200

    # --- split -----------------------------------------------------------
    n_dev: int = 10
    seed: int = 20260913

    # --- model calls -----------------------------------------------------
    models: List[str] = field(default_factory=lambda: [
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "google/gemini-3-flash",
        "qwen/qwen3-7-plus",
    ])
    temperature: float = 0.0
    max_repairs: int = 1
    max_side: int = 1536
    jpeg_quality: int = 90
    sleep_s: float = 3.0
    timeout: int = 300
    max_retries: int = 2
    upload_mode: str = "files"
    save_predictions: bool = True
    max_tokens_total: Optional[int] = None

    # --- credentials -----------------------------------------------------
    replicate_api_token: Optional[str] = None

    # --- data source -----------------------------------------------------
    drive_zip_url: Optional[str] = None
    drive_folder_url: Optional[str] = None

    # ---------------------------------------------------------------------

    @property
    def runs_dir(self) -> Path:
        return self.output_root / "runs"

    @property
    def eval_dir(self) -> Path:
        return self.output_root / "eval"

    @property
    def gt_path(self) -> Path:
        return self.output_root / "gt_layer1.json"

    @property
    def dev_runs_dir(self) -> Path:
        """Development-plan records, moved out of `runs_dir` before the main run."""
        return self.output_root / "runs_dev"

    @property
    def figure_dir(self) -> Path:
        return Path(self.figures_dir) if self.figures_dir else self.output_root / "figures"

    @property
    def split_path(self) -> Path:
        return self.output_root / "split.json"

    def manifest_path(self, plan: str) -> Path:
        return self.output_root / f"manifest_{plan}.json"

    def ensure_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.data_root.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "replicate_api_token":
                out[f.name] = "***" if value else None
            else:
                out[f.name] = str(value) if isinstance(value, Path) else value
        return out


# Recognised environment variables, mapped to Config field names.
ENV_MAP = {
    "RELIEF_DATA_ROOT": ("data_root", Path),
    "RELIEF_OUTPUT_ROOT": ("output_root", Path),
    "RELIEF_FIGURES_DIR": ("figures_dir", Path),
    "RELIEF_SEED": ("seed", int),
    "RELIEF_TEMPERATURE": ("temperature", float),
    "RELIEF_MAX_REPAIRS": ("max_repairs", int),
    "RELIEF_SLEEP_S": ("sleep_s", float),
    "RELIEF_UPLOAD_MODE": ("upload_mode", str),
    "REPLICATE_API_TOKEN": ("replicate_api_token", str),
    "RELIEF_DRIVE_ZIP_URL": ("drive_zip_url", str),
    "RELIEF_DRIVE_FOLDER_URL": ("drive_folder_url", str),
}


def _load_dotenv(path: Path) -> None:
    """Read a simple .env file without extra dependencies."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Build a Config from YAML + environment + explicit overrides."""
    _load_dotenv(Path(os.environ.get("RELIEF_ENV_FILE", PROJECT_ROOT / ".env")))

    data: Dict[str, Any] = {}
    if path is None and os.environ.get("RELIEF_CONFIG"):
        path = Path(os.environ["RELIEF_CONFIG"]).expanduser()
    if path is None:
        default = PROJECT_ROOT / "configs" / "default.yaml"
        path = default if default.exists() else None
    if path is not None:
        if yaml is None:
            raise ImportError("PyYAML is required to read a config file")
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    known = {f.name: f for f in fields(Config)}
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"unknown config key: {key!r}")
        is_path = known[key].type in (Path, "Path", Optional[Path], "Optional[Path]")
        kwargs[key] = Path(value).expanduser() if is_path and value is not None else value

    for env_key, (name, caster) in ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw:
            kwargs[name] = caster(raw) if caster is not Path else Path(raw).expanduser()

    for key, value in (overrides or {}).items():
        if value is not None:
            kwargs[key] = value

    cfg = Config(**kwargs)
    cfg.data_root = Path(cfg.data_root).expanduser().resolve()
    cfg.output_root = Path(cfg.output_root).expanduser().resolve()
    if cfg.figures_dir is not None:
        cfg.figures_dir = Path(cfg.figures_dir).expanduser().resolve()
    if cfg.replicate_api_token:
        os.environ.setdefault("REPLICATE_API_TOKEN", cfg.replicate_api_token)
    return cfg
