"""
relief.logging_setup
====================
Logging configuration for local runs.

`force=True` is needed on Colab, where the runtime already installs a handler
on the root logger. Elsewhere it is kept so that repeated calls (e.g. from
tests) do not stack handlers.

Logs are also written to a file: runs lasting hours outgrow the terminal
scrollback, and the cause of a failed call is usually investigated only after
the run has finished.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    quiet_libraries: bool = True,
) -> Path | None:
    """Install a console handler and, if `log_dir` is given, a file handler."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    path: Path | None = None

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )

    if quiet_libraries:
        # httpx logs every request at INFO, including every prediction
        # status poll. On long runs that drowns out everything else.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    return path
