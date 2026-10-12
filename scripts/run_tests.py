"""
Standalone test runner that does not need pytest.

The test files in this project use only three pytest features: the
`@pytest.fixture` decorator, the `pytest.raises` context manager, and the two
built-in fixtures `tmp_path` and `monkeypatch`. All of them are replaced here
by a stand-in module injected into `sys.modules` before the test files are
imported, so the whole suite runs on a plain Python install.

If pytest is available, `pytest tests/ -v` is still preferred because it
gives a richer report. This script is for when pytest cannot be installed.

    python scripts/run_tests.py
    python scripts/run_tests.py -q        # summary only

Works from any working directory; test files are located relative to the
repository root.
"""

import contextlib
import inspect
import io
import logging
import re
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TEST_FILES = ["tests/test_usage.py", "tests/test_pipeline.py"]


# --------------------------------------------------------------------------
# Minimal pytest stand-ins
# --------------------------------------------------------------------------

class MonkeyPatch:
    """Stand-in for the `monkeypatch` fixture: only setattr and undo."""

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value, raising=True):
        old = getattr(target, name)
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def undo(self):
        for target, name, old in reversed(self._undo):
            setattr(target, name, old)
        self._undo.clear()


def install_fake_pytest() -> None:
    """Insert a stand-in `pytest` module into sys.modules."""
    m = types.ModuleType("pytest")

    def fixture(fn=None, **_kw):
        def mark(f):
            f.__is_fixture__ = True
            return f
        return mark(fn) if fn is not None else mark

    @contextlib.contextmanager
    def raises(exc_type, match=None):
        try:
            yield
        except exc_type as exc:
            if match and not re.search(match, str(exc)):
                raise AssertionError(
                    f"exception raised but its message does not match "
                    f"{match!r}: {exc}"
                ) from None
        else:
            raise AssertionError(f"{exc_type.__name__} was not raised")

    m.fixture = fixture
    m.raises = raises
    m.skip = lambda *a, **k: None
    sys.modules["pytest"] = m


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def load_module(path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def run_one(module, name: str) -> str:
    """
    Resolve the requested fixtures, run the test, then clean up.

    Test output is captured and only returned when the test fails, mimicking
    pytest. Without this, progress bars and runner logs flood the terminal
    and the report becomes unreadable.
    """
    fn = getattr(module, name)
    mp = MonkeyPatch()
    temp_dirs: list[Path] = []
    cache: dict[str, object] = {}

    def resolve(param: str):
        if param in cache:
            return cache[param]
        if param == "tmp_path":
            d = Path(tempfile.mkdtemp())
            temp_dirs.append(d)
            value = d
        elif param == "monkeypatch":
            value = mp
        else:
            f = getattr(module, param, None)
            if f is None or not getattr(f, "__is_fixture__", False):
                raise RuntimeError(f"unknown fixture '{param}'")
            value = f(*[resolve(q) for q in inspect.signature(f).parameters])
        cache[param] = value
        return value

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            fn(*[resolve(p) for p in inspect.signature(fn).parameters])
    finally:
        mp.undo()
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
    return captured.getvalue()


def main() -> int:
    quiet = "-q" in sys.argv
    install_fake_pytest()
    logging.disable(logging.CRITICAL)   # silence runner logs during the tests

    passed = failed = 0
    print(f"running tests from {ROOT}")
    print("(pytest not required; fixtures replaced by a stand-in module)\n")

    for rel in TEST_FILES:
        path = ROOT / rel
        if not path.exists():
            print(f"  SKIP    {rel} (missing)")
            continue
        module = load_module(path)
        test_names = sorted(n for n in dir(module)
                            if n.startswith("test_") and callable(getattr(module, n)))
        print(f"{rel}  ({len(test_names)} tests)")
        print("-" * 68)
        for name in test_names:
            try:
                run_one(module, name)
            except Exception:
                failed += 1
                print(f"  FAIL    {name}")
                if not quiet:
                    traceback.print_exc(limit=3)
            else:
                passed += 1
                print(f"  PASS    {name}")
        print()

    print("=" * 68)
    print(f"SUMMARY: {passed} passed, {failed} failed, {passed + failed} tests in total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
