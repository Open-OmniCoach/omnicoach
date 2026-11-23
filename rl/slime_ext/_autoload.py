"""Apply Omni patches after the first Slime import in each Python process.

The installed ``.pth`` hook registers a finder; its loader applies patches
only after Slime finishes importing, avoiding partially initialized modules.
"""

import importlib.util
import sys
from pathlib import Path

_installed = False
_applied = False


def real_slime_ext():
    """Recover the source package if Ray resolved a checkout directory as a namespace."""
    mod = sys.modules.get("slime_ext")
    if mod is not None and getattr(mod, "__file__", None) is not None:
        return mod

    pkg_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "slime_ext", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["slime_ext"] = mod  # override the namespace-package stub
    spec.loader.exec_module(mod)
    return mod


def _safe_apply():
    global _applied
    if _applied:
        return
    _applied = True
    try:
        real_slime_ext().apply()
    except Exception:  # pragma: no cover - never break the host import
        import logging

        _applied = False  # allow a later retry (e.g. wrapper entrypoint)
        logging.getLogger("slime_ext").exception("slime_ext auto-apply failed")


class _ApplyAfterSlimeFinder:
    """Notice the first ``slime`` import and apply patches *after* it loads."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "slime":
            return None
        # Remove this finder before resolving Slime to avoid recursion.
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            _install()  # restore; let normal machinery raise/handle
            return None

        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            _safe_apply()

        spec.loader.exec_module = exec_module
        return spec


_finder = _ApplyAfterSlimeFinder()


def install():
    global _installed
    if _installed:
        return
    if "slime" in sys.modules:
        _safe_apply()
    else:
        _install()
    _installed = True


def _install():
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)
