"""Run Slime's torch-dist-to-HF converter with ``slime_ext`` applied.

The conversion implementation remains owned by upstream Slime at
``third_party/slime/tools/convert_torch_dist_to_hf.py``.  This script registers
Omni's Megatron-to-HF mappings, resolves the tool from the installed/editable
Slime checkout, and delegates its CLI unchanged.

Invoke it with the same arguments as the upstream tool::

    python tools/checkpoints/convert_torch_dist_to_hf.py --help
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _upstream_tool() -> Path:
    import slime

    tool = Path(slime.__file__).resolve().parents[1] / "tools" / "convert_torch_dist_to_hf.py"
    if not tool.is_file():
        raise FileNotFoundError(f"Upstream Slime conversion tool does not exist: {tool}")
    return tool


def main() -> None:
    from slime_ext._autoload import real_slime_ext

    real_slime_ext().apply()
    runpy.run_path(str(_upstream_tool()), run_name="__main__")


if __name__ == "__main__":
    main()
