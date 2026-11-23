"""Apply Omni extensions before dispatching to the pinned Slime entrypoint."""

import runpy
from pathlib import Path


def run(name: str) -> None:
    from slime_ext._autoload import real_slime_ext

    real_slime_ext().apply()
    import slime

    entrypoint = Path(slime.__file__).resolve().parents[1] / f"{name}.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"upstream entrypoint not found: {entrypoint}")
    runpy.run_path(str(entrypoint), run_name="__main__")
