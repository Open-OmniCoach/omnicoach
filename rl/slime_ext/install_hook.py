from __future__ import annotations

import argparse
import sysconfig
from pathlib import Path

PTH_NAME = "slime_ext_autoload.pth"


def _pth_path() -> Path:
    return Path(sysconfig.get_path("purelib")) / PTH_NAME


def install() -> Path:
    pth = _pth_path()
    pth.write_text("import slime_ext._autoload; slime_ext._autoload.install()\n", encoding="utf-8")
    return pth


def uninstall() -> bool:
    pth = _pth_path()
    if pth.exists():
        pth.unlink()
        return True
    return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Install or remove the slime_ext startup autoload hook.")
    parser.add_argument("--uninstall", action="store_true", help="Remove slime_ext_autoload.pth if installed.")
    args = parser.parse_args(argv)
    if args.uninstall:
        removed = uninstall()
        status = "removed" if removed else "not present"
        print(f"slime_ext: autoload hook {status} ({_pth_path()})")
        return
    pth = install()
    print(f"slime_ext: wrote auto-apply hook to {pth}")


if __name__ == "__main__":
    main()
