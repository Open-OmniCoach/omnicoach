"""Load flat training YAML configs and resolve launcher defaults."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - bare bootstrap python during DRY_RUN=1
    yaml = None


def _require_yaml() -> Any:
    if yaml is None:
        raise ImportError("PyYAML is required to load training configs")
    return yaml


def _load(path: Path) -> dict[str, Any]:
    payload = _require_yaml().safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a YAML mapping")
    return payload


def resolve_train_config(path: str | Path) -> dict[str, Any]:
    """Load one self-contained flat training YAML config."""
    target = Path(path)
    payload = _load(target)
    if "_base_" in payload:
        raise ValueError(f"layered _base_ configs are no longer supported; use a flat YAML: {target}")
    return payload


def flat_train_config(path: str | Path) -> Path:
    """Validate and return the self-contained flat YAML path for Swift."""
    source = Path(path)
    resolve_train_config(source)
    return source


def _cli_override(extra: Sequence[str], key: str) -> list[str] | None:
    flag = f"--{key}"
    for index in range(len(extra) - 1, -1, -1):
        if extra[index].startswith(flag + "="):
            return [extra[index].split("=", 1)[1]]
        if extra[index] == flag:
            values: list[str] = []
            for value in extra[index + 1 :]:
                if value.startswith("--"):
                    break
                values.append(value)
            return values
    return None


def resolve_launcher_defaults(
    config_path: str | Path,
    extra_args: Sequence[str] = (),
    *,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Validate absolute model/dataset paths and compute launcher defaults.

    Returns ``(is_qwen25_trace, model_seq_len_default)``.
    """
    if yaml is None:
        if dry_run:
            return 0, 65536
        _require_yaml()

    config = resolve_train_config(config_path)
    datasets = _cli_override(extra_args, "dataset") or config.get("dataset") or []
    dataset_list = [datasets] if isinstance(datasets, str) else list(datasets)
    model_values = _cli_override(extra_args, "model") or [config.get("model")]
    paths = [*model_values, *dataset_list]
    missing = [
        value for value in paths if isinstance(value, str) and value.startswith("/") and not Path(value.split("#", 1)[0]).exists()
    ]
    if missing:
        print("Model/dataset paths not found on this machine:", *missing, sep="\n  ", file=sys.stderr)
        print("Edit a copy of the config, or pass --model/--dataset after the config path.", file=sys.stderr)
        if not dry_run:
            raise SystemExit(2)
    qwen25_trace = config.get("model_type") == "qwen2_5_omni" and bool(config.get("omni_pack_length_profile"))
    seq_len = int(config.get("max_length") or 65536) if qwen25_trace else 65536
    return int(qwen25_trace), seq_len


def find_cuda_site_packages() -> Path | None:
    """Return the first sys.path root holding cuDNN, NVRTC, and cuRAND wheels."""
    required = (
        Path("nvidia/cudnn/lib"),
        Path("nvidia/cuda_nvrtc/lib"),
        Path("nvidia/curand/lib"),
    )
    for raw_path in sys.path:
        if not raw_path:
            continue
        root = Path(raw_path)
        if all((root / relative).is_dir() for relative in required):
            return root
    return None


def _main(argv: Sequence[str]) -> None:
    if len(argv) >= 3 and argv[0] == "--launcher-defaults":
        stage_default, seq_len = resolve_launcher_defaults(
            argv[1],
            argv[3:],
            dry_run=argv[2] == "1",
        )
        print(stage_default)
        print(seq_len)
        return
    if argv == ["--cuda-site-packages"]:
        root = find_cuda_site_packages()
        if root is not None:
            print(root)
        return
    raise SystemExit("Usage: python -m omnicoach_sft.train.config --launcher-defaults <config> <dry_run> [extra...] | --cuda-site-packages")


if __name__ == "__main__":
    _main(sys.argv[1:])
