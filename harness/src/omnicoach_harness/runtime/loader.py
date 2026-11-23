"""`load_entrypoint` and `build_route`, which resolve a `module:function` string."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from typing import Any

_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")


def load_entrypoint(value: str) -> Callable[..., Any]:
    if _ENTRYPOINT_RE.fullmatch(value) is None:
        raise ValueError("entrypoint must use a valid module:function path")
    module_name, function_name = value.split(":", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, function_name, None)
    if not callable(target):
        raise TypeError(f"entrypoint is not callable: {value}")
    return target


def build_route(value: str) -> Any:
    route = load_entrypoint(value)()
    if not isinstance(getattr(route, "route_id", None), str):
        raise TypeError("route must expose a string route_id")
    return route
