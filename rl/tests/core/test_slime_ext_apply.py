import importlib
import sys
from types import ModuleType

import slime_ext


def test_apply_does_not_require_megatron_bridge(monkeypatch, tmp_path):
    bridge_modules = {name for name in sys.modules if name.startswith("megatron.bridge")}
    slime = ModuleType("slime")
    slime.__file__ = str(tmp_path / "slime" / "__init__.py")
    monkeypatch.setitem(sys.modules, "slime", slime)

    patch_modules = (
        "slime_ext.backends.megatron.arguments",
        "slime_ext.backends.megatron.data",
        "slime_ext.backends.megatron.loss",
        "slime_ext.backends.megatron.update_weight",
        "slime_ext.models.qwen3_omni.weights.hf_to_megatron",
        "slime_ext.models.qwen3_omni.weights.megatron_to_hf",
        "slime_ext.backends.megatron.actor",
        "slime_ext.backends.megatron.rolling_checkpoint",
    )
    applied = []
    for name in patch_modules:
        module = ModuleType(name)
        module.patch = lambda name=name: applied.append(name)
        monkeypatch.setitem(sys.modules, name, module)
        parent_name, attribute = name.rsplit(".", 1)
        monkeypatch.setattr(importlib.import_module(parent_name), attribute, module, raising=False)

    monkeypatch.setattr(slime_ext, "_applied", False)
    slime_ext.apply()

    assert applied == list(patch_modules)
    assert {name for name in sys.modules if name.startswith("megatron.bridge")} == bridge_modules


def test_qwen3_omni_model_package_does_not_import_megatron_bridge():
    before = {name for name in sys.modules if name.startswith("megatron.bridge")}

    importlib.import_module("slime_ext.models.qwen3_omni")

    assert {name for name in sys.modules if name.startswith("megatron.bridge")} == before
