"""Static AST and architectural isolation invariant tests."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLIME_EXT = ROOT / "slime_ext"
EXAMPLES = ROOT / "examples"
TOOLS = ROOT / "tools"
TESTS = ROOT / "tests"
SLIME_ROOT = ROOT / "third_party" / "slime"
SLIME_PATCH = ROOT / "tools" / "env" / "patches" / "slime-omni.patch"


def _collect_imports(directory: Path) -> dict[str, set[str]]:
    """Return {relative_py_path: set_of_imported_module_strings} via AST."""
    results: dict[str, set[str]] = {}
    for py_file in sorted(directory.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        results[str(py_file.relative_to(ROOT))] = imports
    return results


def _has_prefix(imports: set[str], forbidden_prefixes: tuple[str, ...]) -> list[str]:
    hits = []
    for mod in sorted(imports):
        for prefix in forbidden_prefixes:
            if mod == prefix or mod.startswith(prefix + "."):
                hits.append(mod)
    return hits


def _resolve_local_module(modname: str, base_root: Path) -> Path | None:
    p = base_root.joinpath(*modname.split("."))
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py")
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    return None


def _collect_block_names(stmts: list[ast.stmt], names: set[str]) -> None:
    for node in stmts:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Try):
            _collect_block_names(node.body, names)
            for handler in node.handlers:
                _collect_block_names(handler.body, names)
            _collect_block_names(node.orelse, names)
            _collect_block_names(node.finalbody, names)
        elif isinstance(node, ast.If):
            _collect_block_names(node.body, names)
            _collect_block_names(node.orelse, names)


def _module_exported_names(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    names: set[str] = set()
    _collect_block_names(tree.body, names)
    return names


class TestExampleIsolation(unittest.TestCase):
    def test_slime_ext_zero_examples_coupling(self) -> None:
        files = sorted(p for p in SLIME_EXT.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        self.assertTrue(files, "expected files under slime_ext/")
        examples_hits = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            rel = str(f.relative_to(ROOT))
            for idx, line in enumerate(text.splitlines(), 1):
                if "examples" in line.lower().split():
                    examples_hits.append(f"{rel}:{idx}: {line.strip()}")
        self.assertEqual(
            examples_hits, [], "examples references found inside slime_ext/:\n" + "\n".join(examples_hits)
        )

        imports_by_file = _collect_imports(SLIME_EXT)
        for rel, mods in imports_by_file.items():
            hits = _has_prefix(mods, ("examples",))
            self.assertEqual(hits, [], f"{rel} illegally imports {hits}")

    def test_slime_omni_patch_exists(self) -> None:
        self.assertTrue(SLIME_PATCH.is_file(), f"missing patch {SLIME_PATCH}")

    def test_qwen3_omni_rl_zero_agent_or_cross_example_imports(self) -> None:
        imports_by_file = _collect_imports(EXAMPLES / "qwen3_omni_rl")
        self.assertTrue(imports_by_file, "expected python files in examples/qwen3_omni_rl")
        forbidden = (
            "slime_ext.agent",
            "examples.qwen3_omni_agent_gspo",
        )
        for rel, mods in imports_by_file.items():
            hits = _has_prefix(mods, forbidden)
            self.assertEqual(hits, [], f"{rel} illegally imports {hits}")

    def test_qwen3_omni_agent_gspo_zero_cross_example_imports(self) -> None:
        imports_by_file = _collect_imports(EXAMPLES / "qwen3_omni_agent_gspo")
        self.assertTrue(imports_by_file, "expected python files in examples/qwen3_omni_agent_gspo")
        for rel, mods in imports_by_file.items():
            hits = _has_prefix(mods, ("examples.qwen3_omni_rl",))
            self.assertEqual(hits, [], f"{rel} illegally imports {hits}")

    def test_examples_common_zero_cross_example_imports(self) -> None:
        common_dir = EXAMPLES / "_common"
        files = sorted(p for p in common_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        self.assertTrue(files, "expected files under examples/_common/")
        imports_by_file = _collect_imports(common_dir)
        forbidden = (
            "examples.qwen3_omni_rl",
            "examples.qwen3_omni_agent_gspo",
        )
        for rel, mods in imports_by_file.items():
            hits = _has_prefix(mods, forbidden)
            self.assertEqual(hits, [], f"{rel} illegally imports {hits}")

    def test_zero_broken_local_or_slime_imports_across_repo(self) -> None:
        errors: list[str] = []
        roots = (SLIME_EXT, EXAMPLES, TOOLS, TESTS)
        for scan_dir in roots:
            for py_file in sorted(scan_dir.rglob("*.py")):
                if "__pycache__" in py_file.parts:
                    continue
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
                rel = str(py_file.relative_to(ROOT))

                def _check_mod(modname: str, lineno: int, _rel: str = rel) -> Path | None:
                    if modname == "slime" or modname.startswith("slime."):
                        target = _resolve_local_module(modname, SLIME_ROOT)
                        if target is None:
                            errors.append(f"{_rel}:{lineno}: unresolved slime module {modname!r}")
                        return target
                    if modname in {"slime_ext", "examples"} or modname.startswith(("slime_ext.", "examples.")):
                        target = _resolve_local_module(modname, ROOT)
                        if target is None:
                            errors.append(f"{_rel}:{lineno}: unresolved local module {modname!r}")
                        return target
                    return None

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            _check_mod(alias.name, node.lineno)
                    elif isinstance(node, ast.ImportFrom):
                        if node.level > 0:
                            pkg_parts = list(py_file.relative_to(ROOT).parts[:-1])
                            if node.level > len(pkg_parts):
                                errors.append(f"{rel}:{node.lineno}: relative import escapes repo root")
                                continue
                            base_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                            if node.module:
                                base_parts.extend(node.module.split("."))
                            full_mod = ".".join(base_parts)
                            target = _resolve_local_module(full_mod, ROOT)
                            if target is None:
                                errors.append(f"{rel}:{node.lineno}: unresolved relative module {full_mod!r}")
                                continue
                            exported = _module_exported_names(target)
                            raw_src = target.read_text(encoding="utf-8")
                            for alias in node.names:
                                if alias.name == "*":
                                    continue
                                sub = _resolve_local_module(f"{full_mod}.{alias.name}", ROOT)
                                if sub is None and alias.name not in exported and "__getattr__" not in raw_src:
                                    errors.append(
                                        f"{rel}:{node.lineno}: missing symbol {alias.name!r} in {full_mod!r}"
                                    )
                        elif node.module:
                            target = _check_mod(node.module, node.lineno)
                            if target is not None:
                                base = (
                                    SLIME_ROOT
                                    if (node.module == "slime" or node.module.startswith("slime."))
                                    else ROOT
                                )
                                exported = _module_exported_names(target)
                                raw_src = target.read_text(encoding="utf-8")
                                for alias in node.names:
                                    if alias.name == "*":
                                        continue
                                    sub = _resolve_local_module(f"{node.module}.{alias.name}", base)
                                    if sub is None and alias.name not in exported and alias.name not in raw_src:
                                        errors.append(
                                            f"{rel}:{node.lineno}: missing symbol {alias.name!r} in {node.module!r}"
                                        )
        self.assertEqual(errors, [], "Broken local/slime imports detected:\n" + "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
