"""Guard against Modal test images missing a dependency the test suite imports at collection time."""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# import name -> pip distribution name, for every third-party module imported at module level
# by tests/, loramux/ or bench/ (imports inside functions or importorskip are optional and not listed).
DIST = {"torch": "torch", "numpy": "numpy", "safetensors": "safetensors", "pytest": "pytest",
        "pandas": "pandas", "matplotlib": "matplotlib", "scipy": "scipy", "diffusers": "diffusers",
        "transformers": "transformers", "accelerate": "accelerate", "peft": "peft"}
GPU_ONLY = {"triton"}


def _image_deps() -> set[str]:
    tree = ast.parse((ROOT / "modal_app.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_PY_DEPS" for t in node.targets):
            return {ast.literal_eval(e).split("==")[0].split(">=")[0] for e in node.value.elts}
    raise AssertionError("_PY_DEPS not found in modal_app.py")


def _top_level_imports(path: pathlib.Path) -> set[str]:
    mods = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def test_modal_images_cover_collection_imports():
    deps = _image_deps()
    local = {"loramux", "bench", "tests", "tiny_models", "conftest", "vllm_shim"}
    stdlib = set(sys.stdlib_module_names)
    missing = {}
    for d in ("tests", "loramux", "bench"):
        for f in (ROOT / d).rglob("*.py"):
            for m in _top_level_imports(f) - local - stdlib - GPU_ONLY:
                dist = DIST.get(m, m)
                if dist not in deps:
                    missing.setdefault(dist, []).append(str(f.relative_to(ROOT)))
    assert not missing, f"Modal images miss {sorted(missing)} imported at module level by {missing}"
