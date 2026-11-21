"""Generic key resolution against an engine's module names."""
from __future__ import annotations

from collections.abc import Callable

KeyMap = Callable[[str], "str | list[tuple[str, slice]] | None"]


def make_keymap(engine_linear_names: list[str], aliases: dict | None = None) -> KeyMap:
    """Build keymap(module_name) -> engine key | [(engine_key, out_slice), ...] | None.

    A source module name (prefix already stripped) resolves by exact dotted match, or by its underscore form
    (`a.b_c.0.d` <-> `a_b_c_0_d`, as Kohya writes it) built from the engine names, so nothing is guessed.
    If two names share an underscore form, looking that form up raises ValueError (ambiguous).

    `aliases` adds source names that differ from the engine's (original / Comfy naming): {source_name: target}
    where target is an engine key or a split list [(engine_key, out_slice), ...] for a source module that is
    fused in the file but separate in the engine. Aliases resolve exactly or by underscore form like engine names.
    """
    table: dict[str, object] = {n: n for n in engine_linear_names}
    for src, tgt in (aliases or {}).items():
        if src in table and table[src] != tgt:
            raise ValueError(f"alias {src!r} collides with an engine module name")
        table[src] = tgt
    flat: dict[str, list[str]] = {}
    for src in table:
        flat.setdefault(src.replace(".", "_"), []).append(src)

    def keymap(module: str):
        if module in table:
            return table[module]
        cands = flat.get(module)
        if cands is None:
            return None
        if len(cands) > 1:
            raise ValueError(f"ambiguous underscore name {module!r}: matches {sorted(cands)}")
        return table[cands[0]]

    return keymap
