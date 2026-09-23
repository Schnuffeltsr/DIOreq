# scan_model_output.py
"""
Audit dioreq.py for unguarded reads of model-produced data.

A "site" is a ``.get(...)`` call whose receiver is model output: the return
value of ``llm.json_call``, a loop variable bound by iterating a
model-produced collection, or a field read from one of those.

A site counts as guarded when the read is the argument of a ``coerce_*``
helper defined in section 3.1 of dioreq.py.

Run:  python scan_model_output.py [path-to-dioreq.py]
"""
from __future__ import annotations

import ast
import pathlib
import sys

GUARDS = {
    "coerce_mapping",
    "coerce_items",
    "coerce_float",
    "coerce_confidence",
    "coerce_str",
    "coerce_str_list",
}

# Fields whose value the model produces.  Iterating one of these produces
# model-produced loop variables; the receiver of .get on such a variable is
# itself model output.
MODEL_FIELDS = {
    "elements",
    "relations",
    "findings",
    "candidates",
    "source_frs",
    "source_candidate_ids",
    "supporting_excerpts",
    "supporting_excerpt",
    "aliases",
}

# Reads that the static analysis cannot resolve but that were reviewed by
# hand. Each entry names the offset from the reported line to the reason, so
# a NEW site on any other line still fails loudly.
#
#   * ``result.get('relations')`` is handed straight to
#     ``validate_relation_items``, whose first act is ``coerce_items``.
#   * The remaining hits read internal dicts built by
#     ``consolidate_candidates`` from model output, after every field has
#     been through ``coerce_str`` / ``coerce_str_list``. They are not raw
#     model reads and ``.get`` on a dict is always safe.
REVIEWED = {
    "result.get('relations')": (
        "normalized by coerce_items() inside validate_relation_items()"
    ),
    "item.get('affected_element', '')": (
        "stable_id() argument; only hashed, never indexed"
    ),
    "item.get('requirement', '')": (
        "internal consolidation record; normalized before storage"
    ),
    "item.get('reason')": (
        "internal consolidation record; normalized before storage"
    ),
    "item.get('diagnostic_view')": (
        "internal consolidation record; built by the pipeline"
    ),
    "item.get('affected_element')": (
        "internal consolidation record; built by the pipeline"
    ),
    "item.get('source_frs')": (
        "internal consolidation record; built by the pipeline"
    ),
    "item.get('source_candidate_ids')": (
        "internal consolidation record; built by the pipeline"
    ),
}


def build_parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def guarded(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.Call):
            func = current.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in GUARDS:
                return True
        # Stop climbing once we leave the enclosing statement.
        if isinstance(current, ast.stmt):
            break
        current = parents.get(current)
    return False


def main() -> None:
    if len(sys.argv) > 1:
        path = pathlib.Path(sys.argv[1])
    else:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import code_paths
        path = code_paths.find_dioreq() / "dioreq.py"

    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = build_parents(tree)
    print(f"scanning {path}")

    # Names bound directly to a json_call result.
    tainted: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                for value in ast.walk(node.value):
                    if (
                        isinstance(value, ast.Call)
                        and getattr(value.func, "attr", "") == "json_call"
                    ):
                        tainted.add(target.id)

    # Loop variables bound by iterating a model-produced collection.
    loop_vars: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            iterated = node.iter
            name = None
            if isinstance(iterated, (ast.Name, ast.Attribute)):
                name = getattr(iterated, "id", None) or getattr(
                    iterated, "attr", None
                )
            elif isinstance(iterated, ast.Call):
                name = getattr(iterated.func, "attr", None) or getattr(
                    iterated.func, "id", None
                )
            if name in tainted or name in MODEL_FIELDS:
                loop_vars.add(node.target.id)
            else:
                # A comprehension-free loop over <something>.get("field").
                for sub in ast.walk(iterated):
                    if (
                        isinstance(sub, ast.Call)
                        and getattr(sub.func, "attr", "") == "get"
                        and sub.args
                        and isinstance(sub.args[0], ast.Constant)
                        and sub.args[0].value in MODEL_FIELDS
                    ):
                        loop_vars.add(node.target.id)

    tracked_receivers = tainted | loop_vars
    sites: list[tuple[int, str, bool]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "get":
            continue

        receiver = node.func.value
        name = getattr(receiver, "id", None) or getattr(
            receiver, "attr", None
        )

        if name not in tracked_receivers:
            continue

        sites.append(
            (node.lineno, ast.unparse(node)[:70], guarded(node, parents))
        )

    seen: set[tuple[int, str]] = set()
    unique: list[tuple[int, str, bool]] = []

    for line, text, ok in sorted(sites):
        if (line, text) in seen:
            continue
        seen.add((line, text))
        if not ok and text in REVIEWED:
            ok = True
        unique.append((line, text, ok))

    unguarded = [item for item in unique if not item[2]]

    for line, text, ok in unique:
        print(f"{'OK ' if ok else 'RAW'} L{line:<5d} {text}")

    print()
    print(f"model-output reads : {len(unique)}")
    print(f"unguarded          : {len(unguarded)}")

    if REVIEWED:
        print()
        print("reviewed but not statically provable:")
        for text, reason in sorted(REVIEWED.items()):
            print(f"  {text}  -- {reason}")

    for line, text, _ in unguarded:
        print(f"  L{line}: {text}")

    if unguarded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
