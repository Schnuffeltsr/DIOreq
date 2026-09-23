# -*- coding: utf-8 -*-
"""
Extract every LLM prompt template from dioreq.py into a prompts/ tree,
organised by the five pipeline stages of the paper's Section 3.

The .txt files hold the exact strings the code sends (byte-identical to the
string constants / literals in dioreq.py). All metadata -- placeholders,
call site, stage mapping -- lives in prompts/README.md so that the .txt
files stay pure prompt text.
"""
import ast
import os
import shutil
import string
import sys
from pathlib import Path

HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0, HERE)

import code_paths

# The method module moved under DIOReq/RQ1, so the prompts are extracted from
# wherever it currently lives rather than from a hard-coded absolute path.
DIOREQ_DIR = code_paths.install()

ROOT = os.path.join(HERE, "prompts")
import dioreq

SRC = str(DIOREQ_DIR / "dioreq.py")
tree = ast.parse(open(SRC, encoding="utf-8").read())

# ------------------------------------------------------------------ stages
STAGES = [
    ("stage1_semantic_dependency_graph", "3.1",
     "Typed Directional Semantic Dependency Graph"),
    ("stage2_diagnostic_propagation_dps", "3.2",
     "Diagnostic Propagation and DPS"),
    ("stage3_multi_view_nomination", "3.3",
     "Multi-View Candidate Nomination"),
    ("stage4_validation", "3.4",
     "Evidence, Coverage, and Boundary Validation"),
    ("stage5_generation_consolidation", "3.5",
     "Constrained Requirement Generation and Consolidation"),
]

# (stage folder, base filename, call-site tag)
LAYOUT = {
    "extract_elements": [
        ("stage1_semantic_dependency_graph", "element_extraction"),
    ],
    "extract_dependency_relations": [
        ("stage1_semantic_dependency_graph", "relation_extraction.intra"),
        ("stage1_semantic_dependency_graph", "relation_extraction.cross"),
    ],
    "nominate_dependency_findings": [
        ("stage3_multi_view_nomination", "dependency_nomination"),
    ],
    "nominate_isolation_findings": [
        ("stage3_multi_view_nomination", "isolation_nomination"),
    ],
    "nominate_operation_findings": [
        ("stage3_multi_view_nomination", "operation_nomination"),
    ],
    "validate_findings": [
        ("stage4_validation", "validation"),
    ],
    "generate_candidates": [
        ("stage5_generation_consolidation", "generation"),
    ],
    "consolidate_candidates": [
        ("stage5_generation_consolidation", "consolidation"),
    ],
}

ROLE = {
    "element_extraction": "Typed requirement-element extraction (one call per requirement block)",
    "relation_extraction.intra": "Pass 1 - intra-requirement directional relations",
    "relation_extraction.cross": "Pass 2 - cross-requirement directional relations",
    "dependency_nomination": "Dependency-view gap nomination",
    "isolation_nomination": "Isolation-view gap nomination",
    "operation_nomination": "Operation-view gap nomination",
    "validation": "Evidence / coverage / boundary validation",
    "generation": "Constrained requirement generation",
    "consolidation": "Cross-view consolidation, refinement, deduplication, filtering",
}

order = {folder: i for i, (folder, _, _) in enumerate(STAGES)}


# ------------------------------------------------------------------ scan
def resolve(node):
    if node is None:
        return None, "missing"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, "literal"
    if isinstance(node, ast.Name):
        val = getattr(dioreq, node.id, None)
        return (val, node.id) if isinstance(val, str) else (None, "?")
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "format":
            return resolve(f.value)
    if isinstance(node, ast.Attribute) and node.attr == "format":
        return resolve(node.value)
    try:
        val = ast.literal_eval(node)
        if isinstance(val, str):
            return val, "literal"
    except Exception:
        pass
    return None, type(node).__name__


parents = {}
for parent in ast.walk(tree):
    for child in ast.iter_child_nodes(parent):
        parents[child] = parent


def enclosing_func(node):
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
    return "<module>"


records = []
for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue
    if not (isinstance(node.func, ast.Attribute)
            and node.func.attr == "json_call"):
        continue
    kw = {k.arg: k.value for k in node.keywords}
    sys_val, sys_src = resolve(kw.get("system_prompt"))
    usr_val, usr_src = resolve(kw.get("user_prompt"))
    records.append({
        "line": node.lineno,
        "func": enclosing_func(node),
        "system": sys_val,
        "system_src": sys_src,
        "user": usr_val,
        "user_src": usr_src,
    })

records.sort(key=lambda r: r["line"])

groups = {}
for r in records:
    groups.setdefault(r["func"], []).append(r)

for func, wanted in LAYOUT.items():
    got = len(groups.get(func, []))
    assert got == len(wanted), (
        f"{func}: code has {got} json_call sites, layout expects {len(wanted)}")


def placeholders(template):
    names = []
    for _, field, _, _ in string.Formatter().parse(template):
        if field and field not in names:
            names.append(field)
    return names


# ------------------------------------------------------------------ write
shutil.rmtree(ROOT, ignore_errors=True)
for folder, _, _ in STAGES:
    os.makedirs(os.path.join(ROOT, folder), exist_ok=True)

written = []
for func, wanted in LAYOUT.items():
    for (folder, base), rec in zip(wanted, groups[func]):
        for role, text in (("system", rec["system"]), ("user", rec["user"])):
            rel = f"{folder}/{base}.{role}.txt"
            path = os.path.join(ROOT, folder, f"{base}.{role}.txt")
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            written.append({
                "rel": rel,
                "folder": folder,
                "role": role,
                "type": "system prompt" if role == "system" else "user template",
                "base": base,
                "func": func,
                "line": rec["line"],
                "src": rec["system_src"] if role == "system" else rec["user_src"],
                "chars": len(text),
                "placeholders": placeholders(text),
            })

written.sort(key=lambda w: (order[w["folder"]], w["base"], w["role"]))

# stage 2 has no prompts -- document that explicitly
with open(os.path.join(ROOT, "stage2_diagnostic_propagation_dps", "README.md"),
          "w", encoding="utf-8", newline="") as fh:
    fh.write(
        "# Stage 2 - Diagnostic Propagation and DPS (Section 3.2)\n\n"
        "This stage issues **no LLM calls**. It is a deterministic numerical\n"
        "layer implemented in `dioreq.py`, so there are no prompt templates here.\n\n"
        "Components:\n\n"
        "- `DependencyGraphs.build()` - semantic `MultiDiGraph` plus the acyclic\n"
        "  computational `DiGraph` (lowest-supported edge dropped on cycles,\n"
        "  incoming edges above `max_parents` dropped).\n"
        "- `DependencyGraphs.propagate()` - Equation (3), forward activation in\n"
        "  topological order with optional source clamping.\n"
        "- `DependencyGraphs.dps()` - Equation (4), the difference between the\n"
        "  clamped-at-1 and clamped-at-0 runs.\n"
        "- `deterministic_shortest_path()` / `maximum_geometric_path_support()`\n"
        "  / `AllPathSupportIndex` - the Topology and Extraction Support ranking\n"
        "  signals compared against DPS in RQ3.\n"
    )

# ------------------------------------------------------------------ README
lines = []
A = lines.append
A("# DIOReq prompt templates\n")
A("Every prompt that `dioreq.py` sends to a language model, extracted "
  "verbatim.\n")
A("The `.txt` files hold the **exact strings the code sends**: each one is "
  "byte-identical to the corresponding string constant or literal in "
  "`dioreq.py`. Two consequences of that fidelity are worth knowing before "
  "you copy a template:\n")
A("- Multi-line templates begin and end with the newline that comes from "
  "the source triple-quoted literal.")
A("- Single-line system prompts have no trailing newline at all.\n")
A("Placeholders written `{like_this}` are substituted by `.format(...)` at "
  "the call site. Literal JSON braces are escaped as `{{` and `}}` in the "
  "template and appear as `{` and `}` in the string that is actually sent.\n")
A("## Stage mapping\n")
A("Section 3 of the paper groups the pipeline into five stages. The "
  "framework-overview sentence in the Section 3 introduction describes the "
  "same pipeline as \"four main stages\", which splits the first stage and "
  "merges the third and fourth. This directory follows the five Section 3 "
  "subsections, which is the finer-grained and directly executable "
  "decomposition.\n")
A("| Directory | Paper section | Stage | Prompt files |")
A("|---|---|---|---|")
for folder, sec, title in STAGES:
    n = len([w for w in written if w["folder"] == folder])
    A(f"| `{folder}/` | {sec} | {title} | "
      f"{n if n else 'none &mdash; deterministic, no LLM call'} |")
A("")
A("## Files\n")
A("| File | Type | Role | Called from | Placeholders |")
A("|---|---|---|---|---|")
for w in written:
    ph = ", ".join(f"`{{{p}}}`" for p in w["placeholders"]) or "&mdash;"
    A(f"| `{w['rel']}` | {w['type']} | {ROLE[w['base']]} | "
      f"`{w['func']}()` L{w['line']} | {ph} |")
A("")
A("## Notes\n")
A("- The relation-extraction system prompt is identical for both extraction "
  "passes. It is duplicated into the two call-site files so that each pair "
  "of templates is self-contained; the two passes differ only in their user "
  "template.")
A("- `validation.user.txt` returns three independent decisions from a "
  "single call (`evidence`, `coverage`, `boundary`), each `YES`, `NO`, or "
  "`UNCERTAIN`. Automatic generation requires `YES` / `NO` / `YES`.")
A("- `consolidation.user.txt` receives the four pipeline switches "
  "(`merge`, `refine`, `deduplicate`, `filtering`) as booleans. That is how "
  "the RQ2 consolidation ablations change behaviour without editing the "
  "prompt text.")
A("- Comparison-baseline prompts are intentionally absent: the released "
  "code implements the proposed method only.")
A("")
A("## Scope: where the pipeline stops\n")
A("Stage 5 ends with consolidation. The **analyst review that follows is a "
  "human step and has no prompt template**, because the framework "
  "deliberately stops at evidence-linked proposals:\n")
A("- Section 3 introduction: outputs \"are therefore treated as "
  "evidence-linked proposals for analyst review rather than as logically "
  "necessary consequences of the source document\".")
A("- Section 3.4: findings whose coverage or responsibility is uncertain "
  "\"are retained for analyst review but are not automatically forwarded to "
  "the generator\".")
A("- Section 3.5: \"Final acceptance, rejection, or modification remains the "
  "responsibility of requirements analysts and domain stakeholders\".")
A("- Section 6 lists \"human-in-the-loop review settings\" as future work.\n")
A("The released pipeline therefore produces candidates and stops. No model "
  "call is made for acceptance, rejection, or modification, and no prompt "
  "template exists for it by design rather than by omission.")
A("")
A("Regenerate with `python build_prompts.py` from the repository root. The "
  "script re-reads `dioreq.py`, so the templates cannot drift from the "
  "implementation.")

with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8",
          newline="") as fh:
    fh.write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ verify
bad = []
for w in written:
    path = os.path.join(ROOT, w["rel"].replace("/", os.sep))
    on_disk = open(path, encoding="utf-8", newline="").read()
    rec = next(r for r in records if r["line"] == w["line"])
    expect = rec["system"] if w["role"] == "system" else rec["user"]
    if on_disk != expect:
        bad.append(w["rel"])

print(f"prompt files written : {len(written)}")
print(f"README written       : prompts/README.md")
print(f"byte-fidelity check  : "
      f"{'ALL OK' if not bad else 'MISMATCH ' + str(bad)}")
for folder, sec, _ in STAGES:
    n = len([w for w in written if w["folder"] == folder])
    print(f"  {folder:42s} {n:>2d} file(s)")
