# -*- coding: utf-8 -*-
"""Exercise the LLM-free logic of dioreq.py to surface real runtime defects.

Run:  py test_dioreq_logic.py   (dioreq.py must sit in the same directory)
Only pure functions and graph classes are exercised; no API calls are made.
"""
import os
import re
import sys
import traceback

import math

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import code_paths

code_paths.install()

import dioreq
from dioreq import (DependencyGraphs, RequirementElement, DependencyRelation,
                 RankingRecord, rank_records, merge_elements, flatten_unique,
                 split_into_frs, requirement_context, normalize_decision,
                 stable_id, Usage, validate_relation_items,
                 deterministic_shortest_path, maximum_geometric_path_support,
                 requirements_scope, original_fr_major, last_fr_number)

fails = []


def case(name, fn):
    try:
        out = fn()
        print(f"  [ok]   {name}" + (f"  -> {out}" if out is not None else ""))
    except Exception as e:
        fails.append((name, e))
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)


print("=" * 70)
print("A. split_into_frs on five real document dialects")
print("=" * 70)

docs = {
    "FunctionRequirement": "Function Requirement 1: Register User\nInput: a\n\nFunction Requirement 2: Login\nInput: b",
    "FR-dash": "FR-001\tSend Email\nInput: a\nFR-002\tReceive\nInput: b",
    "numbered": "1.1 Complaint Registration\nInput: a\n1.2 View Details\nInput: b",
    "markdown": "## 1.1 Alpha\nInput: a\n## 1.2 Beta\nInput: b",
    "FRcolon": "FR 1: Complaint Registration\nInput: a\nFR 2: Tracking\nInput: b",
    "Fnum": "F1 User Registration\n- Input: a\nF2 User Login\n- Input: b",
    "three-level": "Functional Requirement\n3.1 Cart\n3.1.1 Add Dish\nInput: a\n3.1.2 Remove Dish\nInput: b",
    "md-FR-heading": "## 1. Vehicles\n### FR-01 Registration\nInput: a\n### FR-02 Deletion\nInput: b",
    "compound-FR": "Functional Requirement\n2. Functional Requirements\n2.1 Ingestion\nFR-DCI-001: accept real-time input\nFR-DCI-002: validate input quality",
}
for k, v in docs.items():
    case(f"split {k}", lambda v=v: [(i, t.splitlines()[0][:28]) for i, t in split_into_frs(v)])


def deepest_level_wins():
    """
    Sections numbered '2.1' must not swallow the 'FR-001' requirements
    inside them: picking the first matching dialect instead of the deepest
    turned every section into a single 4,000-character requirement.
    """
    doc = (
        "Functional Requirement\n"
        "2. Functional Requirements\n"
        "2.1 User Account Management\n"
        "FR-001\nUsers can create an account.\n"
        "FR-002\nUsers can log in.\n"
        "FR-003\nUsers can log out.\n"
        "2.2 Product Catalog\n"
        "FR-004\nProducts are displayed.\n"
        "FR-005\nAdmins manage products.\n"
    )
    ids = [name for name, _ in split_into_frs(doc)]
    assert ids == ["FR-001", "FR-002", "FR-003", "FR-004", "FR-005"], ids
    return f"2-section document split into {len(ids)} FR blocks, not 2"


case("deepest numbering level wins", deepest_level_wins)


def three_level_beats_two_level():
    doc = (
        "项目文档\nFunctional Requirement\n"
        "3.1 User Management\n"
        "3.1.1 User Registration\nDescription: d\n"
        "3.1.2 User Login\nDescription: d\n"
        "3.2 Restaurant Browsing\n"
        "3.2.1 Browse Restaurants\nDescription: d\n"
        "3.2.2 Search Restaurants\nDescription: d\n"
    )
    ids = [name for name, _ in split_into_frs(doc)]
    assert ids == ["3.1.1", "3.1.2", "3.2.1", "3.2.2"], ids
    return f"x.y.z nesting wins over x.y -> {ids}"


case("three-level numbering wins", three_level_beats_two_level)


def scope_excludes_non_requirements():
    """
    A trailing '3. External Interfaces' section reuses x.y numbering but is
    not a requirement; it must not become four extra requirements.
    """
    doc = (
        "Functional Requirement\n"
        "2. Functional Requirements\n"
        "2.1 Asset Registration\nInput: a\n"
        "2.2 Asset Usage Tracking\nInput: b\n"
        "3. External Interfaces\n"
        "3.1 Authentication System Integration\nInterface Type: API\n"
        "3.2 Email Notification System\nInterface Type: SMTP\n"
    )
    ids = [name for name, _ in split_into_frs(doc)]
    assert ids == ["2.1", "2.2"], ids
    assert "External Interfaces" not in requirements_scope(doc)
    return f"scope stops at the next top-level section -> {ids}"


case("functional-requirements scope", scope_excludes_non_requirements)


def numbering_does_not_collide():
    """
    original_fr_major() read the first requirement's major while
    last_fr_number() read the highest (major, minor) pair anywhere in the
    document. When a later non-requirement section reused the numbering,
    the two disagreed and generated numbers collided with existing ones.
    """
    doc = (
        "Functional Requirement\n"
        "2. Functional Requirements\n"
        "2.1 Asset Registration\nInput: a\n"
        "2.2 Asset Usage Tracking\nInput: b\n"
        "2.3 Asset Transfer\nInput: c\n"
        "3. External Interfaces\n"
        "3.1 Authentication\nInterface Type: API\n"
        "3.2 Email\nInterface Type: SMTP\n"
    )
    major = original_fr_major(doc)
    assert major == "2", major
    assert last_fr_number(doc, major) == 3, last_fr_number(doc, major)
    assert last_fr_number(doc) == 3, "scoped count must ignore section 3"

    numbered = dioreq.assign_requirement_numbers(
        doc, [{"requirement": "The system shall X."}])
    assert numbered[0]["fr_number"] == "2.4", numbered[0]["fr_number"]

    existing = set(re.findall(r"(?m)^(\d+\.\d+)\s", doc))
    assert numbered[0]["fr_number"] not in existing, "number collides"
    return "major=2, last=3 -> next is 2.4, no collision with 3.x"


case("numbering stays inside its major section", numbering_does_not_collide)


def preamble_is_not_a_requirement():
    doc = ("Functional Requirement\n"
           "3.1 Complaint Registration\nInput: a\n"
           "3.2 Investigation Tracking\nInput: b\n")
    ids = [name for name, _ in split_into_frs(doc)]
    assert ids == ["3.1", "3.2"], ids
    assert all(not name.startswith("BLOCK-") for name in ids)
    return "leading title block dropped"


case("preamble block dropped", preamble_is_not_a_requirement)


def undialected_document_keeps_every_block():
    """The no-dialect fallback must not lose content."""
    prose = "just prose\n\nmore prose\n\neven more"
    ids = [name for name, _ in split_into_frs(prose)]
    assert ids == ["BLOCK-1", "BLOCK-2", "BLOCK-3"], ids
    return f"{len(ids)} blocks preserved when nothing is numbered"


case("undialected fallback", undialected_document_keeps_every_block)


print()
print("=" * 70)
print("B. DependencyGraphs: cycle removal + parent limit + DPS")
print("=" * 70)

els = [RequirementElement(name=n, element_type="FUNCTION", confidence=0.9,
                          source_frs=["FR-1"], supporting_excerpts=["x"])
       for n in ["A", "B", "C", "D"]]

cyc = [
    DependencyRelation("A", "B", "ENABLES", 0.9, ["FR-1"], ["x"]),
    DependencyRelation("B", "C", "TRIGGERS", 0.8, ["FR-1"], ["x"]),
    DependencyRelation("C", "A", "MODIFIES", 0.5, ["FR-1"], ["x"]),   # low weight -> dropped
    DependencyRelation("A", "D", "PRODUCES", 0.95, ["FR-1"], ["x"]),
]


def build_cycle():
    g = DependencyGraphs(max_parents=12)
    g.build(els, cyc)
    import networkx as nx
    return dict(is_dag=nx.is_directed_acyclic_graph(g.computational_graph),
                exclusions=[e["reason"] for e in g.exclusions],
                edges=sorted(g.computational_graph.edges()))


case("cyclic input -> DAG", build_cycle)


def dps_case():
    g = DependencyGraphs(max_parents=12)
    g.build(els, cyc)
    return dict(A_to_D=g.dps("A", "D"), self_dps=g.dps("A", "A"),
                no_path=g.dps("D", "A"))


case("DPS values", dps_case)


def prop_case():
    g = DependencyGraphs(max_parents=12)
    g.build(els, cyc)
    return {k: round(v, 3) for k, v in g.propagate(None, None).items()}


case("propagate (no clamp)", prop_case)


def parent_limit():
    many = [DependencyRelation(f"P{i}", "T", "ENABLES", 0.5 + i * 0.01, ["FR-1"], ["x"])
            for i in range(20)]
    g = DependencyGraphs(max_parents=3)
    names = ["T"] + [f"P{i}" for i in range(20)]
    e2 = [RequirementElement(name=n, element_type="FUNCTION", confidence=0.9,
                             source_frs=["FR-1"], supporting_excerpts=["x"]) for n in names]
    g.build(e2, many)
    return dict(in_degree_T=g.computational_graph.in_degree("T"),
                excluded=sum(1 for e in g.exclusions if e["reason"] == "parent_limit"))


case("max_parents limit", parent_limit)

print()
print("=" * 70)
print("C. ranking_records + all five ranking strategies")
print("=" * 70)


def rank_case():
    g = DependencyGraphs(max_parents=12)
    g.build(els, cyc)
    recs = g.ranking_records()
    out = {}
    for s in ["dps", "extraction_support", "topology", "unranked", "random"]:
        r = rank_records(recs, s, seed=1)
        out[s] = [x.record_id[:14] for x in r]
    return f"{len(recs)} records; orders differ={len({tuple(v) for v in out.values()})}"


case("rank_records", rank_case)


def bad_strategy():
    g = DependencyGraphs()
    g.build(els, cyc)
    try:
        rank_records(g.ranking_records(), "nope")
        return "NO ERROR RAISED"
    except ValueError as e:
        return f"ValueError ok: {e}"


case("unknown strategy rejected", bad_strategy)

print()
print("=" * 70)
print("D. robustness against malformed LLM fields (the risky ones)")
print("=" * 70)


def nonnumeric_confidence_relation():
    """validate_relation_items does float(item['confidence']) with no guard."""
    items = [{"source": "A", "target": "B", "relation_type": "ENABLES",
              "confidence": "high", "supporting_excerpts": ["x"]}]
    return validate_relation_items(items, els, 0.5, False)


case("relation confidence='high'", nonnumeric_confidence_relation)


def null_confidence_relation():
    items = [{"source": "A", "target": "B", "relation_type": "ENABLES",
              "confidence": None, "supporting_excerpts": ["x"]}]
    return validate_relation_items(items, els, 0.5, False)


case("relation confidence=None", null_confidence_relation)


def excerpt_none():
    items = [{"source": "A", "target": "B", "relation_type": "ENABLES",
              "confidence": 0.9, "supporting_excerpts": None}]
    return validate_relation_items(items, els, 0.5, False)


case("supporting_excerpts=None", excerpt_none)

print()
print("=" * 70)
print("E. requirement_context: empty id list")
print("=" * 70)


def ctx_empty():
    doc = "1.1 Alpha\n" + "x" * 40 + "\n\n1.2 Beta\n" + "y" * 40
    out = requirement_context(doc, [])
    return f"returned {len(out)} chars (whole doc={len(doc)} chars)"


case("requirement_context(doc, [])", ctx_empty)


def ctx_missing():
    doc = "1.1 Alpha\nbody\n\n1.2 Beta\nbody"
    out = requirement_context(doc, ["FR-999"])
    return f"returned {len(out)} chars (whole doc={len(doc)} chars)"


case("requirement_context(doc, ['FR-999'])", ctx_missing)

print()
print("=" * 70)
print("F. misc helpers")
print("=" * 70)
case("stable_id deterministic",
     lambda: stable_id("x", "a", "b") == stable_id("x", "a", "b"))
case("normalize_decision junk", lambda: normalize_decision("maybe"))
case("flatten_unique", lambda: flatten_unique([["a", "b"], ["b", "c"]]))
case("merge_elements dedup", lambda: len(merge_elements([
    RequirementElement("Alpha", "FUNCTION", 0.8, ["FR-1"], ["e1"]),
    RequirementElement("alpha", "FUNCTION", 0.95, ["FR-2"], ["e2"])])))
case("Usage positional ctor", lambda: Usage(1, 2, 3).__dict__)

print()
print("=" * 70)
print("G. Extraction Support must use ALL directed paths (paper Eq. 8)")
print("=" * 70)


def es_uses_all_paths():
    """Shortest path gives 0.60; a longer admissible path gives 0.90."""
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=0.60, extraction_support=0.60)
    graph.add_edge("B", "D", weight=0.60, extraction_support=0.60)
    graph.add_edge("A", "C", weight=0.90, extraction_support=0.90)
    graph.add_edge("C", "E", weight=0.90, extraction_support=0.90)
    graph.add_edge("E", "D", weight=0.90, extraction_support=0.90)

    score, path = maximum_geometric_path_support(graph, "A", "D")
    assert math.isclose(score, 0.90, rel_tol=1e-12, abs_tol=1e-12), score
    assert path == ["A", "C", "E", "D"], path

    shortest = deterministic_shortest_path(graph, "A", "D")
    assert shortest == ["A", "B", "D"], shortest
    return f"ES={score:.4f} via {path}; shortest={shortest}"


case("ES picks the best path, not the shortest", es_uses_all_paths)


def es_tie_break_deterministic():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=0.80, extraction_support=0.80)
    graph.add_edge("B", "D", weight=0.80, extraction_support=0.80)
    graph.add_edge("A", "C", weight=0.80, extraction_support=0.80)
    graph.add_edge("C", "D", weight=0.80, extraction_support=0.80)

    score, path = maximum_geometric_path_support(graph, "A", "D")
    assert math.isclose(score, 0.80, rel_tol=1e-12, abs_tol=1e-12), score
    assert path == ["A", "B", "D"], path
    return f"tie -> lexicographically smallest {path}"


case("ES tie-break is deterministic", es_tie_break_deterministic)


def es_matches_brute_force():
    """Cross-check the log-space DP against explicit path enumeration."""
    import itertools
    import random as _random

    rng = _random.Random(7)
    checked = 0
    for trial in range(40):
        g = nx.DiGraph()
        nodes = [f"n{i}" for i in range(6)]
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                if rng.random() < 0.45:
                    w = round(rng.uniform(0.2, 1.0), 2)
                    g.add_edge(nodes[a], nodes[b], weight=w,
                               extraction_support=w)
        if not ({"n0", "n5"} <= set(g.nodes)):
            continue
        if not nx.has_path(g, "n0", "n5"):
            continue

        best = 0.0
        for length in range(2, len(nodes) + 1):
            for chain in itertools.permutations(nodes, length):
                if chain[0] != "n0" or chain[-1] != "n5":
                    continue
                if not all(g.has_edge(chain[k], chain[k + 1])
                           for k in range(length - 1)):
                    continue
                ws = [g[chain[k]][chain[k + 1]]["weight"]
                      for k in range(length - 1)]
                best = max(best, math.prod(ws) ** (1 / len(ws)))

        got, _ = maximum_geometric_path_support(g, "n0", "n5")
        assert math.isclose(got, best, rel_tol=1e-9, abs_tol=1e-9), \
            f"trial {trial}: dp={got} brute={best}"
        checked += 1
    return f"matched brute force on {checked} random DAGs"


case("DP == brute-force max over all paths", es_matches_brute_force)

print()
print("=" * 70)
print("H. configuration hygiene")
print("=" * 70)


def baseline_fields_removed():
    import dataclasses
    names = [f.name for f in dataclasses.fields(dioreq.ModelConfig)]
    assert "baseline_model" not in names, names
    assert "baseline_temperature" not in names, names
    return f"ModelConfig fields = {names}"


case("baseline fields removed", baseline_fields_removed)


def no_baseline_env_var():
    src = (
        code_paths.find_dioreq() / "dioreq.py"
    ).read_text(encoding="utf-8")
    assert "DIOREQ_BASELINE_MODEL" not in src
    assert "DEFAULT_MODEL_CONFIG" not in src
    return "no DIOREQ_BASELINE_MODEL / DEFAULT_MODEL_CONFIG left"


case("dead env var / constant gone", no_baseline_env_var)


def lazy_config_reads_late_env():
    """The whole point of the fix: env set AFTER import must still apply."""
    os.environ["DIOREQ_EXTRACTION_MODEL"] = "late-set-model"
    os.environ["DIOREQ_VALIDATION_MODEL"] = "late-v"
    os.environ["DIOREQ_GENERATION_MODEL"] = "late-g"
    try:
        cfg = dioreq.default_model_config()
        assert cfg.extraction_model == "late-set-model", cfg
        assert cfg.validation_model == "late-v", cfg
        assert cfg.generation_model == "late-g", cfg

        pipe = dioreq.DIOReqPipeline(llm=None)
        assert pipe.model_config.extraction_model == "late-set-model", \
            pipe.model_config

        # an explicit config must still win over the environment
        explicit = dioreq.ModelConfig("e", "v", "g")
        pipe2 = dioreq.DIOReqPipeline(llm=None, model_config=explicit)
        assert pipe2.model_config is explicit
        return "env honoured at call time; explicit config still overrides"
    finally:
        for k in ("DIOREQ_EXTRACTION_MODEL", "DIOREQ_VALIDATION_MODEL",
                  "DIOREQ_GENERATION_MODEL"):
            os.environ.pop(k, None)


case("default_model_config() is lazy", lazy_config_reads_late_env)


def pipeline_defaults_resolved():
    pipe = dioreq.DIOReqPipeline(llm=None)
    assert isinstance(pipe.model_config, dioreq.ModelConfig)
    assert isinstance(pipe.config, dioreq.PipelineConfig)
    assert pipe.config.dependency_budget == 20
    return (f"defaults resolved: model={pipe.model_config.extraction_model!r}, "
            f"budget={pipe.config.dependency_budget}")


case("pipeline defaults resolved on construction", pipeline_defaults_resolved)

print()
print("=" * 70)
print("I. Stage-5 refined requirements document")
print("=" * 70)

# label -> (document, expected format, expected major, expected last number,
#           expected numbers for two appended requirements)
DIALECTS = {
    "dot": ("1.1 A\nInput: a\n\n1.2 B\nInput: b\n", "dot", "1", 2,
            ["1.3", "1.4"]),
    "mddot": ("## 1.1 A\nInput: a\n\n## 1.2 B\nInput: b\n", "mddot", "1", 2,
              ["1.3", "1.4"]),
    "dash": ("FR-001\tSend\nInput: a\nFR-002\tRecv\nInput: b\n", "dash", "", 2,
             ["FR-003", "FR-004"]),
    "word": ("Function Requirement 1: A\nInput: a\n\n"
             "Function Requirement 2: B\nInput: b\n", "word", "", 2,
             ["Function Requirement 3", "Function Requirement 4"]),
    "letter": ("F1 A\n- Input: a\nF2 B\n- Input: b\n", "letter", "", 2,
               ["F3", "F4"]),
    "space": ("FR 1: A\nInput: a\nFR 2: B\nInput: b\n", "space", "", 2,
              ["FR 3:", "FR 4:"]),
    "mdash-heading": ("### FR-01 A\nInput: a\n### FR-02 B\nInput: b\n",
                      "dash", "", 2, ["FR-003", "FR-004"]),
    "dot3": ("3.1.1 A\nInput: a\n3.1.2 B\nInput: b\n", "dot3", "3.1", 2,
             ["3.1.3", "3.1.4"]),
    "compound": ("FR-DCI-001: A\nInput: a\nFR-DCI-002: B\nInput: b\n",
                 "compound", "", 2, ["FR-DCI-003", "FR-DCI-004"]),
}


def fr_dialects():
    two = [
        {"requirement": "The system shall do X.", "diagnostic_view": "dependency",
         "gap_type": "G1", "source_frs": [], "supporting_excerpts": [],
         "affected_element": None, "dps": None},
        {"requirement": "The system shall do Y.", "diagnostic_view": "isolation",
         "gap_type": "G2", "source_frs": [], "supporting_excerpts": [],
         "affected_element": None, "dps": None},
    ]
    problems = []
    for label, (doc, fmt, major, last, expected) in DIALECTS.items():
        got_fmt = dioreq.detect_fr_format(doc)
        got_major = dioreq.original_fr_major(doc)
        got_last = dioreq.last_fr_number(doc)
        got_nums = [r["fr_number"]
                    for r in dioreq.assign_requirement_numbers(doc, two)]
        if (got_fmt != fmt or got_major != major or got_last != last
                or got_nums != expected):
            problems.append((label, got_fmt, got_major, got_last, got_nums))
    assert not problems, problems
    return f"all {len(DIALECTS)} dialects numbered correctly"


case("FR dialect detection and renumbering", fr_dialects)


def numbering_follows_segmentation():
    """
    Supplementary requirements must continue the numbering of the
    requirement level that was actually segmented, not of the section level
    above it. This document is split at 'FR-001', so appending '3.7' would
    leave the two numbering schemes mixed in one document.
    """
    doc = (
        "Functional Requirement\n"
        "3. Functional Requirements\n"
        "3.1 User Account Management\n"
        "FR-001\nUsers can create an account.\n"
        "FR-002\nUsers can log in.\n"
        "3.2 Product Catalog\n"
        "FR-003\nProducts are displayed.\n"
    )
    ids = [name for name, _ in split_into_frs(doc)]
    assert ids == ["FR-001", "FR-002", "FR-003"], ids

    context = dioreq.numbering_context(doc)
    assert context["format"] == "dash", context
    assert context["last"] == "3", context
    assert context["major"] == "", context

    numbered = dioreq.assign_requirement_numbers(
        doc, [{"requirement": "The system shall X."}])
    assert numbered[0]["fr_number"] == "FR-004", numbered[0]["fr_number"]
    return "split at FR-xxx -> next number is FR-004, not 3.3"


case("numbering follows the segmented level", numbering_follows_segmentation)


def markdown_headings_are_preserved():
    doc = "## 1.1 Alpha\nInput: a\n## 1.2 Beta\nInput: b\n"
    text = dioreq.build_refined_document(
        doc, [{"requirement": "The system shall X.",
               "diagnostic_view": "dependency", "gap_type": "G",
               "source_frs": ["1.1"], "supporting_excerpts": ["e"]}])
    assert "## 1.3" in text, text
    return "appended requirement keeps the document's '##' heading style"


case("markdown heading style preserved", markdown_headings_are_preserved)


def refined_document():
    doc = "1.1 Complaint Registration\nInput: a\nOutput: b\n"
    cands = [{
        "candidate_id": "c1",
        "requirement": "The system shall notify the citizen.",
        "diagnostic_view": "dependency",
        "gap_type": "MISSING_NOTIFICATION",
        "source_frs": ["1.1"],
        "supporting_excerpts": ["excerpt A"],
        "affected_element": "Complaint",
        "dps": 0.4213,
    }]
    out = dioreq.build_refined_document(doc, cands)
    assert doc.strip() in out, "original text must be preserved verbatim"
    assert "1.2 Complaint" in out, out
    assert "Requirement: The system shall notify the citizen." in out
    assert "Source requirements: 1.1" in out
    assert "Evidence: excerpt A" in out
    assert "Diagnostic view: dependency" in out
    assert "Gap type: MISSING_NOTIFICATION" in out
    assert "DPS: 0.4213" in out
    return f"{len(out)} chars, original preserved, 1.2 numbered"


case("build_refined_document", refined_document)


def refined_empty_candidates():
    doc = "1.1 A\nInput: a\n"
    out = dioreq.build_refined_document(doc, [])
    assert out.strip() == doc.strip(), out
    return "no candidates -> document unchanged"


case("refined document with no candidates", refined_empty_candidates)


def design_constraints_document():
    demoted = [{
        "requirement": "Persist audit rows in a separate PostgreSQL schema.",
        "reason": "describes an implementation detail",
        "diagnostic_view": "operation",
        "affected_element": "AuditLog",
        "source_frs": ["1.2"],
    }]
    removed = [{
        "requirement": "The system shall email the customer.",
        "reason": "duplicate of an existing requirement",
    }]
    text = dioreq.build_design_constraints_document(demoted, removed)
    assert "DESIGN CONSTRAINTS AND REJECTED CANDIDATES" in text
    assert "NOT part of the refined requirements document" in text
    assert "DC-001" in text
    assert "Persist audit rows" in text
    assert "describes an implementation detail" in text
    assert "REMOVED-001" in text
    assert "duplicate of an existing requirement" in text
    return f"{len(text)} chars, 1 design constraint + 1 rejected"


case("build_design_constraints_document", design_constraints_document)


def design_constraints_empty():
    text = dioreq.build_design_constraints_document([], [])
    assert "DC-001" not in text and "(none)" in text
    return "empty input renders (none) sections"


case("design constraints with nothing demoted", design_constraints_empty)


def refinement_report():
    doc = "1.1 A\nInput: a\n\n1.2 B\nInput: b\n"
    kept = [
        {"fr_number": "1.3", "requirement": "The system shall notify.",
         "diagnostic_view": "dependency", "gap_type": "G1",
         "source_frs": ["1.1"]},
        {"fr_number": "1.4", "requirement": "The system shall log.",
         "diagnostic_view": "operation", "gap_type": "G2", "source_frs": []},
    ]
    demoted = [{"requirement": "Use Redis.", "reason": "technology choice"}]
    removed = [{"requirement": "Duplicate.", "reason": "already covered"}]
    text = dioreq.build_refinement_report(doc, kept, demoted, removed)
    assert text.startswith("# Stage 5 - Requirement Refinement Report")
    assert "| Original requirements in the source document | 2 |" in text
    assert "| Refined (kept) | 2 |" in text
    assert "| Demoted to design constraints | 1 |" in text
    assert "| Rejected | 1 |" in text
    assert "**4**" in text, "total refined length must be 2 + 2"
    assert "`1.3` The system shall notify." in text
    assert "Use Redis." in text
    assert "Duplicate." in text
    return f"{len(text)} chars, totals 2 original + 2 kept = 4"


case("build_refinement_report", refinement_report)


def consolidation_result_partitions():
    """consolidate_candidates must return a ConsolidationResult."""
    import dataclasses
    fields = [f.name
              for f in dataclasses.fields(dioreq.ConsolidationResult)]
    assert fields == ["kept", "demoted", "removed"], fields

    class _NoLLM:
        def json_call(self, **kwargs):
            raise AssertionError("must not be called when disabled")

    from rq2 import BASE_CONFIG
    import dataclasses as dc
    off = dc.replace(
        BASE_CONFIG,
        consolidation=False, refinement=False,
        deduplication=False, final_filtering=False,
    )
    cands = [dioreq.GeneratedCandidate(
        candidate_id="c1", requirement="The system shall X.",
        diagnostic_view="dependency", gap_type="G1",
        source_frs=["1.1"], supporting_excerpts=["e"])]
    out = dioreq.consolidate_candidates(
        _NoLLM(), "1.1 A\n", cands, dioreq.default_model_config(), off)
    assert isinstance(out, dioreq.ConsolidationResult)
    assert len(out.kept) == 1 and not out.demoted and not out.removed
    return "disabled consolidation passes candidates straight through"


case("ConsolidationResult contract", consolidation_result_partitions)

print()
print("=" * 70)
print("J. Malformed model output must never abort a run")
print("=" * 70)

# A fake client that returns the same payload for every call, so each
# pipeline function is fed the exact same malformed shape.
class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def json_call(self, **kwargs):
        self.calls += 1
        return self.payload


MATRIX_DOC = (
    "1.1 Register a complaint\n"
    "Input: complaint\nOutput: id\n"
    "1.2 Notify the complainant\n"
    "Input: id\nOutput: message\n"
)

# Shapes an endpoint can plausibly return for a well-formed prompt.
MALFORMED_PAYLOADS = [
    ("empty object", {}),
    ("all fields null", {
        key: None for key in (
            "elements", "relations", "findings", "candidates",
            "evidence", "coverage", "boundary", "source_frs",
            "source_candidate_ids", "supporting_excerpts",
            "supporting_excerpt", "aliases", "requirement", "label",
            "gap_type", "affected_element", "reason", "decision",
            "type", "confidence", "name", "description",
            "justification", "source", "target", "relation_type",
        )
    }),
    ("collections as strings", {
        "elements": "none", "relations": "none", "findings": "none",
        "candidates": "none", "source_frs": "1.1",
        "source_candidate_ids": "c1", "supporting_excerpts": "excerpt",
        "aliases": "alias",
    }),
    ("collections as scalars", {
        "elements": 7, "relations": 7, "findings": 7, "candidates": 7,
        "source_frs": 7, "supporting_excerpts": 7, "aliases": 7,
    }),
    ("items are not objects", {
        "elements": [1, "x", None, [1]],
        "relations": [1, None],
        "findings": ["a", None],
        "candidates": [None, 3],
    }),
    ("field types are wrong", {
        "elements": [{
            "type": 5, "confidence": [], "name": {}, "aliases": {},
            "supporting_excerpt": 0, "description": None,
        }],
        "relations": [{
            "source": [], "target": {}, "relation_type": 1,
            "confidence": "abc", "supporting_excerpts": {"a": 1},
            "source_frs": 3, "justification": None,
        }],
        "findings": [{
            "gap_type": [], "affected_element": None,
            "source_frs": 1, "supporting_excerpts": "x", "reason": None,
        }],
        "candidates": [{
            "decision": [], "source_candidate_ids": 1,
            "requirement": None, "reason": None,
        }],
    }),
    ("confidence as percentage", {
        "elements": [
            {"name": "Complaint", "type": "DATA", "confidence": 90,
             "supporting_excerpt": "complaint"},
            {"name": "Notify", "type": "FUNCTION", "confidence": "0.9",
             "supporting_excerpt": "Notify"},
            {"name": "Bad", "type": "ACTOR", "confidence": 500,
             "supporting_excerpt": "Bad"},
            {"name": "Bool", "type": "EVENT", "confidence": True,
             "supporting_excerpt": "Bool"},
        ],
        "relations": [
            {"source": "Notify", "target": "Complaint",
             "relation_type": "MODIFIES", "confidence": 80,
             "supporting_excerpts": ["Notify"]},
        ],
    }),
    ("valid findings but null sub-fields", {
        "elements": [
            {"name": "Complaint", "type": "DATA", "confidence": 0.9,
             "supporting_excerpt": "complaint"},
            {"name": "Notify", "type": "FUNCTION", "confidence": 0.9,
             "supporting_excerpt": "Notify"},
        ],
        "relations": [
            {"source": "Notify", "target": "Complaint",
             "relation_type": "ENABLES", "confidence": 0.9,
             "supporting_excerpts": ["Notify"]},
        ],
        "findings": [{
            "gap_type": "MISSING_PRECONDITION",
            "affected_element": None,
            "source_frs": None,
            "supporting_excerpts": None,
            "reason": None,
        }],
        "label": "GENUINE_GAP",
        "evidence": {"decision": "YES", "reason": None},
        "coverage": {"decision": "NO", "reason": None},
        "boundary": {"decision": "YES", "reason": None},
        "requirement": "The system shall notify the complainant.",
        "candidates": [{
            "decision": None,
            "source_candidate_ids": None,
            "requirement": None,
            "reason": None,
        }],
    }),
]


def run_pipeline_functions(payload):
    """Feed one payload to every model-consuming function in the pipeline."""
    llm = _FakeLLM(payload)
    config = dioreq.default_model_config()
    pipeline_config = dioreq.PipelineConfig()

    elements = dioreq.extract_elements(llm, MATRIX_DOC, config)
    relations = dioreq.extract_dependency_relations(
        llm, MATRIX_DOC, elements, config)

    graphs = DependencyGraphs(max_parents=12)
    graphs.build(elements, relations)
    records = graphs.ranking_records()

    findings = []
    findings += dioreq.nominate_dependency_findings(
        llm, MATRIX_DOC, graphs, records, config)
    findings += dioreq.nominate_isolation_findings(
        llm, MATRIX_DOC, graphs, config)
    findings += dioreq.nominate_operation_findings(
        llm, MATRIX_DOC, elements, config)

    validated = dioreq.validate_findings(
        llm, MATRIX_DOC, findings, config, pipeline_config)
    generated = dioreq.generate_candidates(
        llm, MATRIX_DOC, elements, validated, config)
    result = dioreq.consolidate_candidates(
        llm, MATRIX_DOC, generated, config, pipeline_config)

    total = (
        len(result.kept) + len(result.demoted) + len(result.removed)
    )
    assert total == len(generated), (
        "consolidation dropped candidates: "
        f"{total} partitioned vs {len(generated)} generated"
    )

    return (
        f"elements={len(elements)} relations={len(relations)} "
        f"findings={len(validated)} generated={len(generated)} "
        f"kept={len(result.kept)}"
    )


for _label, _payload in MALFORMED_PAYLOADS:
    case(f"survives: {_label}", lambda p=_payload: run_pipeline_functions(p))


def confidence_normalization():
    """Confidence must be read on the unit interval, never saturated."""
    c = dioreq.coerce_confidence
    assert c(0.9) == 0.9
    assert c("0.85") == 0.85
    assert c(90) == 0.9, "a percentage must not saturate to 1.0"
    assert c(100) == 1.0
    assert abs(c(1.5) - 0.015) < 1e-9
    assert c(500) == 0.0
    assert c(-3) == 0.0
    assert c(None) == 0.0
    assert c("abc") == 0.0
    assert c(True) == 0.0, "booleans are not confidences"
    assert c([]) == 0.0
    return "0.9 / '0.85' / 90 / 1.5 / 500 / None / True all handled"


case("coerce_confidence contract", confidence_normalization)


def coercion_helpers():
    assert dioreq.coerce_mapping(None) == {}
    assert dioreq.coerce_mapping([1]) == {}
    assert dioreq.coerce_mapping({"a": 1}) == {"a": 1}
    assert dioreq.coerce_items(None) == []
    assert dioreq.coerce_items("abc") == []
    assert dioreq.coerce_items([{"a": 1}, 2, None]) == [{"a": 1}]
    assert dioreq.coerce_str(None) == ""
    assert dioreq.coerce_str("  x  ") == "x"
    assert dioreq.coerce_str_list(None) == []
    assert dioreq.coerce_str_list("x") == ["x"]
    assert dioreq.coerce_str_list(["a", None, "", 5]) == ["a", "5"]
    return "mapping/items/str/str_list all degrade safely"


case("coercion helpers", coercion_helpers)


def every_prompt_mentions_json():
    """
    response_format=json_object is rejected unless the messages say JSON.

    Only the templates that request a structured return are required to say
    it: the two system-only prompts are always paired with such a template,
    and json_call additionally injects a guard when neither message
    mentions JSON.
    """
    import re
    src = (
        code_paths.find_dioreq() / "dioreq.py"
    ).read_text(encoding="utf-8")
    names = sorted(set(
        re.findall(r"^([A-Z][A-Z0-9_]*)\s*=\s*\"\"\"", src, re.M)
    ))
    assert names, "no prompt constants found"

    templates = [n for n in names if "{{" in getattr(dioreq, n)]
    assert templates, "no return templates found"

    missing = [
        name for name in templates
        if not re.search(r"json", getattr(dioreq, name), re.IGNORECASE)
    ]
    assert not missing, f"templates without the word JSON: {missing}"

    system_only = [n for n in names if n not in templates]
    return (
        f"all {len(templates)} return templates mention JSON; "
        f"{len(system_only)} system-only prompt(s) covered by the guard"
    )


case("prompt constants satisfy json_object", every_prompt_mentions_json)


# ------------------------------------------------------------
# json_call behaviour, driven by a stubbed OpenAI client
# ------------------------------------------------------------

class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content, usage=None):
        self.choices = [_Choice(content)]
        self.usage = usage


class _Completions:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.behaviour(kwargs)


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class _StubClient:
    def __init__(self, behaviour):
        self.chat = _Chat(_Completions(behaviour))
        self.base_url = "stub"


def stub_llm(behaviour):
    client = object.__new__(dioreq.LLMClient)
    client.client = _StubClient(behaviour)
    client.usage = Usage()
    return client


def json_guard_is_injected():
    def behaviour(kwargs):
        text = " ".join(m["content"] for m in kwargs["messages"])
        assert "json" in text.lower(), "messages must mention JSON"
        return _Response('{"ok": true}')

    llm = stub_llm(behaviour)
    out = llm.json_call("m", "You are a helper.", "Do the thing.", 0.0)
    assert out == {"ok": True}
    return "guard appended when the prompt never says JSON"


case("json_call injects the JSON guard", json_guard_is_injected)


def temperature_fallback():
    """A model that rejects the temperature must be retried without it."""
    def behaviour(kwargs):
        if "temperature" in kwargs:
            raise Exception(
                "Error code: 400 - Unsupported value: 'temperature' does "
                "not support 0.0 with this model. Only the default (1) "
                "value is supported."
            )
        return _Response('{"ok": true}')

    llm = stub_llm(behaviour)
    out = llm.json_call("m", "JSON please.", "Give JSON.", 0.0)
    calls = llm.client.chat.completions.calls
    assert out == {"ok": True}
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1], "retry must drop the field"
    return f"{len(calls)} calls; second call has no temperature"


case("json_call drops a rejected temperature", temperature_fallback)


def non_object_response_is_retried():
    """A JSON array where an object was required must be retried."""
    state = {"n": 0}

    def behaviour(kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return _Response("[1, 2, 3]")
        return _Response('{"ok": true}')

    llm = stub_llm(behaviour)
    out = llm.json_call("m", "JSON.", "JSON.", 1.0)
    assert out == {"ok": True}
    assert state["n"] == 2, f"expected one retry, saw {state['n']} calls"
    return "array response retried until an object arrived"


case("json_call retries a non-object response", non_object_response_is_retried)


def temperature_env_override():
    keys = {
        "DIOREQ_EXTRACTION_TEMPERATURE": None,
        "DIOREQ_VALIDATION_TEMPERATURE": None,
        "DIOREQ_GENERATION_TEMPERATURE": None,
    }
    saved = {k: os.environ.get(k) for k in keys}

    try:
        os.environ["DIOREQ_EXTRACTION_TEMPERATURE"] = "none"
        os.environ["DIOREQ_VALIDATION_TEMPERATURE"] = "0.7"
        os.environ["DIOREQ_GENERATION_TEMPERATURE"] = "junk"
        config = dioreq.default_model_config()
        assert config.extraction_temperature is None
        assert config.validation_temperature == 0.7
        assert config.generation_temperature == 0.2, "junk -> documented default"

        for key in keys:
            os.environ.pop(key, None)
        config = dioreq.default_model_config()
        assert config.extraction_temperature == 0.0
        assert config.generation_temperature == 0.2
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return "'none' disables, numeric parses, junk falls back"


case("temperature environment override", temperature_env_override)


def failed_run_is_recorded():
    """One failing document-run must not abort the batch."""
    import json as _json
    import tempfile

    dataset = [
        {"system_id": "S", "document_id": "bad", "text": "1.1 A\n"},
        {"system_id": "S", "document_id": "good", "text": "1.1 B\n"},
    ]

    class _Boom:
        def __init__(self, **kwargs):
            self.usage = Usage()

        def json_call(self, **kwargs):
            raise RuntimeError("endpoint exploded")

    calls = {"n": 0}

    def fake_run(self, document_text, ranking_strategy="dps",
                 random_seed=0, progress=None):
        calls["n"] += 1
        if document_text.startswith("1.1 A"):
            raise RuntimeError("endpoint exploded")
        return {
            "elements": [], "relations": [],
            "computational_exclusions": [], "ranking_records": [],
            "findings": [], "raw_candidates": [], "candidates": [],
            "design_constraints": [], "removed_candidates": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "api_calls": 0},
        }

    original_client = dioreq.LLMClient
    original_run = dioreq.DIOReqPipeline.run
    dioreq.LLMClient = _Boom
    dioreq.DIOReqPipeline.run = fake_run

    try:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, "dataset.json")
            output_path = os.path.join(tmp, "out.json")
            with open(dataset_path, "w", encoding="utf-8") as fh:
                _json.dump(dataset, fh)

            dioreq.run_dioreq(
                dataset_path=dataset_path,
                output_path=output_path,
                repeated_runs=1,
                dependency_budget=5,
                refined_dir=None,
            )

            with open(output_path, encoding="utf-8") as fh:
                results = _json.load(fh)
    finally:
        dioreq.LLMClient = original_client
        dioreq.DIOReqPipeline.run = original_run

    assert calls["n"] == 2, "the second document must still be attempted"
    assert len(results) == 2, f"expected 2 records, got {len(results)}"
    assert "error" in results[0], "the failing run must record its error"
    assert "error" not in results[1], "the good run must be unaffected"
    return "bad run recorded, good run preserved, batch not aborted"


case("run-level failure isolation", failed_run_is_recorded)


def cli_options_reach_the_client():
    """--model / --base-url must actually reach the client and the config."""
    import json as _json
    import tempfile

    captured = {}

    class _Spy:
        def __init__(self, api_key=None, base_url=None, **kwargs):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.usage = Usage()

        def json_call(self, **kwargs):
            raise RuntimeError("not used")

    seen = {}

    def fake_run(self, document_text, ranking_strategy="dps",
                 random_seed=0, progress=None):
        seen["models"] = (
            self.model_config.extraction_model,
            self.model_config.validation_model,
            self.model_config.generation_model,
        )
        return {
            "elements": [], "relations": [], "computational_exclusions": [],
            "ranking_records": [], "findings": [], "raw_candidates": [],
            "candidates": [], "design_constraints": [],
            "removed_candidates": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "api_calls": 0},
        }

    original_client = dioreq.LLMClient
    original_run = dioreq.DIOReqPipeline.run
    dioreq.LLMClient = _Spy
    dioreq.DIOReqPipeline.run = fake_run

    try:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, "d.json")
            with open(dataset_path, "w", encoding="utf-8") as fh:
                _json.dump([{"system_id": "S", "document_id": "D",
                             "text": "1.1 A\n"}], fh)

            dioreq.run_dioreq(
                dataset_path=dataset_path,
                output_path=os.path.join(tmp, "o.json"),
                repeated_runs=1,
                dependency_budget=5,
                refined_dir=None,
                model="custom-model",
                base_url="https://example.invalid/v1",
                api_key="sk-test",
            )
    finally:
        dioreq.LLMClient = original_client
        dioreq.DIOReqPipeline.run = original_run

    assert captured["base_url"] == "https://example.invalid/v1", captured
    assert captured["api_key"] == "sk-test", captured
    assert seen["models"] == ("custom-model",) * 3, seen
    return "model, base_url and api_key all honoured"


case("runner CLI options", cli_options_reach_the_client)


def sharding_is_a_partition():
    """
    Process-level sharding is how a multi-day protocol becomes feasible, so
    the slices must cover every document exactly once.
    """
    documents = [
        dioreq.DocumentRecord(system_id="S", document_id=f"D{i}", text="x")
        for i in range(10)
    ]

    assert dioreq.shard_documents(documents, None) == documents

    collected = []
    for index in range(3):
        part = dioreq.shard_documents(documents, (index, 3))
        assert part, f"shard {index}/3 is empty"
        collected += [d.document_id for d in part]

    assert sorted(collected) == [d.document_id for d in documents], collected
    assert len(collected) == len(set(collected)), "a document was duplicated"

    assert dioreq.parse_shard(None) is None
    assert dioreq.parse_shard("2/5") == (2, 5)
    assert dioreq.parse_shard(" 1 / 4 ") == (1, 4)

    for bad in ("abc", "4/4", "1/0", "-1/2", "1"):
        try:
            dioreq.parse_shard(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_shard accepted {bad!r}")

    return "3 shards partition 10 documents exactly once; bad specs rejected"


case("--shard partitions the corpus", sharding_is_a_partition)

print()
print("=" * 70)
print(f"RESULT: {len(fails)} failure(s)")
for n, e in fails:
    print(f"  - {n}: {type(e).__name__}: {e}")
