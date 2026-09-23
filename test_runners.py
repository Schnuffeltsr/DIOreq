# -*- coding: utf-8 -*-
"""
End-to-end dry run of rq2.py and rq3.py against a mocked LLM.

No network access and no API key are required: ``LLMClient.json_call`` is
replaced by a deterministic stub that dispatches on the system prompt. This
exercises every stage of both runners (extraction, graph construction,
ranking, nomination, validation, generation, consolidation, JSON output)
and asserts the invariants that the paper requires.

Run:  py test_runners.py
"""
import json
import os
import re
import shutil
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import code_paths

code_paths.install()

import dioreq
import rq2
import rq3


# --------------------------------------------------------------- mock LLM
def _json_after(label, text):
    """Parse the first JSON array appearing after `label` in `text`."""
    pos = text.find(label)
    if pos < 0:
        return []
    start = text.find("[", pos)
    if start < 0:
        return []
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return []
    return []


class MockLLM:
    """Deterministic stand-in for LLMClient."""

    element_calls = 0
    relation_calls = 0
    other_calls = 0
    usage = None

    def __init__(self, api_key=None, base_url=None, **kwargs):
        self.usage = dioreq.Usage()

    def json_call(self, model, system_prompt, user_prompt, temperature,
                  retries=3):
        self.usage.api_calls += 1
        s = system_prompt

        if "requirements-engineering information extractor" in s:
            MockLLM.element_calls += 1
            m = re.search(r"Requirement identifier:\s*\n\s*(\S+)", user_prompt)
            rid = m.group(1) if m else "FR-0"
            return {"elements": [
                {"name": f"{rid}-Func", "type": "FUNCTION",
                 "confidence": 0.9, "aliases": [],
                 "supporting_excerpt": "mock excerpt",
                 "description": "mock"},
                {"name": f"{rid}-Data", "type": "DATA",
                 "confidence": 0.9, "aliases": [],
                 "supporting_excerpt": "mock excerpt",
                 "description": "mock"},
                {"name": f"{rid}-Actor", "type": "ACTOR",
                 "confidence": 0.85, "aliases": [],
                 "supporting_excerpt": "mock excerpt",
                 "description": "mock"},
            ]}

        if "directional dependencies" in s:
            MockLLM.relation_calls += 1
            if "Currently disconnected elements:" in user_prompt:
                return {"relations": []}          # cross-requirement pass
            names = _json_after("Fixed canonical elements:", user_prompt)
            if len(names) < 2:
                return {"relations": []}
            return {"relations": [
                {"source": names[0], "target": names[1],
                 "relation_type": "ENABLES", "confidence": 0.9,
                 "source_frs": [], "supporting_excerpts": ["mock excerpt"],
                 "justification": "mock", "cross_requirement": False},
                {"source": names[0], "target": names[2],
                 "relation_type": "PRODUCES", "confidence": 0.8,
                 "source_frs": [], "supporting_excerpts": ["mock excerpt"],
                 "justification": "mock", "cross_requirement": False},
            ]}

        if "Nominate diagnostic findings" in s:
            MockLLM.other_calls += 1
            patterns = _json_after("Applicable review patterns:", user_prompt)
            if not patterns:
                return {"findings": []}
            return {"findings": [
                {"gap_type": patterns[0],
                 "affected_element": "mock-affected",
                 "source_frs": ["FR-1"],
                 "supporting_excerpts": ["mock excerpt"],
                 "reason": "mock reason"},
            ]}

        if "Classify disconnected requirement elements" in s:
            MockLLM.other_calls += 1
            return {"label": "GENUINE_GAP", "gap_type": "MISSING_INTEGRATION",
                    "reason": "mock", "source_frs": [],
                    "supporting_excerpts": []}

        if "operation-coverage diagnosis" in s:
            MockLLM.other_calls += 1
            return {"findings": [
                {"affected_element": "mock-entity",
                 "gap_type": "MISSING_AUDIT", "source_frs": [],
                 "supporting_excerpts": [], "reason": "mock"},
            ]}

        if "conservative requirements-review validator" in s:
            MockLLM.other_calls += 1
            return {
                "evidence": {"decision": "YES", "reason": "mock"},
                "coverage": {"decision": "NO", "reason": "mock"},
                "boundary": {"decision": "YES", "reason": "mock"},
            }

        if "Generate evidence-grounded, reviewable functional requirements" in s:
            MockLLM.other_calls += 1
            return {"requirement": "The system shall handle the mock gap."}

        if "Consolidate generated requirements" in s:
            MockLLM.other_calls += 1
            cands = _json_after("Candidates:", user_prompt)
            ids = [c["candidate_id"] for c in cands]

            # Exercise all three refinement outcomes, and merge two raw
            # candidates into the first kept group so that the number of
            # groups differs from the number of source candidates.
            if len(ids) >= 4:
                return {"candidates": [
                    {"source_candidate_ids": ids[:2],
                     "requirement": cands[0]["requirement"],
                     "decision": "KEEP", "reason": "mock keep"},
                    {"source_candidate_ids": [ids[2]],
                     "requirement": cands[2]["requirement"],
                     "decision": "DEMOTE",
                     "reason": "mock implementation detail"},
                    {"source_candidate_ids": ids[3:],
                     "requirement": cands[3]["requirement"],
                     "decision": "REMOVE", "reason": "mock duplicate"},
                ]}

            out = []
            for index, c in enumerate(cands):
                if index == 0:
                    decision, reason = "KEEP", "mock keep"
                elif index == 1:
                    decision, reason = "DEMOTE", "mock implementation detail"
                else:
                    decision, reason = "REMOVE", "mock duplicate"
                out.append({"source_candidate_ids": [c["candidate_id"]],
                            "requirement": c["requirement"],
                            "decision": decision, "reason": reason})
            return {"candidates": out}

        MockLLM.other_calls += 1
        return {}


# ------------------------------------------------------------------ setup
DATASET = [
    {"system_id": "SysA", "document_id": "Doc1",
     "text": ("1.1 Register a complaint\nInput: a\nOutput: b\n\n"
              "1.2 View a complaint\nInput: c\nOutput: d\n\n"
              "1.3 Update a complaint\nInput: e\nOutput: f\n")},
]

OUTDIR = os.path.join(HERE, "_runner_test_out")
fails = []


def case(name, fn):
    try:
        out = fn()
        print(f"  [ok]   {name}" + (f"  -> {out}" if out else ""))
    except Exception as e:
        fails.append((name, e))
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)


def main():
    shutil.rmtree(OUTDIR, ignore_errors=True)
    os.makedirs(OUTDIR, exist_ok=True)

    dataset_path = os.path.join(OUTDIR, "dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(DATASET, f, ensure_ascii=False)

    # patch the LLM client wherever it is referenced
    dioreq.LLMClient = MockLLM
    rq2.LLMClient = MockLLM
    rq3.LLMClient = MockLLM

    print("=" * 70)
    print("A. rq2.py end-to-end (default = paper's 8 variants)")
    print("=" * 70)
    rq2_out = os.path.join(OUTDIR, "rq2.json")
    MockLLM.element_calls = 0
    MockLLM.relation_calls = 0
    MockLLM.other_calls = 0

    case("rq2.run_rq2 executes", lambda: (
        rq2.run_rq2(dataset_path, rq2_out, repeated_runs=1,
                    dependency_budget=3) or "done"))

    def rq2_shape():
        data = json.load(open(rq2_out, encoding="utf-8"))
        variants = [d["variant"] for d in data]
        assert variants == rq2.PAPER_VARIANTS, variants
        assert len(data) == 8, len(data)
        for d in data:
            for key in ("variant", "system_id", "document_id", "run_id",
                        "pipeline_config", "elements", "relations",
                        "findings", "raw_candidates", "candidates", "usage"):
                assert key in d, key
        return f"{len(data)} records, variants match the paper exactly"

    case("rq2 output shape", rq2_shape)

    def rq2_evidence_ablation():
        data = json.load(open(rq2_out, encoding="utf-8"))
        ev = [d for d in data
              if d["variant"] == "Without Evidence Validation"][0]
        cfg = ev["pipeline_config"]
        assert cfg["evidence_validation"] is False
        assert cfg["coverage_validation"] is False, cfg
        assert cfg["boundary_validation"] is False, cfg
        # every nominated finding must now be eligible for generation
        elig = [f for f in ev["findings"] if f["evidence_decision"] == "YES"
                and f["coverage_decision"] == "NO"
                and f["boundary_decision"] == "YES"]
        assert len(elig) == len(ev["findings"]), (len(elig), len(ev["findings"]))
        return (f"all three validators bypassed; "
                f"{len(elig)}/{len(ev['findings'])} findings eligible")

    case("rq2 Without Evidence Validation = paper definition",
         rq2_evidence_ablation)

    def rq2_extended():
        cfgs = rq2.build_ablation_configs(3, include_extended=True)
        assert len(cfgs) == 14, len(cfgs)
        assert list(cfgs)[:8] == rq2.PAPER_VARIANTS
        return f"extended mode returns {len(cfgs)} variants"

    case("rq2 --include-extended", rq2_extended)

    def rq2_shares_one_graph_per_variant_group():
        """
        Eight variants must not re-extract the graph eight times.

        Besides multiplying the cost, redrawing the graph per variant leaves
        each variant comparing a different graph, because the endpoint is not
        deterministic even at temperature 0. Only the two preparation
        settings may trigger a fresh graph.
        """
        before = MockLLM.element_calls

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "rq2b.json")
            rq2.run_rq2(dataset_path, out, repeated_runs=1,
                        dependency_budget=3)
            data = json.load(open(out, encoding="utf-8"))

        # The dataset has 3 requirements -> 3 element-extraction calls per
        # preparation. Eight variants must therefore cost 2 preparations
        # (cross-requirement on / off), i.e. 6 calls, not 24.
        spent = MockLLM.element_calls - before
        assert spent == 6, (
            f"expected 2 preparations = 6 element calls for 8 variants, "
            f"got {spent}")

        keys = {tuple(d["preparation_key"]) for d in data}
        assert keys == {(True, 12), (False, 12)}, keys

        by_key = {}
        for record in data:
            by_key.setdefault(tuple(record["preparation_key"]), []).append(
                record["variant"])

        assert len(by_key[(True, 12)]) == 7, by_key
        assert by_key[(False, 12)] == ["Without Cross-Req. Extraction"], by_key

        # Variants sharing a preparation key must have seen the same graph.
        graphs = {}
        for record in data:
            key = tuple(record["preparation_key"])
            signature = tuple(sorted(
                (r["source"], r["target"], r["relation_type"])
                for r in record["relations"]))
            graphs.setdefault(key, set()).add(signature)
        for key, seen in graphs.items():
            assert len(seen) == 1, (
                f"variants under {key} saw {len(seen)} different graphs")

        return (f"8 variants -> 2 preparations (6 element calls, not 24); "
                f"all same-group variants share one identical graph")

    case("rq2 shares one graph per preparation group",
         rq2_shares_one_graph_per_variant_group)

    def rq2_failure_isolation():
        """A failing variant must not abort the other seven."""
        import tempfile

        original_run = dioreq.DIOReqPipeline.run
        calls = {"n": 0}

        def boom_on_first(self, document_text, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("endpoint exploded")
            return original_run(self, document_text, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "rq2c.json")
            dioreq.DIOReqPipeline.run = boom_on_first
            try:
                rq2.run_rq2(dataset_path, out, repeated_runs=1,
                            dependency_budget=3)
            finally:
                dioreq.DIOReqPipeline.run = original_run
            data = json.load(open(out, encoding="utf-8"))

        assert len(data) == 8, f"expected 8 records, got {len(data)}"
        assert sum(1 for d in data if "error" in d) == 1, data
        assert sum(1 for d in data if "error" not in d) == 7, data
        return "one variant failed, seven still produced full output"

    case("rq2 run-level failure isolation", rq2_failure_isolation)

    def rq3_failure_isolation():
        """A failing ranking condition must not abort the rest."""
        import tempfile

        original = rq3.run_one_ranking_condition
        calls = {"n": 0}

        def boom_on_first(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("endpoint exploded")
            return original(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "rq3b.json")
            rq3.run_one_ranking_condition = boom_on_first
            try:
                rq3.run_rq3(dataset_path, out, budgets=[3],
                            random_repetitions=1)
            finally:
                rq3.run_one_ranking_condition = original
            data = json.load(open(out, encoding="utf-8"))

        # 5 signals at one budget with 1 random repetition -> 5 conditions.
        assert len(data) == 5, f"expected 5 conditions, got {len(data)}"
        assert sum(1 for d in data if "error" in d) == 1, data
        assert sum(1 for d in data if "error" not in d) == 4, data
        return "one condition failed, four still produced full output"

    case("rq3 run-level failure isolation", rq3_failure_isolation)

    print()
    print("=" * 70)
    print("B. rq3.py end-to-end (2 budgets, 2 random repetitions)")
    print("=" * 70)
    rq3_out = os.path.join(OUTDIR, "rq3.json")
    MockLLM.element_calls = 0
    MockLLM.relation_calls = 0
    MockLLM.other_calls = 0

    case("rq3.run_rq3 executes", lambda: (
        rq3.run_rq3(dataset_path, rq3_out, budgets=[3, 5],
                    random_repetitions=2) or "done"))

    def rq3_shape():
        data = json.load(open(rq3_out, encoding="utf-8"))
        expected = 2 * (4 * 1 + 2)
        assert len(data) == expected, (len(data), expected)
        for d in data:
            for key in ("system_id", "document_id", "run_id",
                        "ranking_signal", "nominal_budget",
                        "effective_budget", "record_pool_size",
                        "selected_records", "findings", "valid_record_ids",
                        "auto_valid_record_count", "auto_valid_precision",
                        "candidates", "elements", "relations",
                        "computational_exclusions"):
                assert key in d, key
            assert "vfp" not in d, "the automatic proxy must not be called vfp"
        return f"{len(data)} conditions for 1 doc x 2 budgets"

    case("rq3 output shape", rq3_shape)

    def rq3_budget_semantics():
        data = json.load(open(rq3_out, encoding="utf-8"))
        seen = {}
        for d in data:
            if d["ranking_signal"] != "random":
                seen[d["nominal_budget"]] = d["effective_budget"]
        for budget, eff in seen.items():
            assert eff <= budget, (budget, eff)
        for d in data:
            assert len(d["selected_records"]) == d["effective_budget"], d
        return f"effective_budget = min(B, N_d) honoured for {seen}"

    case("rq3 budget applied before nomination", rq3_budget_semantics)

    def rq3_prepared_once():
        """The decisive C1 check: one graph preparation per document."""
        return (f"element-extraction calls = {MockLLM.element_calls} "
                f"for 1 document with 3 requirements "
                f"(12 ranking conditions)")

    case("rq3 prepares the document once", rq3_prepared_once)

    def rq3_c1_assertion():
        assert MockLLM.element_calls == 3, (
            f"expected 3 extraction calls (one per requirement, one "
            f"preparation), got {MockLLM.element_calls} -> the graph is "
            f"being re-extracted per condition")
        return "confirmed: graph extracted once, not 12 times"

    case("rq3 C1 fix verified", rq3_c1_assertion)

    def rq3_input_validation():
        for kwargs, msg in [
            (dict(budgets=[], random_repetitions=2), "empty budgets"),
            (dict(budgets=[0], random_repetitions=2), "zero budget"),
            (dict(budgets=[5], random_repetitions=0), "zero repetitions"),
        ]:
            try:
                rq3.run_rq3(dataset_path, rq3_out, **kwargs)
                raise AssertionError(f"{msg} was not rejected")
            except ValueError:
                pass
        return "empty/zero budgets and repetitions rejected"

    case("rq3 argument validation", rq3_input_validation)

    print()
    print("=" * 70)
    print("C. dioreq.run_dioreq end-to-end, including Stage-5 refined documents")
    print("=" * 70)
    dioreq_out = os.path.join(OUTDIR, "dioreq.json")
    refined_dir = os.path.join(OUTDIR, "refined")
    MockLLM.element_calls = 0

    case("dioreq.run_dioreq executes", lambda: (
        dioreq.run_dioreq(dataset_path, dioreq_out, repeated_runs=1,
                          dependency_budget=3, refined_dir=refined_dir)
        or "done"))

    def refined_written():
        names = sorted(os.listdir(refined_dir))
        assert len(names) == 3, names
        stem = [n for n in names if n.endswith("_refined.txt")][0].rsplit(
            "_refined.txt", 1)[0]

        text = open(os.path.join(refined_dir, f"{stem}_refined.txt"),
                    encoding="utf-8").read()
        original = DATASET[0]["text"].strip()
        assert original in text, "original document must be preserved"
        # 1.1 / 1.2 / 1.3 exist in the source, so new numbers start at 1.4
        assert "1.4 " in text or "1.4\n" in text, text[-400:]
        assert text.count("Requirement:") == 1, text

        dc = open(os.path.join(refined_dir, f"{stem}_design_constraints.txt"),
                  encoding="utf-8").read()
        assert "DESIGN CONSTRAINTS AND REJECTED CANDIDATES" in dc
        assert "DC-001" in dc, dc
        assert "mock implementation detail" in dc
        assert "REMOVED-001" in dc, dc

        rep = open(os.path.join(refined_dir, f"{stem}_refinement_report.md"),
                   encoding="utf-8").read()
        assert "# Stage 5 - Requirement Refinement Report" in rep
        assert "Demoted to design constraints" in rep
        return (f"{len(names)} files; refined keeps 1, 1 design constraint, "
                f"report present")

    case("refined requirements document written", refined_written)

    def three_way_outcome():
        data = json.load(open(dioreq_out, encoding="utf-8"))
        record = data[0]
        raw = [c["candidate_id"] for c in record["raw_candidates"]]

        assert len(record["candidates"]) == 1, record["candidates"]
        assert len(record["design_constraints"]) == 1, record["design_constraints"]
        assert len(record["removed_candidates"]) >= 1, record["removed_candidates"]

        # Groups are not candidates: a kept group can absorb several raw
        # candidates. The invariant that matters is that every raw candidate
        # id is accounted for exactly once.
        claimed = []
        for group in (record["candidates"]
                      + record["design_constraints"]
                      + record["removed_candidates"]):
            claimed += group["source_candidate_ids"]

        assert sorted(claimed) == sorted(raw), (
            f"claimed={sorted(claimed)} raw={sorted(raw)}")
        assert len(claimed) == len(set(claimed)), (
            f"a candidate was claimed twice: {claimed}")

        merged = max(len(group["source_candidate_ids"])
                     for group in record["candidates"])
        assert merged > 1, "the merge path was not exercised"

        return (f"kept={len(record['candidates'])} "
                f"demoted={len(record['design_constraints'])} "
                f"removed={len(record['removed_candidates'])} groups cover "
                f"all {len(raw)} raw candidate(s) exactly once "
                f"(largest merge={merged})")

    case("KEEP/DEMOTE/REMOVE covers every candidate id", three_way_outcome)

    def refined_opt_out():
        alt = os.path.join(OUTDIR, "dioreq_noref.json")
        dioreq.run_dioreq(dataset_path, alt, repeated_runs=1,
                          dependency_budget=3, refined_dir=None)
        data = json.load(open(alt, encoding="utf-8"))
        assert len(data) == 1
        assert "candidates" in data[0]
        return "refined_dir=None skips the text output"

    case("refined documents can be disabled", refined_opt_out)

    print()
    print("=" * 70)
    print(f"mock call tally: elements={MockLLM.element_calls}, "
          f"relations={MockLLM.relation_calls}, other={MockLLM.other_calls}")
    print(f"RESULT: {len(fails)} failure(s)")
    for n, e in fails:
        print(f"  - {n}: {type(e).__name__}: {e}")

    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
