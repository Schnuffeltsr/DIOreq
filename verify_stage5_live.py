# verify_stage5_live.py
"""
Re-run only Stage 5 (consolidation) of a completed DIOReq run against the
saved raw candidates, to check the merge/traceability invariant on real
model output without paying for all 184 calls again.

Usage:
    set OPENAI_API_KEY=...
    set DIOREQ_VALIDATION_MODEL=gpt-5.6-sol
    python verify_stage5_live.py _tmp/out/simple1.json
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import code_paths

code_paths.install()

import dioreq


def main() -> None:
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "_tmp/out/simple1.json")
    dataset_path = pathlib.Path(
        sys.argv[2] if len(sys.argv) > 2 else "_tmp/dataset_simple1.json"
    )

    record = json.loads(path.read_text(encoding="utf-8"))[0]
    documents = dioreq.load_documents(dataset_path)
    document_text = next(
        item.text
        for item in documents
        if item.document_id == record["document_id"]
    )

    candidates = [
        dioreq.GeneratedCandidate(**item)
        for item in record["raw_candidates"]
    ]

    llm = dioreq.LLMClient()
    result = dioreq.consolidate_candidates(
        llm,
        document_text,
        candidates,
        dioreq.default_model_config(),
        dioreq.PipelineConfig(),
    )

    raw = [item.candidate_id for item in candidates]
    claims: dict[str, int] = {}

    for group in result.kept:
        for value in group.source_candidate_ids:
            claims[value] = claims.get(value, 0) + 1

    for group in result.demoted + result.removed:
        for value in group.get("source_candidate_ids", []):
            claims[value] = claims.get(value, 0) + 1

    missing = sorted(set(raw) - set(claims))
    duplicated = sorted(v for v, n in claims.items() if n > 1)
    unclaimed = sorted(set(claims) - set(raw))

    print(f"raw candidates     : {len(raw)}")
    print(f"kept groups        : {len(result.kept)}")
    print(f"demoted groups     : {len(result.demoted)}")
    print(f"removed groups     : {len(result.removed)}")
    print(f"largest merge      : "
          f"{max((len(g.source_candidate_ids) for g in result.kept), default=0)}")
    print(f"ids claimed        : {len(claims)}")
    print(f"missing ids        : {len(missing)} {missing[:5]}")
    print(f"duplicated ids     : {len(duplicated)} {duplicated[:5]}")
    print(f"unknown ids        : {len(unclaimed)} {unclaimed[:5]}")
    print(f"api calls          : {llm.usage.api_calls}")

    ok = not missing and not duplicated and not unclaimed
    print(f"\nPARTITION INVARIANT: {'OK' if ok else 'VIOLATED'}")

    for group in result.kept[:3]:
        print(f"\n- ({len(group.source_candidate_ids)} merged) "
              f"{group.requirement[:110]}")
        print(f"  ids: {group.source_candidate_ids}")


if __name__ == "__main__":
    main()
