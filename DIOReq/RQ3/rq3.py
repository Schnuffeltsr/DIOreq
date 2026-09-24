# rq3.py
"""
RQ3 runner: equal-budget comparison of ranking signals.

The document is prepared ONCE per document and the resulting elements,
relations, graph, and source-target record pool are reused by every ranking
condition, so the conditions share the same extracted graph and differ only
in the ordering signal.

The budget is applied before nomination and before validation. Precision is
reported as ``auto_valid_record_count`` and ``auto_valid_precision``,
computed over the ``min(B, N_d)`` selected records.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path


def _import_dioreq():
    """
    Import the shared method module, wherever this runner happens to live.

    See ``rq2.py`` for the same resolution: the released layout keeps the
    method in ``DIOReq/RQ1/dioreq.py`` and this runner in ``DIOReq/RQ3/``.
    """
    here = Path(__file__).resolve().parent

    for candidate in (
        here,                          # flat checkout
        here.parent / "RQ1",           # DIOReq/RQ3 -> DIOReq/RQ1
        here / "DIOReq" / "RQ1",       # repository root
        here.parent,                   # DIOReq/RQ3 -> DIOReq
        here.parent.parent,            # DIOReq/RQ3 -> repository root
    ):
        if (candidate / "dioreq.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return

    raise ImportError(
        "dioreq.py not found next to this file or in a sibling RQ1 folder. "
        "Expected DIOReq/RQ1/dioreq.py relative to DIOReq/RQ3/rq3.py."
    )


_import_dioreq()

from dioreq import (
    DIOReqPipeline,
    PipelineConfig,
    add_client_arguments,
    add_shard_argument,
    apply_cli_environment,
    generate_candidates,
    load_documents,
    nominate_dependency_findings,
    parse_shard,
    rank_records,
    resolve_client,
    save_json,
    shard_documents,
    validate_findings,
)


RANKING_SIGNALS = [
    "dps",
    "extraction_support",
    "topology",
    "unranked",
    "random",
]


def build_rq3_config(
    dependency_budget: int,
) -> PipelineConfig:
    """
    Dependency-view-only configuration used for every ranking condition.

    Only the ranking signal changes across conditions. Isolation and
    operation views are disabled so that their reference targets are not
    attributed to the ranking signal.
    """
    return PipelineConfig(
        dependency_view=True,
        isolation_view=False,
        operation_view=False,
        cross_requirement_extraction=True,
        use_dps_ranking=True,
        evidence_validation=True,
        coverage_validation=True,
        boundary_validation=True,
        consolidation=False,
        refinement=False,
        deduplication=False,
        final_filtering=False,
        dependency_budget=dependency_budget,
        max_parents=12,
    )


def run_one_ranking_condition(
    pipeline: DIOReqPipeline,
    prepared: tuple,
    document_text: str,
    ranking_signal: str,
    budget: int,
    seed: int,
) -> dict:
    """
    Evaluate one ranking signal at one nominal budget.

    ``prepared`` is the (elements, relations, graphs, records) tuple
    produced once per document by ``DIOReqPipeline.prepare_document``.
    Reusing it keeps the graph, the review patterns, the validation
    procedure, the generator, and the model settings identical across
    conditions; only the ordering of the record pool changes.

    The budget is applied to the eligible record pool before relation-aware
    gap nomination and before validation.
    """
    elements, relations, graphs, records = prepared

    ranked_records = rank_records(
        records,
        strategy=ranking_signal,
        seed=seed,
    )

    effective_budget = min(budget, len(ranked_records))
    selected_records = ranked_records[:effective_budget]

    findings = nominate_dependency_findings(
        llm=pipeline.llm,
        document_text=document_text,
        graphs=graphs,
        records=selected_records,
        model_config=pipeline.model_config,
    )

    validated_findings = validate_findings(
        llm=pipeline.llm,
        document_text=document_text,
        findings=findings,
        model_config=pipeline.model_config,
        config=pipeline.config,
    )

    valid_finding_count = sum(
        1
        for finding in validated_findings
        if finding.eligible_for_generation()
    )

    # A selected source-target region counts as automatically valid when at
    # least one finding anchored on that region passes all three validators.
    valid_record_ids = set()

    for record in selected_records:
        for finding in validated_findings:
            if (
                finding.source_element == record.source
                and finding.target_element == record.target
                and finding.eligible_for_generation()
            ):
                valid_record_ids.add(record.record_id)
                break

    auto_valid_record_count = len(valid_record_ids)
    auto_valid_precision = (
        auto_valid_record_count / effective_budget
        if effective_budget > 0
        else 0.0
    )

    candidates = generate_candidates(
        llm=pipeline.llm,
        document_text=document_text,
        elements=elements,
        findings=validated_findings,
        model_config=pipeline.model_config,
    )

    return {
        "ranking_signal": ranking_signal,
        "nominal_budget": budget,
        "effective_budget": effective_budget,
        "record_pool_size": len(records),
        "selected_records": [
            asdict(record)
            for record in selected_records
        ],
        "findings": [
            asdict(finding)
            for finding in validated_findings
        ],
        "valid_record_ids": sorted(valid_record_ids),
        "auto_valid_record_count": auto_valid_record_count,
        "auto_valid_precision": auto_valid_precision,
        "valid_finding_count": valid_finding_count,
        "candidates": [
            asdict(candidate)
            for candidate in candidates
        ],
        "elements": [
            asdict(element)
            for element in elements
        ],
        "relations": [
            asdict(relation)
            for relation in relations
        ],
        "computational_exclusions": list(graphs.exclusions),
    }


def run_rq3(
    dataset_path: str,
    output_path: str,
    budgets: list[int],
    random_repetitions: int,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    shard: tuple[int, int] | None = None,
) -> None:
    if not budgets:
        raise ValueError("budgets must contain at least one value")

    if any(budget < 1 for budget in budgets):
        raise ValueError("every budget must be at least 1")

    if random_repetitions < 1:
        raise ValueError("random_repetitions must be at least 1")

    documents = shard_documents(load_documents(dataset_path), shard)

    if not documents:
        print("No documents assigned to this shard; nothing to do.")
        return

    llm, model_config = resolve_client(
        model=model, base_url=base_url, api_key=api_key
    )

    results = []
    failures = 0

    for document in documents:
        # Prepared once and reused by every ranking condition below. RQ3
        # holds the document, the graph and the model settings fixed and
        # varies only the ordering of the record pool, so the graph must not
        # be redrawn between conditions.
        prepared = DIOReqPipeline(
            llm=llm,
            model_config=model_config,
            pipeline_config=build_rq3_config(budgets[0]),
        ).prepare_document(document.text)

        for budget in budgets:
            for ranking_signal in RANKING_SIGNALS:
                repetitions = (
                    random_repetitions
                    if ranking_signal == "random"
                    else 1
                )

                pipeline = DIOReqPipeline(
                    llm=llm,
                    model_config=model_config,
                    pipeline_config=build_rq3_config(budget),
                )

                for repetition in range(repetitions):
                    seed = repetition

                    print(
                        f"[RQ3] signal={ranking_signal}, "
                        f"document={document.document_id}, "
                        f"budget={budget}, "
                        f"repetition={repetition + 1}/{repetitions}"
                    )

                    try:
                        condition = run_one_ranking_condition(
                            pipeline=pipeline,
                            prepared=prepared,
                            document_text=document.text,
                            ranking_signal=ranking_signal,
                            budget=budget,
                            seed=seed,
                        )
                    except Exception as error:
                        failures += 1
                        print(
                            f"          FAILED ({type(error).__name__}): "
                            f"{error}"
                        )
                        results.append({
                            "system_id": document.system_id,
                            "document_id": document.document_id,
                            "run_id": repetition + 1,
                            "ranking_signal": ranking_signal,
                            "nominal_budget": budget,
                            "error": f"{type(error).__name__}: {error}",
                        })
                        save_json(output_path, results)
                        continue

                    results.append({
                        "system_id": document.system_id,
                        "document_id": document.document_id,
                        "run_id": repetition + 1,
                        **condition,
                    })

                    save_json(output_path, results)

    print(
        f"RQ3 completed: {len(results) - failures}/{len(results)} "
        f"conditions; results saved to {output_path}"
    )

    if failures:
        print(
            f"{failures} condition(s) failed and are recorded with an "
            f"'error' field in {output_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RQ3 equal-budget comparison of ranking signals."
        )
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output",
        default="results/rq3_raw.json",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help="Nominal pre-validation budgets, as in the paper's Table 6.",
    )
    parser.add_argument(
        "--random-repetitions",
        type=int,
        default=30,
        help="Repetitions averaged for the random ranking signal.",
    )

    add_client_arguments(parser)
    add_shard_argument(parser)

    args = parser.parse_args()

    apply_cli_environment(args)

    run_rq3(
        dataset_path=args.dataset,
        output_path=args.output,
        budgets=args.budgets,
        random_repetitions=args.random_repetitions,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        shard=parse_shard(args.shard),
    )


if __name__ == "__main__":
    main()
