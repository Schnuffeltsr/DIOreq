# rq2.py
"""
RQ2 runner: component and diagnostic-view ablations.

By default this reproduces exactly the eight variants reported in the
paper's Section 4.3 / Table 4. Finer-grained sub-ablations are available
through ``--include-extended`` and are kept out of the default output so
that the default run matches the paper one-to-one.

Note on "Without Evidence Validation": the paper defines this variant as
bypassing evidence, coverage, and boundary validation together. This
runner therefore disables all three validators for that variant, which is
what makes ``eligible_for_generation()`` accept every nominated finding.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path


def _import_dioreq():
    """
    Import the shared method module, wherever this runner happens to live.

    The released layout keeps the method in ``DIOReq/RQ1/dioreq.py`` and the
    RQ2 runner in ``DIOReq/RQ2/``. A flat development checkout keeps all
    three modules side by side. Resolving the directory instead of relying
    on the working directory means ``python DIOReq/RQ2/rq2.py`` works from
    the repository root, from inside the folder, and from anywhere else.
    """
    here = Path(__file__).resolve().parent

    for candidate in (
        here,                          # flat checkout
        here.parent / "RQ1",           # DIOReq/RQ2 -> DIOReq/RQ1
        here / "DIOReq" / "RQ1",       # repository root
        here.parent,                   # DIOReq/RQ2 -> DIOReq
        here.parent.parent,            # DIOReq/RQ2 -> repository root
    ):
        if (candidate / "dioreq.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return

    raise ImportError(
        "dioreq.py not found next to this file or in a sibling RQ1 folder. "
        "Expected DIOReq/RQ1/dioreq.py relative to DIOReq/RQ2/rq2.py."
    )


_import_dioreq()

from dioreq import (
    DIOReqPipeline,
    PipelineConfig,
    add_client_arguments,
    add_shard_argument,
    apply_cli_environment,
    load_documents,
    parse_shard,
    resolve_client,
    save_json,
    shard_documents,
)


BASE_CONFIG = PipelineConfig(
    dependency_view=True,
    isolation_view=True,
    operation_view=True,
    cross_requirement_extraction=True,
    use_dps_ranking=True,
    evidence_validation=True,
    coverage_validation=True,
    boundary_validation=True,
    consolidation=True,
    refinement=True,
    deduplication=True,
    final_filtering=True,
    dependency_budget=20,
    max_parents=12,
)


# Ordered exactly as in the paper's Table 4.
PAPER_VARIANTS = [
    "Full DIOReq",
    "Without Dependency View",
    "Without Operation View",
    "Without Isolation View",
    "Without Cross-Req. Extraction",
    "DPS \u2192 Unranked",
    "Without Evidence Validation",
    "Without Consolidation and Refinement",
]

EXTENDED_VARIANTS = [
    "Without Coverage Validation",
    "Without Boundary Validation",
    "Without Merge",
    "Without Refinement",
    "Without Deduplication",
    "Without Final Filtering",
]


def build_ablation_configs(
    dependency_budget: int,
    include_extended: bool = False,
) -> dict[str, PipelineConfig]:
    """
    Build the ablation conditions.

    Parameters
    ----------
    dependency_budget:
        Nominal pre-validation dependency budget for every variant.
    include_extended:
        When True, append the finer-grained sub-ablations that are not
        reported in the paper. When False (default), the returned mapping
        contains exactly the paper's eight Table 4 variants, in order.
    """
    base = replace(
        BASE_CONFIG,
        dependency_budget=dependency_budget,
    )

    configs: dict[str, PipelineConfig] = {
        "Full DIOReq": base,

        "Without Dependency View": replace(
            base,
            dependency_view=False,
        ),

        "Without Operation View": replace(
            base,
            operation_view=False,
        ),

        "Without Isolation View": replace(
            base,
            isolation_view=False,
        ),

        "Without Cross-Req. Extraction": replace(
            base,
            cross_requirement_extraction=False,
        ),

        "DPS \u2192 Unranked": replace(
            base,
            use_dps_ranking=False,
        ),

        # Paper definition: bypasses evidence, coverage AND boundary
        # validation together.
        "Without Evidence Validation": replace(
            base,
            evidence_validation=False,
            coverage_validation=False,
            boundary_validation=False,
        ),

        "Without Consolidation and Refinement": replace(
            base,
            consolidation=False,
            refinement=False,
            deduplication=False,
            final_filtering=False,
        ),
    }

    if include_extended:
        configs.update({
            "Without Coverage Validation": replace(
                base,
                coverage_validation=False,
            ),

            "Without Boundary Validation": replace(
                base,
                boundary_validation=False,
            ),

            "Without Merge": replace(
                base,
                consolidation=False,
            ),

            "Without Refinement": replace(
                base,
                refinement=False,
            ),

            "Without Deduplication": replace(
                base,
                deduplication=False,
            ),

            "Without Final Filtering": replace(
                base,
                final_filtering=False,
            ),
        })

    return configs


def preparation_key(config: PipelineConfig) -> tuple:
    """
    The settings that change what prepare_document produces.

    Everything else in PipelineConfig acts after preparation, so variants
    that agree on this key must share a graph.
    """
    return (
        config.cross_requirement_extraction,
        config.max_parents,
    )


def run_rq2(
    dataset_path: str,
    output_path: str,
    repeated_runs: int,
    dependency_budget: int,
    include_extended: bool = False,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    shard: tuple[int, int] | None = None,
) -> None:
    if repeated_runs < 1:
        raise ValueError("repeated_runs must be at least 1")

    if dependency_budget < 1:
        raise ValueError("dependency_budget must be at least 1")

    documents = shard_documents(load_documents(dataset_path), shard)

    if not documents:
        print("No documents assigned to this shard; nothing to do.")
        return

    llm, model_config = resolve_client(
        model=model, base_url=base_url, api_key=api_key
    )

    configs = build_ablation_configs(
        dependency_budget=dependency_budget,
        include_extended=include_extended,
    )

    results: list[dict] = []

    total_runs = len(configs) * len(documents) * repeated_runs
    completed_runs = 0
    failures = 0
    preparations = 0

    for document in documents:
        # One preparation per repetition per distinct preparation setting,
        # shared by every variant that only differs downstream. Re-extracting
        # the graph per variant would both multiply the cost by the number of
        # variants and leave each variant comparing a different graph, since
        # the endpoint is not deterministic even at temperature 0.
        for run_id in range(1, repeated_runs + 1):
            prepared_cache: dict[tuple, tuple] = {}

            for variant_name, config in configs.items():
                completed_runs += 1

                key = preparation_key(config)

                if key not in prepared_cache:
                    prepared_cache[key] = DIOReqPipeline(
                        llm=llm,
                        model_config=model_config,
                        pipeline_config=config,
                    ).prepare_document(document.text)
                    preparations += 1

                print(
                    f"[RQ2] run={completed_runs}/{total_runs}, "
                    f"variant={variant_name}, "
                    f"document={document.document_id}, "
                    f"repetition={run_id}"
                )

                pipeline = DIOReqPipeline(
                    llm=llm,
                    model_config=model_config,
                    pipeline_config=config,
                )

                try:
                    output = pipeline.run(
                        document_text=document.text,
                        ranking_strategy=(
                            "dps"
                            if config.use_dps_ranking
                            else "unranked"
                        ),
                        random_seed=run_id,
                        prepared=prepared_cache[key],
                        progress=lambda message: print(f"          {message}"),
                    )
                except Exception as error:
                    failures += 1
                    print(
                        f"          FAILED ({type(error).__name__}): "
                        f"{error}"
                    )
                    results.append({
                        "variant": variant_name,
                        "system_id": document.system_id,
                        "document_id": document.document_id,
                        "run_id": run_id,
                        "pipeline_config": asdict(config),
                        "error": f"{type(error).__name__}: {error}",
                    })
                    save_json(output_path, results)
                    continue

                results.append({
                    "variant": variant_name,
                    "system_id": document.system_id,
                    "document_id": document.document_id,
                    "run_id": run_id,
                    "pipeline_config": asdict(config),
                    "preparation_key": list(key),
                    **output,
                })

                save_json(output_path, results)

    print(
        f"RQ2 completed: {total_runs - failures}/{total_runs} runs over "
        f"{len(configs)} variants ({preparations} graph preparation(s)); "
        f"results saved to {output_path}"
    )

    if failures:
        print(
            f"{failures} run(s) failed and are recorded with an "
            f"'error' field in {output_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RQ2 component and diagnostic-view ablations."
        )
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output",
        default="results/rq2_raw.json",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of independent runs per document and variant.",
    )
    parser.add_argument(
        "--dependency-budget",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help=(
            "Also run the finer-grained sub-ablations that are not "
            "reported in the paper."
        ),
    )

    add_client_arguments(parser)
    add_shard_argument(parser)

    args = parser.parse_args()

    apply_cli_environment(args)

    run_rq2(
        dataset_path=args.dataset,
        output_path=args.output,
        repeated_runs=args.runs,
        dependency_budget=args.dependency_budget,
        include_extended=args.include_extended,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        shard=parse_shard(args.shard),
    )


if __name__ == "__main__":
    main()
