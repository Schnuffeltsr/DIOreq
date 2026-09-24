# DIOReq prompt templates

Every prompt that `dioreq.py` sends to a language model, extracted verbatim.

The `.txt` files hold the **exact strings the code sends**: each one is byte-identical to the corresponding string constant or literal in `dioreq.py`. Two consequences of that fidelity are worth knowing before you copy a template:

- Multi-line templates begin and end with the newline that comes from the source triple-quoted literal.
- Single-line system prompts have no trailing newline at all.

Placeholders written `{like_this}` are substituted by `.format(...)` at the call site. Literal JSON braces are escaped as `{{` and `}}` in the template and appear as `{` and `}` in the string that is actually sent.

## Stage mapping

The pipeline divides into five stages. Section 3 of the paper describes them as five subsections, which is the directly executable decomposition, and this directory follows that grouping.

| Directory | Paper section | Stage | Prompt files |
|---|---|---|---|
| `stage1_semantic_dependency_graph/` | 3.1 | Typed Directional Semantic Dependency Graph | 6 |
| `stage2_diagnostic_propagation_dps/` | 3.2 | Diagnostic Propagation and DPS | none &mdash; deterministic, no LLM call |
| `stage3_multi_view_nomination/` | 3.3 | Multi-View Candidate Nomination | 6 |
| `stage4_validation/` | 3.4 | Evidence, Coverage, and Boundary Validation | 2 |
| `stage5_generation_consolidation/` | 3.5 | Constrained Requirement Generation and Consolidation | 4 |

## Files

| File | Type | Role | Called from | Placeholders |
|---|---|---|---|---|
| `stage1_semantic_dependency_graph/element_extraction.system.txt` | system prompt | Typed requirement-element extraction (one call per requirement block) | `extract_elements()` L834 | &mdash; |
| `stage1_semantic_dependency_graph/element_extraction.user.txt` | user template | Typed requirement-element extraction (one call per requirement block) | `extract_elements()` L834 | `{requirement_id}`, `{requirement_text}` |
| `stage1_semantic_dependency_graph/relation_extraction.cross.system.txt` | system prompt | Pass 2 - cross-requirement directional relations | `extract_dependency_relations()` L1132 | &mdash; |
| `stage1_semantic_dependency_graph/relation_extraction.cross.user.txt` | user template | Pass 2 - cross-requirement directional relations | `extract_dependency_relations()` L1132 | `{elements}`, `{existing_relations}`, `{disconnected_elements}`, `{document_text}` |
| `stage1_semantic_dependency_graph/relation_extraction.intra.system.txt` | system prompt | Pass 1 - intra-requirement directional relations | `extract_dependency_relations()` L1098 | &mdash; |
| `stage1_semantic_dependency_graph/relation_extraction.intra.user.txt` | user template | Pass 1 - intra-requirement directional relations | `extract_dependency_relations()` L1098 | `{elements}`, `{requirement_id}`, `{requirement_text}` |
| `stage3_multi_view_nomination/dependency_nomination.system.txt` | system prompt | Dependency-view gap nomination | `nominate_dependency_findings()` L1992 | &mdash; |
| `stage3_multi_view_nomination/dependency_nomination.user.txt` | user template | Dependency-view gap nomination | `nominate_dependency_findings()` L1992 | `{source}`, `{target}`, `{path}`, `{relation_types}`, `{patterns}`, `{source_requirements}` |
| `stage3_multi_view_nomination/isolation_nomination.system.txt` | system prompt | Isolation-view gap nomination | `nominate_isolation_findings()` L2065 | &mdash; |
| `stage3_multi_view_nomination/isolation_nomination.user.txt` | user template | Isolation-view gap nomination | `nominate_isolation_findings()` L2065 | `{element}`, `{element_type}`, `{source_requirements}`, `{supporting_excerpts}`, `{document_text}` |
| `stage3_multi_view_nomination/operation_nomination.system.txt` | system prompt | Operation-view gap nomination | `nominate_operation_findings()` L2135 | &mdash; |
| `stage3_multi_view_nomination/operation_nomination.user.txt` | user template | Operation-view gap nomination | `nominate_operation_findings()` L2135 | `{elements}`, `{document_text}` |
| `stage4_validation/validation.system.txt` | system prompt | Evidence / coverage / boundary validation | `validate_findings()` L2235 | &mdash; |
| `stage4_validation/validation.user.txt` | user template | Evidence / coverage / boundary validation | `validate_findings()` L2235 | `{finding}`, `{document_text}` |
| `stage5_generation_consolidation/consolidation.system.txt` | system prompt | Cross-view consolidation, refinement, deduplication, filtering | `consolidate_candidates()` L2477 | &mdash; |
| `stage5_generation_consolidation/consolidation.user.txt` | user template | Cross-view consolidation, refinement, deduplication, filtering | `consolidate_candidates()` L2477 | `{candidates}`, `{document_text}`, `{merge}`, `{refine}`, `{deduplicate}`, `{filtering}` |
| `stage5_generation_consolidation/generation.system.txt` | system prompt | Constrained requirement generation | `generate_candidates()` L2396 | &mdash; |
| `stage5_generation_consolidation/generation.user.txt` | user template | Constrained requirement generation | `generate_candidates()` L2396 | `{finding}`, `{source_requirements}`, `{element_names}` |

## Notes

- The relation-extraction system prompt is identical for both extraction passes. It is duplicated into the two call-site files so that each pair of templates is self-contained; the two passes differ only in their user template.
- `validation.user.txt` returns three independent decisions from a single call (`evidence`, `coverage`, `boundary`), each `YES`, `NO`, or `UNCERTAIN`. Automatic generation requires `YES` / `NO` / `YES`.
- `consolidation.user.txt` receives the four pipeline switches (`merge`, `refine`, `deduplicate`, `filtering`) as booleans. That is how the RQ2 consolidation ablations change behaviour without editing the prompt text.

## Scope: where the pipeline stops

Stage 5 ends with consolidation. The **analyst review that follows is a human step and has no prompt template**, because the framework deliberately stops at evidence-linked proposals:

- Section 3 introduction: outputs "are therefore treated as evidence-linked proposals for analyst review rather than as logically necessary consequences of the source document".
- Section 3.4: findings whose coverage or responsibility is uncertain "are retained for analyst review but are not automatically forwarded to the generator".
- Section 3.5: "Final acceptance, rejection, or modification remains the responsibility of requirements analysts and domain stakeholders".
- Section 6 lists "human-in-the-loop review settings" as future work.

The released pipeline therefore produces candidates and stops. No model call is made for acceptance, rejection, or modification, and no prompt template exists for it by design rather than by omission.

Regenerate with `python build_prompts.py` from the repository root. The script re-reads `dioreq.py`, so the templates cannot drift from the implementation.
