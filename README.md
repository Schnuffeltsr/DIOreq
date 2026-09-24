# DIOReq

Reference implementation of **DIOReq**: multi-view dependency-guided
requirement-gap diagnosis and evidence-grounded completion.

The method reads a natural-language requirements specification, builds a
typed directional dependency graph over it, propagates documentary support to
score dependency regions, diagnoses missing obligations through three
independent views, validates every candidate finding against the source
document, and finally emits a *refined requirements document* in which the
accepted supplementary requirements continue the numbering the source
document already uses.

This repository contains the proposed method and its three experiment
runners. Comparison baselines are not included.

---

## Contents

```
DIOreq/
├── README.md
├── requirements.txt
├── code_paths.py              resolves the source directories for the tests
│
├── build_dataset.py           .docx corpus  ->  JSON dataset
├── build_prompts.py           extracts every prompt from dioreq.py
├── compare_fr_counts.py       segmenter regression check against the audit
│
├── test_dioreq_logic.py       LLM-free unit tests
├── test_runners.py            end-to-end tests against a mocked LLM
│
├── fr_counts_audit.json       per-document requirement counts, audited by
│                              hand before the segmenter was written
│
├── prompts/                   every prompt template, by pipeline section
├── smoke_run/                 real end-to-end runs kept as evidence
│
└── DIOReq/
    ├── RQ1/dioreq.py          the method + RQ1 runner
    ├── RQ2/rq2.py             component and view ablations
    ├── RQ3/rq3.py             equal-budget ranking comparison
    ├── Data/<Project>/Document/*.docx       42 source specifications
    └── Referencedata/<Project>/*.docx       42 reference specifications
```

`DIOReq/RQ1/dioreq.py` is the shared module. `rq2.py` and `rq3.py` import it
directly from the sibling folder, so all three run from any working
directory:

```bat
python DIOReq\RQ1\dioreq.py --help
python DIOReq\RQ2\rq2.py    --help
python DIOReq\RQ3\rq3.py    --help
```

---

## Installation

Python 3.10 or newer.

```bat
pip install -r requirements.txt
```

The method needs an OpenAI-compatible chat-completions endpoint that
supports `response_format={"type": "json_object"}`.

### Configuration

Credentials and the model are read from the environment, so a key never has
to appear on the command line:

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | API key |
| `OPENAI_BASE_URL` | Optional endpoint override for an OpenAI-compatible service |
| `DIOREQ_EXTRACTION_MODEL` | Model for element and relation extraction |
| `DIOREQ_VALIDATION_MODEL` | Model for nomination, validation, consolidation |
| `DIOREQ_GENERATION_MODEL` | Model for requirement generation |
| `DIOREQ_EXTRACTION_TEMPERATURE` | Sampling temperature for element and relation extraction (default `0.0`) |
| `DIOREQ_VALIDATION_TEMPERATURE` | Sampling temperature for classification and for merging, filtering and deduplication (default `0.1`) |
| `DIOREQ_GENERATION_TEMPERATURE` | Sampling temperature for requirement generation (default `0.2`) |

The stage models default to `gpt-5.5`. Every runner also accepts `--model`,
`--base-url`, `--api-key` and `--temperature`, which override the
environment. `--model` sets all three stage models at once.

The temperature defaults are the setting reported in Section 4.1.2 of the
paper. Each value covers the steps listed below; the mapping is also the
comment on `ModelConfig` in `DIOReq/RQ1/dioreq.py`.

| Stage | Step | Sampling temperature |
|---|---|---|
| 1 | Typed requirement element extraction | `0.0` — `DIOREQ_EXTRACTION_TEMPERATURE` |
| 2 | Intra- and cross-requirement dependency extraction | `0.0` — `DIOREQ_EXTRACTION_TEMPERATURE` |
| 3 | Dependency, isolation and operation view nomination (classification) | `0.1` — `DIOREQ_VALIDATION_TEMPERATURE` |
| 4 | Evidence, coverage and boundary validation (classification) | `0.1` — `DIOREQ_VALIDATION_TEMPERATURE` |
| 4 | Requirement generation | `0.2` — `DIOREQ_GENERATION_TEMPERATURE` |
| 5 | Merging, filtering and deduplication (consolidation) | `0.1` — `DIOREQ_VALIDATION_TEMPERATURE` |
| 5 | Requirement renumbering | deterministic — no model call |

Renumbering is the last step of stage 5 and is the only step in the paper's
`0.0` group that issues no request: continuing the document's own integer
sequence needs no model, so `assign_requirement_numbers()` is pure Python and
reproduces byte-identically.

A temperature of `none` omits the `temperature` field entirely. Reasoning
models that accept only their built-in temperature then work unchanged, and
so do models that intermittently reject a configured value.

```bat
set OPENAI_API_KEY=...
python DIOReq\RQ1\dioreq.py --dataset dataset_all.json ^
                            --output results\rq1.json ^
                            --model gpt-5.5 --runs 1
```

### Building a dataset

The runners read a JSON array of documents. Build one from the `.docx` corpus
first — this is the step that feeds everything else:

```bat
python build_dataset.py --source DIOReq\Data --output dataset_all.json
python build_dataset.py --source DIOReq\Referencedata --output dataset_reference.json
```

Each entry is `{"system_id": ..., "document_id": ..., "text": ...}`. The
datasets are generated, so they are not committed; the commands below assume
you have built them.

---

## The five stages and the functions that implement them

Figure 1 of the paper divides the pipeline into five stages. Section 3
describes the same pipeline as five subsections. Both maps are given below;
line numbers refer to `DIOReq/RQ1/dioreq.py`.

| Figure 1 stage | Section | Where it lives |
|---|---|---|
| 1. Typed requirement element extraction | 3.1 | `split_into_frs`, `extract_elements`, `merge_elements` |
| 2. Semantic dependency graph construction | 3.1 | `extract_dependency_relations`, `validate_relation_items`, `deduplicate_relations`, `DependencyGraphs.build` |
| 3. Dependency prioritization and multi-view diagnosis | 3.2 + 3.3 | `DependencyGraphs.dps`, `rank_records`, `nominate_*_findings` |
| 4. Evidence-validated requirement generation | 3.4 + 3.5 | `validate_findings`, `generate_candidates` |
| 5. Consolidation and analyst review | 3.5 | `consolidate_candidates`, `assign_requirement_numbers`, `build_refined_document` |

The full table, with the equation each function evaluates:

| Stage | Function | Line | Paper |
|---|---|---|---|
| 3.1 | `split_into_frs` | 469 | segments a specification into requirement blocks; picks the deepest numbering level present |
| 3.1 | `extract_elements` | 817 | the typed element extraction call, one per requirement block |
| 3.1 | `merge_elements` | 873 | canonicalises duplicate element names across blocks |
| 3.1 | `extract_dependency_relations` | 1068 | two-pass relation extraction: intra-requirement, then cross-requirement |
| 3.1 | `validate_relation_items` | 1022 | rejects unknown relation types, self-loops, and unsupported edges |
| 3.1 | `deduplicate_relations` | 1161 | collapses duplicate directed edges |
| 3.1 | `DependencyGraphs.build` | 1470 | semantic `MultiDiGraph` plus the acyclic computational graph |
| 3.2 | `DependencyGraphs.propagate` | 1635 | Eq. (3), forward activation in topological order with source clamping |
| 3.2 | `DependencyGraphs.dps` | 1664 | Eq. (4), the clamped-at-1 minus clamped-at-0 difference; Eq. (5), non-negativity |
| 3.2 | `deterministic_shortest_path` | 1206 | the Topology ranking signal |
| 3.2 | `AllPathSupportIndex` / `maximum_geometric_path_support` | 1364 / 1413 | Eq. (8), maximum geometric-mean support over **all** admissible paths |
| 3.2 | `rank_records` | 1776 | orders the record pool by `dps`, `extraction_support`, `topology`, `unranked` or `random` |
| 3.3 | `nominate_dependency_findings` | 1962 | dependency-view gap nomination over the ranked budget |
| 3.3 | `nominate_isolation_findings` | 2055 | isolation-view nomination over disconnected elements |
| 3.3 | `nominate_operation_findings` | 2121 | operation-view nomination over `DATA` and `FUNCTION` elements |
| 3.4 | `validate_findings` | 2226 | evidence / coverage / boundary decisions in one call per finding |
| 3.4 | `DiagnosticFinding.eligible_for_generation` | 287 | requires `YES` / `NO` / `YES`; `UNCERTAIN` is never forwarded |
| 3.5 | `generate_candidates` | 2367 | one reviewable requirement per eligible finding, with DPS withheld |
| 3.5 | `consolidate_candidates` | 2450 | merge, refine, deduplicate, filter; returns KEEP / DEMOTE / REMOVE |
| 3.5 | `numbering_context` | 2806 | detects the document's numbering dialect and last used number |
| 3.5 | `build_refined_document` | 2863 | the terminal artefact: original text plus numbered additions |
| 3.5 | `build_design_constraints_document` | 2919 | companions for demoted and rejected candidates |
| 3.5 | `build_refinement_report` | 2976 | Markdown summary of the refinement pass |
| — | `DIOReqPipeline` | 3068 | composes the five stages; `run_dioreq` at 3359 drives a dataset |

The three refinement outcomes are a partition: every raw candidate identifier
appears in exactly one of `candidates`, `design_constraints`, or
`removed_candidates`, and each kept record carries the identifiers it merged
so the pass is reversible.

Prompt templates are extracted verbatim into `prompts/`, organised by Section
3 subsections. `prompts/README.md` documents the placeholders and call sites.
Regenerate with `python build_prompts.py`; a byte-fidelity check confirms the
files match the strings the code sends.

---

## Running the experiments

All three runners take `--dataset`, `--output` and `--shard` (see *Sharding*).

### RQ1 — the method end to end

Produces the raw record for every document-run and, with `--refined-dir`, the
Stage-5 artefacts: refined document, design constraints, refinement report.

```bat
python DIOReq\RQ1\dioreq.py ^
       --dataset dataset_all.json ^
       --output results\rq1.json ^
       --refined-dir results\refined ^
       --model gpt-5.5
```

Repeat the run per document with `--runs`, and cap how many ranked dependency
records are reviewed with `--dependency-budget`. Both have defaults, shown by
`--help`.

### RQ2 — component and diagnostic-view ablations

Runs the variants reported in the paper's Table 4 by default.
`--include-extended` adds finer-grained sub-ablations that the paper does not
report.

```bat
python DIOReq\RQ2\rq2.py ^
       --dataset dataset_all.json ^
       --output results\rq2.json ^
       --model gpt-5.5
```

Variants that agree on the preparation settings
(`cross_requirement_extraction`, `max_parents`) share one extracted graph
within a repetition. This is not only an optimisation: an endpoint is not
guaranteed to be deterministic even at temperature 0, so re-extracting the
graph per variant would leave each variant comparing a different graph and
mix the ablation effect with extraction noise. Every same-group variant is
guaranteed to have seen the identical graph.

### RQ3 — equal-budget ranking comparison

Holds the document, the graph and the model settings fixed and varies only
the ordering of the record pool. Compares `dps`, `extraction_support`,
`topology`, `unranked` and `random` at each nominal budget. The budget is
applied before nomination and before validation, and the reported precision
uses Eq. (10), `VFP@B_d = valid selected records / min(B, N_d)`.

```bat
python DIOReq\RQ3\rq3.py ^
       --dataset dataset_all.json ^
       --output results\rq3.json ^
       --model gpt-5.5
```

Budgets are given with `--budgets`; the random signal is averaged over
`--random-repetitions` draws. Both have defaults, shown by `--help`.

The document graph is prepared once per document and reused by every ranking
condition, so the only thing that changes between conditions is the ranking
signal.

---

## Tests

```bat
python test_dioreq_logic.py     # LLM-free: segmentation, DPS, Eq. (8), malformed
                                #   model output, numbering, sharding
python test_runners.py          # end-to-end for all three runners, mocked LLM
python compare_fr_counts.py     # segmenter must reproduce the audit's counts
```

`test_runners.py` never contacts an API: it substitutes a deterministic client
and checks output shape, the Stage-5 partition, budget semantics, and that one
failing run does not abort a batch.

`compare_fr_counts.py` is a data-level regression check. It re-segments the
corpus and compares the block count of every document against an
independently produced requirement audit, so a change to the segmenter cannot
silently alter how many requirements a document is seen to contain. The audit
travels with the repository as `fr_counts_audit.json`; `--export-audit`
regenerates it from the original Excel workbooks. The check reads a built
dataset when one is present and otherwise converts the corpus itself, so it
runs on a fresh clone.

---

## Corpus

`DIOReq/Data` holds 42 source specifications across seven systems (CCTNS,
CityMapper, Email, Inventory, MDOT, QuickEats, WebStore), each in six
variants. `DIOReq/Referencedata` holds the 42 matching reference
specifications.

The documents do not share one numbering style. The segmenter recognises
`Function Requirement n`, `## n.n`, `n.n`, `n.n.n`, `FR-001`, `FR-DCI-001`,
`### FR-01`, `F1` and `FR n:`, and splits at the deepest level present, so a
section numbered `2.1` that contains a dozen `FR-001` items yields twelve
requirement blocks rather than one. A document that continues into a
non-requirement section reusing the same numbering — `3. External Interfaces`,
say — is scoped to its functional-requirements section.

---

## Sharding

Every runner is sequential, and a single document-run issues many model
calls, so a full protocol takes a long time in one process. All three
runners therefore accept `--shard I/N`:

```bat
python DIOReq\RQ3\rq3.py --dataset dataset_all.json --output results\shard0.json --shard 0/4 --model gpt-5.5
python DIOReq\RQ3\rq3.py --dataset dataset_all.json --output results\shard1.json --shard 1/4 --model gpt-5.5
:: ... N processes, one per shard ...
```

The shards partition the documents: every document is processed exactly once
across the slices. Give each process its own `--output` and concatenate the
resulting JSON arrays afterwards. Sharding is at process level rather than
with threads so that no shared mutable state — the usage counter, the result
list — needs a lock.

---

## Notes

- **Endpoint determinism.** A hosted endpoint is not guaranteed to return
  identical output for identical requests, even at temperature 0.
  Repetitions in RQ1–RQ3 measure that variance rather than assuming it away;
  RQ2 and RQ3 additionally keep the graph fixed within a repetition so
  comparisons stay paired.
- **Malformed model output.** Every field read from a model response is
  coerced rather than trusted. A `null` where a list was requested, a
  confidence reported as a percentage, or a non-object JSON response costs at
  most the item that contained it, and a run that fails is recorded with an
  `error` field instead of discarding the batch.
- **Analyst review is a human step.** The pipeline deliberately stops at
  evidence-linked proposals. No model call performs acceptance, rejection or
  modification, and `prompts/README.md` records where each section of the
  paper places that boundary.
- **`smoke_run/`** holds real runs against a live endpoint, with the exact
  input dataset, the raw output, the console log and the Stage-5 documents.

---

## Licence

Not yet specified.
