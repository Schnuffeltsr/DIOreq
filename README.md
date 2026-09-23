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
├── scan_model_output.py       AST audit for unguarded model-output reads
├── verify_stage5_live.py      re-runs only Stage 5 to check traceability
│
├── test_dioreq_logic.py       LLM-free unit tests
├── test_runners.py            end-to-end tests against a mocked LLM
├── check_readme.py            keeps the table below honest about line numbers
│
├── dataset_all.json           the 42-document corpus as a dataset
├── dataset_reference.json     the 42 reference documents as a dataset
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
| `OPENAI_BASE_URL` | Endpoint, e.g. `https://api.deepseek.com/v1` |
| `DIOREQ_EXTRACTION_MODEL` | Model for element and relation extraction |
| `DIOREQ_VALIDATION_MODEL` | Model for nomination, validation, consolidation |
| `DIOREQ_GENERATION_MODEL` | Model for requirement generation |
| `DIOREQ_EXTRACTION_TEMPERATURE` | Override the sampling temperature |
| `DIOREQ_VALIDATION_TEMPERATURE` | " |
| `DIOREQ_GENERATION_TEMPERATURE` | " |

Every runner also accepts `--model`, `--base-url`, `--api-key` and
`--temperature`, which override the environment. `--model` sets all three
stage models at once.

A temperature of `none` omits the `temperature` field entirely. Reasoning
models that accept only their built-in temperature then work unchanged, and
so do models that intermittently reject a configured value.

```bat
set OPENAI_API_KEY=...
set OPENAI_BASE_URL=https://api.deepseek.com/v1
python DIOReq\RQ1\dioreq.py --dataset dataset_all.json ^
                            --output results\rq1.json ^
                            --model deepseek-flash --runs 1
```

### Building a dataset

The runners read a JSON array of documents. `build_dataset.py` produces one
from a folder of `.docx` files:

```bat
python build_dataset.py --source DIOReq\Data --output dataset_all.json
```

Each entry is `{"system_id": ..., "document_id": ..., "text": ...}`. The two
datasets used in the paper are already committed, so this step is only needed
for a new corpus.

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
| 3.1 | `split_into_frs` | 452 | segments a specification into requirement blocks; picks the deepest numbering level present |
| 3.1 | `extract_elements` | 800 | the typed element extraction call, one per requirement block |
| 3.1 | `merge_elements` | 856 | canonicalises duplicate element names across blocks |
| 3.1 | `extract_dependency_relations` | 1051 | two-pass relation extraction: intra-requirement, then cross-requirement |
| 3.1 | `validate_relation_items` | 1005 | rejects unknown relation types, self-loops, and unsupported edges |
| 3.1 | `deduplicate_relations` | 1144 | collapses duplicate directed edges |
| 3.1 | `DependencyGraphs.build` | 1453 | semantic `MultiDiGraph` plus the acyclic computational graph |
| 3.2 | `DependencyGraphs.propagate` | 1618 | Eq. (3), forward activation in topological order with source clamping |
| 3.2 | `DependencyGraphs.dps` | 1647 | Eq. (4), the clamped-at-1 minus clamped-at-0 difference; Eq. (5), non-negativity |
| 3.2 | `deterministic_shortest_path` | 1189 | the Topology ranking signal |
| 3.2 | `AllPathSupportIndex` / `maximum_geometric_path_support` | 1347 / 1396 | Eq. (8), maximum geometric-mean support over **all** admissible paths |
| 3.2 | `rank_records` | 1759 | orders the record pool by `dps`, `extraction_support`, `topology`, `unranked` or `random` |
| 3.3 | `nominate_dependency_findings` | 1945 | dependency-view gap nomination over the ranked budget |
| 3.3 | `nominate_isolation_findings` | 2038 | isolation-view nomination over disconnected elements |
| 3.3 | `nominate_operation_findings` | 2104 | operation-view nomination over `DATA` and `FUNCTION` elements |
| 3.4 | `validate_findings` | 2209 | evidence / coverage / boundary decisions in one call per finding |
| 3.4 | `DiagnosticFinding.eligible_for_generation` | 270 | requires `YES` / `NO` / `YES`; `UNCERTAIN` is never forwarded |
| 3.5 | `generate_candidates` | 2350 | one reviewable requirement per eligible finding, with DPS withheld |
| 3.5 | `consolidate_candidates` | 2433 | merge, refine, deduplicate, filter; returns KEEP / DEMOTE / REMOVE |
| 3.5 | `numbering_context` | 2789 | detects the document's numbering dialect and last used number |
| 3.5 | `build_refined_document` | 2846 | the terminal artefact: original text plus numbered additions |
| 3.5 | `build_design_constraints_document` | 2902 | companions for demoted and rejected candidates |
| 3.5 | `build_refinement_report` | 2959 | Markdown summary of the refinement pass |
| — | `DIOReqPipeline` | 3051 | composes the five stages; `run_dioreq` at 3342 drives a dataset |

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
       --runs 3 --dependency-budget 20 --model <model>
```

### RQ2 — component and diagnostic-view ablations

Runs the eight variants reported in the paper's Table 4 by default.
`--include-extended` adds six finer-grained sub-ablations that the paper does
not report.

```bat
python DIOReq\RQ2\rq2.py ^
       --dataset dataset_all.json ^
       --output results\rq2.json ^
       --runs 3 --dependency-budget 20 --model <model>
```

Variants that agree on the preparation settings
(`cross_requirement_extraction`, `max_parents`) share one extracted graph
within a repetition. This is not only an optimisation: the endpoint is not
deterministic even at temperature 0, so re-extracting the graph per variant
would leave each variant comparing a different graph and mix the ablation
effect with extraction noise. Eight variants therefore cost two graph
preparations, not eight, and every same-group variant is guaranteed to have
seen the identical graph.

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
       --budgets 5 10 15 20 --random-repetitions 30 --model <model>
```

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
python scan_model_output.py     # no unguarded read of model output
python check_readme.py          # the function table above matches the code
```

`test_runners.py` never contacts an API: it substitutes a deterministic client
and checks output shape, the Stage-5 partition, budget semantics, and that one
failing run does not abort a batch.

`compare_fr_counts.py` is a data-level regression check. It re-segments the
corpus and compares the block count of every document against the
independently produced requirement audit, so a change to the segmenter cannot
silently alter how many requirements a document is seen to contain.

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

Every runner is sequential, and one document-run costs roughly 150–200 model
calls. A full protocol is therefore days of wall-clock time in a single
process, so all three accept `--shard I/N`:

```bat
python DIOReq\RQ3\rq3.py --dataset dataset_all.json --output results\rq3_shard0.json --shard 0/20 --model <model>
python DIOReq\RQ3\rq3.py --dataset dataset_all.json --output results\rq3_shard1.json --shard 1/20 --model <model>
:: ... 20 processes ...
```

The shards partition the documents: every document is processed exactly once
across the slices. Give each process its own `--output` and concatenate the
resulting JSON arrays afterwards. Sharding is at process level rather than
with threads so that no shared mutable state — the usage counter, the result
list — needs a lock.

---

## Notes

- **Endpoint determinism.** On the endpoint used during development, three
  identical requests at `temperature=0.0` returned three different
  consolidations. Repetitions in RQ1–RQ3 measure that variance rather than
  assuming it away; RQ2 and RQ3 additionally keep the graph fixed within a
  repetition so comparisons stay paired.
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
