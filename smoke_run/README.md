# Smoke runs

Real end-to-end runs of the RQ1 runner against a live endpoint, kept as
evidence that the released code executes and produces the artefacts the paper
describes. The `WebStore_simple3_tabular/` run keeps the exact dataset that
was passed in, the raw JSON that came out, the console log, and the Stage-5
documents.

Per-stage counts and token usage are in `run.log`; they are not repeated here.

The method module lives in `DIOReq/RQ1/dioreq.py`. Re-run a smoke case with:

```bat
set OPENAI_API_KEY=...
python DIOReq\RQ1\dioreq.py ^
       --dataset smoke_run\<folder>\input_dataset.json ^
       --output  smoke_run\<folder>\dioreq_output.json ^
       --refined-dir smoke_run\<folder>\refined ^
       --model gpt-5.5 --runs 1
```

To spread a full corpus over several processes, add `--shard i/N` with a
distinct `--output` per process and concatenate the JSON arrays at the end.

## WebStore_simple3_tabular

`filtered_WebStore_simple3_doc.docx` is the awkward case in the corpus: it
holds a run of `FR-001`-style requirement rows grouped under `2.x` sections,
with `Inputs:`/`Outputs:` stated once per section rather than per requirement.
The segmenter cuts it at the requirement level.

What this run demonstrates:

- Segmentation reaches the requirement level: the findings cite `FR-001`,
  `FR-002`, ..., not `2.1`, `2.2`.
- The KEEP / DEMOTE / REMOVE partition is complete: every raw candidate
  identifier is claimed exactly once across the three outputs, including the
  groups that merged several candidates.
- The DEMOTE branch fired, producing a design-constraint companion entry
  rather than discarding the content.
- Stage 5 wrote a refined document that continues the source numbering.

## CCTNS_simple1

The smallest document in the corpus, and the quickest way to check a fresh
API configuration. Run it with the command above.
