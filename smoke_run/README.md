# Smoke runs

Real end-to-end runs of the RQ1 runner against the live API, kept as
evidence that the released code executes and produces the artefacts the
paper describes. Each folder holds the exact dataset that was passed in, the
raw JSON that came out, the console log, and the Stage-5 documents.

The method module lives in `DIOReq/RQ1/dioreq.py`. Re-run a smoke case with:

```bat
set OPENAI_API_KEY=...
set OPENAI_BASE_URL=https://your-endpoint/v1
python DIOReq\RQ1\dioreq.py ^
       --dataset smoke_run\<folder>\input_dataset.json ^
       --output  smoke_run\<folder>\dioreq_output.json ^
       --refined-dir smoke_run\<folder>\refined ^
       --model <your-model> --runs 1
```

To spread a full corpus over several processes, add `--shard i/N` with a
distinct `--output` per process and concatenate the JSON arrays at the end.

## WebStore_simple3_tabular

`filtered_WebStore_simple3_doc.docx` is the awkward case in the corpus: the
document holds 19 requirement rows (`FR-001`..`FR-019`) grouped under five
`2.x` sections, with `Inputs:`/`Outputs:` stated once per section rather than
per requirement. The segmenter cuts it at the requirement level.

```
stage 1-2: 66 element(s), 57 relation(s), 169 ranking record(s)
stage 3 dependency view: 20 record(s) reviewed
stage 3: 59 finding(s) nominated
stage 4: 59 validated, 38 eligible for generation
stage 5: 38 candidate(s) -> 19 kept, 1 demoted, 3 removed
usage: 124,048 prompt tokens, 78,604 completion tokens, 167 API calls
```

What this run demonstrates:

- Segmentation reaches the requirement level: the findings cite `FR-001`,
  `FR-002`, ..., not `2.1`, `2.2`.
- The KEEP / DEMOTE / REMOVE partition is complete: all 38 raw candidate
  identifiers are claimed exactly once across the three outputs, with one
  group merging three candidates.
- The DEMOTE branch fired, producing a design-constraint companion entry
  rather than discarding the content.
- Stage 5 wrote a refined document that continues the source numbering.

## CCTNS_simple1

Only the input dataset is kept here. It is the smallest document in the
corpus and the fastest way to check a fresh API configuration; run it with
the command above. A full run costs roughly 180 API calls.

## Cost note

Each call to a reasoning model such as `gpt-5.6-sol` takes about 15 seconds,
and one document-run issues 160-190 calls. That is 35-45 minutes per
document. The paper's full protocol is 42 documents x 3 repetitions, so plan
for tens of hours of wall-clock time, not minutes.
