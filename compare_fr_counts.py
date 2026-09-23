# compare_fr_counts.py
"""
Data-level regression check for the requirement segmenter.

`split_into_frs` decides how many requirements a document is seen to
contain. That number drives how many extraction calls a document costs and
what every finding cites as its source, so a change to the segmenter must
not silently alter it.

This check re-segments the corpus and compares the block count of every
document against an independent audit of the same documents, produced before
the segmenter was written.

The audit ships as ``fr_counts_audit.json`` so the check runs on a fresh
clone with no extra inputs. If the original Excel workbooks are present they
are used instead, and ``--export-audit`` rewrites the JSON from them.

Usage
-----
    python compare_fr_counts.py
    python compare_fr_counts.py --export-audit
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zipfile
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import code_paths

code_paths.install()

import dioreq

AUDIT_JSON = HERE / "fr_counts_audit.json"

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# label, dataset file, audit key, author's workbook, filename column,
# count column
SOURCES = [
    ("测试数据 (DIOReq/Data)", "dataset_all.json", "corpus",
     "原始文件.xlsx", 4, 5),
    ("参考基准 (DIOReq/Referencedata)", "dataset_reference.json", "reference",
     "本文方法生成的统计.xlsx", 3, 4),
]

# Documents whose recorded count is known to be wrong, with the evidence.
# Listed here rather than silently tolerated: the check still fails for any
# other mismatch, and prints these so they are not forgotten.
KNOWN_RECORDED_COUNT_ERRORS = {
    "filtered_WebStore_simple3_doc": (
        "recorded 5, document holds 19 independent 'FR-001'..'FR-019' rows. "
        "The audit counted 'Input/Output' field pairs, and this tabular "
        "variant states Inputs/Outputs once per section. Its sibling "
        "filtered_WebStore_simple2_doc has 14 rows and 14 field pairs and was "
        "counted 14, and the matching reference document is counted per FR "
        "row (56), so 19 is the consistent reading."
    ),
}


# ------------------------------------------------------------------ workbook
def read_sheet(path: pathlib.Path) -> list[list[str]]:
    """Read the first worksheet of an xlsx without any third-party library."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []

        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{NS}si"):
                shared.append(
                    "".join(t.text or "" for t in item.iter(f"{NS}t"))
                )

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows = []

    for row in sheet.iter(f"{NS}row"):
        cells = []

        for cell in row.findall(f"{NS}c"):
            value = cell.find(f"{NS}v")

            if value is None:
                cells.append("")
            elif cell.get("t") == "s":
                cells.append(shared[int(value.text)])
            else:
                cells.append(value.text or "")

        rows.append(cells)

    return rows


def counts_from_workbook(
    path: pathlib.Path,
    name_column: int,
    count_column: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for row in read_sheet(path):
        if len(row) <= max(name_column, count_column):
            continue

        name = row[name_column].strip()
        count = row[count_column].strip()

        if name.endswith(".docx") and count.isdigit():
            counts[pathlib.Path(name).stem] = int(count)

    return counts


def build_audit() -> dict[str, dict[str, int]]:
    """Rebuild the audit from the author's Excel workbooks."""
    audit: dict[str, dict[str, int]] = {}

    for _, _, key, workbook, name_column, count_column in SOURCES:
        path = HERE / workbook

        if not path.is_file():
            raise SystemExit(f"Workbook not found, cannot export: {path}")

        audit[key] = counts_from_workbook(path, name_column, count_column)

    return audit


def load_audit() -> tuple[dict[str, dict[str, int]], str]:
    """Return the audit and where it came from."""
    workbooks = [HERE / source[3] for source in SOURCES]

    if all(path.is_file() for path in workbooks):
        return build_audit(), "the Excel workbooks in the working copy"

    if AUDIT_JSON.is_file():
        return (
            json.loads(AUDIT_JSON.read_text(encoding="utf-8")),
            AUDIT_JSON.name,
        )

    raise SystemExit(
        f"Neither {AUDIT_JSON.name} nor the source workbooks are available, "
        "so there is nothing to compare against."
    )


# ------------------------------------------------------------------ check
# Where to find the corpus if the dataset JSON has not been built. The
# datasets are generated artefacts and are not tracked, so a fresh clone
# converts the .docx files on the fly.
CORPUS_DIRS = {
    "dataset_all.json": "DIOReq/Data",
    "dataset_reference.json": "DIOReq/Referencedata",
}


def find_dataset(name: str) -> pathlib.Path | None:
    """
    Locate a pre-built dataset JSON, or return None.

    The corpus may sit beside this script, under DIOReq/, or next to the
    documents it was built from.
    """
    candidates = [HERE / name, HERE / "DIOReq" / name]

    for base in (HERE, HERE / "DIOReq"):
        if base.is_dir():
            candidates.extend(sorted(base.glob(f"**/{name}")))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def documents_from_corpus(name: str) -> list | None:
    """
    Segment the corpus directly, without a pre-built dataset file.

    Uses the same conversion build_dataset.py performs, so the counts are
    identical whether the JSON was built beforehand or not.
    """
    import build_dataset

    source = HERE / CORPUS_DIRS[name]

    if not source.is_dir():
        return None

    return [
        dioreq.DocumentRecord(
            system_id=project,
            document_id=document_id,
            text=build_dataset.document_to_text(path),
        )
        for project, document_id, path in build_dataset.discover_documents(
            source
        )
    ]


def load_documents(name: str) -> tuple[list, str]:
    """Return the documents for a corpus and a description of their origin."""
    dataset = find_dataset(name)

    if dataset is not None:
        return dioreq.load_documents(dataset), str(dataset)

    documents = documents_from_corpus(name)

    if documents is None:
        raise SystemExit(
            f"Neither {name} nor {CORPUS_DIRS[name]} is present, so there is "
            "nothing to check."
        )

    return documents, f"{CORPUS_DIRS[name]} (converted on the fly)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-audit",
        action="store_true",
        help="Rewrite fr_counts_audit.json from the Excel workbooks and exit.",
    )
    args = parser.parse_args()

    if args.export_audit:
        audit = build_audit()
        AUDIT_JSON.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        for key, counts in audit.items():
            print(f"{key:12s} {len(counts)} documents")
        print(f"wrote {AUDIT_JSON}")
        return

    audit, provenance = load_audit()
    print(f"audit source: {provenance}\n")

    unexpected = 0
    known = 0

    for label, dataset_name, key, _, _, _ in SOURCES:
        documents, origin = load_documents(dataset_name)
        expected = audit[key]

        print("=" * 96)
        print(f"{label}  --  {len(documents)} documents, "
              f"{len(expected)} recorded counts")
        print(f"corpus: {origin}")
        print("=" * 96)

        matched = 0
        missing = []
        mismatched = []

        for document in documents:
            got = len(dioreq.split_into_frs(document.text))
            want = expected.get(document.document_id)

            if want is None:
                missing.append(document.document_id)
            elif got == want:
                matched += 1
            else:
                mismatched.append((document.document_id, got, want))

        print(f"exact match : {matched}/{len(documents)}")
        print(f"unrecorded  : {len(missing)}")
        print(f"mismatch    : {len(mismatched)}")

        for name, got, want in sorted(mismatched, key=lambda item: item[1]):
            if name in KNOWN_RECORDED_COUNT_ERRORS:
                known += 1
                print(f"   KNOWN {name:50s} split={got:4d} recorded={want:4d}")
                print(f"         -> {KNOWN_RECORDED_COUNT_ERRORS[name]}")
            else:
                unexpected += 1
                print(f"   BAD   {name:50s} split={got:4d} recorded={want:4d}")

        for name in missing:
            print(f"   (no recorded count) {name}")

        print()

    print(f"unexpected mismatches : {unexpected}")
    print(f"known audit errors    : {known}")

    if unexpected:
        raise SystemExit(1)

    print("RESULT: segmenter agrees with every recorded count")


if __name__ == "__main__":
    main()
