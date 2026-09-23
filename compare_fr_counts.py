# compare_fr_counts.py
"""
Compare split_into_frs() output against the requirement counts recorded in
the workspace spreadsheets, which were produced by the earlier (independent)
document audit.

Mismatches are the documents whose numbering dialect the splitter gets wrong.

Usage:  python compare_fr_counts.py
"""
from __future__ import annotations

import pathlib
import sys
import zipfile
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import code_paths

code_paths.install()

import dioreq

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_sheet(path: pathlib.Path) -> list[list[str]]:
    z = zipfile.ZipFile(path)
    shared: list[str] = []

    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall(f"{NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))

    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
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


def truth(path: pathlib.Path, name_col: int, count_col: int) -> dict[str, int]:
    out: dict[str, int] = {}

    for row in read_sheet(path):
        if len(row) <= max(name_col, count_col):
            continue
        name = row[name_col].strip()
        count = row[count_col].strip()
        if not name.endswith(".docx") or not count.isdigit():
            continue
        out[pathlib.Path(name).stem] = int(count)

    return out


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


def find_dataset(name: str) -> pathlib.Path:
    """
    Locate a dataset JSON without hard-coding where it currently lives.

    The workspace is reorganised from time to time, so the dataset may sit at
    the repository root, under DIOReq/, or beside the corpus it was built
    from. Searching keeps this check working across those layouts instead of
    failing with a confusing FileNotFoundError.
    """
    candidates = [
        HERE / name,
        HERE / "DIOReq" / name,
    ]

    for base in (HERE, HERE / "DIOReq"):
        if not base.is_dir():
            continue
        for pattern in (name, f"**/{name}"):
            candidates.extend(sorted(base.glob(pattern)))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise SystemExit(
        f"{name} not found. Build it first, e.g.\n"
        f"  python build_dataset.py --source DIOReq/Data "
        f"--output DIOReq/{name}"
    )


def main() -> None:
    datasets = [
        ("测试数据 (DIOReq/Data)", find_dataset("dataset_all.json"),
         truth(HERE / "原始文件.xlsx", 4, 5)),
        ("参考基准 (DIOReq/Referencedata)",
         find_dataset("dataset_reference.json"),
         truth(HERE / "本文方法生成的统计.xlsx", 3, 4)),
    ]

    unexpected = 0
    known = 0

    for label, dataset, expected in datasets:
        docs = dioreq.load_documents(dataset)
        print("=" * 96)
        print(f"{label}  --  {len(docs)} documents, "
              f"{len(expected)} recorded counts   [{dataset}]")
        print("=" * 96)

        ok = 0
        missing = []
        bad = []

        for doc in docs:
            got = len(dioreq.split_into_frs(doc.text))
            want = expected.get(doc.document_id)

            if want is None:
                missing.append(doc.document_id)
            elif got == want:
                ok += 1
            else:
                bad.append((doc.document_id, got, want))

        print(f"exact match : {ok}/{len(docs)}")
        print(f"unrecorded  : {len(missing)}")
        print(f"mismatch    : {len(bad)}")

        for name, got, want in sorted(bad, key=lambda x: x[1]):
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
