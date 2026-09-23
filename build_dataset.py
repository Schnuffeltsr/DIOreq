# build_dataset.py
"""
Convert a folder of Word requirement documents into the JSON dataset that
``dioreq.py`` / ``rq1.py`` / ``rq2.py`` / ``rq3.py`` expect.

Dataset schema (one entry per document):

    [{"system_id": "...", "document_id": "...", "text": "..."}, ...]

Text extraction
---------------
Paragraphs are emitted in body order.  A blank line is inserted before a
paragraph whose style name starts with ``Heading`` and after every
paragraph, so that requirement blocks stay separated in the flattened text.
This matters because ``split_into_frs`` in ``dioreq.py`` splits on
requirement-number starts (``3.1``, ``FR-001``, ``Function Requirement 1``,
...) and on blank lines as a fallback.

Usage
-----
    python build_dataset.py --source DIOReq\\Data --output DIOReq\\dataset.json

``--source`` may be either
  * a dataset root holding one folder per project, each with a ``Document``
    sub-folder (the layout used under ``DIOReq/Data``), or
  * a single project folder that directly contains ``.docx`` files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


# ------------------------------------------------------------
# Word text extraction
# ------------------------------------------------------------

def iter_block_items(parent):
    """
    Yield paragraphs and tables of a document body in true document order.
    """
    body = parent.element.body

    for child in body.iterchildren():
        tag = child.tag

        if tag.endswith('}p'):
            yield Paragraph(child, parent)
        elif tag.endswith('}tbl'):
            yield Table(child, parent)


def table_as_text(table: Table) -> str:
    lines = []

    for row in table.rows:
        cells = [
            " ".join(cell.text.split())
            for cell in row.cells
        ]
        lines.append(" | ".join(cells))

    return "\n".join(line for line in lines if line.strip(" |"))


def document_to_text(path: Path) -> str:
    document = Document(str(path))
    blocks: list[str] = []

    for block in iter_block_items(document):
        if isinstance(block, Table):
            text = table_as_text(block)
        else:
            style = (block.style.name if block.style is not None else "") or ""
            text = block.text

            if style.lower().startswith("heading"):
                text = text.strip()
                if text:
                    blocks.append("")          # blank line before heading

        text = text.strip()

        if text:
            blocks.append(text)

    # Collapse runs of blank lines, keep exactly one.
    lines: list[str] = []

    for line in blocks:
        if line == "" and (not lines or lines[-1] == ""):
            continue
        lines.append(line)

    return "\n".join(lines).strip() + "\n"


# ------------------------------------------------------------
# Folder discovery
# ------------------------------------------------------------

def discover_documents(source: Path) -> list[tuple[str, str, Path]]:
    """
    Return ``(project, document_id, path)`` triples for every .docx file
    below ``source``.

    Hidden Word lock files (``~$...``) are skipped.
    """
    found: list[tuple[str, str, Path]] = []

    def add(project: str, folder: Path) -> None:
        for path in sorted(folder.glob("*.docx")):
            if path.name.startswith("~$"):
                continue
            found.append((project, path.stem, path))

    projects = sorted(
        item for item in source.iterdir()
        if item.is_dir() and not item.name.startswith((".", "_"))
    )

    if projects:
        for project in projects:
            folder = project / "Document"
            add(project.name, folder if folder.is_dir() else project)
    else:
        # A single project folder was passed directly.  If it is itself the
        # "Document" sub-folder, use the parent's name as the system id.
        project = (
            source.parent.name
            if source.name.lower() == "document"
            else source.name
        )
        add(project, source)

    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DIOReq JSON dataset from Word documents."
    )
    parser.add_argument(
        "--source", required=True,
        help="Dataset root or a single project folder containing .docx files.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path of the JSON dataset to write.",
    )
    parser.add_argument(
        "--text-dir", default=None,
        help="Optional folder to also dump each document's extracted text to.",
    )

    args = parser.parse_args()

    source = Path(args.source)

    if not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")

    documents = discover_documents(source)

    if not documents:
        raise SystemExit(f"No .docx files found under: {source}")

    dataset = []

    for project, document_id, path in documents:
        text = document_to_text(path)
        dataset.append({
            "system_id": project,
            "document_id": document_id,
            "text": text,
        })
        print(f"{project}/{document_id}: {len(text)} chars")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"\nWrote {len(dataset)} document(s) -> {output}")

    if args.text_dir:
        text_dir = Path(args.text_dir)
        text_dir.mkdir(parents=True, exist_ok=True)

        for project, document_id, path in documents:
            target = text_dir / f"{project}_{document_id}.txt"
            with target.open("w", encoding="utf-8", newline="\n") as file:
                file.write(document_to_text(path))


if __name__ == "__main__":
    main()
