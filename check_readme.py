# check_readme.py
"""
Verify that README.md's function table matches the code it documents.

Every function named in the stage table must exist in dioreq.py, and every
line number quoted beside it must be that definition's actual line. A README
that points at the wrong line is worse than one that quotes none.

Run:  python check_readme.py
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import code_paths

CODE = code_paths.find_dioreq() / "dioreq.py"
README = code_paths.ROOT / "README.md"

TICKED = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")


def main() -> None:
    locations: dict[str, int] = {}

    for node in ast.walk(ast.parse(CODE.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            locations.setdefault(node.name, node.lineno)

    rows = [
        line
        for line in README.read_text(encoding="utf-8").splitlines()
        if re.match(r"^\|\s*(3\.\d|—)\s*\|", line)
    ]

    problems = []
    checked = 0
    exact = 0

    for line in rows:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue

        names = TICKED.findall(cells[1])
        numbers = [int(value) for value in re.findall(r"\b(\d{3,4})\b", cells[2])]

        for name in names:
            key = name.split(".")[-1]
            if key not in locations:
                problems.append(f"{name} does not exist in dioreq.py")
            else:
                checked += 1

        if names and numbers:
            key = names[0].split(".")[-1]
            if locations.get(key) == numbers[0]:
                exact += 1
            elif key in locations:
                problems.append(
                    f"{key}: README says line {numbers[0]}, "
                    f"definition is at {locations[key]}"
                )

    print(f"table rows                 : {len(rows)}")
    print(f"function names checked     : {checked}")
    print(f"line numbers exact         : {exact}")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("\nRESULT: README matches dioreq.py")


if __name__ == "__main__":
    main()
