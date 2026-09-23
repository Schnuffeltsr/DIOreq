# code_paths.py
"""
Locate the DIOReq source directories from anywhere.

The repository keeps the method in ``DIOReq/RQ1/dioreq.py`` and the two
comparison runners in ``DIOReq/RQ2`` and ``DIOReq/RQ3``. The tests and the
tooling scripts live at the repository root, so they need those directories
on ``sys.path`` before they can ``import dioreq``.

Importing this module and calling ``install()`` is the supported way to do
that. A flat development checkout, where ``dioreq.py``, ``rq2.py`` and
``rq3.py`` all sit beside this file, keeps working too.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT / "DIOReq"

RQ1 = CODE_ROOT / "RQ1"
RQ2 = CODE_ROOT / "RQ2"
RQ3 = CODE_ROOT / "RQ3"

# Search order: the released layout first, then a flat layout.
SOURCE_DIRS = (RQ1, RQ2, RQ3, ROOT)


def find_dioreq() -> Path:
    """Return the directory holding ``dioreq.py``."""
    for directory in SOURCE_DIRS:
        if (directory / "dioreq.py").is_file():
            return directory

    raise ImportError(
        "dioreq.py not found. Looked in: "
        + ", ".join(str(directory) for directory in SOURCE_DIRS)
    )


def install() -> Path:
    """
    Put every existing source directory on ``sys.path`` and return the one
    that holds ``dioreq.py``.

    Appending rather than prepending keeps a deliberately installed package
    earlier in the search order.
    """
    for directory in SOURCE_DIRS:
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.append(str(directory))

    return find_dioreq()
