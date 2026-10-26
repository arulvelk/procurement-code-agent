"""Loading the committed taxonomy files into memory.

The corpus is small, a few thousand rows, so it is held in a plain list and a
dict. There is no database here on purpose. Reaching for one would add operational
weight without buying anything at this size.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..domain import Taxonomy, TaxonomyCode
from ..errors import NotFound

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

FILENAMES: dict[str, str] = {
    "psc": "psc.jsonl",
    "naics": "naics.jsonl",
    "arxiv": "arxiv.jsonl",
}


class TaxonomyCatalog:
    """Every code in one taxonomy, addressable by code and iterable in order."""

    def __init__(self, taxonomy: Taxonomy, codes: list[TaxonomyCode]) -> None:
        self.taxonomy = taxonomy
        self.codes = codes
        self._by_code = {code.code.upper(): code for code in codes}

    def __len__(self) -> int:
        return len(self.codes)

    def get(self, code: str) -> TaxonomyCode:
        try:
            return self._by_code[code.strip().upper()]
        except KeyError:
            raise NotFound(
                f"{code!r} is not a {self.taxonomy} code", taxonomy=self.taxonomy, code=code
            ) from None

    def contains(self, code: str) -> bool:
        return code.strip().upper() in self._by_code


def load_catalog(taxonomy: Taxonomy, data_dir: Path | None = None) -> TaxonomyCatalog:
    """Read one taxonomy from disk.

    Rows carrying a ``year_retired`` are dropped. Retired codes still appear in
    historical awards, but putting them in the candidate pool only gives the
    reranker more ways to be wrong on a live classification.
    """
    directory = data_dir or DATA_DIR
    path = directory / FILENAMES[taxonomy]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python scripts/fetch_taxonomy.py` to build it."
        )

    codes: list[TaxonomyCode] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if row.get("year_retired"):
                continue
            row.pop("year_retired", None)
            codes.append(TaxonomyCode(**row))

    if not codes:
        raise ValueError(f"{path} contained no usable rows")
    return TaxonomyCatalog(taxonomy, codes)
