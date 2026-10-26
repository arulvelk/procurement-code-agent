"""Build data/eval/labeled_awards.jsonl from real federal contract awards.

Every row is a purchase description written by a contracting officer, paired with
the PSC and NAICS codes that officer actually assigned. That is the whole reason
this repo can report accuracy at all rather than asking a model to grade itself.

A caveat that belongs in front of the numbers, not behind them: these labels are
not a gold standard. They are what the agency recorded, and agencies miscode.
Spot checking turns up descriptions whose assigned code is defensible at best.
The evals therefore report agreement with the recorded code, which is a floor on
real accuracy rather than an estimate of it.

    python scripts/build_eval_set.py
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _http import request_json, write_jsonl

API = "https://api.usaspending.gov/api/v2"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "data", "eval", "labeled_awards.jsonl")

TARGET_ROWS = 400
PAGE_SIZE = 100
PAGES = range(1, 9)
"""Pages per window. Results are sorted by award value descending, so the first
page of any quarter is mega contracts for aircraft and R&D. Going eight pages
deep is what brings in the ordinary office furniture and janitorial work that
makes the sample look like federal purchasing rather than like the defence
budget."""
MAX_PER_PSC_GROUP = 14
"""Cap per two character PSC group. Federal spending is wildly concentrated, and
without a cap roughly a third of any random sample is professional services,
which would let a model score well by learning one answer."""

# Quarters spread over three years, so the sample is not one procurement cycle.
WINDOWS = [
    ("2022-01-01", "2022-03-31"),
    ("2022-07-01", "2022-09-30"),
    ("2023-01-01", "2023-03-31"),
    ("2023-07-01", "2023-09-30"),
    ("2024-01-01", "2024-03-31"),
    ("2024-07-01", "2024-09-30"),
    ("2025-01-01", "2025-03-31"),
    ("2025-07-01", "2025-09-30"),
]

# Federal descriptions are full of filler. These are the common ones.
BOILERPLATE = re.compile(
    r"^(igf::\w+::igf|n/?a|none|see attach\w*|no description|not available|\W*)$",
    re.IGNORECASE,
)
IGF_PREFIX = re.compile(r"^igf::\w+::igf\s*", re.IGNORECASE)

MIN_DESCRIPTION_CHARS = 25


def fetch_page(start: str, end: str, page: int) -> list[dict[str, Any]]:
    payload = {
        "filters": {
            "award_type_codes": ["A", "B", "C", "D"],
            "time_period": [{"start_date": start, "end_date": end}],
        },
        "fields": [
            "Award ID",
            "Description",
            "naics_code",
            "naics_description",
            "psc_code",
            "psc_description",
            "Award Amount",
        ],
        "page": page,
        "limit": PAGE_SIZE,
        "sort": "Award Amount",
        "order": "desc",
    }
    return request_json(f"{API}/search/spending_by_award/", payload=payload)["results"]


def clean_description(raw: str | None) -> str | None:
    text = " ".join((raw or "").split())
    text = IGF_PREFIX.sub("", text).strip()
    if len(text) < MIN_DESCRIPTION_CHARS or BOILERPLATE.match(text):
        return None
    return text


def split_for(award_id: str) -> str:
    """Assign dev or test deterministically from the award id.

    Hashing the id rather than shuffling means the split survives regenerating
    the file, so a number tuned on dev can never quietly leak into test.
    """
    digest = hashlib.sha256(award_id.encode()).hexdigest()
    return "dev" if int(digest[:8], 16) % 100 < 35 else "test"


def main() -> None:
    seen_descriptions: set[str] = set()
    per_group: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for start, end in WINDOWS:
        for page in PAGES:
            for record in fetch_page(start, end, page):
                description = clean_description(record.get("Description"))
                psc = (record.get("psc_code") or "").strip().upper()
                naics = (record.get("naics_code") or "").strip()
                award_id = (record.get("Award ID") or "").strip()

                if not (description and psc and naics and award_id):
                    continue
                if len(psc) != 4 or len(naics) != 6:
                    continue

                key = description.lower()
                if key in seen_descriptions:
                    continue
                group = psc[:2]
                if per_group[group] >= MAX_PER_PSC_GROUP:
                    continue

                seen_descriptions.add(key)
                per_group[group] += 1
                rows.append(
                    {
                        "award_id": award_id,
                        "description": description,
                        "psc_code": psc,
                        "psc_title": " ".join((record.get("psc_description") or "").split()),
                        "naics_code": naics,
                        "naics_title": " ".join((record.get("naics_description") or "").split()),
                        "amount": record.get("Award Amount"),
                        "split": split_for(award_id),
                    }
                )
        print(f"{start}..{end}: {len(rows)} rows kept so far")

    rows.sort(key=lambda row: row["award_id"])
    rows = rows[:TARGET_ROWS]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    write_jsonl(OUT_PATH, rows)

    splits = Counter(row["split"] for row in rows)
    groups = Counter(row["psc_code"][:2] for row in rows)
    print(f"wrote {len(rows)} rows to {OUT_PATH}")
    print(f"  splits: {dict(splits)}")
    print(f"  distinct psc groups: {len(groups)}, most common: {groups.most_common(5)}")


if __name__ == "__main__":
    main()
