"""Build data/arxiv.jsonl and data/eval/arxiv_papers.jsonl.

Why this corpus exists
----------------------
It is the control. The procurement corpora measured badly, and a single bad
result cannot tell you whether the pipeline is weak or the task is hard. Running
the identical pipeline against a corpus where the text really does determine the
label separates those two explanations.

arXiv fits because the abstract is written to say what the paper is about, the
author picks the primary category, and arXiv publishes a prose description of
every category. That is the opposite of a federal award description, which is an
internal memo whose code was set by a contract vehicle.

    python scripts/fetch_arxiv.py

Both the taxonomy page and the API are public and need no key. The API asks
callers for a delay of a few seconds between requests, which this honours.
"""

from __future__ import annotations

import html
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import urllib.request

from _http import write_jsonl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TAXONOMY_URL = "https://arxiv.org/category_taxonomy"
API_URL = "https://export.arxiv.org/api/query"

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

API_DELAY_SECONDS = 3.0
"""arXiv asks for a few seconds between calls. Respected rather than tuned."""

PER_CATEGORY = 16
MAX_PER_CATEGORY = 10
TARGET_ROWS = 400
MAX_DESCRIPTION_CHARS = 400

# Spread across groups on purpose. A sample drawn only from cs would make the
# task look easier than it is, because a classifier that never has to decide
# between machine learning and econometrics is not being asked much.
SEED_CATEGORIES = [
    "cs.AI", "cs.CL", "cs.CR", "cs.CV", "cs.DB", "cs.DC", "cs.DS", "cs.GT",
    "cs.HC", "cs.IR", "cs.LG", "cs.NI", "cs.PL", "cs.RO", "cs.SE",
    "math.AG", "math.AP", "math.CO", "math.NT", "math.OC", "math.PR", "math.ST",
    "astro-ph.CO", "astro-ph.GA", "astro-ph.SR",
    "cond-mat.mes-hall", "cond-mat.soft", "cond-mat.stat-mech",
    "hep-ph", "hep-th", "gr-qc", "quant-ph", "nucl-th",
    "physics.optics", "physics.flu-dyn", "physics.bio-ph",
    "q-bio.NC", "q-bio.PE", "q-fin.PM", "q-fin.ST",
    "stat.ME", "stat.ML", "eess.AS", "eess.IV", "eess.SP", "econ.EM",
]

GROUP_RE = re.compile(
    r'<h2 class="accordion-head"[^>]*>.*?<button[^>]*>.*?</span>\s*(.*?)\s*</button>',
    re.S,
)
CATEGORY_RE = re.compile(
    r"<h4>([\w.\-]+)\s*<span>\(([^)]*)\)</span></h4>"
    r".*?<div class=\"column\"><p>(.*?)</p>",
    re.S,
)
TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    return " ".join(html.unescape(TAG_RE.sub(" ", text)).split())


def _truncate(text: str, limit: int) -> str:
    """Cut to a word boundary. Long descriptions distort BM25 length
    normalisation and dilute the embedding away from the category name."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[: cut.rfind(" ")].rstrip(" ,;:.") if " " in cut else cut


def fetch_taxonomy() -> list[dict[str, Any]]:
    request = urllib.request.Request(
        TAXONOMY_URL, headers={"User-Agent": "procurement-code-agent/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        page = response.read().decode("utf-8", errors="replace")

    # Split the page per group so each category inherits the right heading.
    group_names = GROUP_RE.findall(page)
    sections = GROUP_RE.split(page)[1:]
    rows: list[dict[str, Any]] = []
    for group_name, section in zip(group_names, sections[1::2] or sections, strict=False):
        for code, title, description in CATEGORY_RE.findall(section):
            rows.append(
                {
                    "taxonomy": "arxiv",
                    "code": code,
                    "title": _clean(title),
                    "parent_code": code.split(".")[0],
                    "parent_title": _clean(group_name),
                    "category": _clean(group_name),
                    "description": _truncate(_clean(description), MAX_DESCRIPTION_CHARS),
                }
            )

    by_code = {row["code"]: row for row in rows}
    return sorted(by_code.values(), key=lambda row: row["code"])


def fetch_papers(category: str) -> list[dict[str, Any]]:
    query = (
        f"{API_URL}?search_query=cat:{category}"
        f"&start=0&max_results={PER_CATEGORY}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    request = urllib.request.Request(
        query, headers={"User-Agent": "procurement-code-agent/0.1"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        feed = ET.fromstring(response.read())

    papers = []
    for entry in feed.findall(f"{ATOM}entry"):
        primary = entry.find(f"{ARXIV}primary_category")
        summary = entry.findtext(f"{ATOM}summary") or ""
        title = entry.findtext(f"{ATOM}title") or ""
        identifier = entry.findtext(f"{ATOM}id") or ""
        if primary is None or not summary.strip():
            continue
        papers.append(
            {
                "paper_id": identifier.rsplit("/", 1)[-1],
                "title": _clean(title),
                "abstract": _clean(summary),
                # The author's own primary category, not the category we queried.
                # Cross listed papers would otherwise be silently relabelled to
                # whatever we happened to ask for, which would make the eval a
                # measure of our query plan rather than of the classifier.
                "label": primary.get("term", ""),
            }
        )
    return papers


def split_for(paper_id: str) -> str:
    import hashlib

    digest = hashlib.sha256(paper_id.encode()).hexdigest()
    return "dev" if int(digest[:8], 16) % 100 < 35 else "test"


def main() -> None:
    os.makedirs(os.path.join(DATA_DIR, "eval"), exist_ok=True)

    taxonomy = fetch_taxonomy()
    if len(taxonomy) < 100:
        raise SystemExit(f"only parsed {len(taxonomy)} categories, the page layout changed")
    write_jsonl(os.path.join(DATA_DIR, "arxiv.jsonl"), taxonomy)
    print(f"wrote {len(taxonomy)} arxiv categories")

    known = {row["code"] for row in taxonomy}
    per_label: Counter[str] = Counter()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    for position, category in enumerate(SEED_CATEGORIES, start=1):
        try:
            papers = fetch_papers(category)
        except Exception as exc:
            print(f"  {category}: failed, {exc}")
            time.sleep(API_DELAY_SECONDS)
            continue

        for paper in papers:
            label = paper["label"]
            if label not in known or paper["paper_id"] in seen:
                continue
            if per_label[label] >= MAX_PER_CATEGORY:
                continue
            seen.add(paper["paper_id"])
            per_label[label] += 1
            rows.append(
                {
                    "paper_id": paper["paper_id"],
                    # Title and abstract together, which is what a human sees
                    # when deciding where a paper belongs.
                    "description": f"{paper['title']}. {paper['abstract']}",
                    "label": label,
                    "split": split_for(paper["paper_id"]),
                }
            )
        print(f"  [{position}/{len(SEED_CATEGORIES)}] {category}: {len(rows)} rows so far")
        time.sleep(API_DELAY_SECONDS)

    rows.sort(key=lambda row: row["paper_id"])
    rows = rows[:TARGET_ROWS]
    write_jsonl(os.path.join(DATA_DIR, "eval", "arxiv_papers.jsonl"), rows)

    splits = Counter(row["split"] for row in rows)
    print(f"wrote {len(rows)} papers across {len(per_label)} categories, splits {dict(splits)}")


if __name__ == "__main__":
    main()
