"""Build data/psc.jsonl and data/naics.jsonl from the USAspending reference API.

Run once. The output is committed so that the repo, its tests and its evals work
with no network access at all.

    python scripts/fetch_taxonomy.py

PSC is walked as category -> two digit group -> four character code. NAICS is
walked as two digit sector -> four digit industry group -> six digit industry.
Both walks keep the parent titles, because a bare code title like "GUNS, OVER
300MM" retrieves far better when the group title "WEAPONS" travels with it.
"""

from __future__ import annotations

import os
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _http import quote, request_json, write_jsonl

API = "https://api.usaspending.gov/api/v2"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Serial. Responses are cached on disk by _http, so a rerun costs nothing and
# there is no reason to lean on someone else's free public API for speed.
WORKERS = 1
PSC_CODE_LENGTH = 4


def fetch_psc() -> list[dict[str, Any]]:
    """Walk the PSC tree to its leaves.

    The three top level categories do not nest to the same depth. Products go
    category -> two digit group -> four digit code, while services and R&D carry
    an extra level (B -> B5 -> B502). So rather than assume a depth, expand level
    by level.

    Every PSC leaf is exactly four characters, so a node whose id has reached four
    characters is never expanded. That single rule drops roughly 1800 pointless
    requests, because the API answers a request for a leaf's children with a 404
    rather than an empty list.
    """
    tree = request_json(f"{API}/references/filter_tree/psc/")
    categories = [node["id"] for node in tree["results"]]
    print(f"psc: {len(categories)} categories -> {categories}")

    def expand(path: tuple[str, ...]) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
        url = f"{API}/references/filter_tree/psc/" + "".join(quote(seg) + "/" for seg in path)
        try:
            return path, request_json(url)["results"]
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return path, []  # the API 404s rather than returning an empty leaf
            raise

    nodes: dict[tuple[str, ...], dict[str, Any]] = {}
    leaves: list[tuple[str, ...]] = []
    frontier: list[tuple[str, ...]] = [(category,) for category in categories]

    level = 0
    while frontier:
        level += 1
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            expanded = list(pool.map(expand, frontier))

        next_frontier: list[tuple[str, ...]] = []
        for path, children in expanded:
            for child in children:
                child_path = (*path, child["id"])
                nodes[child_path] = child
                if len(child["id"]) >= PSC_CODE_LENGTH:
                    leaves.append(child_path)
                else:
                    next_frontier.append(child_path)

        print(
            f"psc: level {level} expanded {len(frontier)} nodes, "
            f"{len(leaves)} leaves so far, {len(next_frontier)} still to walk"
        )
        frontier = next_frontier

    rows = []
    for path in leaves:
        node = nodes[path]
        parent = nodes.get(path[:-1])
        rows.append(
            {
                "taxonomy": "psc",
                "code": node["id"],
                "title": _clean(node["description"]),
                "parent_code": path[-2],
                "parent_title": _clean(parent["description"]) if parent else "",
                "category": path[0],
            }
        )
    return _dedupe(rows)


def fetch_naics() -> list[dict[str, Any]]:
    sectors = request_json(f"{API}/references/naics/")["results"]
    print(f"naics: {len(sectors)} sectors")

    def industry_groups(sector: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        node = request_json(f"{API}/references/naics/{quote(sector['naics'])}/")["results"][0]
        return [(sector, child) for child in node.get("children") or []]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pairs = [pair for batch in pool.map(industry_groups, sectors) for pair in batch]
    print(f"naics: {len(pairs)} industry groups")

    def industries(pair: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
        sector, group = pair
        node = request_json(f"{API}/references/naics/{quote(group['naics'])}/")["results"][0]
        return [
            {
                "taxonomy": "naics",
                "code": child["naics"],
                "title": _clean(child["naics_description"]),
                "parent_code": group["naics"],
                "parent_title": _clean(group["naics_description"]),
                "category": _clean(sector["naics_description"]),
                "year_retired": child.get("year_retired"),
            }
            for child in node.get("children") or []
        ]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = [row for batch in pool.map(industries, pairs) for row in batch]

    return _dedupe(rows)


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per code and sort, so reruns produce byte identical files."""
    by_code = {row["code"]: row for row in rows}
    return sorted(by_code.values(), key=lambda row: row["code"])


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    psc = fetch_psc()
    write_jsonl(os.path.join(DATA_DIR, "psc.jsonl"), psc)
    print(f"wrote {len(psc)} psc codes")

    naics = fetch_naics()
    write_jsonl(os.path.join(DATA_DIR, "naics.jsonl"), naics)
    print(f"wrote {len(naics)} naics codes")


if __name__ == "__main__":
    main()
