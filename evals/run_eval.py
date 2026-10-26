"""Run the eval suite and write evals/report.md.

    python evals/run_eval.py                       # retrieval ablation, PSC, test split
    python evals/run_eval.py --taxonomy naics
    python evals/run_eval.py --decider claude      # adds the end to end model run
    python evals/run_eval.py --split dev --limit 50

What it measures, and why in this order:

1. **The retrieval ablation.** Four pipelines over the same queries: BM25 alone,
   dense alone, the two fused, and the fused list reranked by a cross encoder.
   This is the table that says whether each stage is paying for itself. A stage
   that does not move recall is a stage to delete.
2. **End to end agreement.** What fraction of final answers matched the code the
   contracting officer recorded, for whichever decider is configured. The
   baseline runs with no API key, so this column always has a number in it, and
   any model result has something to be compared against.
3. **Calibration and routing.** Whether the confidence score means anything, and
   what a given review threshold actually buys in automation and accuracy.

Splits are fixed in the data file and assigned by hashing the award id, so dev
can be tuned on without leaking into the test numbers quoted in the README.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Pinned before fastembed is imported so that two runs on the same machine are
# comparable. Left unset, ONNX picks a thread count from the current load.
os.environ.setdefault("PCAGENT_ONNX_THREADS", "4")

from metrics import (  # noqa: E402
    Calibration,
    LatencySummary,
    RoutingOutcome,
    mean_reciprocal_rank,
    recall_at_k,
)

from pcagent.bootstrap import RETRIEVAL_SETTINGS  # noqa: E402
from pcagent.domain import Candidate, Taxonomy  # noqa: E402
from pcagent.llm.base import BaselineDecider, Decider, DecisionRequest  # noqa: E402
from pcagent.retrieval.dense import DenseIndex  # noqa: E402
from pcagent.retrieval.hybrid import RetrievalSettings, reciprocal_rank_fusion  # noqa: E402
from pcagent.retrieval.lexical import BM25Index  # noqa: E402
from pcagent.retrieval.rerank import CrossEncoderReranker  # noqa: E402
from pcagent.taxonomy.loader import load_catalog  # noqa: E402

EVAL_FILES: dict[str, tuple[str, str]] = {
    # taxonomy -> (eval file, the field holding the true label)
    "psc": ("labeled_awards.jsonl", "psc_code"),
    "naics": ("labeled_awards.jsonl", "naics_code"),
    "arxiv": ("arxiv_papers.jsonl", "label"),
}
EVAL_DIR = ROOT / "data" / "eval"
REPORT_DIR = ROOT / "evals"

RECALL_DEPTHS = (1, 5, 10, 20)
STRATEGIES = ("bm25", "dense", "hybrid", "hybrid+rerank")


@dataclass
class Row:
    award_id: str
    description: str
    truth: str


def load_rows(taxonomy: Taxonomy, split: str, limit: int | None) -> list[Row]:
    filename, field = EVAL_FILES[taxonomy]
    path = EVAL_DIR / filename
    if not path.exists():
        builder = (
            "scripts/fetch_arxiv.py" if taxonomy == "arxiv" else "scripts/build_eval_set.py"
        )
        raise SystemExit(f"{path} is missing. Run `python {builder}` first.")

    rows: list[Row] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if split != "all" and record["split"] != split:
                continue
            rows.append(
                Row(
                    # arXiv labels are case sensitive (cs.LG, not CS.LG), so the
                    # catalogue side is upper cased to match rather than the
                    # other way round.
                    award_id=record.get("award_id") or record["paper_id"],
                    description=record["description"],
                    truth=record[field].strip().upper(),
                )
            )
    return rows[:limit] if limit else rows


def rank_of(truth: str, ordering: list[tuple[int, float]], codes: list[str]) -> int | None:
    for position, (index, _score) in enumerate(ordering, start=1):
        if codes[index] == truth:
            return position
    return None


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", choices=("psc", "naics", "arxiv"), default="psc")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decider", choices=("baseline", "claude"), default="baseline")
    parser.add_argument("--depth", type=int, default=RetrievalSettings().rerank_depth)
    args = parser.parse_args()

    taxonomy: Taxonomy = args.taxonomy
    rows = load_rows(taxonomy, args.split, args.limit)
    if not rows:
        raise SystemExit(f"no rows for split {args.split!r}")

    catalog = load_catalog(taxonomy)
    codes = [code.code.upper() for code in catalog.codes]
    documents = [code.index_text for code in catalog.codes]

    print(f"{taxonomy}: {len(rows)} rows from the {args.split} split, {len(codes)} codes")

    in_corpus = sum(1 for row in rows if row.truth in set(codes))
    if in_corpus < len(rows):
        # Worth printing loudly. A label that is not in the corpus is an
        # automatic miss, and quoting a recall number without saying so would
        # be quoting a number that no pipeline change can ever improve.
        print(
            f"  note: {len(rows) - in_corpus} labels are not in the current corpus "
            "(retired or reclassified codes). They count as misses."
        )

    # The ablation always measures all four strategies. What the corpus is
    # configured to actually serve decides which of them feeds the decider.
    configured = RETRIEVAL_SETTINGS.get(taxonomy, RetrievalSettings())
    settings = RetrievalSettings(
        rerank_depth=args.depth,
        rerank_enabled=configured.rerank_enabled,
        rerank_query_chars=configured.rerank_query_chars,
    )
    serving = "hybrid+rerank" if settings.rerank_enabled else "hybrid"
    print(f"  serving strategy for this corpus: {serving}")
    bm25 = BM25Index(documents)
    dense = DenseIndex(documents)
    reranker = CrossEncoderReranker(threads=int(os.environ["PCAGENT_ONNX_THREADS"]))

    print("warming models ...")
    warm_started = time.perf_counter()
    await asyncio.to_thread(dense.warm)
    await asyncio.to_thread(reranker.warm)
    print(f"  warm in {time.perf_counter() - warm_started:.1f}s")

    ranks: dict[str, list[int | None]] = {name: [] for name in STRATEGIES}
    stage_latency: dict[str, list[float]] = {name: [] for name in STRATEGIES}
    shortlists: list[list[Candidate]] = []

    for position, row in enumerate(rows, start=1):
        if position % 50 == 0:
            print(f"  retrieval {position}/{len(rows)}")

        started = time.perf_counter()
        lexical_hits = bm25.search(row.description, settings.lexical_k)
        stage_latency["bm25"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        dense_hits = await asyncio.to_thread(
            dense.search, row.description, settings.dense_k
        )
        stage_latency["dense"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [hits for hits in (lexical_hits, dense_hits) if hits], rrf_k=settings.rrf_k
        )
        stage_latency["hybrid"].append(
            stage_latency["bm25"][-1]
            + stage_latency["dense"][-1]
            + (time.perf_counter() - started) * 1000
        )

        head = fused[: settings.rerank_depth]
        started = time.perf_counter()
        scores = await asyncio.to_thread(
            reranker.score,
            row.description[: settings.rerank_query_chars],
            [documents[i] for i, _ in head],
        )
        rerank_ms = (time.perf_counter() - started) * 1000
        stage_latency["hybrid+rerank"].append(stage_latency["hybrid"][-1] + rerank_ms)

        reranked = sorted(
            zip((i for i, _ in head), scores, strict=True), key=lambda p: -p[1]
        )
        final = reranked + fused[settings.rerank_depth :]

        ranks["bm25"].append(rank_of(row.truth, lexical_hits, codes))
        ranks["dense"].append(rank_of(row.truth, dense_hits, codes))
        ranks["hybrid"].append(rank_of(row.truth, fused, codes))
        ranks["hybrid+rerank"].append(rank_of(row.truth, final, codes))

        fused_rank = {index: rank for rank, (index, _) in enumerate(fused, start=1)}
        served = final if settings.rerank_enabled else fused
        shortlists.append(
            [
                Candidate(
                    code=catalog.codes[index].code,
                    title=catalog.codes[index].title,
                    taxonomy=taxonomy,
                    score=float(score),
                    retrieved_rank=fused_rank[index],
                    reranked_rank=slot if settings.rerank_enabled else None,
                )
                for slot, (index, score) in enumerate(served[: settings.final_k], start=1)
            ]
        )

    decider: Decider = BaselineDecider()
    if args.decider == "claude":
        from pcagent.llm.claude import LLMDecider

        decider = LLMDecider(fallback_to_baseline=False)

    print(f"deciding with {decider.name} ...")
    decisions: list[tuple[float, bool]] = []
    decide_latency: list[float] = []
    correct = 0
    for position, (row, shortlist) in enumerate(
        zip(rows, shortlists, strict=True), start=1
    ):
        if position % 50 == 0:
            print(f"  decide {position}/{len(rows)}")
        started = time.perf_counter()
        decision = await decider.decide(
            DecisionRequest(
                description=row.description,
                taxonomy=taxonomy,
                candidates=tuple(shortlist),
            )
        )
        decide_latency.append((time.perf_counter() - started) * 1000)
        hit = decision.code.upper() == row.truth
        correct += int(hit)
        decisions.append((decision.confidence, hit))

    accuracy = correct / len(rows)
    calibration = Calibration.build(decisions)
    routing = [RoutingOutcome.at(decisions, t) for t in (0.5, 0.65, 0.8, 0.9)]

    report = render_report(
        taxonomy=taxonomy,
        split=args.split,
        rows=len(rows),
        corpus_size=len(codes),
        missing_labels=len(rows) - in_corpus,
        depth=settings.rerank_depth,
        ranks=ranks,
        stage_latency=stage_latency,
        decider_name=decider.name,
        serving=serving,
        accuracy=accuracy,
        decide_latency=decide_latency,
        calibration=calibration,
        routing=routing,
    )
    # The split is in the filename. Without it a dev run silently overwrites
    # the test report, which is exactly the kind of quiet mix up that puts a
    # tuned-on number in a README.
    report_path = REPORT_DIR / f"report-{taxonomy}-{args.split}.md"
    report_path.write_text(report, encoding="utf-8")

    results_path = REPORT_DIR / f"results-{taxonomy}-{args.split}.json"
    results_path.write_text(
        json.dumps(
            {
                "taxonomy": taxonomy,
                "split": args.split,
                "rows": len(rows),
                "decider": decider.name,
                "accuracy_at_1": accuracy,
                "recall": {
                    name: {f"@{k}": recall_at_k(ranks[name], k) for k in RECALL_DEPTHS}
                    for name in STRATEGIES
                },
                "mrr": {name: mean_reciprocal_rank(ranks[name]) for name in STRATEGIES},
                "expected_calibration_error": calibration.expected_calibration_error,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(report)
    print(f"written to {report_path} and {results_path}")


def render_report(
    *,
    taxonomy: str,
    split: str,
    rows: int,
    corpus_size: int,
    missing_labels: int,
    depth: int,
    ranks: dict[str, list[int | None]],
    stage_latency: dict[str, list[float]],
    decider_name: str,
    serving: str,
    accuracy: float,
    decide_latency: list[float],
    calibration: Calibration,
    routing: list[RoutingOutcome],
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Eval report")
    add("")
    add(f"Generated by `python evals/run_eval.py --taxonomy {taxonomy} --split {split}`.")
    add("")
    add(f"- Taxonomy: **{taxonomy}**, {corpus_size} active codes")
    add(f"- Split: **{split}**, {rows} labelled examples")
    add(f"- Rerank depth: {depth} candidates")
    add(f"- Decider: `{decider_name}`")
    add(f"- Strategy served to the decider: **{serving}**")
    if missing_labels:
        add(
            f"- {missing_labels} labels are not in the current corpus and count as "
            "misses for every strategy"
        )
    add("")
    add("## Retrieval ablation")
    add("")
    add("Does each stage pay for itself? Recall is the ceiling on the final answer.")
    add("")
    header = "| strategy | " + " | ".join(f"recall@{k}" for k in RECALL_DEPTHS)
    add(header + " | MRR | p50 ms | p95 ms |")
    add("|---" * (len(RECALL_DEPTHS) + 4) + "|")
    for name in STRATEGIES:
        cells = [f"{recall_at_k(ranks[name], k):.3f}" for k in RECALL_DEPTHS]
        latency = LatencySummary.of(stage_latency[name])
        add(
            f"| {name} | "
            + " | ".join(cells)
            + f" | {mean_reciprocal_rank(ranks[name]):.3f}"
            + f" | {latency.p50_ms:.1f} | {latency.p95_ms:.1f} |"
        )
    add("")
    add(
        "Reranking cannot change recall@20 when the rerank depth is 30, because it "
        "only reorders what fusion already found. Read it on recall@1 and MRR."
    )
    add("")
    add("## End to end")
    add("")
    decide = LatencySummary.of(decide_latency)
    add(f"- Agreement with the recorded code (accuracy@1): **{accuracy:.3f}**")
    add(f"- Decision latency: p50 {decide.p50_ms:.1f}ms, p95 {decide.p95_ms:.1f}ms")
    add(f"- Expected calibration error: **{calibration.expected_calibration_error:.3f}**")
    add("")
    add("### Calibration")
    add("")
    add("| confidence | n | mean confidence | actual accuracy | gap |")
    add("|---|---|---|---|---|")
    for bucket in calibration.buckets:
        if not bucket.count:
            continue
        gap = bucket.mean_confidence - bucket.accuracy
        add(
            f"| {bucket.label} | {bucket.count} | {bucket.mean_confidence:.3f} "
            f"| {bucket.accuracy:.3f} | {gap:+.3f} |"
        )
    add("")
    add("A positive gap is overconfidence, which is the dangerous direction.")
    add("")
    add("### What a review threshold buys")
    add("")
    add(
        "| threshold | auto handled | accuracy of auto slice "
        "| sent to review | accuracy of that slice |"
    )
    add("|---|---|---|---|---|")
    for outcome in routing:
        add(
            f"| {outcome.threshold:.2f} | {outcome.auto_count} "
            f"({outcome.automation_rate:.0%}) | {outcome.auto_accuracy:.3f} "
            f"| {outcome.review_count} | {outcome.review_accuracy:.3f} |"
        )
    add("")
    add(
        "The right threshold is the lowest one whose auto slice clears the accuracy "
        "the process needs. Overall accuracy is the wrong number to optimise."
    )
    add("")
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
