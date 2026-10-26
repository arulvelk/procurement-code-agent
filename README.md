# procurement-code-agent

An MCP server, an HTTP API and a small agent that classify free text purchase
descriptions into United States federal procurement codes.

The classification problem is the excuse. What I wanted to show is the part that
usually gets skipped in agent demos: typed tool contracts, authorization
enforced where a model cannot reach it, idempotency on the one call that
changes anything, async that actually yields, and an eval suite honest enough to
have changed my mind about the design three times while I was building it.

It runs on a laptop with no API key. `pip install -e ".[dev,graph]" && pytest`
is the whole setup.

---

## Quickstart

```bash
python -m venv .venv && . .venv/Scripts/activate   # or .venv/bin/activate
pip install -e ".[dev,graph]"
pytest
```

96 tests, under 10 seconds, no network and no model downloads.

To run the service:

```bash
uvicorn pcagent.api:create_app --factory --port 8000
```

First start downloads about 210MB of ONNX embedding weights and warms the
indexes, which takes roughly 20 seconds. After that it is CPU only and offline.

```bash
curl -s localhost:8000/v1/classify \
  -H "Authorization: Bearer dev-reader-token" \
  -H "Content-Type: application/json" \
  -d '{"description":"JANITORIAL AND CUSTODIAL SERVICES FOR FEDERAL BUILDING"}'
```

```json
{
  "classification_id": "cls_df74127ea8d3fbac",
  "taxonomy": "psc",
  "code": "S201",
  "title": "HOUSEKEEPING- CUSTODIAL JANITORIAL",
  "confidence": 0.8149,
  "alternatives": [
    { "code": "M111", "title": "OPERATION OF OFFICE BUILDINGS" },
    { "code": "M1AA", "title": "OPERATION OF OFFICE BUILDINGS" }
  ],
  "decided_by": "baseline-top1",
  "latency_ms": 19
}
```

`classification_id` is derived from the taxonomy and the normalised
description, so the same text asked twice is the same classification and the
record is updated rather than duplicated. `decided_by` says what produced the
answer, so a degraded or fallback result is never mistaken for a model one.

As an MCP server, for Claude Desktop or any other client:

```json
{
  "mcpServers": {
    "procurement-codes": {
      "command": "pcagent-mcp",
      "env": { "PCAGENT_TOKEN": "dev-reviewer-token" }
    }
  }
}
```

---

## The five decisions worth arguing about

### 1. The tool layer is the security boundary, not the prompt

Every tool has the same shape:

```python
async def submit_review(ctx: ToolContext, params: SubmitReviewParams) -> Review:
    ctx.require(Scope.REVIEWS_WRITE)
    ...
```

`params` is what the caller, model included, gets to specify. `ctx` is what only
the transport can supply, because it is built after authentication.

The consequence is the bit I care about. `submit_review` has no `reviewer`
field. The reviewer is `ctx.principal.subject`, read from the token. A model
cannot record a decision in someone else's name because there is nowhere to put
the name. Prompt injection against this agent can at worst produce a badly
chosen tool call. It cannot produce an unauthorised one.

The scope check is the first line of the function body rather than a decorator.
A decorator is invisible at the call site and silently stops applying the day
someone calls the underlying function directly. Instead the decorator only
records the scope, and a test walks the registry, calls every tool with a
principal holding every scope except the declared one, and fails if the call
succeeds. A tool that forgets its check passes every other test in the suite and
fails that one.

### 2. Two transports, one core

`api.py` and `mcp_server.py` are both thin. Neither contains a business rule.
That is what lets the same capability be offered over HTTP and MCP without the
two drifting apart.

They differ in exactly one place, on purpose. Over HTTP the idempotency key
arrives in the `Idempotency-Key` header, where an HTTP client expects to put it.
Over MCP it is a tool argument, because MCP has no headers. Same guarantee, each
transport spelling it the way its callers already speak.

Identity follows the same pattern. HTTP resolves a bearer token per request. An
MCP server over stdio is a process started on behalf of one user, so the
principal is resolved once at startup from the environment. Both get it from the
thing that authenticated the session, and neither gets it from the model.

### 3. Idempotency belongs in the transaction, not above it

`submit_review` is the only tool that changes anything, and it is exactly the
kind of call an agent retries. A timeout on the caller's side says nothing about
whether the write landed.

The protocol is the one payment APIs settled on, and it lives in `store/db.py`:

- `begin_idempotent` reserves the key inside a write transaction. First caller
  gets `Proceed`. Same key and same body gets `Replay` with the original
  response. Same key and a **different** body gets a 409, because one of the two
  is a bug and guessing which would be worse than failing.
- `complete_idempotent` writes the review and stores the replayable response in
  one transaction. A crash cannot leave a review recorded with no way to replay
  it, nor a replayable response for a review that was never recorded.
- `release_idempotent` hands the key back when the work failed, so an honest
  retry is not locked out.

Three details that only show up once it is running:

**Keys are scoped per caller.** Two callers both using `retry-1` are making two
different requests. A global key space would hand the second one the first
one's answer.

**Validation happens before the reservation.** Otherwise a caller who sent a bad
body once could never retry with a corrected one under the same key, which is
precisely what a client library with automatic retries does.

**A shared connection needs a lock as well as `BEGIN IMMEDIATE`.** I had this
wrong. A transaction is a property of the connection, not of the coroutine, so
eight coroutines racing on the same key all issued `BEGIN IMMEDIATE` on one
connection and got `cannot start a transaction within a transaction`. The
asyncio lock serialises coroutines sharing the connection. `BEGIN IMMEDIATE`
serialises this process against any other process on the file. Both are needed,
and `test_only_one_of_many_racing_callers_proceeds` is the test that found it.

### 4. Async that actually yields

The ONNX embedding and reranking calls are synchronous and CPU bound. Wrapping
them in `async def` would not make them yield, so they go through
`asyncio.to_thread`, including during startup, because startup is when a
readiness probe is waiting.

BM25 runs inline. It is tens of microseconds over a few thousand short
documents, and a thread hop would cost more than the work. The rule is do not
block the loop, not never compute on the loop.

This is tested rather than asserted. `test_async_discipline.py` puts a
`time.sleep` inside a stub index and counts heartbeats from a coroutine ticking
alongside it. A loop that stayed responsive gets most of its ticks, a blocked
one gets almost none. The assertion is on the heartbeat, so the test still means
something after a refactor.

The batch agent in `agent/batch.py` is hand written for the same reason: the
interesting decisions are all about restraint. A semaphore caps in flight work,
`asyncio.wait_for` puts a deadline on every item, each failure is caught per
item so one bad row cannot cancel nine thousand others, and results come back in
input order regardless of completion order. There is a LangGraph version of the
same flow in `agent/graph.py`, which exists to show the tools do not care what
is orchestrating them.

### 5. Retrieval is hybrid, and the reranker did not survive the measurement

BM25 and a dense index both run over the whole corpus and their rankings are
fused with reciprocal rank fusion.

Fusion is by rank, not by score, because there is no honest way to put an
unbounded sum of IDF terms on the same scale as a cosine similarity that lives
between 0.6 and 0.7. Normalising each to 0 and 1 per query just invents a scale
that changes with every query. RRF throws the scores away and keeps the
ordering, which is the part both rankers agree is meaningful.

There is also a cross encoder reranking stage, fully built, and it is **off**.
That was not the plan, and the sequence is the point:

1. Built retrieve, fuse, rerank, the standard shape.
2. Measured three rerankers. ms-marco-MiniLM-L-6 did nothing at all, spreading
   four candidates across a single logit on a query with one obviously correct
   answer. jina-reranker-v1-turbo was worth 28% relative recall@5. Switched.
3. Measured two embedders and upgraded bge-small to bge-base.
4. Re-measured the reranker against the better embedder, and it now **lowered**
   MRR on all three corpora. PSC 0.123 to 0.068. NAICS 0.171 to 0.157. arXiv
   0.398 to 0.251.
5. Turned it off, for 170ms a query back and better results.

A cross encoder earns its place by fixing a bi-encoder's mistakes. Once the
bi-encoder stopped making those mistakes, the reranker was left reordering
candidates it understood worse than the embedder did. The stage stays in the
codebase and in the ablation rather than being deleted, because this is a
property of these corpora and these models, and swapping either could flip the
answer back. The ablation is what would tell me.

Both ONNX stages are optional at runtime. If the weights cannot be loaded,
retrieval falls back to BM25 alone and says so on the result rather than
failing. Someone who clones this on a plane gets answers.

---

## What the evals say

`evals/run_eval.py` measures four pipelines over the same queries, then end to
end agreement, calibration, and what a review threshold actually buys.

### The headline

Test split, served pipeline (BM25 and dense, fused, no reranking):

| corpus | classes | recall@1 | recall@5 | recall@20 | MRR | calibration error |
|---|---|---|---|---|---|---|
| arXiv categories | 155 | 0.237 | 0.525 | 0.817 | 0.372 | 0.149 |
| federal NAICS | 1,011 | 0.126 | 0.285 | 0.472 | 0.204 | 0.165 |
| federal PSC | 3,408 | 0.102 | 0.171 | 0.285 | 0.143 | 0.160 |

Random guessing would be 0.0065, 0.001 and 0.0003 respectively, so all three are
far better than nothing and only one of them is useful. Full tables, including
the four way ablation and the routing thresholds, are in
`evals/report-{corpus}-{split}.md`.

### The finding that changed the project

I built this against federal procurement data first. It measures badly, and not
because the pipeline is weak.

The `Description` field on a federal award is an internal memo, not a product
description. The eval surfaces this the moment you look at the failures:

- `"EXPRESS REPORT: QUARTER 4"` is labelled G007, government health insurance
  programs
- `"AWARD TASK ORDER IGF::OT::IGF"` is labelled R706, logistics support
- `"VA MICROSOFT ENTERPRISE LICENSE AGREEMENT"` is labelled DA01, application
  development. The pipeline returned W070, lease or rental of IT software, which
  fits the words better than the label does

A single bad result cannot tell you whether the pipeline is weak or the task is
hard, so I added a second corpus as a control. arXiv abstracts into the arXiv
subject taxonomy is the same engineering against text that genuinely determines
its label, and the identical pipeline gets recall@20 of 0.85 on 155 classes.

That is the useful conclusion, and it is worth more than a flattering number:
**federal award descriptions do not contain enough signal to predict procurement
codes, and the right product here is shortlist plus human review, not
automation.** Everything in the routing table is built around that.

### Things I tried that did not work

Reported because the negative results were more informative than the positive
one, and because a list of only the things that worked is not a record of what
happened.

| change | result |
|---|---|
| Lowercasing the corpus before embedding | No change whatsoever. bge is an uncased model, so the tokenizer already folds case. A reasonable hypothesis, cleanly falsified. |
| Expanding PSC abbreviations, EQPT to equipment and so on | Slightly worse |
| Rerank depth 30 to 80 | Clearly worse, MRR 0.140 to 0.126. A reranker only moderately better than fusion promotes noise when you hand it more of it |
| Filtering to descriptions with real content | Barely moved PSC, which is how I learned the problem was not boilerplate |
| Splitting by product, service and R&D | No bucket is meaningfully easier |

### Things that did work

| change | result |
|---|---|
| Swapping ms-marco-MiniLM-L-6 for jina-reranker-v1-turbo | recall@5 up 28% relative, MRR 0.119 to 0.140 |
| bge-small to bge-base | About a point of accuracy at 1, for 210MB instead of 67MB |
| Truncating the query before reranking | MRR 0.185 to 0.251 on arXiv. A cross encoder reads query and document in one window, so a 1,400 character abstract pushes the document out of it |
| Turning reranking off entirely | The biggest single win, and the one I did not expect. See decision 5 |

The last two rows are worth reading together. I spent real effort making the
reranking stage better, and the thing that improved it most was deleting it
from the serving path. Both facts are in the repo because the second one only
became true after the embedder upgrade, and a record that showed only the final
state would make the decision look obvious when it was not.

### Protocol

Configuration is chosen on the dev split and reported on test. The split is
assigned by hashing the record id rather than by shuffling, so it survives
regenerating the data files and a parameter tuned on dev cannot leak into a test
number.

Two things I deliberately did **not** tune. Dense alone beats fusion on NAICS by
0.011 MRR, and fusion beats dense alone on PSC and arXiv by similar margins. On
a dev split of 150 rows those differences are noise, so hybrid stays everywhere.
The reranking differences are three to six times larger and point the same way
on all three corpora, so those I acted on. Knowing which gaps are real at a
given sample size is most of what an eval suite is for.

### Confidence is weaker than it looks

The retrieval only baseline reports a confidence derived from the margin between
the top two candidates. My first version squashed the raw margin through a
sigmoid, which works for cross encoder logits and is meaningless for RRF scores
where the entire shortlist spans about 0.0003. The calibration table showed all
257 predictions sitting in one bucket at exactly 0.500, which is what a broken
confidence signal looks like from the outside. Dividing by the spread across the
shortlist makes it scale free and dropped expected calibration error from 0.263
to 0.149.

It is still not predictive. Accuracy is roughly flat across the confidence
buckets, which means retrieval margin is close to useless as a signal about
correctness. That is the honest argument for the LLM adjudication step rather
than a reason to hide the baseline.

---

## The model step

Off by default. `PCAGENT_DECIDER=claude` turns it on and it needs
`ANTHROPIC_API_KEY`.

The model picks from the shortlist, it does not generate a code. A model asked
for "the PSC for this" will happily return a well formed code that does not
exist, and constraining it to a list turns a hallucination into a wrong but
checkable answer. The tool then verifies the choice against the list anyway,
because a prompt is a request and not a guarantee. Quoted evidence is checked
against the input text and any span that is not really there is dropped, since a
paraphrase presented as a quote is worse than no quote.

Everything outbound is bounded: a per call timeout, one retry and only for an
off list answer, and a documented fallback to the retrieval baseline that labels
itself in `decided_by` so a degraded answer is never mistaken for a model answer.

The baseline exists so the model has something to beat. Reporting them side by
side is the only way to know whether the model is adding judgement or just
adding latency.

---

## Layout

```
src/pcagent/
  domain.py          pydantic contracts, shared by everything
  auth.py            Principal, Scope, TokenRegistry
  errors.py          the error vocabulary, with stable codes
  config.py          settings from the environment
  bootstrap.py       builds the service graph once
  taxonomy/loader.py loads the committed corpora
  retrieval/         lexical (hand written BM25), dense, rerank, hybrid
  llm/               Decider protocol, baseline, Claude, the prompt
  store/             SQLite and the idempotency protocol
  tools/             the typed core: validation and authorization live here
  api.py             FastAPI, thin
  mcp_server.py      FastMCP, thin
  agent/             hand written batch loop, plus a LangGraph adapter
data/                committed corpora and labelled eval sets, with provenance
evals/               runner, metrics, generated reports
tests/               96 tests, no network, no model downloads
scripts/             one off data builders, stdlib only
```

Read in this order if you have ten minutes: `tools/review.py`,
`store/db.py`, `retrieval/hybrid.py`, `tests/test_auth_contract.py`.

## Tests

```bash
pytest                      # 96 tests, under 10 seconds
ruff check . && mypy        # lint and types
python evals/run_eval.py --taxonomy arxiv --split test
```

The suite stubs the ONNX models. That is not a shortcut, it is a boundary: these
tests are about the tool contract, the authorization rules and the idempotency
protocol, and none of those get more true with a real embedding model behind
them. Model quality is measured in `evals/`, which is where it belongs. The
stubs are also how the degradation paths get tested at all.

## What I would do next

- **The labels deserve a second opinion.** Agreement with a noisy label is a
  floor, not an estimate. A few hundred hand adjudicated rows would turn that
  floor into a real number and is the single highest value thing left.
- **Hierarchical scoring.** Getting PSC group 70 right and the fourth character
  wrong is not the same failure as returning a weapons code, and the current
  metric cannot tell them apart.
- **Fine tune the bi-encoder on the taxonomy.** The off the shelf models have
  never seen a PSC title. A few thousand synthetic pairs would likely beat
  everything tried here.
- **Postgres and pgvector** if the corpus ever outgrows an in memory index.
  Today it is 3,400 rows and reaching for a database would be weight without
  benefit.

## What this is not

Not a framework. Not production infrastructure. The token registry is a stand in
for an identity provider, the store is SQLite, and there is no rate limiting,
tracing or multi tenancy. Those are deliberate omissions in something meant to
be read in ten minutes, and each one is a line in `config.py` or `store/db.py`
rather than an assumption spread through the codebase.

## Licence

MIT. The data is United States government public domain and arXiv public
metadata. See `data/README.md`.
