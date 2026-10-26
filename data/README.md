# Where this data came from

Every file here is committed so the repo, its tests and its evals run with no
network access. Every file is also regenerable by the script named against it.
All four sources are United States or academic public data and need no API key.

## Taxonomies

| file | rows | source | script |
|---|---|---|---|
| `psc.jsonl` | 3,408 | USAspending.gov PSC reference tree | `scripts/fetch_taxonomy.py` |
| `naics.jsonl` | 1,376 | USAspending.gov NAICS reference tree | `scripts/fetch_taxonomy.py` |
| `arxiv.jsonl` | 155 | arxiv.org/category_taxonomy | `scripts/fetch_arxiv.py` |

`naics.jsonl` carries 365 rows with a `year_retired`. The loader drops them,
which leaves 1,011 active six digit codes and matches the NAICS 2022 vintage.
They are kept in the file rather than filtered at fetch time because a historical
award can legitimately carry a retired code, and the eval needs to be able to
tell "we never had this code" apart from "we ranked it badly".

## Labelled evaluation sets

| file | rows | label | source |
|---|---|---|---|
| `eval/labeled_awards.jsonl` | 400 | `psc_code`, `naics_code` | USAspending award search |
| `eval/arxiv_papers.jsonl` | 400 | `label` | arXiv API, author assigned primary category |

Both carry a `split` field of `dev` or `test`, assigned by hashing the record id
rather than by shuffling. That way the split survives regenerating the file, so
a parameter tuned on dev cannot quietly leak into a test number.

### About the award labels

These are not a gold standard. They are the codes a contracting officer
recorded, and contracting officers miscode. Spot checks turn up cases where the
recorded code is hard to defend:

- `"EXPRESS REPORT: QUARTER 4"` carries G007, government health insurance
  programs. Nothing in the text supports that, or anything else.
- `"VA MICROSOFT ENTERPRISE LICENSE AGREEMENT"` carries DA01, application
  development. W070, lease or rental of IT software, fits the words better.

So the evals report *agreement with the recorded code*, which is a floor on real
accuracy rather than an estimate of it. The README explains what that implies
and why the arXiv set is here as a control.

### About the arXiv labels

The primary category is chosen by the submitting author from a published list,
and the abstract is written to explain what the paper is about. The label is
therefore much closer to being determined by the text, which is exactly the
property the procurement data lacks and the reason both sets are in this repo.

## Regenerating

```bash
python scripts/fetch_taxonomy.py    # psc.jsonl and naics.jsonl
python scripts/build_eval_set.py    # eval/labeled_awards.jsonl
python scripts/fetch_arxiv.py       # arxiv.jsonl and eval/arxiv_papers.jsonl
```

The scripts are serial and throttled, and they cache every HTTP response under
`.cache/http/`, so a rerun asks the upstream APIs for nothing it already has.
That is partly politeness and partly self defence: an earlier parallel version
of the taxonomy fetch got this project rate limited out of USAspending for about
an hour, which is how the cache came to exist.
