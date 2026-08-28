# Benchmarking the faithfulness eval

Scores LitRAG's two-stage citation-faithfulness judge against the faithfulness
metrics of [RAGAS](https://github.com/explodinggradients/ragas) and
[DeepEval](https://github.com/confident-ai/deepeval) on a hand-labelled claim set —
the "benchmark the eval" roadmap item.

## The claim set

[`claims.jsonl`](claims.jsonl): 61 claims about the repo's semaglutide corpus
([`data/sample_abstracts.md`](../data/sample_abstracts.md)), each with the quote a
RAG system might cite for it and two gold annotations:

| Field | Meaning |
|---|---|
| `gold_support` | support level of the *claim* against the source passage: `supports` / `partial` / `contradicts` / `not_found` |
| `quote_status` | whether the *cited quote* actually appears in the passage: `verbatim` / `fabricated` |
| `gold_faithful` | `quote_status == verbatim` **and** `gold_support == supports` |

Mix: 27 faithful, 34 unfaithful (9 of them fabricated quotes — including one where
the claim is **true** but the quote is invented, the case that separates citation
checking from mere claim checking). Every `verbatim` quote is mechanically verified
to locate in its passage via LitRAG's own `locate()`, and every `fabricated` quote
verified *not* to (`--dry-run` re-checks this).

## What's being compared

All three systems judge each claim against its source abstract, using the **same
Claude judge model** (`faithfulness.JUDGE_MODEL`) so the comparison isolates the
method rather than the model:

- **litrag** — stage 1 deterministic quote locator, then stage 2 LLM judge only if
  the quote is located. Sees claim *and* cited quote.
- **ragas** — `ragas.metrics.collections.Faithfulness` (statement decomposition +
  NLI-style verification). Sees the claim only — it has no concept of a cited quote.
- **deepeval** — `deepeval.metrics.FaithfulnessMetric` (truths/claims extraction +
  verdicts). Sees the claim only.

That asymmetry is the point of the fabricated-quote cases: a metric that never looks
at the quote cannot flag a fabricated citation attached to a true claim.
`--with-quote` runs the variant where the frameworks *do* see the quote appended to
the answer text.

## Running it

```bash
# One-time: separate venv (ragas needs older langchain than the main repo)
python -m venv benchmark/.venv && source benchmark/.venv/bin/activate
pip install -r benchmark/requirements.txt

# Offline sanity check — validates the dataset, no API key
python benchmark/run_benchmark.py --dry-run

# Full run (~180 judge calls; needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python benchmark/run_benchmark.py

# Variants
python benchmark/run_benchmark.py --systems litrag,ragas --limit 10   # smoke run
python benchmark/run_benchmark.py --with-quote                        # frameworks see the quote
python benchmark/run_benchmark.py --allow-partial                     # litrag counts partial as faithful
```

Results land in `benchmark/results/results-<timestamp>.json` and a comparison table
prints at the end.

## Metrics

Binary faithful/unfaithful per system, scored against `gold_faithful`, with
**unfaithful as the positive class** — so `hallucination_recall` reads as "what
fraction of unfaithful claims did the system catch", the number that matters in a
medical setting. Accuracy, precision, F1, per-category accuracy
(`quote_status/gold_support`), error counts, and wall-clock time are reported per
system; `judge_calls` for litrag shows how many stage-2 calls the deterministic
stage 1 saved.

RAGAS and DeepEval return continuous scores in [0, 1]; they are binarized at
`--threshold` (default 0.5).

## Results (28 Aug 2026)

61 claims, judge model `claude-sonnet-4-6` for all three systems, default
thresholds. Unfaithful is the positive class.

| System | Accuracy | Hallucination precision | Hallucination recall | F1 | Wall time | Judge calls |
|---|---|---|---|---|---|---|
| **LitRAG** (two-stage) | **0.984** | 0.971 | **1.000** | **0.986** | **177 s** | 52 of 61 |
| RAGAS Faithfulness | 0.934 | 0.969 | 0.912 | 0.939 | 355 s | 61 × 2 stages |
| DeepEval FaithfulnessMetric | 0.803 | 1.000 | 0.647 | 0.786 | 863 s | 61 × 2 stages |

### What the errors say

**LitRAG — 1 error in 61**, and it fails in the safe direction: one genuinely
supported claim (STEP program → FDA approval of Wegovy) was graded `partial` by the
judge — a conservative false alarm, not a missed hallucination. All 34 unfaithful
claims were caught, and the 9 fabricated quotes were caught by the deterministic
locator *before* any judge call — which is also why it makes the fewest LLM calls
and runs fastest.

**RAGAS — 4 errors.** It missed the fabricated-quote-with-true-claim case
(scored 1.00 — it never sees the quote, so a true claim with an invented citation
passes), let two fabricated-quote cases through at exactly 0.50 (the default
threshold's edge), and false-flagged the same FDA-approval claim LitRAG was
conservative about (scored 0.00).

**DeepEval — 12 errors, every one a miss scored exactly 1.00.** It caught all
contradicted claims but passed nearly every `not_found` case (a real quote from a
passage that says nothing about the claim), half the `partial` overreach cases, and
all three "plausible" fabricated-quote cases. This is consistent with its
documented scoring, in which claims that don't *contradict* the retrieval context
count as faithful — absence of support is not penalized. That design choice is
defensible for chatbots; over medical abstracts it passes claims like "semaglutide
reduces the risk of dementia" cited to a passage that never mentions dementia.

### The headline case

`c61` — *"Semaglutide produced roughly 15% average weight loss in STEP 1"* (true
per the passage) citing a quote that appears nowhere in it:

| System | Verdict | Why |
|---|---|---|
| LitRAG | flagged `hallucinated_quote`, 0 judge calls | the locator can't find the quote |
| RAGAS | passed, score 1.00 | judges the claim only; never sees the quote |
| DeepEval | passed, score 1.00 | judges the claim only; never sees the quote |

A citation is a checkable promise about *provenance*, not just truth. Claim-level
faithfulness metrics can grade the claim; only a citation-level check can grade the
citation.

### Caveats

Small hand-labelled set (n=61) on one domain, labelled by the benchmark's author —
gold `partial`/`not_found` boundaries involve judgment. All systems used one Claude
model; framework results may differ with their default judges (`--with-quote` and
other variants are one flag away). RAGAS and DeepEval are general-purpose metrics
run with default settings, compared here on a task — citation integrity — that
LitRAG was purpose-built for and they were not.

Raw run outputs: [`results/`](results/) (JSON, per-claim predictions included).
