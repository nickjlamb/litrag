"""Benchmark LitRAG's citation-faithfulness judge against RAGAS and DeepEval.

Scores three systems on the labelled claim set in ``claims.jsonl`` (claims about
the semaglutide corpus with gold faithfulness labels):

  litrag    two-stage: deterministic quote locator -> LLM judge (faithfulness.py)
  ragas     ragas.metrics.collections.Faithfulness
  deepeval  deepeval.metrics.FaithfulnessMetric

All three use the SAME Claude judge model (faithfulness.JUDGE_MODEL), so the
comparison isolates the method, not the model. RAGAS and DeepEval judge the claim
text against the source passage; they have no concept of a cited quote — which is
exactly the gap the fabricated-quote cases probe.

Usage:
  python benchmark/run_benchmark.py --dry-run          # no API key needed
  python benchmark/run_benchmark.py                    # full run, all systems
  python benchmark/run_benchmark.py --systems litrag,ragas --limit 10
  python benchmark/run_benchmark.py --with-quote       # append the cited quote to
                                                       # the answer text the
                                                       # frameworks see

Run from the repo root. Requires benchmark/requirements.txt in a SEPARATE venv
(ragas pins an older langchain than the main repo). ANTHROPIC_API_KEY required
for everything except --dry-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from faithfulness import JUDGE_MODEL, grade_support, locate  # noqa: E402
from ingest import load_corpus  # noqa: E402

QUESTION = "What does the clinical evidence on semaglutide show?"


def load_claims(path: Path, limit: int | None = None) -> list[dict]:
    claims = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return claims[:limit] if limit else claims


def load_passages() -> dict[str, str]:
    return {r.pmid: r.text for r in load_corpus(str(REPO_ROOT / "data" / "sample_abstracts.md"))}


def response_text(claim: dict, with_quote: bool) -> str:
    if with_quote:
        return f"{claim['claim']} (cited quote: \"{claim['cited_quote']}\")"
    return claim["claim"]


# --- Systems -----------------------------------------------------------------


def run_litrag(claims: list[dict], passages: dict[str, str], allow_partial: bool) -> list[dict]:
    """Stage 1 locator, then stage 2 judge only when the quote is located."""
    out = []
    for c in claims:
        t0 = time.perf_counter()
        passage = passages[c["pmid"]]
        loc = locate(c["cited_quote"], passage)
        if not loc["found"]:
            pred, detail, calls = False, "hallucinated_quote", 0
        else:
            v = grade_support(c["claim"], passage)
            pred = v["support"] == "supports" or (allow_partial and v["support"] == "partial")
            detail, calls = v["support"], 1
        out.append({"id": c["id"], "pred_faithful": pred, "detail": detail,
                    "judge_calls": calls, "seconds": round(time.perf_counter() - t0, 2)})
    return out


def run_ragas(claims: list[dict], passages: dict[str, str], threshold: float, with_quote: bool) -> list[dict]:
    import asyncio

    from anthropic import AsyncAnthropic
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    llm = llm_factory(JUDGE_MODEL, provider="anthropic", client=AsyncAnthropic())
    # Newer Claude models reject `temperature` and `top_p` specified together;
    # ragas 0.4.3's anthropic path sends both by default, so drop top_p.
    if isinstance(getattr(llm, "model_args", None), dict):
        llm.model_args.pop("top_p", None)
    metric = Faithfulness(llm=llm)

    async def score_all() -> list[dict]:
        out = []
        for c in claims:
            t0 = time.perf_counter()
            try:
                r = await metric.ascore(
                    user_input=QUESTION,
                    response=response_text(c, with_quote),
                    retrieved_contexts=[passages[c["pmid"]]],
                )
                score = float(r.value)
                out.append({"id": c["id"], "pred_faithful": score >= threshold,
                            "detail": f"score={score:.2f}",
                            "seconds": round(time.perf_counter() - t0, 2)})
            except Exception as exc:  # keep going; record the failure
                out.append({"id": c["id"], "pred_faithful": None,
                            "detail": f"error: {exc}",
                            "seconds": round(time.perf_counter() - t0, 2)})
        return out

    return asyncio.run(score_all())


def run_deepeval(claims: list[dict], passages: dict[str, str], threshold: float, with_quote: bool) -> list[dict]:
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.models import AnthropicModel
    from deepeval.test_case import LLMTestCase

    model = AnthropicModel(model=JUDGE_MODEL, temperature=0)
    out = []
    for c in claims:
        t0 = time.perf_counter()
        metric = FaithfulnessMetric(threshold=threshold, model=model,
                                    include_reason=False, async_mode=False)
        try:
            score = metric.measure(
                LLMTestCase(input=QUESTION,
                            actual_output=response_text(c, with_quote),
                            retrieval_context=[passages[c["pmid"]]]),
                _show_indicator=False,
            )
            score = float(score)
            out.append({"id": c["id"], "pred_faithful": score >= threshold,
                        "detail": f"score={score:.2f}",
                        "seconds": round(time.perf_counter() - t0, 2)})
        except Exception as exc:
            out.append({"id": c["id"], "pred_faithful": None,
                        "detail": f"error: {exc}",
                        "seconds": round(time.perf_counter() - t0, 2)})
    return out


# --- Scoring -----------------------------------------------------------------


def score(claims: list[dict], preds: list[dict]) -> dict:
    """Binary metrics with 'unfaithful' as the positive class (hallucination detection)."""
    by_id = {p["id"]: p for p in preds}
    tp = fp = tn = fn = errors = 0
    per_category: dict[str, list[int]] = {}
    for c in claims:
        p = by_id[c["id"]]
        if p["pred_faithful"] is None:
            errors += 1
            continue
        gold_unfaithful = not c["gold_faithful"]
        pred_unfaithful = not p["pred_faithful"]
        cat = f"{c['quote_status']}/{c['gold_support']}"
        hit = int(pred_unfaithful == gold_unfaithful)
        per_category.setdefault(cat, []).append(hit)
        if gold_unfaithful and pred_unfaithful:
            tp += 1
        elif gold_unfaithful:
            fn += 1
        elif pred_unfaithful:
            fp += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_scored": n, "errors": errors,
        "accuracy": round((tp + tn) / n, 3) if n else 0.0,
        "hallucination_precision": round(precision, 3),
        "hallucination_recall": round(recall, 3),
        "hallucination_f1": round(f1, 3),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_category_accuracy": {k: round(sum(v) / len(v), 3) for k, v in sorted(per_category.items())},
        "total_seconds": round(sum(p["seconds"] for p in preds), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", default="litrag,ragas,deepeval")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N claims")
    ap.add_argument("--threshold", type=float, default=0.5, help="binarization threshold for ragas/deepeval scores")
    ap.add_argument("--allow-partial", action="store_true", help="LitRAG counts 'partial' support as faithful")
    ap.add_argument("--with-quote", action="store_true", help="append the cited quote to the answer text ragas/deepeval judge")
    ap.add_argument("--dry-run", action="store_true", help="validate the dataset offline; no API calls")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    claims = load_claims(Path(__file__).parent / "claims.jsonl", args.limit)
    passages = load_passages()

    print(f"{len(claims)} claims | gold: {sum(c['gold_faithful'] for c in claims)} faithful, "
          f"{sum(not c['gold_faithful'] for c in claims)} unfaithful")
    print("mix:", dict(Counter(f"{c['quote_status']}/{c['gold_support']}" for c in claims)))

    if args.dry_run:
        problems = 0
        flagged = 0
        for c in claims:
            loc = locate(c["cited_quote"], passages[c["pmid"]])
            if loc["found"] != (c["quote_status"] == "verbatim"):
                print(f"  PROBLEM {c['id']}: quote_status={c['quote_status']} but located={loc['found']} (score {loc['score']:.0f})")
                problems += 1
            if not loc["found"]:
                flagged += 1
        print(f"dry run: {problems} dataset problems; stage-1 locator flags {flagged} fabricated quotes "
              f"(expected {sum(c['quote_status'] == 'fabricated' for c in claims)}) — no API calls made")
        sys.exit(1 if problems else 0)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (try: source ../.env or export it). Use --dry-run for the offline check.")

    runners = {"litrag": lambda: run_litrag(claims, passages, args.allow_partial),
               "ragas": lambda: run_ragas(claims, passages, args.threshold, args.with_quote),
               "deepeval": lambda: run_deepeval(claims, passages, args.threshold, args.with_quote)}

    results: dict = {"config": vars(args) | {"judge_model": JUDGE_MODEL, "n_claims": len(claims)}, "systems": {}}
    for name in [s.strip() for s in args.systems.split(",") if s.strip()]:
        print(f"\n=== {name} ===")
        preds = runners[name]()
        summary = score(claims, preds)
        if name == "litrag":
            summary["judge_calls"] = sum(p.get("judge_calls", 0) for p in preds)
        results["systems"][name] = {"summary": summary, "predictions": preds}
        print(json.dumps(summary, indent=2))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"results-{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Compact comparison table
    rows = [(n, r["summary"]) for n, r in results["systems"].items()]
    if rows:
        print(f"\n{'system':<10} {'acc':>6} {'h-prec':>7} {'h-rec':>6} {'h-F1':>6} {'errors':>7} {'seconds':>8}")
        for n, s in rows:
            print(f"{n:<10} {s['accuracy']:>6} {s['hallucination_precision']:>7} "
                  f"{s['hallucination_recall']:>6} {s['hallucination_f1']:>6} {s['errors']:>7} {s['total_seconds']:>8}")


if __name__ == "__main__":
    main()
