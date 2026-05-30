"""Unit tests for the deterministic quote locator.

These run without an API key — they only exercise the Stage-1 quote check
(``locate_quote`` / ``locate``), which is what catches fabricated citations for free.

    pytest test_faithfulness.py
"""

from faithfulness import locate, locate_quote

SOURCE = (
    "A primary cardiovascular end-point event occurred in 569 of the 8803 patients "
    "(6.5%) in the semaglutide group and in 701 of the 8801 patients (8.0%) in the "
    "placebo group (hazard ratio, 0.80; 95% confidence interval, 0.72 to 0.90; P<0.001)."
)


def test_exact_quote_is_found():
    assert locate_quote("hazard ratio, 0.80; 95% confidence interval, 0.72 to 0.90", SOURCE)


def test_fuzzy_quote_is_found():
    # Minor whitespace / casing differences should still match above threshold.
    assert locate_quote("Hazard Ratio, 0.80;  95% confidence  interval, 0.72 to 0.90", SOURCE)


def test_fabricated_quote_is_not_found():
    assert not locate_quote("semaglutide increased all-cause mortality by 35%", SOURCE)


def test_paraphrase_below_threshold_is_rejected():
    # A reworded "quote" is not verbatim — it must not pass as a located citation.
    paraphrase = "the drug cut heart attacks and strokes by about a fifth versus placebo"
    assert not locate_quote(paraphrase, SOURCE)


def test_empty_quote_is_not_found():
    assert not locate_quote("", SOURCE)


def test_locate_reports_score():
    hit = locate("6.5%) in the semaglutide group", SOURCE)
    assert hit["found"] and hit["score"] == 100.0

    miss = locate("a completely unrelated sentence about kidney transplants", SOURCE)
    assert not miss["found"] and miss["score"] < 90.0
