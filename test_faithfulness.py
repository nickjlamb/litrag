"""Unit tests for the deterministic quote locator.

These run without an API key — they only exercise locate_quote.

TODO (build session): implement against faithfulness.locate_quote once written.
"""

import pytest

from faithfulness import locate_quote


def test_exact_quote_is_found():
    source = "Semaglutide reduced major adverse cardiovascular events by 20%."
    assert locate_quote("reduced major adverse cardiovascular events by 20%", source)


def test_fuzzy_quote_is_found():
    source = "Semaglutide reduced major adverse cardiovascular events by 20%."
    # minor whitespace / casing differences should still match
    assert locate_quote("Reduced  major adverse  cardiovascular events by 20%", source)


def test_fabricated_quote_is_not_found():
    source = "Semaglutide reduced major adverse cardiovascular events by 20%."
    assert not locate_quote("increased all-cause mortality by 35%", source)
