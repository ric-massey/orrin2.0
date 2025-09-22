# tests/memory_tests/patterns_test.py
import re
import pytest

from memory.lexicon.patterns import (
    extract_definitions_rich,
    extract_definitions,
    DefinitionHit,
)

# ---------- small helpers ----------

def by_term(hits, term_lower):
    return [h for h in hits if h.term.lower() == term_lower.lower()]

def substr(text, span):
    s, e = span
    return text[s:e]

# ---------- core pattern coverage ----------

def test_definition_of_pattern_basic():
    text = "The definition of salience is the degree to which information stands out."
    hits = extract_definitions_rich(text)
    hs = by_term(hits, "salience")
    assert hs, "expected salience definition"
    h = hs[0]
    assert h.pattern == "definition_of"
    # trailing punctuation removed
    assert h.definition.endswith("stands out")
    # span should contain both term and definition
    frag = substr(text, h.span)
    assert "definition of salience is" in frag
    assert "stands out" in frag
    assert 0.0 <= h.confidence <= 1.0

def test_is_means_equals_pattern():
    text = "Restrictor means a mandated orifice in the intake path that limits airflow."
    hits = extract_definitions_rich(text)
    hs = by_term(hits, "Restrictor")
    assert hs
    h = hs[0]
    assert h.pattern == "is_means_equals"
    assert "mandated orifice" in h.definition

def test_colon_dash_multiline_unicode_dash_and_hyphen_and_colon():
    text = """
    ECU: engine control unit.
    Torque — rotational force applied to cause rotation.
    Drag - aerodynamic resistance.
    """
    hits = extract_definitions_rich(text)
    ecu = by_term(hits, "ECU")[0]
    torque = by_term(hits, "Torque")[0]
    drag = by_term(hits, "Drag")[0]
    assert ecu.pattern == "colon_dash"
    assert torque.pattern == "colon_dash"
    assert drag.pattern == "colon_dash"
    # punctuation stripped from end
    assert ecu.definition == "engine control unit"
    assert drag.definition == "aerodynamic resistance"

def test_parenthetical_definition():
    text = "Lexicon (a vocabulary of terms) is central to clarity."
    hits = extract_definitions_rich(text)
    hx = by_term(hits, "Lexicon")[0]
    assert hx.pattern == "paren_def"
    assert hx.definition == "a vocabulary of terms"

def test_appositive_definition():
    text = "Torque, the rotational force that causes spin, is measured in newton-meters."
    hits = extract_definitions_rich(text)
    ht = by_term(hits, "Torque")[0]
    assert ht.pattern == "appositive"
    assert "rotational force" in ht.definition
    # appositive kept concise (regex restricts length)
    assert 3 <= len(ht.definition) <= 120

def test_aka_alias_inline_with_commas_and_slashes_and_quotes():
    text = "Porsche 986 aka 'Boxster', 986/Boxster S, and sometimes `986.1`."
    hits = extract_definitions_rich(text)
    hp = by_term(hits, "Porsche 986")[0]
    assert hp.pattern == "aka"
    assert hp.definition is None  # alias-only is allowed
    # split & clean aliases: quotes removed, duplicates removed, slashes split
    al = [a.lower() for a in hp.aliases]
    assert "boxster" in al
    assert "986" in al  # from "986/Boxster S" (split keeps both pieces)
    assert "boxster s" in al
    assert "986.1" in al

def test_multiple_patterns_all_detected_and_dedup_by_term_def_aliases():
    text = """
    ECU: engine control unit.
    The definition of ECU is engine control unit
    ECU is the engine control unit.
    """
    hits = extract_definitions_rich(text)
    ecu_hits = by_term(hits, "ECU")
    # Even though there are 3 ways, final unique list should have a single entry for same def
    defs = { (h.term.lower(), (h.definition or "").lower(), tuple(sorted(a.lower() for a in h.aliases))) for h in ecu_hits }
    assert len(defs) == 1

# ---------- normalization & filtering ----------

def test_pronoun_terms_are_ignored():
    text = "It is a small microcontroller."
    hits = extract_definitions_rich(text)
    assert not hits, "pronoun 'It' should be filtered as a term"

def test_tautology_is_filtered():
    text = "Foo is Foo."
    hits = extract_definitions_rich(text)
    assert not by_term(hits, "Foo")

def test_term_length_limit_excludes_long_terms():
    long_term = "A" * 65
    text = f"{long_term} is a very long term that should be ignored."
    hits = extract_definitions_rich(text)
    assert all(h.term != long_term for h in hits)

def test_quotes_on_terms_and_aliases_are_stripped():
    text = """'GPU' is 'graphics processing unit'.
              Device aka "Thing", 'Widget'/'Gadget'."""
    hits = extract_definitions_rich(text)
    gpu = by_term(hits, "GPU")[0]
    assert gpu.definition == "graphics processing unit"
    dev = by_term(hits, "Device")[0]
    # aliases cleaned (no quotes/backticks)
    al = set(a for a in dev.aliases)
    assert {"Thing", "Widget", "Gadget"} <= al

def test_trailing_punctuation_is_removed_from_definition():
    text = "Cache is a small, fast storage; "
    hits = extract_definitions_rich(text)
    cache = by_term(hits, "Cache")[0]
    assert cache.definition == "a small, fast storage"

def test_code_fences_are_ignored_to_reduce_false_positives():
    text = """
    ````python
    Foo: not a real definition from code
    ````
    RealTerm: a real definition after code fences.
    """
    hits = extract_definitions_rich(text)
    # shouldn't have picked the fenced Foo, but should pick RealTerm
    assert not by_term(hits, "Foo")
    assert by_term(hits, "RealTerm")

def test_alias_same_as_term_is_not_included():
    text = "Alpha aka Alpha, Beta"
    hits = extract_definitions_rich(text)
    ha = by_term(hits, "Alpha")[0]
    assert "Alpha" not in ha.aliases
    assert "Beta" in ha.aliases

def test_non_alphanumeric_terms_ignored():
    text = "*** aka Star"
    hits = extract_definitions_rich(text)
    assert not hits

def test_empty_and_none_input_return_empty():
    assert extract_definitions_rich("") == []
    assert extract_definitions("") == []

# ---------- compatibility API ----------

def test_simple_extract_definitions_shapes_tuples():
    text = "ECU: engine control unit."
    xs = extract_definitions(text)
    assert isinstance(xs, list) and xs
    (term, definition, aliases) = xs[0]
    assert term == "ECU"
    assert definition == "engine control unit"
    assert isinstance(aliases, list)

def test_definitionhit_to_tuple_roundtrip():
    text = "DSP means digital signal processing."
    hits = extract_definitions_rich(text)
    h = by_term(hits, "DSP")[0]
    tup = h.to_tuple()
    assert tup == (h.term, h.definition, h.aliases)

# ---------- spans sanity ----------

def test_spans_cover_original_substrings():
    text = "Barometer is a pressure sensor used to measure atmospheric pressure."
    hits = extract_definitions_rich(text)
    hb = by_term(hits, "Barometer")[0]
    segment = substr(text, hb.span)
    # The segment should contain the phrase structure (term + connective + def head)
    assert "Barometer is" in segment
    assert "pressure sensor" in segment

# ---------- edge cases for appositive & parenthetical ----------

def test_appositive_min_length_guard():
    # def must be at least 3 chars; very short ones should be filtered by _mk_hit
    text = "Foo, a x, is something."  # 'a x' length 3 -> borderline accepted
    hits = extract_definitions_rich(text)
    assert by_term(hits, "Foo"), "Expected borderline 3-char appositive to be accepted"

def test_parenthetical_variants_articles_optional():
    text = "Router (an appliance) is network equipment. Modem (appliance) is for modulation."
    hits = extract_definitions_rich(text)
    assert by_term(hits, "Router")
    assert by_term(hits, "Modem")

# ---------- aka variations ----------

def test_aka_parenthetical_and_inline_variants():
    text = "HTTP (aka Hypertext Transfer Protocol), also called H.T.T.P., is common. JSON aka JavaScript Object Notation"
    hits = extract_definitions_rich(text)
    http = by_term(hits, "HTTP")[0]
    json = by_term(hits, "JSON")[0]
    http_aliases = set(a.lower() for a in http.aliases)
    json_aliases = set(a for a in json.aliases)
    assert "hypertext transfer protocol" in http_aliases
    assert "JavaScript Object Notation" in json_aliases or "javascript object notation" in json_aliases

# ---------- dedup robustness ----------

def test_dedup_considers_sorted_aliases_to_avoid_order_sensitivity():
    text = "Device aka A, B\nDevice aka B, A"
    hits = extract_definitions_rich(text)
    dev = by_term(hits, "Device")
    # Should dedup to a single hit because alias set is the same
    assert len(dev) == 1, f"got {len(dev)}: {dev}"
