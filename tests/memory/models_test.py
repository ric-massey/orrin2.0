# tests/memory_tests/models_test.py
import re
from datetime import datetime
import itertools

import pytest

import memory.models as models


ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _is_iso_utc(ts: str) -> bool:
    if not isinstance(ts, str) or not ISO_RE.match(ts):
        return False
    # ensure strptime can parse
    try:
        datetime.strptime(ts, models.ISO)
        return True
    except Exception:
        return False


# ---------------------------
# Event
# ---------------------------
def test_event_basic_fields():
    e = models.Event(kind="chat:user", content="hello", meta={"x": 1})
    assert e.kind == "chat:user"
    assert e.content == "hello"
    assert e.meta == {"x": 1}


def test_event_meta_defaults_to_empty_dict():
    e = models.Event(kind="loop:think", content="tick")
    assert isinstance(e.meta, dict) and e.meta == {}


# ---------------------------
# MemoryItem
# ---------------------------
def test_memoryitem_new_sets_defaults_and_ids_iso():
    it = models.MemoryItem.new(kind="fact", source="chat:user", content="  something  ")
    assert it.id.startswith("mem_")
    # uuid4 hex is 32 chars; prefix + '_' + hex
    assert len(it.id.split("_")[1]) == 32
    assert _is_iso_utc(it.ts)
    assert it.layer == "working"
    assert it.kind == "fact"
    assert it.source == "chat:user"
    assert it.content == "something"  # trimmed
    # defaults
    assert it.embedding_id is None
    assert it.embedding_dim is None
    assert it.model_hint is None
    assert it.salience == 0.0
    assert it.novelty == 0.0
    assert it.goal_relevance == 0.0
    assert it.impact_signal == 0.0
    assert it.freq == 0
    assert it.last_access is None
    assert it.strength == 0.0
    assert it.summary_of == []
    assert it.cross_refs == []
    assert it.pinned is None
    assert it.expiry_hint is None
    assert isinstance(it.meta, dict)


def test_memoryitem_new_meta_is_copied_not_shared():
    meta = {"a": 1}
    it1 = models.MemoryItem.new(kind="fact", source="s", content="c", **meta)
    it2 = models.MemoryItem.new(kind="fact", source="s", content="c", **meta)
    assert it1.meta == {"a": 1}
    assert it2.meta == {"a": 1}
    assert it1.meta is not it2.meta  # separate dicts
    # lists are separate as well
    it1.summary_of.append("x")
    assert it2.summary_of == []


def test_memoryitem_to_dict_roundtrip_and_copies():
    it = models.MemoryItem.new(kind="goal", source="planner", content="do X", layer="long", priority=5)
    it.embedding_id = "vec_123"
    it.embedding_dim = 768
    it.model_hint = "bge-small-en-v1.5"
    it.salience = 0.7
    it.novelty = 0.9
    it.goal_relevance = 0.5
    it.impact_signal = 0.25
    it.freq = 3
    it.last_access = models.now_iso()
    it.strength = 0.42
    it.summary_of = ["mem_a", "mem_b"]
    it.cross_refs = ["mem_c"]
    it.pinned = True
    it.expiry_hint = "2026-01-01"
    d = it.to_dict()

    # basic keys present
    for k in [
        "id","ts","layer","kind","source","content","embedding_id","embedding_dim","model_hint",
        "salience","novelty","goal_relevance","impact_signal","freq","last_access","strength",
        "summary_of","cross_refs","pinned","expiry_hint","meta"
    ]:
        assert k in d

    # copies, not references
    d["summary_of"].append("zzz")
    d["cross_refs"].append("yyy")
    d["meta"]["priority"] = 7
    assert it.summary_of == ["mem_a", "mem_b"]
    assert it.cross_refs == ["mem_c"]
    assert it.meta == {"priority": 5}


def test_memoryitem_ids_unique_over_many():
    ids = set()
    for _ in range(200):
        it = models.MemoryItem.new(kind="fact", source="s", content="c")
        assert it.id not in ids
        ids.add(it.id)


# ---------------------------
# LexiconSense
# ---------------------------
def test_lexiconsense_new_defaults_and_fields_trimmed():
    s = models.LexiconSense.new(
        term="  Airfoil  ",
        sense_id="airfoil:123",
        definition="  a shape designed to produce lift  ",
        source="chat:user",
        aliases=["wing section"],
        examples=["the wing's airfoil"],
        meta={"confidence": 0.9},
    )
    assert s.id.startswith("lex_")
    assert len(s.id.split("_")[1]) == 32
    assert s.sense_id == "airfoil:123"
    assert s.term == "Airfoil"
    assert s.definition == "a shape designed to produce lift"
    assert s.aliases == ["wing section"]
    assert s.examples == ["the wing's airfoil"]
    assert s.sources == ["chat:user"]
    assert s.model_hint is None
    assert s.freq == 0
    assert s.pinned is None
    assert s.meta == {"confidence": 0.9}


def test_lexiconsense_lists_are_not_shared_between_instances():
    s1 = models.LexiconSense.new(term="t", sense_id="t:1", definition="d", aliases=["a"], examples=["e"])
    s2 = models.LexiconSense.new(term="t", sense_id="t:2", definition="d")
    s1.aliases.append("b")
    s1.examples.append("f")
    assert s2.aliases == []
    assert s2.examples == []


def test_lexiconsense_to_dict_has_all_fields_and_is_copy():
    s = models.LexiconSense.new(term="t", sense_id="t:1", definition="d", source="sys", aliases=["a"], examples=["e"], meta={"m": 1})
    s.model_hint = "hash-256"
    s.freq = 5
    s.pinned = True

    d = s.to_dict()
    # all fields present
    for k in ["id","sense_id","term","definition","aliases","examples","sources","model_hint","freq","pinned","meta"]:
        assert k in d

    # ensure returned lists/dicts are copies
    d["aliases"].append("x")
    d["examples"].append("y")
    d["sources"].append("z")
    d["meta"]["m"] = 2
    assert s.aliases == ["a"]
    assert s.examples == ["e"]
    assert s.sources == ["sys"]
    assert s.meta == {"m": 1}


def test_multiple_lexiconsense_ids_unique():
    ids = set()
    for i in range(150):
        s = models.LexiconSense.new(term=f"t{i}", sense_id=f"t:{i}", definition="d")
        assert s.id not in ids
        ids.add(s.id)
