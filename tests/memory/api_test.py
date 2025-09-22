# tests/memory_tests/api_test.py
import types
import numpy as np
import pytest

from memory.lexicon.api import Lexicon

# --- Deterministic embedding/model stubs (monkeypatched into memory.lexicon.api) ---

def _embed_stub(text: str) -> np.ndarray:
    """
    Map keywords to orthogonal basis so we can control cosine similarity.
      - 'apple'  -> e1
      - 'banana' -> e2
      - 'carrot' -> e3
      - otherwise -> zero vector
    If multiple appear, we pick the first match (keeps things simple/predictable).
    """
    t = (text or "").lower()
    if "apple" in t:   return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if "banana" in t:  return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if "carrot" in t:  return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)

def _model_hint_stub() -> str:
    return "test-model"


# --- Minimal in-memory store for lexicon senses ---

class FakeStore:
    """
    Provides:
      - get_lexicon_by_term(term_or_alias)
      - upsert_lexicon(senses)
    Keeps case-insensitive term and alias indexes. Allows id-based lookup for assertions.
    """
    def __init__(self):
        self.by_id = {}              # id -> sense
        self.term_idx = {}           # term_lower -> set(ids)
        self.alias_idx = {}          # alias_lower -> set(ids)

    # Helper to (re)index one sense
    def _index_one(self, s):
        tid = s.id
        term_key = (s.term or "").lower()
        self.term_idx.setdefault(term_key, set()).add(tid)
        for a in (s.aliases or []):
            if not a:
                continue
            self.alias_idx.setdefault(a.lower(), set()).add(tid)

    def _drop_from_indexes(self, old):
        if not old:
            return
        tid = old.id
        if old.term:
            term_key = old.term.lower()
            self.term_idx.get(term_key, set()).discard(tid)
        for a in (old.aliases or []):
            if not a:
                continue
            self.alias_idx.get(a.lower(), set()).discard(tid)

    def upsert_lexicon(self, senses):
        # Accepts list[LexiconSense]
        for s in senses:
            # If replacing existing, drop old index entries
            old = self.by_id.get(s.id)
            if old:
                self._drop_from_indexes(old)
            self.by_id[s.id] = s
            self._index_one(s)

    def get_lexicon_by_term(self, term_or_alias: str):
        k = (term_or_alias or "").lower()
        ids = set()
        ids |= self.term_idx.get(k, set())
        ids |= self.alias_idx.get(k, set())
        return [self.by_id[i] for i in ids if i in self.by_id]


# --- Fixtures ---

@pytest.fixture(autouse=True)
def patch_embedder(monkeypatch):
    """
    Patch memory.lexicon.api.get_embedding + model_hint so similarity logic is stable.
    """
    import memory.lexicon.api as api
    monkeypatch.setattr(api, "get_embedding", _embed_stub, raising=True)
    monkeypatch.setattr(api, "model_hint", _model_hint_stub, raising=True)
    yield


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def lex(store):
    return Lexicon(store)


# --- Tests ---

def test_learn_definition_creates_new_sense_when_none_exist(lex, store):
    sid = lex.learn_definition("Apple", "A fruit that is red")
    senses = store.get_lexicon_by_term("apple")
    assert len(senses) == 1
    s = senses[0]
    assert s.id == sid
    assert s.term == "Apple"
    assert s.definition.strip().startswith("A fruit")
    assert s.freq == 1
    assert s.pinned is True
    assert s.model_hint == "test-model"

def test_learn_definition_uses_context_if_definition_missing(lex, store):
    sid = lex.learn_definition("Banana", definition=None, context_text="banana is yellow")
    s = store.get_lexicon_by_term("banana")[0]
    assert s.id == sid
    assert "banana is yellow" in s.definition

def test_learn_definition_sets_confidence_meta_when_provided(lex, store):
    sid = lex.learn_definition("Apple", "A fruit", confidence=0.7)
    s = store.get_lexicon_by_term("apple")[0]
    assert s.meta.get("confidence") == pytest.approx(0.7)

def test_learn_definition_aliases_are_deduped_and_examples_capped(lex, store):
    sid = lex.learn_definition(
        "Carrot",
        "A root vegetable",
        aliases=["root", "ROOT", "veg", "root"],    # dedup + case-insensitive
        examples=["ex1", "ex2", "ex3", "ex4"],      # MAX_EXAMPLES = 3
    )
    s = store.get_lexicon_by_term("carrot")[0]
    # alias normalized dedupe preserves order of first occurrences
    assert [a.lower() for a in s.aliases] == ["root", "veg"]
    assert len(s.examples) == 3
    assert s.examples == ["ex1", "ex2", "ex3"]

def test_learn_definition_updates_existing_when_similar_and_merges(lex, store):
    # First insert for 'apple'
    first_id = lex.learn_definition("Apple", "apple is tasty", aliases=["pome"], source="chat:u1")
    s1 = store.get_lexicon_by_term("apple")[0]
    old_freq = s1.freq

    # Update path: keep same embedding concept ('apple') but change phrasing/definition + new alias
    upd_id = lex.learn_definition("Apple", "apple is very tasty indeed", aliases=["malus"], source="chat:u2")
    s2 = store.get_lexicon_by_term("apple")[0]

    # Should update same sense (not create a new one)
    assert upd_id == first_id
    assert s2.freq == old_freq + 1
    # Definition should be replaced (materially different string, same embedding bucket)
    assert "very tasty" in s2.definition
    # Aliases merged + deduped
    assert set(a.lower() for a in (s2.aliases or [])) >= {"pome", "malus"}
    # Sources merged + deduped
    assert set(s2.sources or []) >= {"chat:u1", "chat:u2"}

def test_learn_definition_creates_new_sense_when_dissimilar(lex, store):
    id1 = lex.learn_definition("Apple", "apple red")
    id2 = lex.learn_definition("Apple", "banana yellow")  # different embedding bucket
    senses = store.get_lexicon_by_term("apple")
    # Both senses should exist
    assert len(senses) == 2
    assert set(s.id for s in senses) == {id1, id2}

def test_get_definition_none_when_unknown_term(lex):
    assert lex.get_definition("unknown") is None

def test_get_definition_returns_most_frequent_when_no_context(lex, store):
    # Create two senses for Apple and bump freq on the 1st by learning again with same bucket
    id1 = lex.learn_definition("Apple", "apple crunchy")
    id2 = lex.learn_definition("Apple", "banana in apple term")  # new sense (banana bucket)
    # Bump id1 via update path (same bucket 'apple')
    _ = lex.learn_definition("Apple", "apple crunchy and juicy")  # updates existing
    s = lex.get_definition("Apple")
    assert s.id == id1  # most frequent

def test_get_definition_with_context_chooses_best_fit(lex, store):
    id1 = lex.learn_definition("Fruit", "apple thing")    # apple bucket
    id2 = lex.learn_definition("Fruit", "banana thing")   # banana bucket
    s = lex.get_definition("Fruit", context_text="I love banana smoothies")
    assert s.id == id2

def test_add_alias_creates_placeholder_when_term_missing(lex, store):
    lex.add_alias("Widget", "gizmo")
    senses = store.get_lexicon_by_term("widget")
    assert len(senses) == 1
    s = senses[0]
    assert "gizmo" in (s.aliases or [])
    # Because placeholder used context_text=alias, definition should include alias
    assert "gizmo" in s.definition.lower()

def test_add_alias_targets_most_frequent_when_no_sense_id(lex, store):
    id1 = lex.learn_definition("Thing", "apple x")
    id2 = lex.learn_definition("Thing", "banana y")  # second sense
    # bump id2 freq so it becomes the most frequent
    _ = lex.learn_definition("Thing", "banana y indeed")  # update banana bucket => increments freq

    lex.add_alias("Thing", "nickname")
    # Since most frequent is banana sense, alias should map there
    senses = store.get_lexicon_by_term("nickname")
    assert len(senses) == 1
    assert senses[0].id == lex.get_definition("Thing").id

def test_add_alias_targets_specific_sense_when_id_provided(lex, store):
    id1 = lex.learn_definition("Tool", "apple tool")
    id2 = lex.learn_definition("Tool", "banana tool")
    # Add alias to specific sense id2
    lex.add_alias("Tool", "doohickey", sense_id=id2)
    # Ensure the alias resolves specifically to id2
    senses = store.get_lexicon_by_term("doohickey")
    assert len(senses) == 1
    assert senses[0].id == id2

def test_add_alias_ignores_empty_alias(lex, store):
    id1 = lex.learn_definition("Noun", "apple noun")
    lex.add_alias("Noun", "  ")  # should no-op
    # No alias entry appears
    assert store.get_lexicon_by_term("  ") == []

def test_correct_definition_small_change_updates_in_place(lex, store):
    sid = lex.learn_definition("Device", "apple device")
    s0 = store.get_lexicon_by_term("device")[0]
    # Small change (same embedding bucket 'apple'): should update same sense id
    out_id = lex.correct_definition(s0.sense_id, "apple device updated", note="minor tweak", fork_if_large_change=True)
    s1 = store.get_lexicon_by_term("device")[0]
    assert out_id == s0.id
    assert "updated" in s1.definition
    assert s1.meta.get("note") == "minor tweak"

def test_correct_definition_large_change_forks_new_sense(lex, store):
    sid = lex.learn_definition("Gadget", "apple gadget")
    s0 = store.get_lexicon_by_term("gadget")[0]
    out_id = lex.correct_definition(s0.sense_id, "banana gadget", fork_if_large_change=True)
    senses = store.get_lexicon_by_term("gadget")
    assert len(senses) == 2
    assert out_id in {s.id for s in senses}
    # Original should remain with old definition
    olds = [s for s in senses if s.id != out_id][0]
    assert "apple" in olds.definition

def test_correct_definition_raises_when_sense_not_found(lex):
    with pytest.raises(ValueError):
        lex.correct_definition("unknown-sense-id", "new def")

def test_case_insensitive_lookup_by_term_and_alias(lex, store):
    sid = lex.learn_definition("CaseTerm", "apple item", aliases=["AKA"])
    senses1 = store.get_lexicon_by_term("caseterm")
    senses2 = store.get_lexicon_by_term("AKa")  # mixed case
    assert len(senses1) == 1 and len(senses2) == 1
    assert senses1[0].id == senses2[0].id

def test_blank_term_raises_value_error(lex):
    with pytest.raises(ValueError):
        lex.learn_definition("   ", "whatever")

def test_alias_dedup_on_update_path(lex, store):
    sid = lex.learn_definition("AliasTest", "apple base", aliases=["one", "two"])
    _ = lex.learn_definition("AliasTest", "apple base (update)", aliases=["two", "THREE"])
    s = store.get_lexicon_by_term("aliastest")[0]
    # 'two' should not duplicate; 'THREE' added lowercase-equivalent
    assert set(a.lower() for a in (s.aliases or [])) >= {"one", "two", "three"}

def test_examples_merge_then_cap_on_update(lex, store):
    sid = lex.learn_definition("ExTerm", "apple a", examples=["a1", "a2"])
    _ = lex.learn_definition("ExTerm", "apple a update", examples=["a3", "a4", "a5"])
    s = store.get_lexicon_by_term("exterm")[0]
    # merged then capped to MAX_EXAMPLES = 3
    assert s.examples == ["a1", "a2", "a3"]

def test_sources_merge_and_dedup_on_update(lex, store):
    sid = lex.learn_definition("Src", "apple base", source="s1")
    _ = lex.learn_definition("Src", "apple base again", source="s2")
    s = store.get_lexicon_by_term("src")[0]
    assert set(s.sources or []) == {"s1", "s2"}

def test_get_definition_prefers_freq_when_no_context_even_if_vectors_differ(lex, store):
    # two senses: apple and banana; bump banana's freq via update path on banana side
    id1 = lex.learn_definition("FruitX", "apple thing")
    id2 = lex.learn_definition("FruitX", "banana thing")
    _ = lex.learn_definition("FruitX", "banana thing again")  # update to bump freq
    s = lex.get_definition("FruitX")
    assert s.id == id2

def test_update_path_keeps_pinned_true_if_none(lex, store, monkeypatch):
    # Simulate an existing sense whose pinned is None; update should set default pinned=True
    sid = lex.learn_definition("Pin", "apple pin")
    s0 = store.get_lexicon_by_term("pin")[0]
    # Manually mutate to simulate legacy data with pinned=None
    s0 = type(s0).new(**{**s0.to_dict(), "pinned": None}) if hasattr(s0, "to_dict") else s0
    # If no to_dict, we just replace attribute (depends on your LexiconSense impl)
    try:
        s0.pinned = None
    except Exception:
        pass
    store.upsert_lexicon([s0])

    _ = lex.learn_definition("Pin", "apple pin update")
    s1 = store.get_lexicon_by_term("pin")[0]
    assert s1.pinned is True
