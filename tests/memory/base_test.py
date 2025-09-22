# tests/memory_tests/base_test.py
import inspect
from typing import Dict, List, Optional, Iterable, Tuple, get_type_hints
import numpy as np
import pytest

from memory.store.base import VectorStore
from memory.models import MemoryItem, LexiconSense

# ---------------------------------------------------------------------
# A small *test-only* compliant implementation to validate the contract
# ---------------------------------------------------------------------

class _CompliantStore:
    """
    Minimal local implementation that follows VectorStore's docstring semantics.
    NOT your production store; used only to test the base Protocol contract.
    """
    def __init__(self) -> None:
        # items & vectors
        self._items: Dict[str, MemoryItem] = {}
        self._vecs: Dict[str, np.ndarray] = {}         # embedding_id -> 1D float32 (L2-normed)
        self._item_to_eid: Dict[str, str] = {}         # item_id -> embedding_id
        self._eid_to_item: Dict[str, str] = {}         # embedding_id -> item_id
        self._recent_eids: List[str] = []              # append order for novelty

        # lexicon
        self._lex_by_id: Dict[str, LexiconSense] = {}
        self._lex_term_idx: Dict[str, set[str]] = {}
        self._lex_alias_idx: Dict[str, set[str]] = {}

    # ---------------- Upserts ----------------

    def upsert_items(self, items: List[MemoryItem]) -> None:
        for it in items:
            self._items[it.id] = it
            eid = getattr(it, "embedding_id", None)
            if isinstance(eid, str) and eid:
                # if this item had a previous eid, forget the old reverse mapping
                prev = self._item_to_eid.get(it.id)
                if prev and prev in self._eid_to_item:
                    self._eid_to_item.pop(prev, None)
                self._item_to_eid[it.id] = eid
                self._eid_to_item[eid] = it.id

    def upsert_vectors(self, vectors: Dict[str, np.ndarray]) -> None:
        for eid, v in vectors.items():
            # accept lists, tuples, etc.
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(arr))
            if n > 0:
                arr = arr / n
            self._vecs[eid] = arr
            self._recent_eids.append(eid)

    def upsert_lexicon(self, senses: List[LexiconSense]) -> None:
        for s in senses:
            # drop old indexes if replacing
            old = self._lex_by_id.get(s.id)
            if old:
                self._drop_lex_from_index(old)
            self._lex_by_id[s.id] = s
            self._index_lex(s)

    # ---------------- Retrieval ----------------

    def ann_search(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int,
        kind_filter: Optional[List[str]] = None,
        meta_filter: Optional[Dict[str, object]] = None,
    ) -> List[Tuple[str, float]]:
        arr = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        qn = float(np.linalg.norm(arr))
        if qn == 0:
            return []
        q = arr / qn

        kinds = set(k.lower() for k in (kind_filter or [])) or None

        # collect candidates with vectors
        cands: List[str] = []
        for iid, it in self._items.items():
            if kinds is not None:
                if (it.kind or "").lower() not in kinds:
                    continue
            if meta_filter and not _meta_matches(getattr(it, "meta", {}) or {}, meta_filter):
                continue
            eid = self._item_to_eid.get(iid)
            if not eid or eid not in self._vecs:
                continue
            cands.append(iid)

        scored: List[Tuple[str, float]] = []
        for iid in cands:
            v = self._vecs.get(self._item_to_eid[iid])
            if v is None:
                continue
            sim = float(np.dot(q, v))  # both normalized
            scored.append((iid, sim))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: max(0, int(top_k))]

    def get_items(self, ids: List[str]) -> List[MemoryItem]:
        return [self._items[i] for i in ids if i in self._items]

    # ---------------- Lexicon ----------------

    def get_lexicon_by_term(self, term_or_alias: str) -> List[LexiconSense]:
        k = (term_or_alias or "").lower()
        ids = set()
        ids |= self._lex_term_idx.get(k, set())
        ids |= self._lex_alias_idx.get(k, set())
        return [self._lex_by_id[i] for i in ids if i in self._lex_by_id]

    # ---------------- Novelty / Health ----------------

    def get_recent_vectors(self, n: int = 128) -> Iterable[np.ndarray]:
        out: List[np.ndarray] = []
        for eid in reversed(self._recent_eids):
            v = self._vecs.get(eid)
            if v is not None:
                out.append(v)
                if len(out) >= n:
                    break
        return out

    def stats(self) -> Dict[str, int]:
        by_layer = {"working": 0, "long": 0, "summary": 0}
        for it in self._items.values():
            layer = (it.layer or "").lower()
            if layer in by_layer:
                by_layer[layer] += 1

        lag = 0
        for it in self._items.values():
            eid = getattr(it, "embedding_id", None)
            if eid and eid not in self._vecs:
                lag += 1

        v_total = len(self._vecs)
        v_bytes = int(sum(v.nbytes for v in self._vecs.values()))
        return {
            "items_total": len(self._items),
            "items_by_layer": by_layer,  # nested dict (typing hints allow Dict[str, int] top-level)
            "index_lag": lag,
            "vectors_total": v_total,
            "vector_bytes_total": v_bytes,
        }

    # ------------- lexicon index helpers -------------

    def _index_lex(self, s: LexiconSense) -> None:
        t = (s.term or "").lower()
        if t:
            self._lex_term_idx.setdefault(t, set()).add(s.id)
        for a in (s.aliases or []):
            a = (a or "").lower()
            if a:
                self._lex_alias_idx.setdefault(a, set()).add(s.id)

    def _drop_lex_from_index(self, s: LexiconSense) -> None:
        tid = s.id
        t = (s.term or "").lower()
        if t and t in self._lex_term_idx:
            self._lex_term_idx[t].discard(tid)
        for a in (s.aliases or []):
            a = (a or "").lower()
            if a and a in self._lex_alias_idx:
                self._lex_alias_idx[a].discard(tid)


# ---------------- helpers mirrored from base semantics ----------------

def _meta_matches(meta: Dict[str, object], filt: Dict[str, object]) -> bool:
    for k, want in filt.items():
        have = meta.get(k, None)
        if isinstance(want, (list, tuple, set)):
            want_set = set(want)
            if isinstance(have, (list, tuple, set)):
                if not want_set.intersection(set(have)):
                    return False
            else:
                if have not in want_set:
                    return False
        else:
            if isinstance(have, (list, tuple, set)):
                if want not in have:
                    return False
            else:
                if have != want:
                    return False
    return True


# ================================
# Protocol shape / signature tests
# ================================

def test_protocol_has_expected_methods():
    required = {
        "upsert_items", "upsert_lexicon", "upsert_vectors",
        "ann_search", "get_items", "get_lexicon_by_term",
        "get_recent_vectors", "stats",
    }
    for name in required:
        assert hasattr(VectorStore, name), f"VectorStore missing {name}"

def test_ann_search_signature_is_keyword_only_for_filters_and_top_k():
    sig = inspect.signature(VectorStore.ann_search)
    params = list(sig.parameters.values())
    # Expected params: self, query_vec, *, top_k, kind_filter, meta_filter
    assert params[0].name == "self"
    assert params[1].name == "query_vec"
    # Must have a VAR_POSITIONAL star marker via KEYWORD_ONLY params following index 1
    assert any(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params[2:]), "Expected keyword-only args after '*'"
    names = [p.name for p in params]
    assert "top_k" in names and "kind_filter" in names and "meta_filter" in names
    # Ensure top_k is KEYWORD_ONLY
    top_k_param = [p for p in params if p.name == "top_k"][0]
    assert top_k_param.kind is inspect.Parameter.KEYWORD_ONLY

def test_stats_return_type_hint_is_dict_of_ints():
    hints = get_type_hints(VectorStore.stats)
    # For methods, the return type is under 'return'
    assert "return" in hints
    # We only assert that it's annotated to Dict[str, int]-like
    # (exact typing equality isn't necessary for runtime, but check presence)
    assert hints["return"] is dict or hints["return"].__origin__ is dict


# ================================
# Contract behavior tests (using _CompliantStore)
# ================================

@pytest.fixture
def store():
    return _CompliantStore()

def _vec(x):
    return np.asarray(x, dtype=np.float32)

def _mk_item(kind="fact", layer="working", **meta):
    # Avoid passing the same kwarg twice (e.g., content/source in **meta and explicitly)
    meta = dict(meta)  # don't mutate caller
    content = meta.pop("content", "")
    source = meta.pop("source", "test")
    it = MemoryItem.new(kind=kind, source=source, content=content, layer=layer, **meta)
    # give an embedding id so we can attach vectors
    it.embedding_id = f"eid_{it.id}"
    return it


def _mk_sense(term, definition, aliases=None):
    s = LexiconSense.new(term=term, sense_id=f"{term.lower()}:1", definition=definition, aliases=aliases or [])
    return s

def test_upsert_and_basic_retrieval(store):
    it1 = _mk_item(kind="fact", content="a"); it2 = _mk_item(kind="note", content="b")
    store.upsert_items([it1, it2])
    store.upsert_vectors({it1.embedding_id: _vec([1,0]), it2.embedding_id: _vec([0,1])})

    hits = store.ann_search(_vec([1,0]), top_k=1)
    assert hits and hits[0][0] == it1.id and hits[0][1] > 0.9

def test_ann_search_requires_keyword_top_k(store):
    it = _mk_item(); store.upsert_items([it]); store.upsert_vectors({it.embedding_id: _vec([1,0])})
    # Calling with positional top_k should raise TypeError (keyword-only)
    with pytest.raises(TypeError):
        # type: ignore[arg-type]
        store.ann_search(_vec([1,0]), 1)  # wrong: top_k must be keyword

def test_kind_filter_and_meta_filter_semantics(store):
    it1 = _mk_item(kind="fact", importance="high")
    it2 = _mk_item(kind="note", importance=["low", "medium"])
    it3 = _mk_item(kind="fact", importance="low")
    store.upsert_items([it1, it2, it3])
    store.upsert_vectors({
        it1.embedding_id: _vec([1,0]), it2.embedding_id: _vec([1,0]), it3.embedding_id: _vec([1,0])
    })

    # Only kind=fact
    hits = store.ann_search(_vec([1,0]), top_k=10, kind_filter=["FACT"])
    ids = [h[0] for h in hits]
    assert it1.id in ids and it3.id in ids and it2.id not in ids

    # meta scalar equality
    hits = store.ann_search(_vec([1,0]), top_k=10, meta_filter={"importance": "high"})
    ids = [h[0] for h in hits]
    assert it1.id in ids and it2.id not in ids and it3.id not in ids

    # meta list overlap-any (want any of {"low","medium"})
    hits = store.ann_search(_vec([1,0]), top_k=10, meta_filter={"importance": ["low", "medium"]})
    ids = [h[0] for h in hits]
    assert it2.id in ids or it3.id in ids

def test_missing_vectors_are_skipped_and_counted_as_index_lag(store):
    it1 = _mk_item(); it2 = _mk_item()
    store.upsert_items([it1, it2])
    # only give a vector to it1
    store.upsert_vectors({it1.embedding_id: _vec([1,0])})
    hits = store.ann_search(_vec([1,0]), top_k=10)
    ids = [h[0] for h in hits]
    assert it1.id in ids and it2.id not in ids

    st = store.stats()
    assert st["index_lag"] >= 1

def test_zero_norm_query_returns_empty(store):
    it1 = _mk_item(); store.upsert_items([it1]); store.upsert_vectors({it1.embedding_id: _vec([1,0])})
    zeros = np.zeros(2, dtype=np.float32)
    assert store.ann_search(zeros, top_k=5) == []

def test_top_k_limits_results(store):
    it1 = _mk_item(); it2 = _mk_item(); it3 = _mk_item()
    store.upsert_items([it1, it2, it3])
    store.upsert_vectors({
        it1.embedding_id: _vec([1,0]), it2.embedding_id: _vec([1,0]), it3.embedding_id: _vec([1,0]),
    })
    hits = store.ann_search(_vec([1,0]), top_k=2)
    assert len(hits) == 2

def test_get_items_skips_missing_ids(store):
    it1 = _mk_item(); it2 = _mk_item()
    store.upsert_items([it1, it2])
    got = store.get_items([it1.id, "missing", it2.id])
    assert [g.id for g in got] == [it1.id, it2.id]

def test_recent_vectors_returns_most_recent_upserts(store):
    it1 = _mk_item(); it2 = _mk_item(); store.upsert_items([it1, it2])
    v1 = _vec([1,0,0]); v2 = _vec([0,1,0]); v3 = _vec([0,0,1])
    store.upsert_vectors({it1.embedding_id: v1})
    store.upsert_vectors({it2.embedding_id: v2})
    store.upsert_vectors({it1.embedding_id: v3})  # overwrite/another eid upsert order

    rec = list(store.get_recent_vectors(2))
    # Should be [v3_normed, v2_normed] based on upsert order (last two)
    assert len(rec) == 2
    assert np.allclose(rec[0], v3 / np.linalg.norm(v3))
    assert np.allclose(rec[1], v2 / np.linalg.norm(v2))

def test_stats_keys_and_layers_and_vector_bytes(store):
    it1 = _mk_item(layer="working"); it2 = _mk_item(layer="long")
    store.upsert_items([it1, it2])
    v = _vec([1,2,3])
    store.upsert_vectors({it1.embedding_id: v})
    st = store.stats()
    for key in ("items_total", "items_by_layer", "index_lag", "vectors_total", "vector_bytes_total"):
        assert key in st
    assert st["items_total"] == 2
    assert st["items_by_layer"]["working"] == 1
    assert st["items_by_layer"]["long"] == 1
    assert st["vectors_total"] == 1
    # 3 float32s = 12 bytes
    assert st["vector_bytes_total"] >= 12

def test_lexicon_case_insensitive_term_and_alias(store):
    s1 = _mk_sense("GPU", "graphics processing unit", aliases=["graphics card"])
    store.upsert_lexicon([s1])
    got1 = store.get_lexicon_by_term("gpu")
    got2 = store.get_lexicon_by_term("Graphics Card")
    assert len(got1) == 1 and len(got2) == 1
    assert got1[0].id == s1.id == got2[0].id

def test_alias_dedup_and_reindex_on_upsert(store):
    s1 = _mk_sense("DSP", "digital signal processing", aliases=["signal proc"])
    store.upsert_lexicon([s1])

    # upsert updated sense with more aliases; ensure indexes reflect latest set
    s1b = LexiconSense.new(
        term=s1.term, sense_id=s1.sense_id, definition=s1.definition,
        aliases=["signal proc", "d.s.p."],
    )
    s1b.id = s1.id  # replace same id
    store.upsert_lexicon([s1b])

    got = store.get_lexicon_by_term("d.s.p.")
    assert got and got[0].id == s1.id

def test_upserting_item_changes_embedding_id_updates_reverse_map(store):
    it = _mk_item()
    store.upsert_items([it])
    store.upsert_vectors({it.embedding_id: _vec([1,0])})
    # change embedding id on the item and upsert again
    old_eid = it.embedding_id
    it.embedding_id = f"{old_eid}_new"
    store.upsert_items([it])
    store.upsert_vectors({it.embedding_id: _vec([0,1])})

    # The old eid should not map to this item anymore (and shouldn't be used)
    # Retrieval towards [0,1] should still find the item
    hits = store.ann_search(_vec([0,1]), top_k=1)
    assert hits and hits[0][0] == it.id
