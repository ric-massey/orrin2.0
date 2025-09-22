# tests/memory_tests/inmem_test.py
import threading
import time
import numpy as np
import pytest

from memory.store.inmem import InMemoryStore
from memory.models import MemoryItem, LexiconSense


def _vec(arr):
    return np.asarray(arr, dtype=np.float32)


def _mk_item(kind="fact", layer="working", **meta):
    it = MemoryItem.new(kind=kind, source="test", content=meta.get("content", ""), layer=layer, **meta)
    # The inmem store expects items to carry their embedding_id directly
    it.embedding_id = f"eid_{it.id}"
    return it


def _mk_sense(term, definition, aliases=None):
    s = LexiconSense.new(term=term, sense_id=f"{term.lower()}:1", definition=definition, aliases=aliases or [])
    return s


# ----------------- basic upserts & search -----------------

def test_upsert_and_ann_search_basic_similarity_and_order():
    st = InMemoryStore()
    it1 = _mk_item(kind="fact")
    it2 = _mk_item(kind="note")

    st.upsert_items([it1, it2])
    st.upsert_vectors({
        it1.embedding_id: _vec([1, 0, 0]),
        it2.embedding_id: _vec([0.5, 0.0, 0.0]),  # same direction, smaller magnitude (should normalize)
    })

    # Query along x-axis → both should score ~1, but stable order by score then id insertion
    hits = st.ann_search(_vec([10.0, 0.0, 0.0]), top_k=2)
    assert [h[0] for h in hits] == [it1.id, it2.id]
    assert all(0.99 <= sim <= 1.0 for _, sim in hits)


def test_ann_search_kind_filter_and_case_insensitive():
    st = InMemoryStore()
    it1 = _mk_item(kind="Fact"); it2 = _mk_item(kind="Note"); it3 = _mk_item(kind="Fact")
    st.upsert_items([it1, it2, it3])
    v = _vec([1, 0])
    st.upsert_vectors({it1.embedding_id: v, it2.embedding_id: v, it3.embedding_id: v})

    hits = st.ann_search(_vec([1, 0]), top_k=10, kind_filter=["fact"])  # lowercased filter
    ids = [h[0] for h in hits]
    assert it1.id in ids and it3.id in ids and it2.id not in ids


def test_ann_search_meta_filter_matches_scalar_and_list_membership():
    st = InMemoryStore()
    it1 = _mk_item(kind="fact", importance="high")
    it2 = _mk_item(kind="fact", tags=["a", "b"])
    it3 = _mk_item(kind="fact", tags=["c"])
    st.upsert_items([it1, it2, it3])
    v = _vec([1, 0])
    st.upsert_vectors({it1.embedding_id: v, it2.embedding_id: v, it3.embedding_id: v})

    # scalar equality
    hits = st.ann_search(_vec([1, 0]), top_k=10, meta_filter={"importance": "high"})
    assert [h[0] for h in hits] == [it1.id]

    # when meta value is a list: store checks ANY element equals the scalar filter value
    hits = st.ann_search(_vec([1, 0]), top_k=10, meta_filter={"tags": "a"})
    ids = [h[0] for h in hits]
    assert it2.id in ids and it3.id not in ids

    # non-matching scalar against list
    hits = st.ann_search(_vec([1, 0]), top_k=10, meta_filter={"tags": "z"})
    assert hits == []


def test_ann_search_skips_items_without_vectors():
    st = InMemoryStore()
    it1 = _mk_item(); it2 = _mk_item()
    st.upsert_items([it1, it2])
    st.upsert_vectors({it1.embedding_id: _vec([1, 0])})
    hits = st.ann_search(_vec([1, 0]), top_k=10)
    ids = [h[0] for h in hits]
    assert it1.id in ids and it2.id not in ids


def test_ann_search_zero_norm_query_returns_low_scores_but_respects_top_k_floor():
    st = InMemoryStore()
    it1 = _mk_item()
    st.upsert_items([it1])
    st.upsert_vectors({it1.embedding_id: _vec([1, 0])})

    zeros = np.zeros(2, dtype=np.float32)
    # Implementation returns at least 1 hit due to max(1, top_k) and cosine ~ 0
    hits = st.ann_search(zeros, top_k=0)  # even 0 => at least 1 by design
    assert len(hits) == 1
    assert hits[0][0] == it1.id
    assert abs(hits[0][1]) <= 1e-6  # ~0.0 similarity


def test_ann_search_top_k_is_keyword_only():
    st = InMemoryStore()
    it = _mk_item(); st.upsert_items([it]); st.upsert_vectors({it.embedding_id: _vec([1, 0])})
    with pytest.raises(TypeError):
        # type: ignore[arg-type]
        st.ann_search(_vec([1, 0]), 1)  # should require keyword (top_k=1)


# ----------------- lexicon -----------------

def test_lexicon_upsert_and_lookup_case_insensitive_and_aliases():
    st = InMemoryStore()
    s1 = _mk_sense("GPU", "graphics processing unit", aliases=["Graphics Card", "gpu"])
    st.upsert_lexicon([s1])

    got_term = st.get_lexicon_by_term("gpu")
    got_alias = st.get_lexicon_by_term("graphics card")
    assert len(got_term) == 1 and len(got_alias) == 1
    assert got_term[0].id == s1.id == got_alias[0].id


def test_lexicon_upsert_adds_aliases_but_does_not_remove_stale_ones():
    st = InMemoryStore()
    s1 = _mk_sense("DSP", "digital signal processing", aliases=["sigproc"])
    st.upsert_lexicon([s1])

    # Re-upsert with different alias set; this simple indexer only adds (doesn't drop)
    s1b = LexiconSense.new(term=s1.term, sense_id=s1.sense_id, definition=s1.definition, aliases=["dsp"])
    s1b.id = s1.id
    st.upsert_lexicon([s1b])

    # Both old and new aliases should work in this implementation
    assert st.get_lexicon_by_term("sigproc") or st.get_lexicon_by_term("SIGPROC")
    assert st.get_lexicon_by_term("dsp")


def test_get_lexicon_by_term_empty_key_returns_empty():
    st = InMemoryStore()
    assert st.get_lexicon_by_term("") == []
    assert st.get_lexicon_by_term("   ") == []


# ----------------- novelty / health -----------------

def test_get_recent_vectors_returns_last_n_in_chronological_order():
    st = InMemoryStore()
    it1 = _mk_item(); it2 = _mk_item(); it3 = _mk_item()
    st.upsert_items([it1, it2, it3])

    v1 = _vec([1, 0, 0]); v2 = _vec([0, 1, 0]); v3 = _vec([0, 0, 1])
    st.upsert_vectors({it1.embedding_id: v1})
    st.upsert_vectors({it2.embedding_id: v2})
    st.upsert_vectors({it3.embedding_id: v3})

    rec = list(st.get_recent_vectors(2))
    # In this store, order is chronological for the last N (not reversed):
    # expect [v2_norm, v3_norm]
    assert len(rec) == 2
    assert np.allclose(rec[0], v2 / np.linalg.norm(v2))
    assert np.allclose(rec[1], v3 / np.linalg.norm(v3))


def test_stats_counts_layers_index_lag_vectors_and_bytes():
    st = InMemoryStore()
    it1 = _mk_item(layer="working"); it2 = _mk_item(layer="long"); it3 = _mk_item(layer="summary")
    it4 = _mk_item(layer="working")  # will be index-lag (vector missing)
    st.upsert_items([it1, it2, it3, it4])
    v = _vec([1, 2, 3, 4])  # 4 float32 -> 16 bytes
    st.upsert_vectors({it1.embedding_id: v, it2.embedding_id: v})

    stt = st.stats()
    assert stt["items_total"] == 4
    assert stt["items_by_layer"]["working"] == 2
    assert stt["items_by_layer"]["long"] == 1
    assert stt["items_by_layer"]["summary"] == 1
    assert stt["vectors_total"] == 2
    assert stt["index_lag"] == 1
    assert stt["vector_bytes_total"] >= 32  # at least 2 vectors * 16 bytes


# ----------------- concurrency smoke -----------------

def test_threaded_upserts_are_safe_smoke():
    st = InMemoryStore()
    items = [_mk_item() for _ in range(50)]
    st.upsert_items(items)

    def worker(lo, hi):
        for i in range(lo, hi):
            st.upsert_vectors({items[i].embedding_id: _vec([i + 1.0, 0.0, 0.0])})

    threads = [threading.Thread(target=worker, args=(i * 10, (i + 1) * 10)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()

    # All vectors should be present; a quick query should return something
    hits = st.ann_search(_vec([1, 0, 0]), top_k=5)
    assert len(hits) == 5
