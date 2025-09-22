# memory/store/inmem.py
from __future__ import annotations
from typing import List, Dict, Optional, Iterable, Tuple, Any
from collections import deque
import threading
import numpy as np

from ..models import MemoryItem, LexiconSense
from .base import VectorStore

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a); b = _normalize(b)
    if a.shape != b.shape:
        L = max(a.size, b.size)
        if a.size < L: a = np.pad(a, (0, L - a.size))
        if b.size < L: b = np.pad(b, (0, L - b.size))
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


class InMemoryStore(VectorStore):
    """
    Simple, thread-safe, in-memory store for development.
    Items, vectors, and lexicon senses live in dicts.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._items: Dict[str, MemoryItem] = {}
        self._vecs: Dict[str, np.ndarray] = {}
        self._recent_vec_ids: deque[str] = deque(maxlen=10_000)

        self._senses: Dict[str, LexiconSense] = {}
        self._term_index: Dict[str, List[str]] = {}  # lower(term/alias) -> [sense_id]

    # ---------- Items / Vectors ----------
    def upsert_items(self, items: List[MemoryItem]) -> None:
        with self._lock:
            for it in items:
                self._items[it.id] = it

    def upsert_vectors(self, vectors: Dict[str, np.ndarray]) -> None:
        with self._lock:
            for eid, v in vectors.items():
                vv = _normalize(np.asarray(v, dtype=np.float32))
                self._vecs[eid] = vv
                self._recent_vec_ids.append(eid)

    def ann_search(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int,
        kind_filter: Optional[List[str]] = None,
        meta_filter: Optional[Dict[str, object]] = None,
    ) -> List[Tuple[str, float]]:
        q = _normalize(np.asarray(query_vec, dtype=np.float32))
        kind_set = set(k.lower() for k in (kind_filter or []))
        with self._lock:
            # Map item id -> embedding vector
            candidates: List[Tuple[str, np.ndarray]] = []
            for it in self._items.values():
                if kind_set and it.kind.lower() not in kind_set:
                    continue
                if meta_filter:
                    ok = True
                    for mk, mv in meta_filter.items():
                        v = it.meta.get(mk)
                        if isinstance(v, list):
                            ok = any(x == mv for x in v)
                        else:
                            ok = v == mv
                        if not ok: break
                    if not ok:
                        continue
                eid = it.embedding_id
                if not eid or eid not in self._vecs:
                    continue
                candidates.append((it.id, self._vecs[eid]))

            scored = [(mid, _cos(v, q)) for mid, v in candidates]
            scored.sort(key=lambda t: t[1], reverse=True)
            return scored[: max(1, int(top_k))]

    def get_items(self, ids: List[str]) -> List[MemoryItem]:
        with self._lock:
            return [self._items[i] for i in ids if i in self._items]

    # ---------- Lexicon ----------
    def upsert_lexicon(self, senses: List[LexiconSense]) -> None:
        with self._lock:
            for s in senses:
                self._senses[s.id] = s
                # rebuild simple index for this sense
                keys = {s.term.lower().strip()} | {a.lower().strip() for a in (s.aliases or [])}
                for k in keys:
                    self._term_index.setdefault(k, [])
                    if s.id not in self._term_index[k]:
                        self._term_index[k].append(s.id)

    def get_lexicon_by_term(self, term_or_alias: str) -> List[LexiconSense]:
        key = (term_or_alias or "").lower().strip()
        if not key:
            return []
        with self._lock:
            ids = self._term_index.get(key, [])
            return [self._senses[i] for i in ids if i in self._senses]

    # ---------- Novelty / Health ----------
    def get_recent_vectors(self, n: int = 128) -> Iterable[np.ndarray]:
        with self._lock:
            ids = list(self._recent_vec_ids)[-int(n):]
            return [self._vecs[i] for i in ids if i in self._vecs]

    def stats(self) -> Dict[str, Any]:
        items_by_layer = {"working": 0, "long": 0, "summary": 0}
        for it in self._items.values():
            layer = (it.layer or "").lower()
            if layer in items_by_layer:
                items_by_layer[layer] += 1
            else:
                items_by_layer[layer] = items_by_layer.get(layer, 0) + 1  # tolerate unknown layers

        # Count index lag for non-summary items only
        index_lag = 0
        for it in self._items.values():
            layer = (it.layer or "").lower()
            if layer == "summary":
                continue
            emb = getattr(it, "embedding_id", None)
            if emb and emb not in self._vecs:
                index_lag += 1

        vectors_total = len(self._vecs)

        # bytes of raw float32 storage (approx); prefer numpy nbytes when available
        vector_bytes_total = 0
        for v in self._vecs.values():
            try:
                vector_bytes_total += int(getattr(v, "nbytes"))
            except Exception:
                try:
                    vector_bytes_total += int(getattr(v, "size", len(v))) * 4
                except Exception:
                    pass

        return {
            "items_total": len(self._items),
            "items_by_layer": items_by_layer,
            "vectors_total": vectors_total,
            "vector_bytes_total": vector_bytes_total,
            "index_lag": index_lag,
        }
