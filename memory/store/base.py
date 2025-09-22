# memory/store/base.py
# Base interface (Protocol) for Orrin2.0 memory backends. Use relative imports + keyword-only ann_search args + a stats() hook.

from __future__ import annotations
from typing import List, Dict, Optional, Protocol, Iterable, Tuple
import numpy as np

from ..models import MemoryItem, LexiconSense


class VectorStore(Protocol):
    """
    Minimal interface for Orrin2.0's memory backend.

    Implementations should be LOCAL-first (no network). You can later
    provide LanceDB/FAISS/sqlite-vec adapters that satisfy this Protocol.
    """

    # ---- Upserts ----
    def upsert_items(self, items: List[MemoryItem]) -> None:
        """Create or update MemoryItems (metadata only; vectors are supplied separately)."""

    def upsert_lexicon(self, senses: List[LexiconSense]) -> None:
        """Create or update LexiconSense entries (and maintain term/alias indexes)."""

    def upsert_vectors(self, vectors: Dict[str, np.ndarray]) -> None:
        """
        Upsert embedding vectors by embedding_id. This is called by the ingest loop
        after computing embeddings for newly written items.
        """

    # ---- Retrieval ----
    def ann_search(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int,
        kind_filter: Optional[List[str]] = None,
        meta_filter: Optional[Dict[str, object]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return a list of (item_id, similarity) pairs for the ANN search.

        - kind_filter: restrict to certain MemoryItem.kind values.
        - meta_filter: shallow filter on MemoryItem.meta. List values may be treated
          as "overlap-any", scalars as equality. Implementations may choose a simple
          best-effort semantics here.
        """

    def get_items(self, ids: List[str]) -> List[MemoryItem]:
        """Fetch MemoryItems by id, skipping any not found."""

    # ---- Lexicon ----
    def get_lexicon_by_term(self, term_or_alias: str) -> List[LexiconSense]:
        """Fetch all senses matching a term or any alias (case-insensitive)."""

    # ---- Novelty / Health helpers ----
    def get_recent_vectors(self, n: int = 128) -> Iterable[np.ndarray]:
        """Return up to n *most recently upserted* embedding vectors (for novelty calc)."""

    def stats(self) -> Dict[str, int]:
        """
        Return basic counts used by health/metrics:
          {
            "items_total": int,
            "items_by_layer": {"working": int, "long": int, "summary": int},
            "index_lag": int,             # items with embedding_id but missing vector
            "vectors_total": int,
            "vector_bytes_total": int,
          }
        """


# ---------------------------------------------------------------------
# Convenience helper for tests/scripts:
# Avoid passing 'content' twice (explicit kw + **meta) into MemoryItem.new(...)
# ---------------------------------------------------------------------
def safe_make_item(kind: str = "fact", layer: str = "working", **meta) -> MemoryItem:
    """
    Create a MemoryItem while defensively handling a 'content' key in **meta
    to prevent 'multiple values for keyword argument "content"'.

    Example:
        it = safe_make_item(kind="note", content="hello", topic="x")
    """
    meta = dict(meta)  # copy so we don't mutate caller
    content = meta.pop("content", "")
    source = meta.pop("source", "api")
    return MemoryItem.new(kind=kind, source=source, content=content, layer=layer, **meta)
