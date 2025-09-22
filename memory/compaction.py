# memory/compaction.py
# Working→long compaction and clustering with summary generation.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional
import time
import numpy as np

from .models import MemoryItem
from .embedder import get_embedding
from .strength import clamp01

@dataclass
class CompactionStats:
    processed: int = 0
    promoted: int = 0
    summary_items_created: int = 0
    near_duplicates_dropped: int = 0
    clusters_formed: int = 0

def should_compact(working_cache_size: int, last_ts: float, *, cap: int, interval_minutes: int) -> bool:
    if working_cache_size >= int(cap):
        return True
    if last_ts <= 0:
        return True
    return ((time.time() - float(last_ts)) / 60.0) >= float(interval_minutes)

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a); b = _normalize(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

def _get_vecs(store, items: List[MemoryItem]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for it in items:
        v = None
        if getattr(it, "embedding_id", None) and hasattr(store, "_vecs"):
            v = getattr(store, "_vecs").get(it.embedding_id)
        if v is None:
            v = get_embedding(it.content)
        out[it.id] = _normalize(v)
    return out

def _greedy_clusters(items: List[MemoryItem], vecs: Dict[str, np.ndarray], sim_thr: float) -> List[List[MemoryItem]]:
    pool = list(items)
    clusters: List[List[MemoryItem]] = []
    while pool:
        pivot = pool.pop(0)
        group = [pivot]
        rest = []
        pv = vecs[pivot.id]
        for it in pool:
            if _cos(vecs[it.id], pv) >= sim_thr:
                group.append(it)
            else:
                rest.append(it)
        clusters.append(group)
        pool = rest
    return clusters

def _make_summary_text(cluster: List[MemoryItem], *, max_bullets: int, bullet_chars: int) -> str:
    bullets = []
    for it in cluster[:max_bullets]:
        t = it.content.strip().replace("\n", " ")
        if len(t) > bullet_chars:
            t = t[:bullet_chars-1] + "…"
        bullets.append(f"• {t}")
    return "Summary of related items:\n" + "\n".join(bullets)

def compact_and_promote(
    store,
    working_items: List[MemoryItem],
    *,
    sim_threshold: float,
    duplicate_sim: float,
    min_cluster_size: int,
    max_bullets: int,
    bullet_chars: int,
    promote_layer: str,
    wal=None,
) -> CompactionStats:
    stats = CompactionStats(processed=len(working_items))
    if not working_items:
        return stats

    # Fetch/normalize vectors
    vecs = _get_vecs(store, working_items)

    # Cluster
    clusters = _greedy_clusters(working_items, vecs, sim_threshold)
    stats.clusters_formed = len(clusters)

    to_update: List[MemoryItem] = []
    new_summaries: List[MemoryItem] = []

    for group in clusters:
        if len(group) >= max(2, int(min_cluster_size)):
            # Near-dup elimination only when we have enough evidence (3+ items).
            if len(group) >= 3:
                pivot = group[0]
                kept: List[MemoryItem] = [pivot]
                pv = vecs[pivot.id]
                for it in group[1:]:
                    if _cos(vecs[it.id], pv) >= float(duplicate_sim):
                        stats.near_duplicates_dropped += 1
                    else:
                        kept.append(it)
            else:
                # For 2-item clusters, keep both (still summarize/promote).
                kept = list(group)

            # Promote kept members
            for it in kept:
                it.layer = promote_layer
                to_update.append(it)
            stats.promoted += len(kept)

            # Create a summary node for the cluster (even if size==2)
            summary_text = _make_summary_text(kept, max_bullets=max_bullets, bullet_chars=bullet_chars)
            sm = MemoryItem.new(kind="summary", source="compaction", content=summary_text, layer="summary")
            sm.summary_of = [it.id for it in kept]
            sm.strength = 0.30
            sm.embedding_id = f"vec_{sm.id}"
            sm.embedding_dim = None  # set after embedding
            new_summaries.append(sm)
        else:
            # Small cluster: just promote items
            for it in group:
                it.layer = promote_layer
                to_update.append(it)
            stats.promoted += len(group)

    # Upserts: items
    if to_update:
        store.upsert_items(to_update)
        if wal:
            try:
                wal.append_items(to_update)
            except Exception:
                pass

    # Upserts: summaries + vectors
    if new_summaries:
        vec_map: Dict[str, np.ndarray] = {}
        for sm in new_summaries:
            v = get_embedding(sm.content)
            sm.embedding_dim = int(len(v))
            vec_map[sm.embedding_id] = _normalize(v)
        store.upsert_items(new_summaries)
        store.upsert_vectors(vec_map)
        if wal:
            try:
                wal.append_items(new_summaries)
            except Exception:
                pass
        stats.summary_items_created += len(new_summaries)

    return stats
