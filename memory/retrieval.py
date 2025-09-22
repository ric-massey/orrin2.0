# Vector retrieval helpers: ANN search, cosine⊕strength rerank, optional MMR re-ranking, and optional reinforcement on access.

from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Iterable
from datetime import datetime, timezone
import numpy as np

from .config import MEMCFG
from .models import MemoryItem
from .embedder import get_embedding
from .strength import strength_from


# ---------------------------
# Small math & utils
# ---------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = _normalize(a), _normalize(b)
    # both normalized → denom≈1; keep numerical guard
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9
    return float(np.dot(va, vb) / denom)

def _blend(sim: float, strength: float, alpha: float) -> float:
    return float(alpha) * float(sim) + (1.0 - float(alpha)) * float(strength or 0.0)


# ---------------------------
# Vector access for items
# ---------------------------
def _get_item_vec(store, it: MemoryItem) -> Optional[np.ndarray]:
    """
    Best-effort vector lookup:
      1) If store exposes a private _vecs map (InMemoryStore), use it.
      2) Else recompute from content text (cheap dev fallback).
    """
    emb_id = getattr(it, "embedding_id", None)
    if emb_id and hasattr(store, "_vecs"):
        vec = getattr(store, "_vecs").get(emb_id)
        if isinstance(vec, np.ndarray):
            return vec
    # Fallback: recompute from content
    try:
        return get_embedding(it.content)
    except Exception:
        return None

def _get_item_vecs(store, items: List[MemoryItem]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for it in items:
        v = _get_item_vec(store, it)
        if isinstance(v, np.ndarray):
            out[it.id] = _normalize(v)
    return out


# ---------------------------
# MMR (optional)
# ---------------------------
def _mmr_order(
    qvec: np.ndarray,
    ids_in: List[str],
    vecs: Dict[str, np.ndarray],
    *,
    k: int,
    lamb: float = 0.5,
    duplicate_sim: float = 0.96,
) -> List[str]:
    """
    Maximal Marginal Relevance re-ranking:
      score(i) = λ * sim(i, q) - (1-λ) * max_{j∈S} sim(i, j)

    Practical tweaks:
      - Skip candidates whose redundancy to the selected set exceeds `duplicate_sim`
        (near-duplicates). If that would select nothing in an iteration, relax the
        skip for that iteration (so we still fill k).
      - Tie-break by lower redundancy (penalty).
    """
    if not ids_in:
        return []
    k = max(1, min(k, len(ids_in)))
    q = _normalize(qvec)

    # Precompute sims to query
    sim_q = {i: _cos(vecs[i], q) for i in ids_in if i in vecs}

    selected: List[str] = []
    candidates: List[str] = [i for i in ids_in if i in vecs]

    while len(selected) < k and candidates:
        best_id = None
        best_score = -1e9
        best_penalty = 1e9

        # First pass: enforce duplicate cutoff
        for cid in candidates:
            if not selected:
                penalty = 0.0
            else:
                penalty = max(_cos(vecs[cid], vecs[sid]) for sid in selected)
                if penalty >= float(duplicate_sim):
                    # too redundant; skip in first pass
                    continue
            score = float(lamb) * float(sim_q.get(cid, 0.0)) - float(1.0 - lamb) * float(penalty)
            if (score > best_score + 1e-12) or (abs(score - best_score) <= 1e-12 and penalty < best_penalty):
                best_score = score
                best_penalty = penalty
                best_id = cid

        # If every candidate was skipped as a duplicate, relax and pick the best anyway
        if best_id is None:
            for cid in candidates:
                if not selected:
                    penalty = 0.0
                else:
                    penalty = max(_cos(vecs[cid], vecs[sid]) for sid in selected)
                score = float(lamb) * float(sim_q.get(cid, 0.0)) - float(1.0 - lamb) * float(penalty)
                if (score > best_score + 1e-12) or (abs(score - best_score) <= 1e-12 and penalty < best_penalty):
                    best_score = score
                    best_penalty = penalty
                    best_id = cid

        selected.append(best_id)
        candidates.remove(best_id)

    return selected



# ---------------------------
# Public API
# ---------------------------
def retrieve(
    store,
    *,
    query_text: Optional[str] = None,
    query_vec: Optional[np.ndarray] = None,
    top_k: Optional[int] = None,
    kinds: Optional[List[str]] = None,
    meta_filter: Optional[Dict[str, object]] = None,
    alpha: Optional[float] = None,
    use_mmr: bool = False,
    mmr_lambda: float = 0.5,
    overfetch: int = 3,
    reinforce: bool = True,
) -> List[MemoryItem]:
    """
    End-to-end retrieval:
      1) Encode query (unless query_vec provided)
      2) ANN search (overfetch to help re-ranking)
      3) Re-rank by alpha*cosine + (1-alpha)*strength
      4) Optional MMR re-ranking for diversity
      5) Optional reinforcement (freq/strength bump on access)

    Returns a list of MemoryItem (up to top_k).
    """
    if query_vec is None:
        if not isinstance(query_text, str) or not query_text.strip():
            return []
        qv = get_embedding(query_text)
    else:
        qv = _normalize(np.asarray(query_vec, dtype=np.float32))

    k = int(top_k or MEMCFG.RETRIEVE_TOP_K)
    a = float(MEMCFG.RETRIEVE_ALPHA if alpha is None else alpha)

    # 1) ANN search
    hits: List[Tuple[str, float]] = store.ann_search(qv, top_k=max(1, k * max(1, overfetch)), kind_filter=kinds, meta_filter=meta_filter)
    if not hits:
        return []

    # 2) Fetch items
    ids, sims = zip(*hits)
    items = {it.id: it for it in store.get_items(list(ids))}
    # 3) blend score
    blended: List[Tuple[str, float]] = []
    for mid, sim in hits:
        it = items.get(mid)
        if not it:
            continue
        blended.append((mid, _blend(sim, float(it.strength or 0.0), a)))

    blended.sort(key=lambda t: t[1], reverse=True)
    reranked_ids = [mid for mid, _ in blended]

    # 4) Optional MMR (works on the top slice after blend)
    if use_mmr and reranked_ids:
        top_for_mmr = reranked_ids[: max(k * 2, k)]  # keep some headroom
        head_items = [items[i] for i in top_for_mmr if i in items]
        vecs = _get_item_vecs(store, head_items)
        mmr_ids = _mmr_order(qv, [it.id for it in head_items], vecs, k=k, lamb=float(mmr_lambda))
        reranked_ids = mmr_ids + [i for i in reranked_ids if i not in mmr_ids]

    final_ids = reranked_ids[:k]
    out = [items[i] for i in final_ids if i in items]

    # 5) Reinforce on access
    if reinforce and out:
        for it in out:
            it.freq = (it.freq or 0) + 1
            it.last_access = _now_iso()
            tau = MEMCFG.tau_for_layer(it.layer)
            it.strength = strength_from(it.freq, 0.0, float(it.goal_relevance or 0.0), tau)
        store.upsert_items(out)

    return out


def score_only(
    store,
    *,
    query_text: Optional[str] = None,
    query_vec: Optional[np.ndarray] = None,
    top_k: Optional[int] = None,
    kinds: Optional[List[str]] = None,
    meta_filter: Optional[Dict[str, object]] = None,
    alpha: Optional[float] = None,
) -> List[Tuple[MemoryItem, float]]:
    """
    Same as retrieve(), but returns (item, blended_score) and does NOT reinforce.
    Useful for diagnostics or read-only ranking.
    """
    res = retrieve(
        store,
        query_text=query_text,
        query_vec=query_vec,
        top_k=top_k,
        kinds=kinds,
        meta_filter=meta_filter,
        alpha=alpha,
        use_mmr=False,
        overfetch=3,
        reinforce=False,
    )
    # Recompute blended score against the same alpha for transparency
    a = float(MEMCFG.RETRIEVE_ALPHA if alpha is None else alpha)
    qv = get_embedding(query_text) if query_vec is None else _normalize(np.asarray(query_vec, dtype=np.float32))

    # Compute similarity to query for diagnostics
    out: List[Tuple[MemoryItem, float]] = []
    for it in res:
        v = _get_item_vec(store, it)
        sim = _cos(v, qv) if v is not None else 0.0
        out.append((it, _blend(sim, float(it.strength or 0.0), a)))
    # Keep in score-desc order
    out.sort(key=lambda t: t[1], reverse=True)
    return out
