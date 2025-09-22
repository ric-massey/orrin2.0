# memory/ingest.py
# Turn Events into MemoryItems: sanitize meta, embed, score novelty/salience, decide keep, set priors, and return (item, vector, salience).

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np

from .config import MEMCFG
from .models import Event, MemoryItem
from .embedder import get_embedding, model_hint
from .novelty import novelty as novelty_score
from .strength import clamp01

SALIENCE_W_NOVELTY = 0.25
SALIENCE_W_GOALREL = 0.35
SALIENCE_W_SOCIAL  = 0.20
SALIENCE_W_IMPACT  = 0.20

RESERVED_META_KEYS = {
    "id","ts","layer","kind","source","content",
    "embedding_id","embedding_dim","model_hint",
    "salience","novelty","goal_relevance","impact_signal",
    "freq","last_access","strength","summary_of","cross_refs",
    "pinned","expiry_hint","meta","_vec"
}

@dataclass
class IngestResult:
    item: Optional[MemoryItem]
    vector: Optional[np.ndarray]
    salience: float
    kept: bool

def sanitize_meta(meta: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not isinstance(meta, dict):
        return {}
    out = dict(meta)
    for k in list(out.keys()):
        if k in RESERVED_META_KEYS:
            del out[k]
    return out

def social_relevance(ev_kind: str) -> float:
    if not isinstance(ev_kind, str):
        return 0.0
    if ev_kind.startswith("chat:"):
        return 1.0
    if ev_kind.startswith("goal:"):
        return 0.3
    return 0.0

def compute_salience(*, novelty: float, goal_rel: float, social_rel: float, impact: float) -> float:
    s = (
        SALIENCE_W_NOVELTY * float(novelty) +
        SALIENCE_W_GOALREL * float(goal_rel) +
        SALIENCE_W_SOCIAL  * float(social_rel) +
        SALIENCE_W_IMPACT  * float(impact)
    )
    return clamp01(s)

def kind_prior(kind: str) -> float:
    prior = float(MEMCFG.STRENGTH_PRIORS.get(kind, MEMCFG.STRENGTH_PRIORS.get("fact", 0.10)))
    return clamp01(prior)

def decide_keep(*, salience: float, meta: Optional[Dict[str, object]] = None, capture_all: Optional[bool] = None, salience_keep: Optional[float] = None) -> bool:
    cap_all = MEMCFG.CAPTURE_ALL if capture_all is None else bool(capture_all)
    if cap_all:
        return True
    if bool((meta or {}).get("explicit_remember")):
        return True
    thr = float(MEMCFG.SALIENCE_KEEP if salience_keep is None else salience_keep)
    return float(salience) >= thr

def build_item_from_event(
    ev: Event,
    recent_vecs: List[np.ndarray] | Tuple[np.ndarray, ...] | None = None,
    *,
    capture_all: Optional[bool] = None,
    salience_keep: Optional[float] = None,
) -> IngestResult:
    meta = sanitize_meta(getattr(ev, "meta", None))
    kind = (getattr(ev, "meta", {}) or {}).get("kind") if isinstance(getattr(ev, "meta", {}), dict) else None
    kind = (kind or "fact").strip()
    content = (ev.content or "").strip()

    # precomputed joint vector (e.g., from media.ingest_image)
    pre_vec = None
    if isinstance(ev.meta, dict) and "_vec" in ev.meta:
        try:
            arr = np.asarray(ev.meta["_vec"], dtype=np.float32).reshape(-1)
            nrm = np.linalg.norm(arr)
            pre_vec = (arr / nrm) if nrm > 0 else arr
        except Exception:
            pre_vec = None

    vec = pre_vec if pre_vec is not None else get_embedding(content)

    n = novelty_score(vec, recent_vecs or []) if (recent_vecs and len(recent_vecs) > 0) else 1.0
    g = float((getattr(ev, "meta", {}) or {}).get("goal_rel", 0.0))
    s = social_relevance(ev.kind)
    i = float((getattr(ev, "meta", {}) or {}).get("impact", 0.0))
    sal = compute_salience(novelty=n, goal_rel=g, social_rel=s, impact=i)

    keep = decide_keep(salience=sal, meta=getattr(ev, "meta", None), capture_all=capture_all, salience_keep=salience_keep)
    if not keep:
        return IngestResult(item=None, vector=None, salience=float(sal), kept=False)

    it = MemoryItem.new(kind=kind, source=ev.kind, content=content, layer="working", **meta)
    it.embedding_id = f"vec_{it.id}"
    it.embedding_dim = int(len(vec))
    it.model_hint = model_hint()
    it.salience = float(sal)
    it.novelty = float(n)
    it.goal_relevance = float(g)
    it.impact_signal = float(i)
    it.freq = 0
    it.strength = kind_prior(kind)

    return IngestResult(item=it, vector=vec, salience=float(sal), kept=True)
