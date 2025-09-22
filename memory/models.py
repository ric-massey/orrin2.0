# memory/models.py
# Core dataclasses for Orrin2.0 memory: Event, MemoryItem, LexiconSense.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid

__all__ = ["Event", "MemoryItem", "LexiconSense", "now_iso"]

ISO = "%Y-%m-%dT%H:%M:%SZ"

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)

def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

# -------------------------
# Event: raw input into memory
# -------------------------
@dataclass
class Event:
    """
    A single occurrence to ingest into memory.
    - kind: namespaced source, e.g., "chat:user", "chat:assistant", "goal:update", "loop:think", "media:image"
    - content: primary text payload (caption for media)
    - meta: free-form metadata dict (will be sanitized by ingest/daemon)
    """
    kind: str
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, *, kind: str, content: str, meta: Optional[Dict[str, Any]] = None) -> "Event":
        return cls(kind=kind, content=content, meta=dict(meta or {}))

# -------------------------
# MemoryItem: stored unit
# -------------------------
@dataclass
class MemoryItem:
    """
    A retrievable memory node. Most fields are optional and get filled by the daemon.
    """
    id: str
    ts: str

    # Placement
    layer: str                 # "working" | "long" | "summary"
    kind: str                  # "fact" | "definition" | "rule" | "goal" | "decision" | "procedure" | "media" | "introspection" | "summary" | ...

    # Provenance & content
    source: str                # event kind, e.g., "chat:user"
    content: str               # short, single-point text (caption for media)

    # Embedding info
    embedding_id: Optional[str] = None
    embedding_dim: Optional[int] = None
    model_hint: Optional[str] = None

    # Salience & signals
    salience: float = 0.0
    novelty: float = 0.0
    goal_relevance: float = 0.0
    impact_signal: float = 0.0

    # Retrieval reinforcement
    freq: int = 0
    last_access: Optional[str] = None
    strength: float = 0.0

    # Linking
    summary_of: List[str] = field(default_factory=list)
    cross_refs: List[str] = field(default_factory=list)

    # Lifecycle / policy
    pinned: Optional[bool] = None
    expiry_hint: Optional[str] = None

    # Extra metadata (shallow; heavy blobs live in external refs/URIs)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, *, kind: str, source: str, content: str, layer: str = "working", **meta) -> "MemoryItem":
        """
        Convenience factory used across the codebase.
        Reserved fields (id/ts/etc.) are set here, not taken from **meta.
        Also strips reserved keys from meta so callers can pass duplicates safely.
        """
        reserved = {
            "id", "ts", "layer", "kind", "source", "content",
            "embedding_id", "embedding_dim", "model_hint",
            "salience", "novelty", "goal_relevance", "impact_signal",
            "freq", "last_access", "strength",
            "summary_of", "cross_refs",
            "pinned", "expiry_hint", "meta",
        }
        clean_meta = {k: v for k, v in (meta or {}).items() if k not in reserved}
        return cls(
            id=_gen_id("mem"),
            ts=now_iso(),
            layer=layer,
            kind=kind,
            source=source,
            content=(content or "").strip(),
            meta=clean_meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "layer": self.layer,
            "kind": self.kind,
            "source": self.source,
            "content": self.content,
            "embedding_id": self.embedding_id,
            "embedding_dim": self.embedding_dim,
            "model_hint": self.model_hint,
            "salience": self.salience,
            "novelty": self.novelty,
            "goal_relevance": self.goal_relevance,
            "impact_signal": self.impact_signal,
            "freq": self.freq,
            "last_access": self.last_access,
            "strength": self.strength,
            "summary_of": list(self.summary_of or []),
            "cross_refs": list(self.cross_refs or []),
            "pinned": self.pinned,
            "expiry_hint": self.expiry_hint,
            "meta": dict(self.meta or {}),
        }

# -------------------------
# LexiconSense: durable definition of a term
# -------------------------
@dataclass
class LexiconSense:
    """
    A single sense (meaning) for a term in Orrin's lexicon.
    - id: internal unique id (used by the store)
    - sense_id: human-stable id (slug+hash) chosen by the lexicon API
    - term: the headword (canonical form)
    - definition: concise description of the sense
    - aliases: alternative surface forms
    - examples: short usage examples (bounded)
    - sources: where this sense came from (chat:user, system, docs...)
    - freq: how often this sense was selected/used
    - pinned: whether to keep this sense durable (GC-exempt)
    - meta: small notes (e.g., confidence, notes), not for large blobs
    """
    id: str
    sense_id: str
    term: str
    definition: str
    aliases: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    model_hint: Optional[str] = None
    freq: int = 0
    pinned: Optional[bool] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        term: str,
        sense_id: str,
        definition: str,
        source: str = "chat:user",
        aliases: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        # optional to support reconstruction/cloning from to_dict()
        id: Optional[str] = None,
        model_hint: Optional[str] = None,
        freq: Optional[int] = None,
        pinned: Optional[bool] = None,
        sources: Optional[List[str]] = None,
        **_ignored,  # swallow extra keys coming from helpers/tests
    ) -> "LexiconSense":
        return cls(
            id=id or _gen_id("lex"),
            sense_id=str(sense_id),
            term=(term or "").strip(),
            definition=(definition or "").strip(),
            aliases=list(aliases or []),
            examples=list(examples or []),
            sources=list(sources if sources is not None else ([source] if source else [])),
            model_hint=model_hint,
            freq=int(freq) if freq is not None else 0,
            pinned=pinned,
            meta=dict(meta or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sense_id": self.sense_id,
            "term": self.term,
            "definition": self.definition,
            "aliases": list(self.aliases or []),
            "examples": list(self.examples or []),
            "sources": list(self.sources or []),
            "model_hint": self.model_hint,
            "freq": self.freq,
            "pinned": self.pinned,
            "meta": dict(self.meta or {}),
        }
