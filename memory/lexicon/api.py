# memory/lexicon/api.py
# Local lexicon API: learn_definition, get_definition, add_alias, correct_definition.
from __future__ import annotations
from dataclasses import replace
from typing import Optional, List, Iterable
import re
import unicodedata

import numpy as np

# Import your shared pieces
try:
    from ..models import LexiconSense
    from ..embedder import get_embedding, model_hint
except Exception as e:  # minimal fallback for dev
    raise RuntimeError("api.py expects memory/models.py and memory/embedder.py") from e


# -------------------------
# Config / thresholds
# -------------------------
# Use a high threshold so only *very* similar definitions update the same sense.
CONF_NEW_SENSE = 0.95      # >= update existing sense, else create a new sense
CTX_MATCH_FLOOR = 0.75     # min context match to choose a sense by context
DEFAULT_PIN = True         # lexicon entries default to pinned (durable)
MAX_EXAMPLES = 3           # keep senses small and snappy


# -------------------------
# Small helpers
# -------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "term"

def _unique_sense_id(term: str, seed_text: str, existing_ids: Iterable[str]) -> str:
    base = _slug(term)
    h = abs(hash(seed_text)) % 10_000_000
    candidate = f"{base}:{h}"
    # avoid collisions (rare, but cheap to guard)
    bump = 1
    ids = set(existing_ids)
    while candidate in ids:
        bump += 1
        candidate = f"{base}:{h+bump}"
    return candidate

def _dedup_aliases_preserve_case(aliases: Iterable[str]) -> list[str]:
    """
    Case-insensitive, order-preserving alias dedupe.
    Also strips simple surrounding quotes/backticks.
    """
    out: list[str] = []
    seen: set[str] = set()
    for a in (aliases or []):
        if not a:
            continue
        raw = str(a).strip().strip("`'\"")
        if not raw:
            continue
        key = raw.lower()
        if key not in seen:
            seen.add(key)
            out.append(raw)
    return out


# -------------------------
# Public API (class wrapper)
# -------------------------
class Lexicon:
    """
    Local lexicon API (no external services).
    - learn_definition(term, definition, ...) -> sense_id
    - get_definition(term, context_text=...) -> LexiconSense | None
    - add_alias(term, alias, sense_id=None)
    - correct_definition(sense_id, new_definition, note=None) -> new_sense_id (if forked) or existing id (if update)
    """

    def __init__(self, store):
        """
        store must implement:
          - get_lexicon_by_term(term_or_alias: str) -> list[LexiconSense]
          - upsert_lexicon(senses: list[LexiconSense]) -> None
        """
        self.store = store

    # ------------ Learn / Update ------------
    def learn_definition(
        self,
        term: str,
        definition: Optional[str],
        *,
        context_text: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        source: str = "chat:user",
        confidence: Optional[float] = None,
    ) -> str:
        """
        Create or update a lexicon sense.
        Heuristics:
          - Build an embedding from *definition text* (or context if definition missing).
          - If similarity >= CONF_NEW_SENSE to an existing sense, update that sense (aliases, freq++).
          - Else create a new sense (unique sense_id).
        """
        term = term.strip()
        if not term:
            raise ValueError("term must be non-empty")

        # Case-insensitive alias dedupe, preserve order; trim examples to cap
        aliases = _dedup_aliases_preserve_case(aliases or [])
        examples = (examples or [])[:MAX_EXAMPLES]

        # For similarity, compare *definitions* (more discriminative than "term: def")
        seed_text = (definition or (context_text or term)).strip()
        cand_vec = get_embedding(seed_text)
        senses = self.store.get_lexicon_by_term(term)

        # Find best matching existing sense by definition similarity
        best = None
        best_sim = -1.0
        for s in senses:
            s_vec = get_embedding((s.definition or s.term).strip())
            sim = _cosine(cand_vec, s_vec)
            if sim > best_sim:
                best_sim, best = sim, s

        if best is not None and best_sim >= CONF_NEW_SENSE:
            # Update existing sense (non-destructive)
            updated = replace(best)
            # Merge aliases with case-insensitive dedupe
            updated.aliases = _dedup_aliases_preserve_case([*(updated.aliases or []), *aliases])

            # Prefer provided definition if materially different
            if definition and definition.strip() and definition.strip() != updated.definition.strip():
                updated.definition = definition.strip()

            # Merge examples (bounded)
            merged_examples = [*(updated.examples or []), *examples]
            updated.examples = merged_examples[:MAX_EXAMPLES]

            # Merge sources; bump freq
            updated.sources = list(dict.fromkeys([*(updated.sources or []), source]))
            updated.freq = (updated.freq or 0) + 1

            # Default pinned=True if legacy rows had None
            if updated.pinned is None:
                updated.pinned = DEFAULT_PIN

            self.store.upsert_lexicon([updated])
            return updated.id

        # Create new sense
        sense_id = _unique_sense_id(term, seed_text, (s.sense_id for s in senses))
        new = LexiconSense.new(
            term=term,
            sense_id=sense_id,
            definition=seed_text,
            source=source,
            aliases=aliases,
            examples=examples,
        )
        new.model_hint = model_hint()
        new.freq = 1
        new.pinned = DEFAULT_PIN
        if confidence is not None:
            new.meta["confidence"] = float(confidence)

        # We rely on get_embedding downstream for retrieval; we don't store raw vectors here.
        self.store.upsert_lexicon([new])
        return new.id

    # ------------ Retrieve / Disambiguate ------------
    def get_definition(
        self,
        term: str,
        *,
        context_text: Optional[str] = None,
    ) -> Optional[LexiconSense]:
        """
        If context_text is provided, pick the sense whose definition best fits the context.
        Else return the most-used (freq) sense.
        """
        senses = self.store.get_lexicon_by_term(term)
        if not senses:
            return None
        if not context_text:
            return max(senses, key=lambda s: (s.freq or 0))
        ctx_vec = get_embedding(context_text)
        best, best_sim = None, -1.0
        for s in senses:
            s_vec = get_embedding((s.definition or s.term).strip())
            sim = _cosine(ctx_vec, s_vec)
            if sim > best_sim:
                best, best_sim = s, sim
        return best

    # ------------ Curation ------------
    def add_alias(self, term: str, alias: str, *, sense_id: Optional[str] = None) -> None:
        """
        Add an alias to a term. If sense_id is omitted:
          - if the term has one sense → add to that sense
          - else choose the most frequent sense
        Accepts either the internal `id` or the human-facing `sense_id`.
        """
        alias = alias.strip()
        if not alias:
            return
        senses = self.store.get_lexicon_by_term(term)
        if not senses:
            # Create a placeholder from alias alone (low confidence)
            _ = self.learn_definition(
                term,
                definition=None,
                context_text=alias,
                aliases=[alias],
                source="system:alias",
                confidence=0.2,
            )
            return
        target = None
        if sense_id:
            target = next(
                (s for s in senses if getattr(s, "id", None) == sense_id or getattr(s, "sense_id", None) == sense_id),
                None,
            )
        if target is None:
            target = max(senses, key=lambda s: (s.freq or 0))
        updated = replace(target)
        updated.aliases = _dedup_aliases_preserve_case([*(updated.aliases or []), alias])
        self.store.upsert_lexicon([updated])

    def correct_definition(
        self,
        sense_id: str,
        new_definition: str,
        *,
        note: Optional[str] = None,
        fork_if_large_change: bool = True,
    ) -> str:
        """
        Correct or fork a definition.
        - Small change → update the same sense
        - Large semantic change → fork a new sense_id (preserve history)
        Returns the sense.id that should be used going forward.
        """
        # Find by scanning all senses for this simple in-memory store interface
        candidates: List[LexiconSense] = []
        terms_seen = set()
        for alias in [sense_id.split(":")[0]]:
            senses = self.store.get_lexicon_by_term(alias)
            for s in senses:
                if s.id not in terms_seen:
                    candidates.append(s)
                    terms_seen.add(s.id)
        target = next((s for s in candidates if s.sense_id == sense_id or s.id == sense_id), None)
        if target is None:
            raise ValueError(f"sense {sense_id!r} not found")

        old_vec = get_embedding((target.definition or target.term).strip())
        new_vec = get_embedding(new_definition.strip())
        sim = _cosine(old_vec, new_vec)

        if fork_if_large_change and sim < CONF_NEW_SENSE:
            # Fork a new sense (preserve the old one)
            new_sid = _unique_sense_id(target.term, new_definition, [])
            fork = LexiconSense.new(
                term=target.term,
                sense_id=new_sid,
                definition=new_definition.strip(),
                source="curation:correct",
                aliases=target.aliases,
                examples=target.examples,
            )
            fork.model_hint = model_hint()
            fork.freq = max(1, (target.freq or 0))  # inherit some usage
            if note:
                fork.meta["note"] = note
            self.store.upsert_lexicon([fork])
            return fork.id
        else:
            # In-place correction
            updated = replace(target)
            updated.definition = new_definition.strip()
            if note:
                updated.meta["note"] = note
            self.store.upsert_lexicon([updated])
            return updated.id
