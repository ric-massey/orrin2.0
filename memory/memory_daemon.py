# memory/daemon.py
# Background memory service that runs beside Reaper: captures all events, embeds, stores, extracts definitions, promotes/compacts, and serves retrieval.

from __future__ import annotations
import threading, queue, time
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone
import numpy as np

from .config import MEMCFG
from .models import Event, MemoryItem
from .embedder import get_embedding  # model_hint not needed here anymore
from .novelty import novelty as novelty_score
from .strength import strength_from
from .store.base import VectorStore
from .lexicon.api import Lexicon
from .lexicon.patterns import extract_definitions
from .compaction import should_compact, compact_and_promote
from .ingest import build_item_from_event
from .wal import append_event as wal_append_event, append_items as wal_append_items, DEFAULT_WAL
from .metrics import (
    bump_ingest,
    note_item_upserts,
    note_vector_upserts,
    note_retrieval,
    note_compaction,
)

ISO = "%Y-%m-%dT%H:%M:%SZ"
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


class MemoryDaemon:
    """
    Runs a lightweight background loop:
      - drains an ingest queue (capture-everything mode if enabled)
      - embeds content, computes novelty/salience, writes to working memory
      - auto-learns definitions (lexicon) from “X means/is Y” patterns
      - maintains a small working cache and periodically compacts/promotes to long-term
      - provides retrieval (cosine ⊕ strength) and bumps strength with use
    """

    def __init__(self, store: VectorStore, *, tick_hz: Optional[float] = None):
        self.cfg = MEMCFG
        self.store = store

        self.tick_s = 1.0 / (tick_hz if tick_hz is not None else max(self.cfg.TICK_HZ, 0.1))
        self.ingest_q: "queue.Queue[Event]" = queue.Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Working cache of items we most recently wrote at layer='working'
        self._working_cache: Dict[str, MemoryItem] = {}

        # Initialize compaction timestamp to "now" so health doesn't think it's stalled
        self._last_compact_ts: float = time.time()

        # Track WAL flush failures for health reporting
        self._flush_failures: int = 0

        # Lexicon handler (definitions)
        self.lexicon = Lexicon(self.store)

    # ----------------- Public API -----------------

    def start(self) -> None:
        if self.running:
            return
        self.running = True

        # Safety: if something zeroed the ts before start, fix it
        if not self._last_compact_ts:
            self._last_compact_ts = time.time()

        self.thread = threading.Thread(target=self._loop, name="MemoryDaemon", daemon=True)
        self.thread.start()

    def stop(self, join: bool = True) -> None:
        self.running = False
        if join and self.thread:
            self.thread.join(timeout=2.0)

    def ingest(self, ev: Event) -> None:
        """Submit an event (chat, goal update, planner decision, media, etc.)."""
        self.ingest_q.put(ev)

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        kinds: Optional[List[str]] = None,
        meta_filter: Optional[Dict[str, object]] = None,
    ) -> List[MemoryItem]:
        """
        ANN search by text query, re-ranked by alpha*cosine + (1-alpha)*strength.
        Bumps freq/strength of returned items and emits metrics.
        """
        t0 = time.perf_counter()
        alpha = float(self.cfg.RETRIEVE_ALPHA)
        k = int(top_k or self.cfg.RETRIEVE_TOP_K)

        qv = get_embedding(query)
        hits = self.store.ann_search(qv, top_k=k * 3, kind_filter=kinds, meta_filter=meta_filter)  # overfetch a bit
        if not hits:
            note_retrieval(kinds, hits=0, latency_s=(time.perf_counter() - t0))
            return []

        ids, _ = zip(*hits)
        items = {it.id: it for it in self.store.get_items(list(ids))}

        # Blend cosine with strength
        scored: List[Tuple[str, float]] = []
        for mid, sim in hits:
            it = items.get(mid)
            if not it:
                continue
            s = float(it.strength or 0.0)
            score = alpha * float(sim) + (1.0 - alpha) * s
            scored.append((mid, score))

        scored.sort(key=lambda t: t[1], reverse=True)
        keep_ids = [mid for mid, _ in scored[:k]]
        out = [items[i] for i in keep_ids if i in items]

        # Reinforce on access
        for it in out:
            it.freq = (it.freq or 0) + 1
            it.last_access = now_iso()
            tau = self.cfg.tau_for_layer(it.layer)
            # For simplicity we treat "hours since last" as 0 on immediate access (freq term dominates)
            it.strength = strength_from(it.freq, 0.0, float(it.goal_relevance or 0.0), tau)

        if out:
            self.store.upsert_items(out)

        note_retrieval(kinds, hits=len(out), latency_s=(time.perf_counter() - t0))
        return out

    # ----------------- Loop -----------------

    def _loop(self) -> None:
        while self.running:
            try:
                self._tick()
            except Exception:
                # In production, report to Reaper error bus (make_event_from_key("memory_tick_failure"))
                pass
            time.sleep(self.tick_s)

    def _tick(self) -> None:
        drained = 0
        batch_items: List[MemoryItem] = []
        batch_vecs: Dict[str, np.ndarray] = {}

        # Snapshot of recent vectors for novelty scoring
        recent_vecs = list(self.store.get_recent_vectors(128))

        # ---- Drain events this tick ----
        while not self.ingest_q.empty() and drained < 64:
            ev = self.ingest_q.get()
            drained += 1

            # Metrics + WAL for the raw event
            bump_ingest()
            try:
                wal_append_event(ev)
            except Exception:
                self._flush_failures += 1  # count WAL misses
                pass

            # Build item (sanitize meta, precomputed _vec support, salience/novelty, kind priors)
            res = build_item_from_event(
                ev,
                recent_vecs,
                capture_all=self.cfg.CAPTURE_ALL,
                salience_keep=self.cfg.SALIENCE_KEEP,
            )

            # Always learn definitions from the text, kept or not
            self._maybe_learn_definitions(ev, ev.content or "")

            if not res.kept:
                continue

            # Accumulate batch writes
            item = res.item
            vec = res.vector
            batch_items.append(item)
            batch_vecs[item.embedding_id] = vec
            self._working_cache[item.id] = item

            # Let subsequent events in the same tick “see” this vector for novelty
            recent_vecs.append(vec)

        # ---- Persist batch ----
        if batch_items:
            self.store.upsert_items(batch_items)
            try:
                wal_append_items(batch_items)
            except Exception:
                self._flush_failures += 1
                pass
            note_item_upserts(len(batch_items))

        if batch_vecs:
            self.store.upsert_vectors(batch_vecs)
            note_vector_upserts(len(batch_vecs))

        # ---- Compaction / promotion ----
        if should_compact(
            len(self._working_cache),
            self._last_compact_ts,
            cap=self.cfg.WORKING_CAP,
            interval_minutes=self.cfg.COMPACT_INTERVAL_MIN,
        ):
            stats = compact_and_promote(
                self.store,
                list(self._working_cache.values()),
                sim_threshold=self.cfg.SIM_THRESHOLD,
                duplicate_sim=self.cfg.DUPLICATE_SIM,
                min_cluster_size=self.cfg.MIN_CLUSTER_SIZE,
                max_bullets=self.cfg.MAX_SUMMARY_BULLETS,
                bullet_chars=self.cfg.SUMMARY_BULLET_CHARS,
                promote_layer=self.cfg.PROMOTION_LAYER,
                wal=DEFAULT_WAL,  # WAL promotions/summaries
            )
            self._working_cache.clear()

            # Update timestamp each time compaction runs
            self.mark_compaction_now()
            note_compaction(stats, when_ts=self._last_compact_ts)

    # ----------------- Helpers -----------------

    # Expose clean accessors for health/instrumentation
    @property
    def last_compaction_ts(self) -> float:
        """UNIX seconds (float) of the last successful compaction (or daemon start)."""
        return float(self._last_compact_ts or 0.0)

    @property
    def working_cache_size(self) -> int:
        """How many items are in the working cache (pre-compaction)."""
        return len(self._working_cache)

    @property
    def flush_failures(self) -> int:
        """Number of WAL append failures observed since start."""
        return int(self._flush_failures)

    def mark_compaction_now(self) -> None:
        """Set last compaction timestamp to 'now'."""
        self._last_compact_ts = time.time()

    def time_since_compaction_min(self) -> float:
        """Minutes since last compaction (or start)."""
        return max(0.0, (time.time() - float(self._last_compact_ts or 0.0)) / 60.0)

    def get_health_hints(self) -> Dict[str, float | int]:
        """
        Small struct for dashboards:
          - working_cache_size
          - last_compaction_ts
          - compaction_stalled_min (derived)
          - flush_failures
        """
        return {
            "working_cache_size": self.working_cache_size,
            "last_compaction_ts": self.last_compaction_ts,
            "compaction_stalled_min": self.time_since_compaction_min(),
            "flush_failures": self.flush_failures,
        }

    def _maybe_learn_definitions(self, ev: Event, text: str) -> None:
        """Scan text for definitional patterns and teach the lexicon."""
        try:
            hits = extract_definitions(text)
        except Exception:
            hits = []
        if not hits:
            return
        for term, definition, aliases in hits:
            try:
                self.lexicon.learn_definition(
                    term,
                    definition,
                    context_text=text,
                    aliases=aliases,
                    source=ev.kind,
                )
            except Exception:
                continue
