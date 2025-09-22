# memory/config.py
# Central configuration for Orrin2.0 memory: capture-all mode, compaction, retrieval, decay/GC, lexicon, media, paths, and health/metrics.

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import os
import time

# ---------- Paths ----------
ROOT_DIR = Path(__file__).resolve().parent.parent  # .../orrin2.0
DATA_DIR = ROOT_DIR / "data"
MEMORY_DIR = DATA_DIR / "memory"
MEDIA_DIR = DATA_DIR / "media"
WAL_DIR = MEMORY_DIR / "wal"

def _ensure_dirs():
    for p in [DATA_DIR, MEMORY_DIR, MEDIA_DIR, WAL_DIR]:
        p.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
def _to_bool(v: Optional[str], default: bool) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on", "y", "t"}

def _to_int(v: Optional[str], default: int) -> int:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _to_float(v: Optional[str], default: float) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

# ---------- Config Dataclass ----------
@dataclass
class MemoryConfig:
    # Core behavior
    CAPTURE_ALL: bool = True              # write everything into structured memory (working) + WAL
    SALIENCE_KEEP: float = 0.0            # salience threshold (ignored if CAPTURE_ALL=True)
    TICK_HZ: float = 2.0                  # daemon tick frequency
    WORKING_CAP: int = 1000               # working-layer capacity before compaction is encouraged

    # Retrieval
    RETRIEVE_ALPHA: float = 0.7           # blend: alpha*cosine + (1-alpha)*strength
    RETRIEVE_TOP_K: int = 8

    # Decay / Strength time constants (hours)
    TAU_HOURS_WORKING: float = 72.0
    TAU_HOURS_LONG: float = 168.0
    TAU_HOURS_SUMMARY: float = 240.0

    # Strength priors (used when items are first written)
    STRENGTH_PRIORS: Dict[str, float] = field(default_factory=lambda: {
        "definition": 0.40,
        "goal": 0.50,
        "rule": 0.35,
        "procedure": 0.30,
        "decision": 0.30,
        "fact": 0.10,
        "introspection": 0.35,
        "media": 0.15,
        "summary": 0.30,
    })

    # Compaction / Promotion (working -> long)
    COMPACT_INTERVAL_MIN: int = 15
    SIM_THRESHOLD: float = 0.82           # cluster similarity threshold
    DUPLICATE_SIM: float = 0.985          # near-duplicate fold-in threshold
    MIN_CLUSTER_SIZE: int = 3
    MAX_SUMMARY_BULLETS: int = 6
    SUMMARY_BULLET_CHARS: int = 160
    PROMOTION_LAYER: str = "long"

    # Garbage collection (gentle; summaries/pinned/lexicon/rules exempt)
    GC_STRENGTH_FLOOR: float = 0.15
    GC_MIN_AGE_DAYS: int = 30

    # Lexicon (definitions)
    LEXICON_DEFAULT_PIN: bool = True
    LEXICON_UPDATE_THRESHOLD: float = 0.82   # >= update existing sense, else create new
    LEXICON_CONTEXT_MATCH_FLOOR: float = 0.75

    # Embedding settings (adapters may choose to honor these)
    TEXT_EMBED_MODEL: str = "bge-small-en-v1.5"  # hint; embedder has its own fallback
    HASH_FALLBACK_DIM: int = 384                 # if no local model is available

    # Store backend hint (informational; actual wiring happens in main/boot)
    STORE_BACKEND: str = "inmem"   # "inmem" | "lancedb" | "faiss" | "sqlitevec"

    # Media ingestion
    MEDIA_ENABLE_CAPTION: bool = True
    MEDIA_ENABLE_OCR: bool = True
    MEDIA_ENABLE_PHASH: bool = True
    MEDIA_THUMB_SIZE: int = 256  # px, square thumb

    # WAL (write-ahead log) paths
    WAL_EVENTS_PATH: Path = WAL_DIR / "events.jsonl"
    WAL_ITEMS_PATH: Path = WAL_DIR / "items.jsonl"

    # Metrics/health (Reaper integration can read these thresholds)
    METRICS_ENABLED: bool = True
    HEALTH_INDEX_LAG_SOFT: int = 10_000
    HEALTH_COMPACTION_STALLED_MIN: int = 30
    HEALTH_FLUSH_FAILURES_SOFT: int = 3

    # Paths (derived)
    ROOT_DIR: Path = ROOT_DIR
    DATA_DIR: Path = DATA_DIR
    MEMORY_DIR: Path = MEMORY_DIR
    MEDIA_DIR: Path = MEDIA_DIR
    WAL_DIR: Path = WAL_DIR

    # Runtime
    START_TS: float = field(default_factory=time.time)

    # Convenience
    def tau_for_layer(self, layer: str) -> float:
        if layer == "working":
            return self.TAU_HOURS_WORKING
        if layer == "summary":
            return self.TAU_HOURS_SUMMARY
        return self.TAU_HOURS_LONG

# ---------- Build config with env overrides ----------
def _build_from_env() -> MemoryConfig:
    cfg = MemoryConfig()
    # core
    cfg.CAPTURE_ALL = _to_bool(os.getenv("ORRIN_MEM_CAPTURE_ALL"), cfg.CAPTURE_ALL)
    cfg.SALIENCE_KEEP = _to_float(os.getenv("ORRIN_MEM_SALIENCE_KEEP"), cfg.SALIENCE_KEEP)
    cfg.TICK_HZ = _to_float(os.getenv("ORRIN_MEM_TICK_HZ"), cfg.TICK_HZ)
    cfg.WORKING_CAP = _to_int(os.getenv("ORRIN_MEM_WORKING_CAP"), cfg.WORKING_CAP)

    # retrieval
    cfg.RETRIEVE_ALPHA = _to_float(os.getenv("ORRIN_MEM_RETRIEVE_ALPHA"), cfg.RETRIEVE_ALPHA)
    cfg.RETRIEVE_TOP_K = _to_int(os.getenv("ORRIN_MEM_RETRIEVE_TOP_K"), cfg.RETRIEVE_TOP_K)

    # decay
    cfg.TAU_HOURS_WORKING = _to_float(os.getenv("ORRIN_MEM_TAU_WORKING"), cfg.TAU_HOURS_WORKING)
    cfg.TAU_HOURS_LONG = _to_float(os.getenv("ORRIN_MEM_TAU_LONG"), cfg.TAU_HOURS_LONG)
    cfg.TAU_HOURS_SUMMARY = _to_float(os.getenv("ORRIN_MEM_TAU_SUMMARY"), cfg.TAU_HOURS_SUMMARY)

    # compaction
    cfg.COMPACT_INTERVAL_MIN = _to_int(os.getenv("ORRIN_MEM_COMPACT_MIN"), cfg.COMPACT_INTERVAL_MIN)
    cfg.SIM_THRESHOLD = _to_float(os.getenv("ORRIN_MEM_SIM_THRESHOLD"), cfg.SIM_THRESHOLD)
    cfg.DUPLICATE_SIM = _to_float(os.getenv("ORRIN_MEM_DUP_SIM"), cfg.DUPLICATE_SIM)
    cfg.MIN_CLUSTER_SIZE = _to_int(os.getenv("ORRIN_MEM_MIN_CLUSTER"), cfg.MIN_CLUSTER_SIZE)
    cfg.MAX_SUMMARY_BULLETS = _to_int(os.getenv("ORRIN_MEM_MAX_BULLETS"), cfg.MAX_SUMMARY_BULLETS)
    cfg.SUMMARY_BULLET_CHARS = _to_int(os.getenv("ORRIN_MEM_BULLET_CHARS"), cfg.SUMMARY_BULLET_CHARS)

    # gc
    cfg.GC_STRENGTH_FLOOR = _to_float(os.getenv("ORRIN_MEM_GC_STRENGTH"), cfg.GC_STRENGTH_FLOOR)
    cfg.GC_MIN_AGE_DAYS = _to_int(os.getenv("ORRIN_MEM_GC_MIN_AGE_DAYS"), cfg.GC_MIN_AGE_DAYS)

    # lexicon
    cfg.LEXICON_DEFAULT_PIN = _to_bool(os.getenv("ORRIN_MEM_LEX_PIN"), cfg.LEXICON_DEFAULT_PIN)
    cfg.LEXICON_UPDATE_THRESHOLD = _to_float(os.getenv("ORRIN_MEM_LEX_UPDATE_THR"), cfg.LEXICON_UPDATE_THRESHOLD)
    cfg.LEXICON_CONTEXT_MATCH_FLOOR = _to_float(os.getenv("ORRIN_MEM_LEX_CTX_FLOOR"), cfg.LEXICON_CONTEXT_MATCH_FLOOR)

    # embed/store
    cfg.TEXT_EMBED_MODEL = os.getenv("ORRIN_MEM_TEXT_EMBED_MODEL", cfg.TEXT_EMBED_MODEL)
    cfg.HASH_FALLBACK_DIM = _to_int(os.getenv("ORRIN_MEM_HASH_DIM"), cfg.HASH_FALLBACK_DIM)
    cfg.STORE_BACKEND = os.getenv("ORRIN_MEM_STORE", cfg.STORE_BACKEND)

    # media
    cfg.MEDIA_ENABLE_CAPTION = _to_bool(os.getenv("ORRIN_MEM_MEDIA_CAPTION"), cfg.MEDIA_ENABLE_CAPTION)
    cfg.MEDIA_ENABLE_OCR = _to_bool(os.getenv("ORRIN_MEM_MEDIA_OCR"), cfg.MEDIA_ENABLE_OCR)
    cfg.MEDIA_ENABLE_PHASH = _to_bool(os.getenv("ORRIN_MEM_MEDIA_PHASH"), cfg.MEDIA_ENABLE_PHASH)
    cfg.MEDIA_THUMB_SIZE = _to_int(os.getenv("ORRIN_MEM_MEDIA_THUMB"), cfg.MEDIA_THUMB_SIZE)

    # metrics/health
    cfg.METRICS_ENABLED = _to_bool(os.getenv("ORRIN_MEM_METRICS"), cfg.METRICS_ENABLED)
    cfg.HEALTH_INDEX_LAG_SOFT = _to_int(os.getenv("ORRIN_MEM_HEALTH_INDEX_LAG"), cfg.HEALTH_INDEX_LAG_SOFT)
    cfg.HEALTH_COMPACTION_STALLED_MIN = _to_int(os.getenv("ORRIN_MEM_HEALTH_COMPACT_MIN"), cfg.HEALTH_COMPACTION_STALLED_MIN)
    cfg.HEALTH_FLUSH_FAILURES_SOFT = _to_int(os.getenv("ORRIN_MEM_HEALTH_FLUSH_FAIL"), cfg.HEALTH_FLUSH_FAILURES_SOFT)

    return cfg

# Build singleton and ensure directories exist
_ensure_dirs()
MEMCFG = _build_from_env()

# ---------- Quick usage notes ----------
# from memory.config import MEMCFG
# if MEMCFG.CAPTURE_ALL: ...
# tau = MEMCFG.tau_for_layer("working")
# paths: MEMCFG.DATA_DIR / "..."
