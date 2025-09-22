# tests/memory_tests/config_test.py
import importlib
import os
import time
from types import ModuleType

import pytest


def _reload_with_env(monkeypatch, **env):
    """
    Helper: reload memory.config with specific env overrides applied.
    Automatically clears ORRIN_MEM_* vars not provided so each reload is clean.
    """
    # Import once so we can inspect defaults for cleanup if needed
    import memory.config as config

    # Clear all ORRIN_MEM_* envs first (clean slate)
    for k in list(os.environ.keys()):
        if k.startswith("ORRIN_MEM_"):
            monkeypatch.delenv(k, raising=False)

    # Apply requested envs
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))

    # Reload module to rebuild MEMCFG with current env
    return importlib.reload(config)


def test_paths_created_on_import():
    import memory.config as config
    # Directories should exist because _ensure_dirs() runs at import
    assert config.DATA_DIR.is_dir()
    assert config.MEMORY_DIR.is_dir()
    assert config.MEDIA_DIR.is_dir()
    assert config.WAL_DIR.is_dir()

    # WAL files are paths (may or may not exist yet) but their parent must exist
    assert config.MEMCFG.WAL_EVENTS_PATH.parent == config.WAL_DIR
    assert config.MEMCFG.WAL_ITEMS_PATH.parent == config.WAL_DIR


def test_tau_for_layer_values():
    import memory.config as config
    cfg = config.MEMCFG
    assert cfg.tau_for_layer("working") == cfg.TAU_HOURS_WORKING
    assert cfg.tau_for_layer("summary") == cfg.TAU_HOURS_SUMMARY
    # anything else defaults to "long"
    assert cfg.tau_for_layer("long") == cfg.TAU_HOURS_LONG
    assert cfg.tau_for_layer("something-else") == cfg.TAU_HOURS_LONG


def test_defaults_have_expected_keys_and_types():
    import memory.config as config
    cfg = config.MEMCFG

    # Core
    assert isinstance(cfg.CAPTURE_ALL, bool)
    assert isinstance(cfg.SALIENCE_KEEP, float)
    assert isinstance(cfg.TICK_HZ, float)
    assert isinstance(cfg.WORKING_CAP, int)

    # Retrieval
    assert isinstance(cfg.RETRIEVE_ALPHA, float)
    assert 0.0 <= cfg.RETRIEVE_ALPHA <= 1.0
    assert isinstance(cfg.RETRIEVE_TOP_K, int)
    assert cfg.RETRIEVE_TOP_K > 0

    # Strength priors includes some key kinds
    for k in {"definition", "goal", "rule", "procedure", "decision", "fact", "introspection", "media", "summary"}:
        assert k in cfg.STRENGTH_PRIORS

    # Paths
    assert config.ROOT_DIR.is_dir()
    assert config.DATA_DIR.is_dir()

    # Start ts
    assert isinstance(cfg.START_TS, float) and cfg.START_TS > 0.0


def test_env_overrides_core(monkeypatch):
    mod = _reload_with_env(
        monkeypatch,
        ORRIN_MEM_CAPTURE_ALL="false",
        ORRIN_MEM_SALIENCE_KEEP="0.42",
        ORRIN_MEM_TICK_HZ="7.5",
        ORRIN_MEM_WORKING_CAP="1234",
    )
    cfg = mod.MEMCFG
    assert cfg.CAPTURE_ALL is False
    assert cfg.SALIENCE_KEEP == pytest.approx(0.42)
    assert cfg.TICK_HZ == pytest.approx(7.5)
    assert cfg.WORKING_CAP == 1234


def test_env_overrides_retrieval_decay(monkeypatch):
    mod = _reload_with_env(
        monkeypatch,
        ORRIN_MEM_RETRIEVE_ALPHA="0.25",
        ORRIN_MEM_RETRIEVE_TOP_K="17",
        ORRIN_MEM_TAU_WORKING="10.0",
        ORRIN_MEM_TAU_LONG="20.0",
        ORRIN_MEM_TAU_SUMMARY="30.0",
    )
    cfg = mod.MEMCFG
    assert cfg.RETRIEVE_ALPHA == pytest.approx(0.25)
    assert cfg.RETRIEVE_TOP_K == 17
    assert cfg.TAU_HOURS_WORKING == pytest.approx(10.0)
    assert cfg.TAU_HOURS_LONG == pytest.approx(20.0)
    assert cfg.TAU_HOURS_SUMMARY == pytest.approx(30.0)


def test_env_overrides_compaction(monkeypatch):
    mod = _reload_with_env(
        monkeypatch,
        ORRIN_MEM_COMPACT_MIN="9",
        ORRIN_MEM_SIM_THRESHOLD="0.77",
        ORRIN_MEM_DUP_SIM="0.991",
        ORRIN_MEM_MIN_CLUSTER="4",
        ORRIN_MEM_MAX_BULLETS="9",
        ORRIN_MEM_BULLET_CHARS="140",
    )
    cfg = mod.MEMCFG
    assert cfg.COMPACT_INTERVAL_MIN == 9
    assert cfg.SIM_THRESHOLD == pytest.approx(0.77)
    assert cfg.DUPLICATE_SIM == pytest.approx(0.991)
    assert cfg.MIN_CLUSTER_SIZE == 4
    assert cfg.MAX_SUMMARY_BULLETS == 9
    assert cfg.SUMMARY_BULLET_CHARS == 140


def test_env_overrides_gc_lexicon(monkeypatch):
    mod = _reload_with_env(
        monkeypatch,
        ORRIN_MEM_GC_STRENGTH="0.33",
        ORRIN_MEM_GC_MIN_AGE_DAYS="45",
        ORRIN_MEM_LEX_PIN="false",
        ORRIN_MEM_LEX_UPDATE_THR="0.9",
        ORRIN_MEM_LEX_CTX_FLOOR="0.8",
    )
    cfg = mod.MEMCFG
    assert cfg.GC_STRENGTH_FLOOR == pytest.approx(0.33)
    assert cfg.GC_MIN_AGE_DAYS == 45
    assert cfg.LEXICON_DEFAULT_PIN is False
    assert cfg.LEXICON_UPDATE_THRESHOLD == pytest.approx(0.9)
    assert cfg.LEXICON_CONTEXT_MATCH_FLOOR == pytest.approx(0.8)


def test_env_overrides_embed_store_media_metrics(monkeypatch):
    mod = _reload_with_env(
        monkeypatch,
        ORRIN_MEM_TEXT_EMBED_MODEL="local-awesome-embed",
        ORRIN_MEM_HASH_DIM="384",
        ORRIN_MEM_STORE="faiss",
        ORRIN_MEM_MEDIA_CAPTION="0",   # -> False
        ORRIN_MEM_MEDIA_OCR="no",      # -> False
        ORRIN_MEM_MEDIA_PHASH="off",   # -> False
        ORRIN_MEM_MEDIA_THUMB="512",
        ORRIN_MEM_METRICS="false",
        ORRIN_MEM_HEALTH_INDEX_LAG="22222",
        ORRIN_MEM_HEALTH_COMPACT_MIN="99",
        ORRIN_MEM_HEALTH_FLUSH_FAIL="7",
    )
    cfg = mod.MEMCFG
    assert cfg.TEXT_EMBED_MODEL == "local-awesome-embed"
    assert cfg.HASH_FALLBACK_DIM == 384
    assert cfg.STORE_BACKEND == "faiss"
    assert cfg.MEDIA_ENABLE_CAPTION is False
    assert cfg.MEDIA_ENABLE_OCR is False
    assert cfg.MEDIA_ENABLE_PHASH is False
    assert cfg.MEDIA_THUMB_SIZE == 512
    assert cfg.METRICS_ENABLED is False
    assert cfg.HEALTH_INDEX_LAG_SOFT == 22222
    assert cfg.HEALTH_COMPACTION_STALLED_MIN == 99
    assert cfg.HEALTH_FLUSH_FAILURES_SOFT == 7


def test_invalid_env_values_fall_back_to_defaults(monkeypatch):
    # Load defaults first
    import memory.config as config_default
    default_cfg = config_default.MEMCFG

    # Now set invalid values; _to_int/_to_float should fall back to defaults
    mod = _reload_with_env(
        monkeypatch,
        ORRIN_MEM_TICK_HZ="not-a-float",
        ORRIN_MEM_WORKING_CAP="not-an-int",
        ORRIN_MEM_RETRIEVE_TOP_K="NaN",
        ORRIN_MEM_GC_MIN_AGE_DAYS="",
        ORRIN_MEM_HASH_DIM="[]",
    )
    cfg = mod.MEMCFG
    assert cfg.TICK_HZ == default_cfg.TICK_HZ
    assert cfg.WORKING_CAP == default_cfg.WORKING_CAP
    assert cfg.RETRIEVE_TOP_K == default_cfg.RETRIEVE_TOP_K
    assert cfg.GC_MIN_AGE_DAYS == default_cfg.GC_MIN_AGE_DAYS
    assert cfg.HASH_FALLBACK_DIM == default_cfg.HASH_FALLBACK_DIM


def test_boolean_parser_accepts_common_truthy_and_falsey(monkeypatch):
    # truthy variants
    for val in ("1", "true", "TRUE", "Yes", "on", "Y", "t"):
        mod = _reload_with_env(monkeypatch, ORRIN_MEM_CAPTURE_ALL=val)
        assert mod.MEMCFG.CAPTURE_ALL is True

    # falsey variants (anything not in truthy set becomes False)
    for val in ("0", "false", "no", "off", "n", "f", "random"):
        mod = _reload_with_env(monkeypatch, ORRIN_MEM_CAPTURE_ALL=val)
        assert mod.MEMCFG.CAPTURE_ALL is False


def test_start_ts_is_set_each_reload(monkeypatch):
    mod1 = _reload_with_env(monkeypatch)
    ts1 = mod1.MEMCFG.START_TS
    time.sleep(0.01)
    mod2 = _reload_with_env(monkeypatch)
    ts2 = mod2.MEMCFG.START_TS
    assert ts2 >= ts1
