# tests/memory_tests/embedder_test.py
import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest


# ---------------------------
# Debug helpers
# ---------------------------

ENV_KEYS_OF_INTEREST = [
    "MEMORY_IMG_BACKEND",
    "MEMORY_IMG_DISABLE_HF",
    "ORRIN_IMAGE_EMBED",
    "MEMORY_IMG_FORCE_HASH",
    "PYTEST_FORCE_HASH_EMBEDDING",
    "MEMORY_IMG_HASH_DIM",
]

def _debug_snapshot(mod=None, note=""):
    lines = []
    lines.append("\n========== DEBUG SNAPSHOT ==========")
    if note:
        lines.append(f"note: {note}")
    # env
    lines.append("---- env ----")
    for k in ENV_KEYS_OF_INTEREST:
        v = os.getenv(k)
        if v is not None:
            lines.append(f"{k}={v!r}")
    # sys.modules presence
    lines.append("---- sys.modules (presence) ----")
    for name in ("transformers", "open_clip", "sentence_transformers", "PIL.Image"):
        lines.append(f"{name}: {'present' if name in sys.modules else 'absent'}")

    # module-level state
    if mod is not None:
        try:
            text_hint = getattr(mod, "text_model_hint", lambda: "<no text_model_hint>")()
        except Exception as e:
            text_hint = f"<error: {e}>"
        try:
            img_hint = getattr(mod, "image_model_hint", lambda: "<no image_model_hint>")()
        except Exception as e:
            img_hint = f"<error: {e}>"
        try:
            tdim = getattr(mod, "text_dim", lambda: None)()
        except Exception as e:
            tdim = f"<error: {e}>"
        try:
            idim = getattr(mod, "image_dim", lambda: None)()
        except Exception as e:
            idim = f"<error: {e}>"

        lines.append("---- module hints ----")
        lines.append(f"text_model_hint(): {text_hint}")
        lines.append(f"image_model_hint(): {img_hint}")
        lines.append(f"text_dim(): {tdim}")
        lines.append(f"image_dim(): {idim}")

        # Try to surface private state if available (best-effort)
        for attr in ("_image_model", "_image_processor", "_image_dim", "_image_hint"):
            val = getattr(mod, attr, "<missing>")
            # just show type/name to avoid giant dumps
            if attr in ("_image_model", "_image_processor") and val not in (None, "<missing>"):
                val = f"<{type(val).__name__}>"
            lines.append(f"{attr}: {val}")

    lines.append("====================================\n")
    return "\n".join(lines)


def _assert_with_debug(cond, mod, msg):
    if not cond:
        pytest.fail(msg + _debug_snapshot(mod))


# ---------------------------
# Reload helper
# ---------------------------

def _reload_embedder(monkeypatch, *, fake_st=None, hash_dim=None, reset=True, force_hash_env=False):
    """
    Reload memory.embedder with optional:
      - fake_st: a fake 'sentence_transformers' module injected into sys.modules
      - hash_dim: override MEMCFG.HASH_FALLBACK_DIM before reload
      - reset:    drop existing memory.embedder modules first
      - force_hash_env: set env to force hash image backend
    """
    # Clean out prior loads of memory.embedder
    if reset:
        for name in list(sys.modules.keys()):
            if name.startswith("memory.embedder"):
                sys.modules.pop(name, None)

    # Optionally force image hash backend through env (several keys supported by code)
    if force_hash_env:
        monkeypatch.setenv("MEMORY_IMG_BACKEND", "hash")
        monkeypatch.setenv("PYTEST_FORCE_HASH_EMBEDDING", "1")

    import memory.config as config
    if hash_dim is not None:
        monkeypatch.setattr(config.MEMCFG, "HASH_FALLBACK_DIM", int(hash_dim), raising=False)

    # sentence_transformers injection (for text path control)
    if fake_st is not None:
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    else:
        sys.modules.pop("sentence_transformers", None)

    import memory.embedder as embedder
    mod = importlib.reload(embedder)

    # If the module exposes a reset hook for image cache, use it so env takes effect
    reset_hook = getattr(mod, "reset_image_backend_cache", None)
    if callable(reset_hook):
        reset_hook()

    return mod


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


# ---------------------------
# Text embedding — fallback path
# ---------------------------

def test_text_fallback_hash_dims_norm_and_determinism(monkeypatch):
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=257, force_hash_env=True)

    v1 = mod.get_text_embedding("hello world")
    v2 = mod.get_text_embedding("hello world")
    v3 = mod.get_text_embedding("different text")

    _assert_with_debug(isinstance(v1, np.ndarray), mod, "v1 must be a numpy array\n")
    _assert_with_debug(v1.dtype == np.float32, mod, "v1 dtype should be float32\n")
    _assert_with_debug(v1.shape == (mod.text_dim(),), mod, "v1 shape mismatch with text_dim()\n")
    _assert_with_debug(mod.text_dim() == 257, mod, "text_dim should reflect overridden hash_dim=257\n")
    _assert_with_debug(np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5), mod, "v1 should be unit-normalized\n")
    _assert_with_debug(np.allclose(v1, v2), mod, "same input must produce same vector\n")
    _assert_with_debug(not np.allclose(v1, v3), mod, "different text should change vector (likely)\n")
    _assert_with_debug(mod.text_model_hint().startswith("hash-"), mod, "text_model_hint should show fallback hash\n")


def test_text_list_input_returns_list_of_vectors(monkeypatch):
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=64, force_hash_env=True)
    vecs = mod.get_text_embedding(["a", "b", "c"])
    _assert_with_debug(isinstance(vecs, list) and len(vecs) == 3, mod, "get_text_embedding(list) should return list of 3\n")
    for v in vecs:
        _assert_with_debug(isinstance(v, np.ndarray), mod, "each vec must be numpy array\n")
        _assert_with_debug(v.shape == (mod.text_dim(),), mod, "each vec shape mismatch with text_dim()\n")
        _assert_with_debug(np.isclose(np.linalg.norm(v), 1.0, atol=1e-5), mod, "each vec must be unit-normalized\n")


def test_get_embedding_alias_matches_get_text_embedding(monkeypatch):
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=128, force_hash_env=True)
    a = mod.get_text_embedding("alias check")
    b = mod.get_embedding("alias check")
    _assert_with_debug(np.allclose(a, b), mod, "get_embedding should alias get_text_embedding\n")


# ---------------------------
# Text embedding — fake sentence-transformers path
# ---------------------------

def test_text_with_fake_sentence_transformers_success(monkeypatch):
    class DummyST:
        def __init__(self, name): self.name = name
        def get_sentence_embedding_dimension(self): return 3
        def encode(self, arr, normalize_embeddings=True, show_progress_bar=False, batch_size=32):
            out = []
            for s in arr:
                s = (s or "").lower()
                if "x" in s:
                    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                elif "y" in s:
                    v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                else:
                    v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                out.append(v / (np.linalg.norm(v) or 1.0) if normalize_embeddings else v)
            return out

    fake_st = types.ModuleType("sentence_transformers")
    setattr(fake_st, "SentenceTransformer", DummyST)

    mod = _reload_embedder(monkeypatch, fake_st=fake_st, hash_dim=77, force_hash_env=True)

    v = mod.get_text_embedding("contains x")
    _assert_with_debug(isinstance(v, np.ndarray) and v.shape == (3,), mod, "fake ST should yield 3-dim vector\n")
    _assert_with_debug(np.allclose(v, np.array([1.0, 0.0, 0.0], dtype=np.float32)), mod, "fake ST mapping incorrect\n")
    _assert_with_debug(isinstance(mod.text_model_hint(), str), mod, "text_model_hint should be a string\n")
    _assert_with_debug(mod.text_dim() == 3, mod, "text_dim should be 3 from fake ST\n")


def test_text_fake_st_runtime_failure_falls_back_to_hash(monkeypatch):
    class DummyFailST:
        def __init__(self, name): pass
        def get_sentence_embedding_dimension(self): return 5
        def encode(self, *a, **k): raise RuntimeError("simulated encode failure")

    fake_st = types.ModuleType("sentence_transformers")
    setattr(fake_st, "SentenceTransformer", DummyFailST)

    mod = _reload_embedder(monkeypatch, fake_st=fake_st, hash_dim=99, force_hash_env=True)

    v = mod.get_text_embedding("fallback please")
    _assert_with_debug(isinstance(v, np.ndarray), mod, "fallback vector must be numpy array\n")
    _assert_with_debug(v.shape == (mod.text_dim(),), mod, "fallback vector shape should match text_dim()\n")
    _assert_with_debug(np.isclose(np.linalg.norm(v), 1.0, atol=1e-5), mod, "fallback vector should be normalized\n")


# ---------------------------
# Image embedding — fallback path
# ---------------------------

def test_image_fallback_bytes_and_path_have_same_result(monkeypatch, tmp_path: Path):
    # Force hash backend regardless of local CLIP/install
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=128, force_hash_env=True)

    raw = b"\x89PNG\r\n\x1a\ncontent-of-image"
    v1 = mod.get_image_embedding(raw)
    _assert_with_debug(isinstance(v1, np.ndarray), mod, "image embedding should be numpy array\n")
    _assert_with_debug(np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5), mod, "image embedding should be normalized\n")

    dim = mod.image_dim()
    _assert_with_debug(isinstance(dim, int) and dim >= 256, mod, "image_dim should floor to >=256\n")
    _assert_with_debug(v1.shape == (dim,), mod, "image vector shape should match image_dim()\n")

    p = tmp_path / "img.bin"
    p.write_bytes(raw)
    v2 = mod.get_image_embedding(str(p))
    _assert_with_debug(np.allclose(v1, v2), mod, "bytes vs path embeddings should match for same content\n")

    v3 = mod.get_image_embedding(b"different-image-content")
    _assert_with_debug(not np.allclose(v1, v3), mod, "different content should change embedding\n")

    _assert_with_debug(mod.image_model_hint().startswith("hash-img-"), mod, "image_model_hint should indicate hash fallback\n")


def test_image_fallback_respects_hash_dim_floor(monkeypatch):
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=64, force_hash_env=True)
    d = mod.image_dim()
    _assert_with_debug(d >= 256, mod, "image_dim should be floored to >=256 when hash fallback is used\n")


# ---------------------------
# Misc / unified hints
# ---------------------------

def test_model_hint_returns_text_hint(monkeypatch):
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=200, force_hash_env=True)
    _assert_with_debug(mod.model_hint() == mod.text_model_hint(), mod, "model_hint should equal text_model_hint\n")
