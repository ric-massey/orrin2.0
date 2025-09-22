# tests/memory_tests/embedder_test.py
import importlib
import sys
import types
import io
from pathlib import Path

import numpy as np
import pytest


# ---------------------------
# Helpers
# ---------------------------

def _reload_embedder(monkeypatch, *, fake_st=None, hash_dim=None, reset=True):
    """
    Reload memory.embedder with optional:
      - fake_st: a fake 'sentence_transformers' module injected into sys.modules
      - hash_dim: override MEMCFG.HASH_FALLBACK_DIM before reload
    """
    # Ensure clean environment
    if reset:
        for name in list(sys.modules.keys()):
            if name.startswith("memory.embedder"):
                sys.modules.pop(name, None)

    import memory.config as config
    if hash_dim is not None:
        monkeypatch.setattr(config.MEMCFG, "HASH_FALLBACK_DIM", int(hash_dim), raising=False)

    if fake_st is not None:
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    else:
        # Make sure import fails by removing if present
        sys.modules.pop("sentence_transformers", None)

    import memory.embedder as embedder
    return importlib.reload(embedder)


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


# ---------------------------
# Text embedding — fallback path
# ---------------------------

def test_text_fallback_hash_dims_norm_and_determinism(monkeypatch):
    # Force fallback by ensuring sentence_transformers is not importable
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=257)

    # Single string
    v1 = mod.get_text_embedding("hello world")
    v2 = mod.get_text_embedding("hello world")
    v3 = mod.get_text_embedding("different text")

    assert isinstance(v1, np.ndarray)
    assert v1.dtype == np.float32
    assert v1.shape == (mod.text_dim(),)
    assert mod.text_dim() == 257  # from our override
    # normalized
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)
    # deterministic same input
    assert np.allclose(v1, v2)
    # different text -> very likely different vector
    assert not np.allclose(v1, v3)
    # hint reflects fallback
    assert mod.text_model_hint().startswith("hash-")


def test_text_list_input_returns_list_of_vectors(monkeypatch):
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=64)
    vecs = mod.get_text_embedding(["a", "b", "c"])
    assert isinstance(vecs, list) and len(vecs) == 3
    for v in vecs:
        assert isinstance(v, np.ndarray)
        assert v.shape == (mod.text_dim(),)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_get_embedding_alias_matches_get_text_embedding(monkeypatch):
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=128)
    a = mod.get_text_embedding("alias check")
    b = mod.get_embedding("alias check")
    assert np.allclose(a, b)


# ---------------------------
# Text embedding — fake sentence-transformers path
# ---------------------------

def test_text_with_fake_sentence_transformers_success(monkeypatch):
    # Build a fake sentence_transformers module
    class DummyST:
        def __init__(self, name):
            # accept any name
            self.name = name
        def get_sentence_embedding_dimension(self):
            return 3
        def encode(self, arr, normalize_embeddings=True, show_progress_bar=False, batch_size=32):
            # map strings to simple 3D vectors
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

    mod = _reload_embedder(monkeypatch, fake_st=fake_st, hash_dim=77)

    v = mod.get_text_embedding("contains x")
    assert isinstance(v, np.ndarray) and v.shape == (3,)
    assert np.allclose(v, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    # hint should be the model name requested in _lazy_init_text
    assert isinstance(mod.text_model_hint(), str)
    assert mod.text_dim() == 3


def test_text_fake_st_runtime_failure_falls_back_to_hash(monkeypatch):
    class DummyFailST:
        def __init__(self, name): pass
        def get_sentence_embedding_dimension(self): return 5
        def encode(self, *a, **k):
            raise RuntimeError("simulated encode failure")

    fake_st = types.ModuleType("sentence_transformers")
    setattr(fake_st, "SentenceTransformer", DummyFailST)

    mod = _reload_embedder(monkeypatch, fake_st=fake_st, hash_dim=99)

    # encode will fail -> code should fall back to hash
    v = mod.get_text_embedding("fallback please")
    assert isinstance(v, np.ndarray)
    assert v.shape == (mod.text_dim(),)  # text_dim is still 5 if model set? No: fallback uses cached hash dim from config
    # In our implementation, _text_dim remains whatever _lazy_init_text set;
    # but encode failure path uses hash with that _text_dim. Ensure normalized.
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


# ---------------------------
# Image embedding — fallback path
# ---------------------------

def test_image_fallback_bytes_and_path_have_same_result(monkeypatch, tmp_path: Path):
    # Ensure image path uses fallback (hash) by reloading and keeping no CLIP/open_clip
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=128)

    raw = b"\x89PNG\r\n\x1a\ncontent-of-image"
    v1 = mod.get_image_embedding(raw)
    assert isinstance(v1, np.ndarray)
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)
    dim = mod.image_dim()
    assert isinstance(dim, int) and dim >= 256  # image_dim uses max(256, hash_dim)
    assert v1.shape == (dim,)

    # Write to a temp file and ensure hashing the file bytes gives the same vector
    p = tmp_path / "img.bin"
    p.write_bytes(raw)
    v2 = mod.get_image_embedding(str(p))
    assert np.allclose(v1, v2)

    # Different bytes should produce a different vector (high probability)
    v3 = mod.get_image_embedding(b"different-image-content")
    assert not np.allclose(v1, v3)

    # Hint reflects hash fallback
    assert mod.image_model_hint().startswith("hash-img-")


def test_image_fallback_respects_hash_dim_floor(monkeypatch):
    # Set text hash dim to a small number; image fallback should floor to >= 256
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=64)
    d = mod.image_dim()
    assert d >= 256


# ---------------------------
# Misc / unified hints
# ---------------------------

def test_model_hint_returns_text_hint(monkeypatch):
    mod = _reload_embedder(monkeypatch, fake_st=None, hash_dim=200)
    assert mod.model_hint() == mod.text_model_hint()
