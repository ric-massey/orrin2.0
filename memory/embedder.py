# memory/embedder.py
# Text & image embedding adapters (offline-first). Uses sentence-transformers/CLIP if available, else a deterministic hash fallback.

from __future__ import annotations
from typing import Union, List, Tuple, Optional
from functools import lru_cache
from pathlib import Path
import os
import io
import hashlib
import numpy as np

from .config import MEMCFG

# ------------------------------
# Internal state (lazy init)
# ------------------------------
_text_model = None
_text_dim: Optional[int] = None
_text_hint: Optional[str] = None

_image_model = None            # optional (e.g., CLIP)
_image_processor = None        # optional (e.g., CLIPProcessor/open_clip preprocess)
_image_dim: Optional[int] = None
_image_hint: Optional[str] = None


# ------------------------------
# Helpers
# ------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)

def _hash_vec(seed_bytes: bytes, dim: int) -> np.ndarray:
    # Deterministic random vector from bytes
    h = hashlib.blake2b(seed_bytes, digest_size=32).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=int(dim)).astype(np.float32)
    return _normalize(v)

@lru_cache(maxsize=8192)
def _hash_text_cached(text: str, dim: int) -> np.ndarray:
    return _hash_vec(text.encode("utf-8", errors="ignore"), dim)

def _ensure_bytes_image(image: Union[bytes, bytearray, memoryview, str, Path, "PIL.Image.Image"]) -> bytes:
    # Accept raw bytes, path, or PIL.Image; return bytes deterministically
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if isinstance(image, (str, Path)):
        with open(image, "rb") as f:
            return f.read()
    # PIL path (optional dependency)
    try:
        from PIL import Image  # type: ignore
        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="PNG", optimize=False)  # deterministic enough for hashing
            return buf.getvalue()
    except Exception:
        pass
    # Fallback: best-effort stringification
    return repr(image).encode("utf-8", errors="ignore")


# ------------------------------
# Text embedding (primary)
# ------------------------------
def _lazy_init_text() -> None:
    global _text_model, _text_dim, _text_hint
    if _text_model is not None:
        return
    # Try sentence-transformers locally
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        name = MEMCFG.TEXT_EMBED_MODEL or "bge-small-en-v1.5"
        _text_model = SentenceTransformer(name)
        try:
            _text_dim = int(_text_model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]
        except Exception:
            _text_dim = int(len(_text_model.encode("dim_probe", normalize_embeddings=True)))
        _text_hint = name
        return
    except Exception:
        # Fallback: deterministic hash vectors
        _text_model = None
        _text_dim = int(MEMCFG.HASH_FALLBACK_DIM or 256)
        _text_hint = f"hash-{_text_dim}"


def get_text_embedding(texts: Union[str, List[str]], normalize: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Returns 1D float32 numpy vector(s). If a list is given, returns a list of vectors.
    Normalizes to unit length by default.
    """
    _lazy_init_text()
    single = isinstance(texts, str)
    arr = [texts] if single else list(texts)

    if _text_model is None:
        vecs = [_hash_text_cached(t, int(_text_dim)) for t in arr]
        return vecs[0] if single else vecs

    # sentence-transformers path
    try:
        embs = _text_model.encode(arr, normalize_embeddings=normalize, show_progress_bar=False, batch_size=32)
        vecs = [np.asarray(e, dtype=np.float32) for e in (embs if isinstance(embs, list) else list(embs))]
        if normalize:
            vecs = [_normalize(v) for v in vecs]
        return vecs[0] if single else vecs
    except Exception:
        # If the model fails at runtime, fall back gracefully
        vecs = [_hash_text_cached(t, int(_text_dim)) for t in arr]
        return vecs[0] if single else vecs


# Backwards-compat alias used elsewhere in the codebase
def get_embedding(texts: Union[str, List[str]], normalize: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
    return get_text_embedding(texts, normalize=normalize)


def text_model_hint() -> str:
    _lazy_init_text()
    return str(_text_hint)

def text_dim() -> int:
    _lazy_init_text()
    return int(_text_dim)


# ------------------------------
# Image embedding (overrideable)
# ------------------------------

def _forced_img_backend() -> Optional[str]:
    """
    Returns "hash" or "hf" if explicitly forced via env; otherwise None.

    Supported env flags (any one works):
      - MEMORY_IMG_BACKEND=hash|hf
      - MEMORY_IMG_DISABLE_HF=1  (forces hash)
      - ORRIN_IMAGE_EMBED=hash|hf
      - MEMORY_IMG_FORCE_HASH=1  (forces hash)
      - PYTEST_FORCE_HASH_EMBEDDING=1  (handy in tests)
    """
    # canonical selector
    forced = (os.getenv("MEMORY_IMG_BACKEND") or os.getenv("ORRIN_IMAGE_EMBED") or "").strip().lower()
    if forced in {"hash", "hf"}:
        return forced

    # boolean switches that force hash
    for flag in ("MEMORY_IMG_DISABLE_HF", "MEMORY_IMG_FORCE_HASH", "PYTEST_FORCE_HASH_EMBEDDING"):
        v = (os.getenv(flag) or "").strip().lower()
        if v in {"1", "true", "yes"}:
            return "hash"

    return None

def _hash_img_dim() -> int:
    """
    Hash fallback embedding dimension. Floor to 32 (tests expect a minimum).
    Env override: MEMORY_IMG_HASH_DIM
    """
    try:
        env_dim = os.getenv("MEMORY_IMG_HASH_DIM")
        dim = int(env_dim) if env_dim else int(MEMCFG.HASH_FALLBACK_DIM or 256)
    except Exception:
        dim = int(MEMCFG.HASH_FALLBACK_DIM or 256)
    return max(256, min(dim, 4096))

def reset_image_backend_cache() -> None:
    """
    Testing/helper hook: clears the lazily-initialized image backend so the next call
    re-reads environment variables and re-initializes models.
    """
    global _image_model, _image_processor, _image_dim, _image_hint
    _image_model = None
    _image_processor = None
    _image_dim = None
    _image_hint = None

def _lazy_init_image() -> None:
    """
    Try to initialize a local image encoder unless a forced backend says otherwise.
    Order:
      - forced hash (skip heavy imports)
      - transformers CLIP
      - open_clip
      - fallback hash
    """
    global _image_model, _image_processor, _image_dim, _image_hint
    if _image_model is not None or _image_hint is not None:
        return

    # Respect forced hash before trying heavy imports
    forced = _forced_img_backend()
    if forced == "hash":
        _image_model = None
        _image_processor = None
        _image_dim = _hash_img_dim()
        _image_hint = f"hash-img-{_image_dim}"
        return

    # Try HuggingFace transformers CLIP
    try:
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
        import torch  # type: ignore
        model_id = "openai/clip-vit-base-patch32"
        _image_model = CLIPModel.from_pretrained(model_id)
        _image_processor = CLIPProcessor.from_pretrained(model_id)
        _image_dim = int(_image_model.visual_projection.out_features)  # type: ignore[attr-defined]
        _image_hint = model_id
        return
    except Exception:
        _image_model = None
        _image_processor = None

    # Try open_clip
    try:
        import open_clip  # type: ignore
        model_name, pretrained = "ViT-B-32", "laion2b_s34b_b79k"
        _image_model, _, _image_processor = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        _image_dim = int(getattr(_image_model.visual, "output_dim", 512))
        _image_hint = f"open_clip:{model_name}:{pretrained}"
        return
    except Exception:
        _image_model = None
        _image_processor = None

    # Fallback (hash-based)
    _image_dim = _hash_img_dim()
    _image_hint = f"hash-img-{_image_dim}"


def get_image_embedding(
    image: Union[bytes, bytearray, memoryview, str, Path, "PIL.Image.Image"],
    normalize: bool = True,
) -> np.ndarray:
    """
    Returns a single 1D float32 numpy vector for the image. Accepts bytes, path, or PIL.Image.Image.
    If no local image model is available OR hash is forced, returns a deterministic hash-based embedding.
    """
    _lazy_init_image()

    # Even after init, respect per-call forced override
    if _forced_img_backend() == "hash" or _image_model is None or _image_processor is None:
        img_bytes = _ensure_bytes_image(image)
        return _hash_vec(img_bytes, int(_image_dim))

    # CLIP / open_clip path
    try:
        from PIL import Image  # type: ignore
        if isinstance(image, (bytes, bytearray, memoryview)):
            image = Image.open(io.BytesIO(bytes(image))).convert("RGB")
        elif isinstance(image, (str, Path)):
            image = Image.open(str(image)).convert("RGB")
        # else: assume already a PIL.Image

        # transformers CLIP
        if _image_hint and _image_hint.startswith("openai/clip"):
            import torch  # type: ignore
            inputs = _image_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                feats = _image_model.get_image_features(**inputs)  # type: ignore[attr-defined]
            v = feats[0].detach().cpu().numpy().astype(np.float32)
            return _normalize(v) if normalize else v

        # open_clip
        try:
            import torch  # type: ignore
            image_tensor = _image_processor(image).unsqueeze(0)  # preprocess -> [1, C, H, W]
            with torch.no_grad():
                v = _image_model.encode_image(image_tensor)  # type: ignore[attr-defined]
            v = v[0].detach().cpu().numpy().astype(np.float32)
            return _normalize(v) if normalize else v
        except Exception:
            img_bytes = _ensure_bytes_image(image)
            return _hash_vec(img_bytes, int(_image_dim))
    except Exception:
        img_bytes = _ensure_bytes_image(image)
        return _hash_vec(img_bytes, int(_image_dim))


def image_model_hint() -> str:
    # Reflect forced override at call time
    forced = _forced_img_backend()
    if forced == "hash":
        return f"hash-img-{_hash_img_dim()}"
    # If explicitly forcing HF, we still go through normal init (CLIP/open_clip first)
    _lazy_init_image()
    return str(_image_hint)

def image_dim() -> int:
    _lazy_init_image()
    return int(_image_dim)


# ------------------------------
# Unified hints (optional)
# ------------------------------
def model_hint() -> str:
    """
    For compatibility with existing code that expects a single model hint.
    We prioritize text since the majority of items are text-based.
    """
    return text_model_hint()
