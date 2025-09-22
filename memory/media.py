# memory/media.py
# Image/media ingest: save file, compute sha256 + pHash, extract EXIF/OCR, make a short caption,
# build embeddings, and return an Event for the daemon.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import hashlib
import io
import os

import numpy as np

from .config import MEMCFG, MEDIA_DIR
from .models import Event, MemoryItem
from .embedder import (
    get_text_embedding as _txt_embed,
    get_image_embedding as _img_embed,
    model_hint as _model_hint,
)

# ----------------------------
# Optional deps — all guarded
# ----------------------------
try:
    from PIL import Image, ExifTags  # type: ignore
    _HAS_PIL = True
except Exception:
    Image = None          # ensure module-level symbol exists (tests monkeypatch this)
    ExifTags = None       # type: ignore
    _HAS_PIL = False

try:
    import pytesseract  # type: ignore
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False


# ----------------------------
# Helpers (I/O & hashing)
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _read_bytes(obj: Union[bytes, bytearray, memoryview, str, Path]) -> bytes:
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj)
    p = Path(obj)
    return p.read_bytes()

def _detect_mime_pil(img: "Image.Image") -> str:
    fmt = (img.format or "").lower() if hasattr(img, "format") else ""
    if fmt in ("jpeg", "jpg"):
        return "image/jpeg"
    if fmt == "png":
        return "image/png"
    if fmt == "webp":
        return "image/webp"
    if fmt == "gif":
        return "image/gif"
    if fmt == "tiff":
        return "image/tiff"
    return "image/octet-stream"

def _safe_exif(img: "Image.Image") -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not ExifTags:
        return out
    try:
        raw = getattr(img, "_getexif", lambda: None)()  # type: ignore[attr-defined]
        if not raw:
            return out
        tagmap = {v: k for k, v in ExifTags.TAGS.items()}  # name->id
        want = [
            "Model", "Make", "DateTime", "FNumber", "FocalLength",
            "LensModel", "ExposureTime", "ISOSpeedRatings", "Orientation",
        ]
        for name in want:
            tag_id = tagmap.get(name)
            if tag_id in raw:
                out[name] = str(raw[tag_id])
    except Exception:
        pass
    return out

# Difference hash (dHash) — robust & simple “pHash-like”
def _phash_dhash(img: "Image.Image", hash_size: int = 8) -> str:
    try:
        # Pillow ≥9.1 has Image.Resampling
        g = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)  # type: ignore[attr-defined]
    except Exception:
        g = img.convert("L").resize((hash_size + 1, hash_size))
    pixels = np.asarray(g, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    # pack bits to hex
    value = 0
    for b in bits:
        value = (value << 1) | int(bool(b))
    width = (hash_size * hash_size) // 4  # hex digits
    return f"{value:0{width}x}"

def _make_thumb(img: "Image.Image", size: int = 256) -> "Image.Image":
    im = img.copy()
    im.thumbnail((size, size))
    return im

def _auto_caption(img: Optional["Image.Image"], tags: Optional[List[str]]) -> str:
    if img is None:
        return "Image (unknown size)"
    w, h = img.size
    tag_str = f" — tags: {', '.join(tags)}" if tags else ""
    return f"Image {w}×{h}{tag_str}".strip()

def _ocr_text(img: Optional["Image.Image"]) -> str:
    if not MEMCFG.MEDIA_ENABLE_OCR or not _HAS_TESS or img is None:
        return ""
    try:
        txt = pytesseract.image_to_string(img)
        return " ".join(txt.split())
    except Exception:
        return ""


# ----------------------------
# Result object
# ----------------------------
@dataclass
class MediaIngestResult:
    event: Event                        # ready to feed `MemoryDaemon.ingest()`
    vector: np.ndarray                  # joint embedding (image ⊕ caption)
    sha256: str
    phash: Optional[str]
    ref_uri: str
    thumb_uri: Optional[str]
    width: Optional[int]
    height: Optional[int]
    mime: Optional[str]
    ocr_text: Optional[str]


# ----------------------------
# Main API
# ----------------------------
def ingest_image(
    image: Union[bytes, bytearray, memoryview, str, Path],
    *,
    tags: Optional[List[str]] = None,
    caption: Optional[str] = None,
    explicit_remember: bool = True,
    save_files: bool = True,
    source: str = "media:image",
) -> MediaIngestResult:
    """
    Ingest an image and return an Event ready for the daemon.
    - Saves file (and a thumbnail) to MEMCFG.MEDIA_DIR
    - Computes sha256 + pHash (dHash)
    - Extracts light EXIF (safe subset)
    - Runs OCR if enabled & pytesseract is present
    - Builds a caption and a joint embedding (image ⊕ caption)
    """
    _ensure_dir(MEDIA_DIR)

    # Read bytes first so sha256 is stable for identical raw files
    b = _read_bytes(image)
    sha = _sha256_bytes(b)
    ref_path = MEDIA_DIR / f"{sha}.bin"
    thumb_path = MEDIA_DIR / f"{sha}_thumb.jpg"
    mime = "image/octet-stream"
    w = h = None
    exif_safe: Dict[str, str] = {}
    phash = None
    pil_img: Optional["Image.Image"] = None

    # Try to load via PIL
    if _HAS_PIL:
        try:
            pil_img = Image.open(io.BytesIO(b)).convert("RGB")
            w, h = pil_img.size
            mime = _detect_mime_pil(pil_img)
            exif_safe = _safe_exif(pil_img)
            phash = _phash_dhash(pil_img)
        except Exception:
            pil_img = None

    # Save original bytes & thumbnail (if we could decode)
    if save_files:
        try:
            ref_path.write_bytes(b)
        except Exception:
            pass
        if pil_img is not None:
            try:
                thumb = _make_thumb(pil_img, size=int(MEMCFG.MEDIA_THUMB_SIZE or 256))
                thumb.save(thumb_path, format="JPEG", quality=85)
            except Exception:
                # could not create/save thumb
                pass

    # Caption / OCR
    ocr = _ocr_text(pil_img)
    auto_cap = caption or _auto_caption(pil_img, tags)
    content = auto_cap if not ocr else f"{auto_cap}. OCR: {ocr[:160]}"

    # Embeddings: combine image + text (average then renormalize)
    try:
        ivec = _img_embed(pil_img if pil_img is not None else b)
    except Exception:
        ivec = None
    tvec = _txt_embed(content)

    if ivec is None:
        joint = tvec
    else:
        v = (np.asarray(ivec, dtype=np.float32) + np.asarray(tvec, dtype=np.float32)) / 2.0
        n = np.linalg.norm(v)
        joint = (v / n) if n > 0 else v.astype(np.float32)

    # Build meta for the daemon → MemoryItem
    thumb_uri = None
    if pil_img is not None and save_files and thumb_path.exists():
        thumb_uri = str(thumb_path)

    meta: Dict[str, object] = {
        "kind": "media",
        "modality": "image",
        "mime": mime,
        "w": w,
        "h": h,
        "sha256": sha,
        "phash": phash,
        "ref_uri": str(ref_path),
        "thumb_uri": thumb_uri,
        "tags": list(tags or []),
        "exif": exif_safe or {},
        "ocr_text": ocr or "",
        "explicit_remember": bool(explicit_remember),
    }

    ev = Event(kind=source, content=content, meta=meta)

    return MediaIngestResult(
        event=ev,
        vector=joint,
        sha256=sha,
        phash=phash,
        ref_uri=str(ref_path),
        thumb_uri=thumb_uri,
        width=w,
        height=h,
        mime=mime,
        ocr_text=(ocr or None),
    )


# ----------------------------
# Optional: direct item maker
# ----------------------------
def make_media_item(
    res: MediaIngestResult,
    *,
    layer: str = "working",
) -> Tuple[MemoryItem, np.ndarray]:
    """
    Convenience: turn a MediaIngestResult into a MemoryItem directly
    (bypasses the Event→Daemon path). You still need to upsert via the store.
    """
    # Strip reserved keys from meta to avoid duplicates with explicit args
    raw_meta = dict(res.event.meta or {})
    for k in ("kind", "source", "content", "layer", "id", "ts", "meta"):
        raw_meta.pop(k, None)

    it = MemoryItem.new(
        kind="media",
        source=res.event.kind,
        content=res.event.content,
        layer=layer,
        **raw_meta,
    )
    it.embedding_id = f"vec_{it.id}"
    it.embedding_dim = int(len(res.vector))
    it.model_hint = _model_hint()
    # Initial priors
    it.freq = 0
    it.salience = 0.8  # media is usually salient
    it.novelty = 1.0
    it.goal_relevance = float((res.event.meta or {}).get("goal_rel", 0.0))
    it.impact_signal = float((res.event.meta or {}).get("impact", 0.0))
    it.strength = 0.15
    return it, res.vector
