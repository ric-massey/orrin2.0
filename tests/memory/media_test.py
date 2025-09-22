# tests/memory_tests/media_test.py
from types import SimpleNamespace
from pathlib import Path
import os
import io
import numpy as np
import pytest

import memory.media as mm
from memory.models import Event


# ---------------------------
# helpers
# ---------------------------

def _unit(v):
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return a if n == 0 else a / n


def _fake_png_bytes():
    # tiny valid PNG header (we won't decode it without PIL anyway)
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\x0bIDAT\x08\xd7c\xf8\x0f\x00\x01"
        b"\x01\x01\x00\x18\xdd\x8d\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
    )


# ---------------------------
# tests
# ---------------------------

def test_ingest_image_without_pil_writes_ref_and_builds_event(tmp_path, monkeypatch):
    # route Media dir to tmp
    monkeypatch.setattr(mm, "MEDIA_DIR", tmp_path, raising=True)
    monkeypatch.setattr(mm.MEMCFG, "MEDIA_DIR", tmp_path, raising=False)

    # force "no PIL"
    monkeypatch.setattr(mm, "_HAS_PIL", False, raising=True)

    # predictable embedders: image = [1,0,0], text = [0,1,0]
    monkeypatch.setattr(mm, "_img_embed", lambda x: _unit([1, 0, 0]), raising=True)
    monkeypatch.setattr(mm, "_txt_embed", lambda s: _unit([0, 1, 0]), raising=True)

    raw = _fake_png_bytes()
    res = mm.ingest_image(raw, tags=["tag1", "tag2"], save_files=True)

    # files/paths
    ref = Path(res.ref_uri)
    assert ref.exists() and ref.read_bytes() == raw
    assert res.thumb_uri is None  # no PIL -> no thumb
    assert res.mime == "image/octet-stream"
    assert res.width is None and res.height is None

    # event/meta
    assert isinstance(res.event, Event)
    meta = res.event.meta or {}
    assert meta["kind"] == "media"
    assert meta["modality"] == "image"
    assert meta["sha256"] == res.sha256
    assert meta["phash"] is None
    assert meta["tags"] == ["tag1", "tag2"]
    assert meta["exif"] == {}
    assert meta["ocr_text"] == ""
    assert meta["explicit_remember"] is True
    assert res.event.kind == "media:image"
    assert "Image (unknown size)" in res.event.content

    # vector = average([1,0,0],[0,1,0]) normalized
    expected = _unit([0.5, 0.5, 0.0])
    assert np.allclose(res.vector, expected)


def test_ingest_image_with_pil_branch_and_ocr_and_thumb(tmp_path, monkeypatch):
    # route Media dir
    monkeypatch.setattr(mm, "MEDIA_DIR", tmp_path, raising=True)
    monkeypatch.setattr(mm.MEMCFG, "MEDIA_DIR", tmp_path, raising=False)

    # Force "PIL present"
    monkeypatch.setattr(mm, "_HAS_PIL", True, raising=True)

    # Fake Image API used by media.py just enough to pass
    class _FakeImg:
        format = "PNG"
        size = (32, 24)
        def convert(self, mode): return self
        def copy(self): return self
        def thumbnail(self, sz): pass
        def save(self, fp, format=None, quality=None):
            # handle Path or file-like
            if hasattr(fp, "write"):  # BytesIO etc
                fp.write(b"JPG")
            else:
                Path(fp).write_bytes(b"JPG")
        # keep _getexif unused by overriding _safe_exif later

    class _FakeImageMod:
        @staticmethod
        def open(b):
            # accept BytesIO or bytes/path; return fake image
            return _FakeImg()

    # Patch Image symbol used in module
    monkeypatch.setattr(mm, "Image", _FakeImageMod, raising=True)

    # Simplify internal helpers so we don't need real PIL arrays
    monkeypatch.setattr(mm, "_detect_mime_pil", lambda img: "image/png", raising=True)
    monkeypatch.setattr(mm, "_safe_exif", lambda img: {"Model": "XCam"}, raising=True)
    monkeypatch.setattr(mm, "_phash_dhash", lambda img: "abcd1234", raising=True)
    monkeypatch.setattr(mm, "_make_thumb", lambda img, size=256: _FakeImg(), raising=True)

    # OCR enabled path
    monkeypatch.setattr(mm, "_HAS_TESS", True, raising=True)
    monkeypatch.setattr(mm, "pytesseract", SimpleNamespace(image_to_string=lambda im: "HELLO OCR"), raising=False)

    # predictable embedders: image = [1,0,0], text = [0,1,0]
    monkeypatch.setattr(mm, "_img_embed", lambda x: _unit([1, 0, 0]), raising=True)
    monkeypatch.setattr(mm, "_txt_embed", lambda s: _unit([0, 1, 0]), raising=True)

    res = mm.ingest_image(_fake_png_bytes(), tags=["car"], caption=None, save_files=True)

    # thumb exists (we saved "JPG")
    assert res.thumb_uri is not None
    assert Path(res.thumb_uri).exists()

    # exif/phash/mime propagated
    assert res.mime == "image/png"
    assert res.phash == "abcd1234"
    assert res.event.meta["exif"] == {"Model": "XCam"}

    # OCR captured into content prefix
    assert "OCR: HELLO OCR" in res.event.content

    # vector as before
    expected = _unit([0.5, 0.5, 0.0])
    assert np.allclose(res.vector, expected)

    # dims from fake size
    assert (res.width, res.height) == (32, 24)


def test_ingest_image_save_files_false_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MEDIA_DIR", tmp_path, raising=True)
    monkeypatch.setattr(mm.MEMCFG, "MEDIA_DIR", tmp_path, raising=False)

    # No PIL path
    monkeypatch.setattr(mm, "_HAS_PIL", False, raising=True)
    monkeypatch.setattr(mm, "_img_embed", lambda x: _unit([1, 0, 0]), raising=True)
    monkeypatch.setattr(mm, "_txt_embed", lambda s: _unit([0, 1, 0]), raising=True)

    res = mm.ingest_image(_fake_png_bytes(), save_files=False)
    # ref_uri computed but file not written
    assert not Path(res.ref_uri).exists()
    assert res.thumb_uri is None


def test_ingest_image_text_only_when_image_embedder_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MEDIA_DIR", tmp_path, raising=True)
    monkeypatch.setattr(mm.MEMCFG, "MEDIA_DIR", tmp_path, raising=False)

    monkeypatch.setattr(mm, "_HAS_PIL", False, raising=True)

    # image embedder fails -> joint = text vector
    def boom(x): raise RuntimeError("img embed fail")
    monkeypatch.setattr(mm, "_img_embed", boom, raising=True)
    monkeypatch.setattr(mm, "_txt_embed", lambda s: _unit([0, 0, 1]), raising=True)

    res = mm.ingest_image(_fake_png_bytes(), save_files=False)
    assert np.allclose(res.vector, _unit([0, 0, 1]))


def test_ingest_image_caption_override_and_tags_present(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MEDIA_DIR", tmp_path, raising=True)
    monkeypatch.setattr(mm.MEMCFG, "MEDIA_DIR", tmp_path, raising=False)

    monkeypatch.setattr(mm, "_HAS_PIL", False, raising=True)
    monkeypatch.setattr(mm, "_img_embed", lambda x: _unit([1, 0, 0]), raising=True)
    monkeypatch.setattr(mm, "_txt_embed", lambda s: _unit([0, 1, 0]), raising=True)

    res = mm.ingest_image(_fake_png_bytes(), tags=["cat", "meme"], caption="My cap", save_files=False)
    assert res.event.content.startswith("My cap")
    assert res.event.meta["tags"] == ["cat", "meme"]


def test_make_media_item_builds_item_with_expected_priors(monkeypatch):
    # Build a minimal MediaIngestResult by calling ingest_image with fixed embedders
    tmp = Path(os.getcwd()) / ".tmp_media_test"
    tmp.mkdir(exist_ok=True)
    try:
        monkeypatch.setattr(mm, "MEDIA_DIR", tmp, raising=True)
        monkeypatch.setattr(mm.MEMCFG, "MEDIA_DIR", tmp, raising=False)
        monkeypatch.setattr(mm, "_HAS_PIL", False, raising=True)
        monkeypatch.setattr(mm, "_img_embed", lambda x: _unit([1, 0, 0]), raising=True)
        monkeypatch.setattr(mm, "_txt_embed", lambda s: _unit([0, 1, 0]), raising=True)
        monkeypatch.setattr(mm, "_model_hint", lambda: "model-hint", raising=True)

        res = mm.ingest_image(_fake_png_bytes(), save_files=False)
        item, vec = mm.make_media_item(res, layer="working")

        assert item.kind == "media"
        assert item.layer == "working"
        assert item.embedding_id.startswith("vec_")
        assert item.embedding_dim == len(vec)
        assert item.model_hint == "model-hint"
        assert item.freq == 0
        assert 0.0 <= item.salience <= 1.0
        assert item.novelty == 1.0
        # strength prior set to ~0.15 in helper
        assert pytest.approx(item.strength, rel=1e-6) == 0.15
    finally:
        try:
            for p in tmp.glob("*"):
                p.unlink()
            tmp.rmdir()
        except Exception:
            pass
