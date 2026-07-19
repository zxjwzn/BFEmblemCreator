"""概括、相似度与近似管线测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bf_emblem_creator.approx.index import DEFAULT_BASIC_STAMPS, StampIndex
from bf_emblem_creator.approx.metrics import compare_images, fit_loss, score_fit
from bf_emblem_creator.approx.models import ApproxConfig
from bf_emblem_creator.approx.pipeline import approximate_image
from bf_emblem_creator.approx.preprocess import abstract_image, save_debug_montage
from bf_emblem_creator.models import EmblemDocument, StampLayer
from bf_emblem_creator.stamps import StampLibrary

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ROOT / "assets" / "stamps"
EMOJI_DIR = ROOT / "examples"
EMOJI_SMILE = EMOJI_DIR / "😄.png"


@pytest.fixture(scope="module")
def stamps_dir() -> Path:
    return STAMPS


def test_abstract_emoji_produces_palette() -> None:
    """emoji 概括应得到少色调色板与有效主体。"""
    cfg = ApproxConfig(stamps_dir=STAMPS, palette_k=5, seed=0)
    target = abstract_image(EMOJI_SMILE, cfg)
    assert target.meta.canvas_size == 320
    assert target.numpy_rgb().shape == (320, 320, 3)
    assert target.numpy_alpha().mean() > 0.05
    assert 1 <= len(target.palette) <= 5
    assert float(target.numpy_weight().mean()) == pytest.approx(1.0, abs=0.15)


def test_similarity_identical_is_high() -> None:
    """相同图像综合分应接近 1。"""
    cfg = ApproxConfig(stamps_dir=STAMPS, palette_k=4)
    target = abstract_image(EMOJI_SMILE, cfg)
    rgb = target.numpy_rgb()
    a = (target.numpy_alpha() * 255).astype(np.uint8)
    rgba = np.dstack([rgb, a])
    report = compare_images(rgba, rgb, target.numpy_alpha(), pass_score=0.9)
    assert report.overall >= 0.9
    assert report.passed


def test_similarity_empty_is_low() -> None:
    """空预测应低分。"""
    cfg = ApproxConfig(stamps_dir=STAMPS)
    target = abstract_image(EMOJI_SMILE, cfg)
    empty = np.zeros((320, 320, 4), dtype=np.uint8)
    report = score_fit(empty, target, pass_score=0.5)
    assert report.overall < 0.35
    assert not report.passed


def test_stamp_index_builds(stamps_dir: Path) -> None:
    lib = StampLibrary(stamps_dir)
    index = StampIndex.build(lib, list(DEFAULT_BASIC_STAMPS)[:8], size=48)
    assert len(index.features) >= 4
    ids = index.recall({"circularity": 0.9, "elongation": 1.1, "fill_ratio": 0.8, "aspect": 1.0}, k=3)
    assert len(ids) == 3


def test_fit_loss_decreases_when_closer() -> None:
    cfg = ApproxConfig(stamps_dir=STAMPS, palette_k=4)
    target = abstract_image(EMOJI_SMILE, cfg)
    rgb = target.numpy_rgb()
    a = target.numpy_alpha()
    good = fit_loss(rgb, a, target)
    bad = fit_loss(np.zeros_like(rgb), np.zeros_like(a), target)
    assert good < bad


def test_approximate_emoji_meets_threshold(stamps_dir: Path) -> None:
    """对笑脸 emoji 拟合应达到综合相似度阈值。"""
    cfg = ApproxConfig(
        stamps_dir=stamps_dir,
        max_layers=20,
        palette_k=5,
        pass_score=0.48,
        refine=True,
        refine_iters=3,
        recall_k=6,
        max_rois_per_iter=3,
        angle_step_deg=30.0,
        render_supersample=1.0,
        seed=1,
        min_gain=0.004,
        stall_patience=2,
    )
    result = approximate_image(EMOJI_SMILE, cfg)
    assert len(result.document) >= 1
    assert len(result.document) <= 40
    assert result.similarity is not None
    assert result.similarity.overall >= cfg.pass_score
    assert result.similarity.passed
    data = result.document.model_dump_list()
    assert isinstance(data, list)
    for layer in result.document:
        assert isinstance(layer, StampLayer)


def test_debug_montage(tmp_path: Path) -> None:
    target = abstract_image(EMOJI_SMILE, ApproxConfig(stamps_dir=STAMPS))
    path = tmp_path / "mont.png"
    save_debug_montage(target, path)
    assert path.is_file()
    im = Image.open(path)
    assert im.size[0] == 320 * 4


def test_document_from_approx_is_renderable(stamps_dir: Path) -> None:
    cfg = ApproxConfig(
        stamps_dir=stamps_dir,
        max_layers=6,
        palette_k=4,
        refine_iters=1,
        recall_k=4,
        max_rois_per_iter=2,
        pass_score=0.0,
        stall_patience=1,
    )
    result = approximate_image(EMOJI_SMILE, cfg)
    doc = EmblemDocument.from_layers(list(result.document))
    assert len(doc) == len(result.document)
