"""严格 num_colors 色量：平面化与标签场。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bf_emblem_creator.approx.label_field import build_label_field, gap_fraction
from bf_emblem_creator.approx.models import AbstractionMode
from bf_emblem_creator.approx.planarize import planarize_image
from bf_emblem_creator.approx.recipe import ImageProcessorConfig

ROOT = Path(__file__).resolve().parents[1]
EMOJI = ROOT / "examples" / "smile.png"


def _two_color(size: int = 64) -> np.ndarray:
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, : size // 2] = (255, 0, 0, 255)
    arr[:, size // 2 :] = (0, 0, 255, 255)
    return arr


def test_strict_kmeans_palette_size() -> None:
    """请求 K 色时，硬分配后调色板长度应等于 K（无空簇时）。"""
    rgb = _two_color(48)[:, :, :3]
    alpha = np.ones((48, 48), dtype=np.float64)
    labels, palette, gf, _ = build_label_field(
        rgb,
        alpha,
        num_colors=2,
        mrf_iters=2,
        enforce_no_gap=True,
        seed=0,
    )
    assert gf == 0.0
    assert len(palette) == 2
    assert int(labels.max()) == 1


def test_num_colors_drives_planarize() -> None:
    """planarize 使用 ImageProcessorConfig.num_colors。"""
    cfg = ImageProcessorConfig(num_colors=4, seed=0, bilateral=False, use_cuda=False)
    labels, palette, alpha, _iq, meta, _src = planarize_image(_two_color(64), cfg, mode=AbstractionMode.logo)
    assert gap_fraction(labels, alpha) == 0.0
    assert 1 <= len(palette) <= 4
    assert meta.num_colors == len(palette)
    assert meta.num_colors <= 4


def test_more_colors_can_increase_palette() -> None:
    """色数提高时，多色图调色板不减。"""
    h, w = 48, 48
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:24, :24] = (255, 0, 0)
    rgb[:24, 24:] = (0, 255, 0)
    rgb[24:, :24] = (0, 0, 255)
    rgb[24:, 24:] = (255, 255, 0)
    alpha = np.ones((h, w), dtype=np.float64)
    _, pal2, _, _ = build_label_field(rgb, alpha, num_colors=2, mrf_iters=1, seed=0)
    _, pal4, _, _ = build_label_field(rgb, alpha, num_colors=4, mrf_iters=1, seed=0)
    assert len(pal4) >= len(pal2)


def test_emoji_num_colors_if_present() -> None:
    if not EMOJI.is_file():
        return
    cfg = ImageProcessorConfig(
        canvas_size=160,
        num_colors=6,
        use_cuda=False,
        mrf_iters=4,
        enforce_no_gap=True,
    )
    labels, palette, alpha, _iq, meta, _src = planarize_image(EMOJI, cfg, mode=AbstractionMode.illustration)
    assert gap_fraction(labels, alpha) == 0.0
    assert 1 <= len(palette) <= 6
    assert meta.num_colors == len(palette)
