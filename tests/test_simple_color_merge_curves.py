"""严格色量 + 精确边界描边 + 共享边。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bf_emblem_creator.approx.label_field import build_label_field, gap_fraction
from bf_emblem_creator.approx.planar_map import (
    _contour_has_interior_chords,
    assert_planar_map_valid,
    build_planar_map,
    face_contour,
    planar_map_to_region_graph,
)
from bf_emblem_creator.approx.planarize import planarize_image


def _two_color_hard(size: int = 64) -> np.ndarray:
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, : size // 2] = (255, 0, 0, 255)
    arr[:, size // 2 :] = (0, 0, 255, 255)
    return arr


def test_strict_palette_size_on_gradient() -> None:
    """连续黄阶：严格 K 色量后调色板长度不超过请求 K。"""
    h, w = 48, 48
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = 24, 24
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    for y in range(h):
        for x in range(w):
            t = min(dist[y, x] / 22.0, 1.0)
            rgb[y, x] = (
                int(255 * (1 - 0.35 * t)),
                int(220 * (1 - 0.55 * t)),
                int(40 * (1 - 0.2 * t)),
            )
    rgb[18:22, 16:20] = (40, 30, 20)
    rgb[18:22, 28:32] = (40, 30, 20)
    alpha = (dist <= 22).astype(np.float64)
    labels, palette, gf, _ = build_label_field(
        rgb,
        alpha,
        num_colors=6,
        mrf_iters=4,
        enforce_no_gap=True,
        seed=0,
    )
    assert gf == 0.0
    assert 1 <= len(palette) <= 6
    sub = alpha >= 0.5
    max_frac = max(float((labels == i).sum()) / float(sub.sum()) for i in range(len(palette)))
    assert max_frac >= 0.15


def test_exact_boundary_and_shared_edge() -> None:
    """两色硬边：共享边存在；轮廓贴 mask，无穿心弦。"""
    rgb = _two_color_hard(48)[:, :, :3]
    alpha = np.ones((48, 48), dtype=np.float64)
    labels, palette, gf, _ = build_label_field(rgb, alpha, num_colors=2, mrf_iters=2, enforce_no_gap=True)
    assert gf == 0.0
    pmap = build_planar_map(
        labels,
        palette,
        alpha,
        gap_frac=gf,
        max_faces=8,
    )
    assert_planar_map_valid(pmap)
    inner = [e for e in pmap.edges if e.left_face >= 0 and e.right_face >= 0]
    assert len(inner) >= 1
    e0 = inner[0]
    poly = np.asarray(e0.polyline, dtype=np.float64)
    assert float(np.std(poly[:, 0])) < 1.5
    ca = face_contour(pmap, e0.left_face)
    cb = face_contour(pmap, e0.right_face)
    assert len(ca) >= 2 and len(cb) >= 2
    graph = planar_map_to_region_graph(pmap)
    for r in graph.regions:
        cont = np.asarray(r.contour, dtype=np.float64)
        mask = np.asarray(r.mask, dtype=bool)
        assert len(cont) >= 4
        assert not _contour_has_interior_chords(cont, mask, min_chord=12.0)


def test_smile_planarize_num_colors() -> None:
    """emoji 笑脸：调色板长度不超过 num_colors。"""
    root = Path(__file__).resolve().parents[1]
    p = root / "examples" / "smile.png"
    if not p.is_file():
        return
    from bf_emblem_creator.approx.models import AbstractionMode
    from bf_emblem_creator.approx.recipe import ImageProcessorConfig

    cfg = ImageProcessorConfig(
        canvas_size=160,
        num_colors=6,
        use_cuda=False,
        mrf_iters=4,
        enforce_no_gap=True,
    )
    labels, palette, alpha, _iq, meta, _src = planarize_image(p, cfg, mode=AbstractionMode.illustration)
    assert gap_fraction(labels, alpha) == 0.0
    assert 1 <= len(palette) <= 6
    assert meta.gap_frac == 0.0 or gap_fraction(labels, alpha) == 0.0
    labs = [pc.rgb for pc in palette]
    has_light = any(r + g + b >= 600 for r, g, b in labs)
    has_dark = any(r + g + b <= 300 for r, g, b in labs)
    assert has_light or has_dark or len(palette) >= 2
