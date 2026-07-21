"""平面图 Face 轮廓：无穿心弦、贴 mask 边界、共享边拓扑可校验。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from bf_emblem_creator.approx.label_field import build_label_field, gap_fraction
from bf_emblem_creator.approx.models import AbstractionMode, PaletteColor, ResampleMode
from bf_emblem_creator.approx.planar_map import (
    _chain_dual_segments,
    _collect_dual_segments,
    _contour_has_interior_chords,
    assert_planar_map_valid,
    build_planar_map,
    face_contour,
    planar_map_to_region_graph,
)
from bf_emblem_creator.approx.planarize import planarize_image
from bf_emblem_creator.approx.recipe import ImageProcessorConfig

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "examples" / "gold.png"

BoolArr = NDArray[np.bool_]
FloatArr = NDArray[np.floating]


def _two_color_hard(size: int = 64) -> np.ndarray:
    """左右硬边两色 RGBA。"""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, : size // 2] = (255, 0, 0, 255)
    arr[:, size // 2 :] = (0, 0, 255, 255)
    return arr


def _l_shape_labels(size: int = 48) -> tuple[np.ndarray, list[PaletteColor], np.ndarray]:
    """L 形两色标签：主色块非凸，PCA 袋排序会失败。"""
    lab = np.full((size, size), -1, dtype=np.int32)
    # 主体 L：左柱 + 底横
    lab[:size, : size // 3] = 0
    lab[size * 2 // 3 :, :] = 0
    # 内角补一块异色
    lab[size // 4 : size * 2 // 3, size // 3 : size * 2 // 3] = 1
    alpha = (lab >= 0).astype(np.float64)
    palette = [
        PaletteColor(hex="#DC2828", fraction=0.7, rgb=(220, 40, 40)),
        PaletteColor(hex="#2828DC", fraction=0.3, rgb=(40, 40, 220)),
    ]
    return lab, palette, alpha


def _chord_count(cont: FloatArr, mask: BoolArr, *, min_chord: float = 10.0) -> int:
    """统计穿心长弦条数。"""
    p = np.asarray(cont, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if len(p) < 3 or not m.any():
        return 0
    h, w = m.shape
    bad = 0
    for i in range(len(p) - 1):
        d = float(np.linalg.norm(p[i + 1] - p[i]))
        if d < min_chord:
            continue
        mid = 0.5 * (p[i] + p[i + 1])
        ix = round(float(mid[0]))
        iy = round(float(mid[1]))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        if not m[iy, ix]:
            continue
        y0, y1 = max(0, iy - 2), min(h, iy + 3)
        x0, x1 = max(0, ix - 2), min(w, ix + 3)
        if bool(m[y0:y1, x0:x1].all()):
            bad += 1
    return bad


def test_dual_chain_preserves_order_on_straight_interface() -> None:
    """竖直硬界面：dual 链应按 y 单调前进，而非 PCA 乱序。"""
    size = 32
    lab = np.zeros((size, size), dtype=np.int32)
    lab[:, size // 2 :] = 1
    segs = _collect_dual_segments(lab)
    chains = _chain_dual_segments(segs)
    # 内部界面链：两侧均为非负
    inner = [c for c in chains if c[0] >= 0 and c[1] >= 0]
    assert len(inner) >= 1
    poly = np.asarray(inner[0][2], dtype=np.float64)
    assert len(poly) >= size
    # x 应稳定在分割线上
    assert float(np.std(poly[:, 0])) < 1e-6
    # 沿链 y 单调（允许闭合反向）
    dy = np.diff(poly[:, 1])
    mono_up = bool(np.all(dy >= -1e-9))
    mono_dn = bool(np.all(dy <= 1e-9))
    assert mono_up or mono_dn


def test_face_contour_two_color_no_interior_chords() -> None:
    """两色硬边：face 轮廓无穿心弦。"""
    rgb = _two_color_hard(40)[:, :, :3]
    alpha = np.ones((40, 40), dtype=np.float64)
    labels, palette, gf, _ = build_label_field(rgb, alpha, num_colors=2, mrf_iters=2, enforce_no_gap=True)
    pmap = build_planar_map(labels, palette, alpha, min_area_frac=0.01, max_faces=8, gap_frac=gf)
    assert_planar_map_valid(pmap)
    for f in pmap.faces:
        cont = face_contour(pmap, f.id)
        mask = np.asarray(f.mask, dtype=bool)
        assert len(cont) >= 4
        assert not _contour_has_interior_chords(cont, mask, min_chord=12.0)
        assert _chord_count(cont, mask, min_chord=12.0) == 0


def test_face_contour_l_shape_no_chords() -> None:
    """非凸 L 形：轮廓不得出现穿越内部的长弦。"""
    lab, palette, alpha = _l_shape_labels(48)
    pmap = build_planar_map(lab, palette, alpha, min_area_frac=0.01, max_faces=8)
    assert_planar_map_valid(pmap)
    assert len(pmap.faces) >= 1
    graph = planar_map_to_region_graph(pmap)
    for reg in graph.regions:
        cont = np.asarray(reg.contour, dtype=np.float64)
        mask = np.asarray(reg.mask, dtype=bool)
        assert len(cont) >= 4
        assert _chord_count(cont, mask, min_chord=14.0) == 0
        # 调试误差字段应真实计算（无穿心时应较小）
        assert reg.contour_area_rel_err < 0.5


def test_region_graph_contour_stays_near_mask_boundary() -> None:
    """轮廓点应落在 mask 边界邻域，而非区域中心。"""
    rgb = _two_color_hard(48)[:, :, :3]
    alpha = np.ones((48, 48), dtype=np.float64)
    labels, palette, gf, _ = build_label_field(rgb, alpha, num_colors=2, enforce_no_gap=True)
    pmap = build_planar_map(labels, palette, alpha, gap_frac=gf)
    graph = planar_map_to_region_graph(pmap)
    for reg in graph.regions:
        cont = np.asarray(reg.contour, dtype=np.float64)
        mask = np.asarray(reg.mask, dtype=bool)
        h, w = mask.shape
        # 边界像素：mask 真且 4 邻有假
        pad = np.pad(mask, 1, constant_values=False)
        boundary = np.zeros_like(mask)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            boundary |= mask & (~pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w])
        ys, xs = np.where(boundary)
        if len(xs) < 4:
            continue
        bpts = np.stack([xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5], axis=1)
        # 子采样轮廓点
        pts = cont
        if len(pts) > 80:
            pts = pts[np.linspace(0, len(pts) - 1, 80).astype(int)]
        d = np.linalg.norm(pts[:, None, :] - bpts[None, :, :], axis=2).min(axis=1)
        # 绝大多数点应贴近边界（dual 角点可偏 ~1px）
        assert float(np.percentile(d, 90)) <= 3.5


def test_gold_planar_contours_no_chords_if_present() -> None:
    """gold.png 平面化后各区域轮廓无穿心弦。"""
    if not GOLD.is_file():
        return

    cfg = ImageProcessorConfig(
        canvas_size=160,
        num_colors=4,
        use_cuda=False,
        mrf_iters=4,
        enforce_no_gap=True,
        resample_mode=ResampleMode.auto,
    )
    labels, palette, alpha, _image_q, meta, _src = planarize_image(GOLD, cfg, mode=AbstractionMode.illustration)
    assert gap_fraction(labels, alpha) == 0.0 or meta.gap_frac == 0.0
    pmap = build_planar_map(
        labels,
        palette,
        alpha,
        min_area_frac=0.004,
        max_faces=24,
        gap_frac=float(meta.gap_frac),
    )
    assert_planar_map_valid(pmap)
    graph = planar_map_to_region_graph(pmap)
    assert len(graph.regions) >= 2
    chord_total = 0
    for reg in graph.regions:
        cont = np.asarray(reg.contour, dtype=np.float64)
        mask = np.asarray(reg.mask, dtype=bool)
        n = _chord_count(cont, mask, min_chord=16.0)
        chord_total += n
        assert not _contour_has_interior_chords(cont, mask, min_chord=16.0), f"region {reg.region_id} 存在穿心弦 n={n}"
    assert chord_total == 0


def test_shared_edge_geometry_unique_for_inner_pair() -> None:
    """内部 face 对应存在共享边，且双侧 face_contour 都覆盖该边邻域。"""
    rgb = _two_color_hard(36)[:, :, :3]
    alpha = np.ones((36, 36), dtype=np.float64)
    labels, palette, gf, _ = build_label_field(rgb, alpha, num_colors=2, enforce_no_gap=True)
    pmap = build_planar_map(labels, palette, alpha, gap_frac=gf)
    inner = [e for e in pmap.edges if e.left_face >= 0 and e.right_face >= 0]
    assert len(inner) >= 1
    e0 = inner[0]
    poly = np.asarray(e0.polyline, dtype=np.float64)
    assert len(poly) >= 2
    ca = face_contour(pmap, e0.left_face)
    cb = face_contour(pmap, e0.right_face)
    # 边上采样点应落在两侧轮廓点云附近
    for c in (ca, cb):
        d = np.linalg.norm(poly[len(poly) // 2] - c, axis=1).min()
        assert float(d) <= 4.0


def test_exact_boundary_not_circle_generalization() -> None:
    """方形 mask 轮廓不得被概括成圆（应保留直角边界）。"""
    lab = np.zeros((40, 40), dtype=np.int32)
    lab[8:32, 8:32] = 0
    alpha = np.ones((40, 40), dtype=np.float64)
    palette = [PaletteColor(hex="#FF0000", fraction=1.0, rgb=(255, 0, 0))]
    # 仅一个 face
    pmap = build_planar_map(lab, palette, alpha, min_area_frac=0.01, max_faces=4)
    assert_planar_map_valid(pmap)
    cont = face_contour(pmap, pmap.faces[0].id)
    # 精确 dual/Moore：应触达包围盒四边附近，而非内缩圆
    xs, ys = cont[:, 0], cont[:, 1]
    assert float(xs.min()) <= 9.0
    assert float(xs.max()) >= 31.0
    assert float(ys.min()) <= 9.0
    assert float(ys.max()) >= 31.0
    # 不应近似单位圆：点到中心距离方差应明显（方 vs 圆）
    cxy = np.array([20.0, 20.0])
    r = np.linalg.norm(cont - cxy, axis=1)
    assert float(np.std(r)) > 1.5
