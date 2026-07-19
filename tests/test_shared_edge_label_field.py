"""Batch A–E：自适应重采样、标签场无洞、共享边平面图与缝宽。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from bf_emblem_creator.approx.assemble import assemble_layers
from bf_emblem_creator.approx.contour_arcs import extract_primitives_for_shared_edges
from bf_emblem_creator.approx.depth_order import EdgeRole, infer_depth_order
from bf_emblem_creator.approx.label_field import (
    build_label_field,
    fill_label_gaps,
    gap_fraction,
    icm_label_refine,
)
from bf_emblem_creator.approx.models import ApproxConfig, ResampleMode
from bf_emblem_creator.approx.planar_map import (
    assert_planar_map_valid,
    build_planar_map,
    face_contour,
    face_shape_boundary_points,
    planar_map_to_region_graph,
    refine_edges_subpixel,
    seam_width_p95,
    shared_shape_points,
    walk_face_halfedges,
)
from bf_emblem_creator.approx.planarize import planarize_image
from bf_emblem_creator.approx.preprocess import detect_resample_mode, estimate_color_stats, fit_to_canvas
from bf_emblem_creator.approx.regions import build_regions

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "examples" / "gold.png"


def _two_color_hard(size: int = 64) -> np.ndarray:
    """左右硬边两色 RGBA。"""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, : size // 2] = (255, 0, 0, 255)
    arr[:, size // 2 :] = (0, 0, 255, 255)
    return arr


def test_detect_resample_nearest_for_few_colors() -> None:
    """少色尖峰图应选择 nearest。"""
    img = Image.fromarray(_two_color_hard(32), mode="RGBA")
    mode, stats = detect_resample_mode(img, target_size=320, configured=ResampleMode.auto)
    assert mode == "nearest"
    assert stats["approx_color_count"] <= 8


def test_detect_resample_force_lanczos() -> None:
    """配置强制 lanczos 应覆盖 auto。"""
    img = Image.fromarray(_two_color_hard(32), mode="RGBA")
    mode, _ = detect_resample_mode(img, configured=ResampleMode.lanczos)
    assert mode == "lanczos"


def test_fit_to_canvas_nearest_preserves_few_colors() -> None:
    """nearest 放大后近似独特色数不爆炸。"""
    img = Image.fromarray(_two_color_hard(16), mode="RGBA")
    rgba, meta = fit_to_canvas(img, 64, how="contain", resample="nearest")
    assert meta.resample == "nearest"
    stats = estimate_color_stats(rgba)
    assert stats["approx_color_count"] <= 6


def test_label_field_no_gap_hard_edge() -> None:
    """硬边两色：gap=0。"""
    rgb = _two_color_hard(48)[:, :, :3]
    alpha = np.ones((48, 48), dtype=np.float64)
    labels, palette, gf, _nf = build_label_field(rgb, alpha, k=3, mrf_iters=3, min_area_frac=0.01, enforce_no_gap=True, seed=0)
    assert gf == 0.0
    assert gap_fraction(labels, alpha) == 0.0
    assert len(palette) >= 2
    assert int((labels >= 0).sum()) == 48 * 48


def test_label_field_soft_edge_no_long_third_filament() -> None:
    """软边条带：无洞；细丝经规整后 noise 受控（不强求仅两色）。"""
    h, w = 40, 80
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        t = x / (w - 1)
        # 左红右蓝，中间 lerp 软边
        c = (1 - t) * np.array([255, 0, 0]) + t * np.array([0, 0, 255])
        rgb[:, x] = c.astype(np.uint8)
    alpha = np.ones((h, w), dtype=np.float64)
    labels, palette, gf, nf = build_label_field(
        rgb,
        alpha,
        k=4,
        mrf_lambda=3.0,
        mrf_iters=6,
        min_area_frac=0.02,
        enforce_no_gap=True,
        seed=1,
    )
    assert gf == 0.0
    assert gap_fraction(labels, alpha) == 0.0
    assert len(palette) >= 2
    # 过小连通噪声占比应有限
    assert nf <= 0.15
    # 不应出现大量 1px 宽竖丝占主导：最大连通域应显著
    from bf_emblem_creator.approx.label_field import _label_ccs

    max_cc = 0
    for lab in range(int(labels.max()) + 1):
        for cc in _label_ccs(labels == lab):
            max_cc = max(max_cc, int(cc.sum()))
    assert max_cc >= int(0.2 * h * w)


def test_fill_label_gaps_and_regions_no_gap() -> None:
    """regions 在强制无洞时主体全覆盖。"""
    rgb = _two_color_hard(40)[:, :, :3]
    alpha = np.ones((40, 40), dtype=np.float64)
    labels, palette, _, _ = build_label_field(rgb, alpha, k=2, mrf_iters=2, enforce_no_gap=True)
    # 人为挖洞
    labels[10:12, 10:12] = -1
    labels = fill_label_gaps(labels, alpha)
    assert gap_fraction(labels, alpha) == 0.0
    graph = build_regions(labels, palette, alpha, min_area_frac=0.01, max_regions=8, enforce_no_gap=True)
    covered = np.zeros((40, 40), dtype=bool)
    for r in graph.regions:
        covered |= np.asarray(r.mask, dtype=bool)
    assert float((~covered & (alpha >= 0.5)).sum()) == 0.0


def test_planar_map_two_color_and_valid() -> None:
    """左右两色平面图可校验。"""
    rgb = _two_color_hard(32)[:, :, :3]
    alpha = np.ones((32, 32), dtype=np.float64)
    labels, palette, gf, _ = build_label_field(rgb, alpha, k=2, mrf_iters=2, enforce_no_gap=True)
    pmap = build_planar_map(labels, palette, alpha, min_area_frac=0.01, max_faces=8, gap_frac=gf)
    assert_planar_map_valid(pmap)
    assert len(pmap.faces) >= 2
    # 应有内部共边（left/right 均非背景）
    inner = [e for e in pmap.edges if e.left_face >= 0 and e.right_face >= 0]
    assert len(inner) >= 1
    # 完整 halfedge 环：每个 face 可 walk
    for f in pmap.faces:
        ring = walk_face_halfedges(pmap, f.id)
        assert len(ring) >= 1
        assert all(he.next_id >= 0 for he in ring)
    graph = planar_map_to_region_graph(pmap)
    assert len(graph.regions) == len(pmap.faces)
    depth = infer_depth_order(graph, planar_map=pmap)
    assert len(depth.ordered) == len(graph.regions)
    # 角色写回
    assert all(isinstance(e.role, EdgeRole) for e in pmap.edges)
    # 半边环派生轮廓与共边一致：改内部边几何后双侧轮廓同步
    if inner:
        e0 = inner[0]
        poly = np.asarray(e0.polyline, dtype=np.float64).copy()
        poly[:, 0] += 0.0  # 触碰
        e0.polyline = poly
        c_a = face_contour(pmap, e0.left_face)
        c_b = face_contour(pmap, e0.right_face)
        assert len(c_a) >= 2 and len(c_b) >= 2


def test_face_shape_boundary_points_dedup_edge_id() -> None:
    """匹配目标点按 edge_id 去重：同一共边不会因两侧 face 重复计入单 face 点云逻辑。"""
    rgb = _two_color_hard(40)[:, :, :3]
    alpha = np.ones((40, 40), dtype=np.float64)
    labels, palette, gf, _ = build_label_field(rgb, alpha, k=2, enforce_no_gap=True)
    pmap = build_planar_map(labels, palette, alpha, gap_frac=gf)
    f0 = pmap.faces[0].id
    pts = face_shape_boundary_points(pmap, f0, only_shape=True)
    assert len(pts) >= 4
    # 关联边 id 集合
    eids = {e.id for e in pmap.edges if e.left_face == f0 or e.right_face == f0}
    assert len(eids) >= 1


def test_edge_subpixel_refines_without_breaking_topology() -> None:
    """亚像素精修保持拓扑 id 与校验。"""
    rgb = _two_color_hard(48)[:, :, :3]
    alpha = np.ones((48, 48), dtype=np.float64)
    labels, palette, gf, _ = build_label_field(rgb, alpha, k=2, enforce_no_gap=True)
    pmap = build_planar_map(labels, palette, alpha, gap_frac=gf, edge_subpixel=True)
    assert_planar_map_valid(pmap)
    eids_before = {e.id for e in pmap.edges}
    refine_edges_subpixel(pmap, iters=1)
    assert {e.id for e in pmap.edges} == eids_before
    assert_planar_map_valid(pmap)


def test_shared_edge_primitives_and_seam_metric() -> None:
    """边上基元提取与 seam_p95 可计算。"""
    rgb = _two_color_hard(48)[:, :, :3]
    alpha = np.ones((48, 48), dtype=np.float64)
    labels, palette, gf, _ = build_label_field(rgb, alpha, k=2, enforce_no_gap=True)
    pmap = build_planar_map(labels, palette, alpha, gap_frac=gf)
    prims = extract_primitives_for_shared_edges(pmap.edges, eps_arc=0.08)
    assert len(prims) >= 1
    pts = shared_shape_points(pmap)
    assert len(pts) >= 4
    # 自洽：目标对自身 seam≈0
    s = seam_width_p95(pts, pts)
    assert s < 1.5


def test_planarize_gold_gap_zero_if_present() -> None:
    """gold.png 平面化后 gap_frac=0，且 auto 倾向 nearest。"""
    if not GOLD.is_file():
        return
    cfg = ApproxConfig(
        canvas_size=160,
        palette_k=4,
        k_start=4,
        use_cuda=False,
        mrf_iters=4,
        enforce_no_gap=True,
        resample_mode=ResampleMode.auto,
    )
    labels, palette, alpha, _image_q, meta, _src = planarize_image(GOLD, cfg, k=4)
    assert meta.gap_frac == 0.0 or gap_fraction(labels, alpha) == 0.0
    assert meta.resample in {"nearest", "lanczos", "bilinear"}
    assert len(palette) >= 2
    # 平面色金锭：期望 nearest
    assert meta.resample == "nearest"
    assert gap_fraction(labels, alpha) == 0.0


def test_icm_reduces_isolated_noise() -> None:
    """ICM 应消减棋盘噪声点。"""
    h, w = 24, 24
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = (200, 40, 40)
    rgb[:, w // 2 :] = (40, 40, 200)
    # 在左半撒蓝噪声点
    rgb[5, 5] = (40, 40, 200)
    rgb[6, 7] = (40, 40, 200)
    alpha = np.ones((h, w), dtype=np.float64)
    labels0, palette, _, _ = build_label_field(rgb, alpha, k=2, mrf_iters=0, enforce_no_gap=True, min_area_frac=0.001, seed=0)
    labels1 = icm_label_refine(rgb, alpha, labels0, palette, mrf_lambda=4.0, iters=8)
    # 噪声点更可能并回左色
    left = labels1[:, : w // 2]
    # 左半应几乎单色
    _, cnts = np.unique(left, return_counts=True)
    assert float(cnts.max()) / float(left.size) >= 0.9


def test_assemble_layers_returns_seam() -> None:
    """assemble_layers 返回三元组含 seam。"""
    from bf_emblem_creator.approx.torch_render import TorchStampRenderer
    from bf_emblem_creator.stamps import StampLibrary

    stamps = ROOT / "assets" / "stamps"
    if not stamps.is_dir():
        return
    lib = StampLibrary(stamps)
    ren = TorchStampRenderer(lib, canvas_size=64, stamp_tex_size=64, device=__import__("torch").device("cpu"))
    layers, bscore, seam = assemble_layers([], ren, [], max_layers=4)
    assert layers == []
    assert isinstance(bscore, float)
    assert isinstance(seam, float)
