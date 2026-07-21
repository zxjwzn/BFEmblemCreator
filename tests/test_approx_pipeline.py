"""近似管线：平面化、弧基元、曲线评分、GPU 粒子与综合评分。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bf_emblem_creator.approx.blocks import abstract_to_blocks
from bf_emblem_creator.approx.contour_arcs import extract_primitives_for_region, fit_segment_primitive, segment_contour
from bf_emblem_creator.approx.curves import extract_outer_contour, rdp
from bf_emblem_creator.approx.depth_order import infer_depth_order
from bf_emblem_creator.approx.line_quality import evaluate_line_quality
from bf_emblem_creator.approx.metrics import score_prediction
from bf_emblem_creator.approx.models import AbstractionMode
from bf_emblem_creator.approx.pipeline import approximate_image
from bf_emblem_creator.approx.planar_map import build_planar_map, planar_map_to_region_graph
from bf_emblem_creator.approx.planarize import planarize_image
from bf_emblem_creator.approx.recipe import ImageProcessorConfig, default_recipe_for_mode
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
from bf_emblem_creator.approx.torch_render import TorchStampRenderer
from bf_emblem_creator.stamps import StampLibrary

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ROOT / "assets" / "stamps"
EMOJI = ROOT / "examples" / "smile.png"


def test_line_quality_smooth_circle_high() -> None:
    """大块光滑色区应得到不触发 hard_fail 的线条分。"""
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[:] = (30, 30, 30)
    yy, xx = np.ogrid[:128, :128]
    disk = (xx - 64) ** 2 + (yy - 64) ** 2 <= 40**2
    rgb[disk] = (255, 200, 0)
    from PIL import Image, ImageFilter

    img = Image.fromarray(rgb).filter(ImageFilter.SMOOTH_MORE)
    rgb = np.asarray(img)
    alpha = np.ones((128, 128), dtype=np.float64)
    report = evaluate_line_quality(rgb, alpha)
    assert not report.hard_fail
    assert report.score >= 0.45


def test_line_quality_noise_low() -> None:
    """强噪声图线条分应低或 hard_fail。"""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    for y in range(128):
        for x in range(128):
            if (x // 2 + y // 2) % 2 == 0:
                rgb[y, x] = (0, 0, 0)
            else:
                rgb[y, x] = (255, 255, 255)
    report = evaluate_line_quality(rgb, np.ones((128, 128)))
    assert report.score <= 0.5 or report.hard_fail


def test_rdp_and_contour() -> None:
    line = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.0], [3.0, 2.0], [4.0, 2.1], [5.0, 2.0]], dtype=np.float64)
    simp = rdp(line, 0.5)
    assert len(simp) >= 3
    assert simp.shape[1] == 2
    mask = np.zeros((64, 64), dtype=bool)
    yy, xx = np.ogrid[:64, :64]
    mask[(xx - 32) ** 2 + (yy - 32) ** 2 <= 18**2] = True
    cont = extract_outer_contour(mask, simplify=1.0)
    assert cont is not None and cont.shape[0] >= 4 and cont.shape[1] == 2


def test_planarize_and_depth_order() -> None:
    """平面化 + SharedEdge 区域 + 层序应产出有序区域。"""

    result = planarize_image(
        EMOJI,
        ImageProcessorConfig(num_colors=6, use_cuda=torch.cuda.is_available()),
        mode=AbstractionMode.illustration,
    )
    labels, palette, alpha = result[0], result[1], result[2]
    assert palette
    assert labels.max() >= 0
    pmap = build_planar_map(labels, palette, alpha, min_area_frac=0.005)
    graph = planar_map_to_region_graph(pmap)
    assert len(graph.regions) >= 1
    depth = infer_depth_order(graph, planar_map=pmap)
    assert len(depth.ordered) == len(graph.regions)
    # 底层面积通常不小于顶层
    if len(depth.ordered) >= 2:
        assert depth.ordered[0].region.area_frac >= depth.ordered[-1].region.area_frac * 0.2


def test_arc_segmentation() -> None:
    """圆轮廓可分段；精确路径下 fit 返回 free（不概括为圆）。"""
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circle = np.stack([40 + 25 * np.cos(t), 40 + 25 * np.sin(t)], axis=1)
    segs = segment_contour(circle, closed=True)
    assert len(segs) >= 1
    ptype, params, res, hard = fit_segment_primitive(segs[0], eps_arc=0.08)
    assert ptype.value == "free"
    assert res >= 0.0
    assert hard
    _ = params


def test_abstract_blocks_emoji() -> None:
    target = abstract_to_blocks(
        EMOJI, default_recipe_for_mode(AbstractionMode.illustration).override(stamps_dir=STAMPS, num_colors=4)
    )
    assert target.canvas_size == 320
    assert len(target.blocks) >= 1
    assert target.blocks[0].area_frac > 0.01
    assert np.asarray(target.blocks[0].contour).ndim == 2


def test_torch_batch_render_shapes() -> None:
    lib = StampLibrary(STAMPS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = TorchStampRenderer(lib, canvas_size=64, stamp_tex_size=32, device=device)
    n = 32
    left = torch.full((n,), 32.0, device=device)
    top = torch.full((n,), 32.0, device=device)
    width = torch.linspace(10, 80, n, device=device)
    height = width.clone()
    angle = torch.linspace(0, 180, n, device=device)
    masks = r.render_batch_masks("Circle", left=left, top=top, width=width, height=height, angle_deg=angle)
    assert masks.shape == (n, 1, 64, 64)
    assert float(masks.max()) > 0.5


def test_moore_contour_with_hole() -> None:
    """带孔圆环：应提取 outer + 至少 1 个 hole，且孔在外轮廓内。"""
    from bf_emblem_creator.approx.curves import extract_all_contours, extract_contour_bundle, signed_area

    m = np.zeros((128, 128), dtype=bool)
    yy, xx = np.ogrid[:128, :128]
    m[(xx - 64) ** 2 + (yy - 64) ** 2 <= 45**2] = True
    m[(xx - 64) ** 2 + (yy - 64) ** 2 <= 18**2] = False  # 内孔
    rings = extract_all_contours(m, simplify=0.5, min_points=6)
    assert any(r.kind == "outer" for r in rings)
    holes = [r for r in rings if r.kind == "hole"]
    assert len(holes) >= 1
    outer, hole_list, _ = extract_contour_bundle(m, simplify=0.5, resample_n=64)
    assert len(outer) >= 8
    assert abs(signed_area(outer)) > abs(signed_area(hole_list[0]))
    assert len(hole_list) >= 1


def test_stamp_curve_library_full_no_shape_bucket() -> None:
    """多环高精度构建 + 描述子召回，无形状桶。"""
    lib = StampLibrary(STAMPS)
    cache = ROOT / "assets" / ".cache" / "stamp_curves_test"
    curves = StampCurveLibrary.build(
        lib,
        ["Circle", "Square", "Triangle", "Star", "OpenCircle", "Line"],
        tex_size=96,
        cache_dir=cache,
        force_refit=True,
    )
    assert "Circle" in curves.by_id
    assert hasattr(curves.by_id["Circle"], "tags")
    assert hasattr(curves.by_id["Circle"], "holes")
    assert curves.by_id["Circle"].tex_size == 96
    ids = curves.recall(curves.by_id["Circle"].descriptor, 0.9, 1.1, k=3)
    assert len(ids) == 3
    # OpenCircle 应识别到孔与 ring 标签
    if "OpenCircle" in curves.by_id:
        oc = curves.by_id["OpenCircle"]
        assert oc.n_holes >= 1
        assert "ring" in oc.tags or oc.n_holes >= 1
        rings = curves.all_rings_normalized("OpenCircle")
        assert len(rings) >= 2
    again = StampCurveLibrary.prefit_directory(STAMPS, cache_dir=cache, tex_size=96)
    assert len(again.entries) >= 1


def test_score_curve_and_color() -> None:
    target = abstract_to_blocks(
        EMOJI, default_recipe_for_mode(AbstractionMode.illustration).override(stamps_dir=STAMPS, num_colors=4)
    )
    rgb = target.numpy_rgb()
    a = (target.numpy_alpha() * 255).astype(np.uint8)
    rgba = np.dstack([rgb, a])
    rep = score_prediction(rgba, target, n_layers=3, pass_sim=0.4, pass_line=0.4, pass_overall=0.3)
    assert rep.sim.score > 0.5
    assert 0.0 <= rep.sim.edge_score <= 1.0
    assert 0.0 <= rep.sim.color_score <= 1.0
    assert rep.simple > 0.5


@pytest.mark.timeout(240)
def test_approximate_emoji() -> None:
    """拟合应在时限内完成并产出 JSON 层。"""
    subset = ["Circle", "Square", "OpenCircle", "Drop", "Triangle", "HalfCircle", "Line", "Banner"]
    recipe = default_recipe_for_mode(AbstractionMode.illustration).override(
        stamps_dir=STAMPS,
        max_layers=12,
        num_colors=4,
        max_faces=12,
        pass_score=0.25,
        asset_allowlist=subset,
        enable_special_fx=False,
        refine=True,
        seed=0,
        use_cuda=torch.cuda.is_available(),
        n_particles=64,
    )
    recipe = recipe.model_copy(update={"match": recipe.match.model_copy(update={"recall_k": 8, "refine_iters": 1})})
    result = approximate_image(EMOJI, recipe, n_particles=64)
    assert len(result.document) >= 1
    assert len(result.document) <= 12
    assert result.elapsed_sec < 200.0
    assert result.score.n_layers == len(result.document)
    assert result.stop_reason
    assert result.blocks_found >= 1
    assert result.k_used == 4
    assert result.preview_rgb is not None
    assert 0.0 <= result.score.sim.edge_score <= 1.0


def test_region_primitives_extract() -> None:

    labels, palette, alpha, _, _, _ = planarize_image(
        EMOJI,
        ImageProcessorConfig(num_colors=4, use_cuda=False),
        mode=AbstractionMode.illustration,
    )
    pmap = build_planar_map(labels, palette, alpha, min_area_frac=0.01)
    graph = planar_map_to_region_graph(pmap)
    if not graph.regions:
        pytest.skip("无区域")
    prims = extract_primitives_for_region(graph.regions[0], depth=0, eps_arc=0.06)
    assert isinstance(prims, list)
