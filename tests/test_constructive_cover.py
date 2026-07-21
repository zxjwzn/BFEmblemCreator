"""构造式同色覆盖：Γ_F 贯穿、未覆盖子弧、画布裁切提案、约束补缝。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bf_emblem_creator.approx.curves import contour_curvature_descriptor, fit_mask_contour_high_precision, mask_to_sdf
from bf_emblem_creator.approx.models import AbstractionMode
from bf_emblem_creator.approx.recipe import UnionCoverConfig, default_recipe_for_mode
from bf_emblem_creator.approx.regions import Region
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
from bf_emblem_creator.approx.torch_render import TorchStampRenderer
from bf_emblem_creator.approx.union_cover import (
    cover_region_with_union_stamps,
    coverage_stats,
    propose_canvas_clip_layers,
    resolve_target_curve,
    uncovered_curve_pts,
    union_boundary_points,
    union_mask_of_layers,
)
from bf_emblem_creator.stamps import StampLibrary

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ROOT / "assets" / "stamps"


def _disk_region(size: int = 64, r: int = 18) -> Region:
    yy, xx = np.ogrid[:size, :size]
    m = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 <= r**2
    outer, _, rs, err = fit_mask_contour_high_precision(m, resample_n=96)
    ys, xs = np.where(m)
    return Region(
        region_id=1,
        color_hex="#E8C040",
        color_rgb=(232, 192, 64),
        area_frac=float(m.mean()),
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        mask=m,
        contour=outer,
        contour_resampled=rs,
        descriptor=contour_curvature_descriptor(rs),
        sdf=mask_to_sdf(m),
        depth=0,
        centroid=(float(xs.mean()), float(ys.mean())),
        contour_area_rel_err=float(err),
    )


def _l_region(size: int = 64) -> Region:
    """L 形：单章难盖满，促发同色多章。"""
    m = np.zeros((size, size), dtype=bool)
    m[:, : size // 3] = True
    m[size * 2 // 3 :, :] = True
    outer, _, rs, err = fit_mask_contour_high_precision(m, resample_n=96)
    ys, xs = np.where(m)
    return Region(
        region_id=2,
        color_hex="#DC2828",
        color_rgb=(220, 40, 40),
        area_frac=float(m.mean()),
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        mask=m,
        contour=outer,
        contour_resampled=rs,
        descriptor=contour_curvature_descriptor(rs),
        sdf=mask_to_sdf(m),
        depth=0,
        centroid=(float(xs.mean()), float(ys.mean())),
        contour_area_rel_err=float(err),
    )


def test_union_cover_config_defaults_constructive() -> None:
    """配方默认对齐构造式：并集 + 裁切 + 补缝开启。"""
    cfg = UnionCoverConfig()
    assert cfg.min_cover >= 0.88
    assert cfg.max_stamps_per_region >= 3
    assert cfg.enable_canvas_clip is True
    assert cfg.enable_constrained_gap_fill is True
    assert cfg.enable_occlusion_carve is True
    r = default_recipe_for_mode(AbstractionMode.illustration)
    assert r.match.union_cover.min_cover >= 0.88


def test_resolve_target_curve_prefers_external_gamma() -> None:
    reg = _disk_region(32, 10)
    gamma = np.stack([np.linspace(0, 20, 30), np.zeros(30)], axis=1)
    got = resolve_target_curve(reg, gamma)
    assert len(got) == 30
    assert float(np.linalg.norm(got[0] - gamma[0])) < 1e-9


def test_uncovered_curve_pts_keeps_far_points() -> None:
    """∂U 只盖住一段时，远端点应保留在 uncovered 中。"""
    gamma = np.stack([np.linspace(0, 100, 50), np.zeros(50)], axis=1)
    ub = np.stack([np.linspace(0, 20, 10), np.zeros(10)], axis=1)
    keep = uncovered_curve_pts(gamma, ub, thr_px=3.0, min_keep=6)
    assert len(keep) >= 6
    assert float(keep[:, 0].mean()) > 30.0


def test_propose_canvas_clip_out_of_canvas() -> None:
    """裁切提案必须包含伸出画布的中心或超画布尺寸（合法构造）。"""
    if not STAMPS.is_dir():
        return
    lib = StampLibrary(STAMPS)
    curves = StampCurveLibrary.build(
        lib,
        ["Circle", "Square", "Oval"],
        tex_size=64,
        cache_dir=None,
        force_refit=True,
        resample_n=48,
    )
    reg = _disk_region(64, 16)
    props = propose_canvas_clip_layers(reg, curves, fill=reg.color_hex, seed=0, max_proposals=12, canvas_size=64)
    assert len(props) >= 1
    ok = any(
        layer.left < -1 or layer.left > 65 or layer.top < -1 or layer.top > 65 or layer.width > 70 or layer.height > 70
        for layer in props
    )
    assert ok, "应有出画布/超画布裁切提案"


def test_cover_region_allows_multiple_same_fill() -> None:
    """并集覆盖可返回多枚同 fill 层（或单枚若已够）。"""
    if not STAMPS.is_dir():
        return
    lib = StampLibrary(STAMPS)
    device = torch.device("cpu")
    ren = TorchStampRenderer(lib, canvas_size=64, stamp_tex_size=64, device=device)
    curves = StampCurveLibrary.build(
        lib,
        ["Circle", "Square", "HalfCircle", "Drop", "Oval"],
        tex_size=64,
        cache_dir=None,
        force_refit=True,
        resample_n=64,
    )
    region = _disk_region(64, 18)
    gamma = np.asarray(region.contour_resampled, dtype=np.float64)
    layers = cover_region_with_union_stamps(
        region,
        curves,
        ren,
        target_curve_pts=gamma,
        n_particles=64,
        recall_k=8,
        seed=0,
        max_stamps=3,
        min_cover=0.75,
        min_cover_gain=0.03,
        max_leak=0.55,
        refine=False,
        layer_budget=10,
        enable_canvas_clip=True,
    )
    assert len(layers) >= 1
    assert len(layers) <= 3
    assert all(str(layer.fill).lower() == region.color_hex.lower() for layer in layers)
    uni = union_mask_of_layers(layers, ren)
    cover, _leak, _iou = coverage_stats(uni, np.asarray(region.mask, dtype=bool))
    assert cover >= 0.35


def test_cover_l_shape_may_use_multiple_stamps() -> None:
    """L 形在较高 cover 要求下允许同色多章。"""
    if not STAMPS.is_dir():
        return
    lib = StampLibrary(STAMPS)
    device = torch.device("cpu")
    ren = TorchStampRenderer(lib, canvas_size=64, stamp_tex_size=64, device=device)
    curves = StampCurveLibrary.build(
        lib,
        ["Circle", "Square", "HalfCircle", "Drop", "Oval", "Rectangle"],
        tex_size=64,
        cache_dir=None,
        force_refit=True,
        resample_n=64,
    )
    region = _l_region(64)
    gamma = np.asarray(region.contour, dtype=np.float64)
    layers = cover_region_with_union_stamps(
        region,
        curves,
        ren,
        target_curve_pts=gamma,
        n_particles=80,
        recall_k=12,
        seed=1,
        max_stamps=4,
        min_cover=0.9,
        min_cover_gain=0.02,
        max_leak=0.45,
        refine=False,
        layer_budget=12,
        enable_canvas_clip=True,
        max_boundary_chamfer=12.0,
    )
    assert len(layers) >= 1
    assert all(str(layer.fill).lower() == region.color_hex.lower() for layer in layers)
    uni = union_mask_of_layers(layers, ren)
    cover, _, _ = coverage_stats(uni, np.asarray(region.mask, dtype=bool))
    assert cover >= 0.25
    ub = union_boundary_points(uni)
    assert len(ub) >= 4


def test_second_stamp_curve_from_gamma_not_moore() -> None:
    """未覆盖子弧必须来自 Γ_F 子集。"""
    gamma = np.stack([np.linspace(10, 50, 40), np.full(40, 20.0)], axis=1)
    ub = np.stack([np.linspace(10, 25, 15), np.full(15, 20.0)], axis=1)
    sub = uncovered_curve_pts(gamma, ub, thr_px=2.0, min_keep=8)
    assert sub[:, 0].min() >= 9.0
    assert sub[:, 0].max() <= 51.0
    assert float(sub[:, 0].mean()) > 25.0
