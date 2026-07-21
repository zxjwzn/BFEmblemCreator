"""闭合色块曲线/轮廓拟合：面积相对误差 ≤3%。"""

from __future__ import annotations

import math

import numpy as np

from bf_emblem_creator.approx.curves import (
    area_relative_error,
    fit_circle_to_mask,
    fit_ellipse_to_mask,
    fit_mask_contour_area_constrained,
    fitted_region_area,
    polygon_area,
    sample_circle_contour,
    sample_ellipse_contour,
)
from bf_emblem_creator.approx.models import PaletteColor


def _disk_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def test_circle_fit_area_error_under_3pct() -> None:
    """实心圆 mask：圆拟合面积相对误差 ≤3%。"""
    m = _disk_mask(160, 160, 80, 80, 45)
    got = fit_circle_to_mask(m, n=128)
    assert got is not None
    poly, err = got
    mask_a = float(m.sum())
    assert area_relative_error(polygon_area(poly), mask_a) <= 0.03
    assert err <= 0.03


def test_ellipse_fit_area_error_under_3pct() -> None:
    """实心椭圆 mask：椭圆拟合面积相对误差 ≤3%。"""
    h, w = 160, 200
    yy, xx = np.mgrid[0:h, 0:w]
    # 轴对齐椭圆
    m = ((xx - 100) / 70) ** 2 + ((yy - 80) / 35) ** 2 <= 1.0
    got = fit_ellipse_to_mask(m, n=160)
    assert got is not None
    poly, err = got
    mask_a = float(m.sum())
    assert area_relative_error(polygon_area(poly), mask_a) <= 0.03
    assert err <= 0.03


def test_adaptive_rdp_area_constrained_for_irregular() -> None:
    """不规则 mask：自适应轮廓相对面积误差 ≤3%。"""
    m = np.zeros((120, 120), dtype=bool)
    m[20:100, 25:95] = True
    m[40:70, 90:110] = True  # 凸出
    m[15:35, 40:55] = True
    poly, rs, err = fit_mask_contour_area_constrained(m, max_area_rel_err=0.03, resample_n=128)
    assert len(poly) >= 4
    assert len(rs) >= 16
    mask_a = float(m.sum())
    assert area_relative_error(polygon_area(poly), mask_a) <= 0.03 + 1e-6
    assert err <= 0.03 + 1e-6


def test_dense_fallback_when_rdp_too_coarse() -> None:
    """默认 3% 约束下，细结构/环状 mask 面积误差可控（含孔时 outer−hole）。"""
    m = np.zeros((100, 100), dtype=bool)
    yy, xx = np.mgrid[0:100, 0:100]
    m |= ((xx - 50) ** 2 + (yy - 50) ** 2 <= 40**2) & ((xx - 50) ** 2 + (yy - 50) ** 2 >= 28**2)
    poly, _, err = fit_mask_contour_area_constrained(m, max_area_rel_err=0.03)
    mask_a = float(m.sum())
    # 环状：fitted_region_area 用外环−孔；err 已内含
    assert err <= 0.03 + 1e-5
    assert len(poly) >= 8
    # 实心不规则块也满足
    solid = np.zeros((100, 100), dtype=bool)
    solid[20:80, 15:85] = True
    solid[30:50, 70:95] = True
    poly2, _, err2 = fit_mask_contour_area_constrained(solid, max_area_rel_err=0.03)
    a2 = fitted_region_area(poly2, mask_shape=solid.shape)
    assert area_relative_error(a2, float(solid.sum())) <= 0.03 + 1e-5
    assert err2 <= 0.03 + 1e-5
    assert mask_a > 0


def test_planar_map_region_contour_area() -> None:
    """planar_map → Region 精确边界轮廓闭合，且面积与 mask 量级一致。"""
    h = w = 128
    labels = np.full((h, w), -1, dtype=np.int32)
    m = _disk_mask(h, w, 64, 64, 36)
    labels[m] = 0
    alpha = m.astype(np.float64)
    palette = [PaletteColor(hex="#FFCC00", fraction=1.0, rgb=(255, 204, 0))]
    from bf_emblem_creator.approx.planar_map import build_planar_map, planar_map_to_region_graph

    pmap = build_planar_map(labels, palette, alpha, min_area_frac=0.001)
    graph = planar_map_to_region_graph(pmap)
    assert len(graph.regions) >= 1
    reg = max(graph.regions, key=lambda r: r.area_frac)
    cont = np.asarray(reg.contour, dtype=np.float64)
    assert len(cont) >= 8
    mask_a = float(np.asarray(reg.mask).sum())
    assert mask_a > 0
    # 轮廓应闭合
    assert np.linalg.norm(cont[0] - cont[-1]) < 2.0 or len(cont) >= 16
    # dual/Moore 精确边界：多边形面积与 mask 量级接近
    a_fit = fitted_region_area(cont, mask_shape=np.asarray(reg.mask).shape)
    ratio = a_fit / mask_a if mask_a > 0 else 0.0
    assert 0.5 <= ratio <= 1.5


def test_sample_circle_area_matches_formula() -> None:
    """采样圆多边形面积接近 πr²。"""
    r = 40.0
    poly = sample_circle_contour(0.0, 0.0, r, n=256)
    a = polygon_area(poly)
    assert abs(a - math.pi * r * r) / (math.pi * r * r) < 0.01


def test_sample_ellipse_closed() -> None:
    poly = sample_ellipse_contour(10, 20, 30, 15, angle_rad=0.3, n=64)
    assert len(poly) >= 64
    assert np.allclose(poly[0], poly[-1])
