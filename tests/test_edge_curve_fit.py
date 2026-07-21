"""共享边宏观贝塞尔拟合、chain 合并与模式默认。"""

from __future__ import annotations

import numpy as np

from bf_emblem_creator.approx.depth_order import EdgeRole
from bf_emblem_creator.approx.edge_curve_fit import (
    _chain_edges_same_pair,
    _macro_corner_anchors,
    fit_polyline_bezier,
    should_curve_fit_policy,
    simplify_planar_map_curves,
)
from bf_emblem_creator.approx.label_field import build_label_field
from bf_emblem_creator.approx.models import AbstractionMode
from bf_emblem_creator.approx.planar_map import SharedEdge, build_planar_map
from bf_emblem_creator.approx.recipe import BoundaryPolicy, default_recipe_for_mode


def test_default_mode_is_illustration() -> None:
    r = default_recipe_for_mode(AbstractionMode.illustration)
    assert r.mode == AbstractionMode.illustration
    assert "auto" not in AbstractionMode.__members__


def test_pixel_disables_curve_fit_policy() -> None:
    assert should_curve_fit_policy(BoundaryPolicy.dense.value) is False
    assert should_curve_fit_policy(BoundaryPolicy.curve_fit.value) is True
    r = default_recipe_for_mode(AbstractionMode.pixel)
    assert r.region.boundary_policy == BoundaryPolicy.dense


def _stair_quarter_circle(n_steps: int = 40) -> np.ndarray:
    """四分之一圆的 dual 风格阶梯折线。"""
    pts: list[list[float]] = [[float(n_steps), 0.0]]
    x, y = float(n_steps), 0.0
    for i in range(n_steps * 2):
        t = (i + 1) / (n_steps * 2) * (np.pi / 2)
        tx = n_steps * np.cos(t)
        ty = n_steps * np.sin(t)
        while x > tx + 0.5:
            x -= 1.0
            pts.append([x, y])
        while y < ty - 0.5:
            y += 1.0
            pts.append([x, y])
    if pts[-1] != [0.0, float(n_steps)]:
        pts.append([0.0, float(n_steps)])
    return np.asarray(pts, dtype=np.float64)


def test_macro_anchors_ignore_pixel_stairs() -> None:
    stair = _stair_quarter_circle(30)
    anchors = _macro_corner_anchors(
        stair,
        corner_deg=50.0,
        closed=False,
        smooth_radius_px=8.0,
        min_anchor_spacing_px=10.0,
    )
    assert anchors[0] == 0
    assert anchors[-1] == len(stair) - 1
    assert len(anchors) < max(6, len(stair) // 4)


def test_short_dense_edge_fits_by_arc_length() -> None:
    """顶点数 ≤ max_vertices 但弧长足够时仍应拟合。"""
    line = np.stack([np.arange(20, dtype=np.float64), np.zeros(20)], axis=1)
    for i in range(1, 19, 2):
        line[i, 1] = 1.0
    fitted, status = fit_polyline_bezier(
        line,
        max_vertices=24,
        min_arc_length_px=6.0,
        line_flat_eps_px=2.0,
        corner_deg=50.0,
        samples_per_seg=4,
        smooth_radius_px=5.0,
        min_anchor_spacing_px=8.0,
    )
    assert status == "fitted"
    assert len(fitted) < len(line)


def test_stair_arc_is_fitted_lossy() -> None:
    stair = _stair_quarter_circle(40)
    fitted, status = fit_polyline_bezier(
        stair,
        max_vertices=8,
        min_arc_length_px=6.0,
        line_flat_eps_px=2.0,
        corner_deg=50.0,
        samples_per_seg=6,
        smooth_radius_px=8.0,
        min_anchor_spacing_px=10.0,
    )
    assert status == "fitted"
    assert len(fitted) < len(stair) * 0.5


def test_macro_l_corner_is_kept() -> None:
    horiz = np.stack([np.arange(0, 41, dtype=np.float64), np.zeros(41)], axis=1)
    vert = np.stack([np.full(40, 40.0), np.arange(1, 41, dtype=np.float64)], axis=1)
    lpoly = np.vstack([horiz, vert])
    fitted, status = fit_polyline_bezier(
        lpoly,
        max_vertices=8,
        min_arc_length_px=6.0,
        line_flat_eps_px=2.0,
        corner_deg=50.0,
        samples_per_seg=6,
        smooth_radius_px=6.0,
        min_anchor_spacing_px=8.0,
    )
    assert status in {"fitted", "skip"}
    pts = fitted if not (len(fitted) > 1 and np.allclose(fitted[0], fitted[-1])) else fitted[:-1]
    d = np.linalg.norm(pts - np.array([40.0, 0.0]), axis=1).min()
    assert d <= 3.0


def test_chain_edges_same_pair_connects() -> None:
    e0 = SharedEdge(
        id=0,
        v0=0,
        v1=1,
        left_face=0,
        right_face=1,
        polyline=np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float64),
        length=20.0,
        role=EdgeRole.shape_boundary,
    )
    e1 = SharedEdge(
        id=1,
        v0=1,
        v1=2,
        left_face=0,
        right_face=1,
        polyline=np.array([[20.0, 0.0], [30.0, 5.0], [40.0, 10.0]], dtype=np.float64),
        length=25.0,
        role=EdgeRole.shape_boundary,
    )
    chains = _chain_edges_same_pair([e0, e1])
    assert len(chains) == 1
    assert len(chains[0]) == 2


def test_simplify_planar_map_reduces_vertices() -> None:
    from bf_emblem_creator.approx.planar_map import assert_planar_map_valid

    size = 64
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, : size // 2] = (255, 0, 0)
    rgb[:, size // 2 :] = (0, 0, 255)
    alpha = np.ones((size, size), dtype=np.float64)
    labels, palette, gf, _ = build_label_field(rgb, alpha, num_colors=2, mrf_iters=2, enforce_no_gap=True)
    pmap = build_planar_map(labels, palette, alpha, gap_frac=gf)
    assert_planar_map_valid(pmap)
    n_before = sum(len(np.asarray(e.polyline)) for e in pmap.edges)
    rep = simplify_planar_map_curves(pmap)
    assert_planar_map_valid(pmap)
    n_after = sum(len(np.asarray(e.polyline)) for e in pmap.edges)
    assert rep.edges_total >= 1
    assert n_after < n_before
    assert rep.edges_fitted >= 1 or rep.edges_skipped >= 1
