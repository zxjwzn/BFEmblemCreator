"""由简入繁 / 弧基元主路径相关单元测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bf_emblem_creator.approx.budget_loop import BudgetState, initial_k, is_coarse_phase, next_k
from bf_emblem_creator.approx.contour_arcs import PrimitiveType, fit_segment_primitive
from bf_emblem_creator.approx.match_curve import low_complexity_assets, region_simplicity
from bf_emblem_creator.approx.models import ApproxConfig
from bf_emblem_creator.approx.pipeline import approximate_image
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary, geometric_complexity
from bf_emblem_creator.stamps import StampLibrary

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ROOT / "assets" / "stamps"
EMOJI = ROOT / "examples" / "😄.png"


def test_initial_k_is_coarse_not_max() -> None:
    """由简入繁：起始 K 取小者。"""
    cfg = ApproxConfig(k_start=4, palette_k=10, k_max=12)
    assert initial_k(cfg) == 4
    assert is_coarse_phase(4, cfg)
    assert is_coarse_phase(6, cfg)
    assert not is_coarse_phase(10, cfg)


def test_next_k_requires_layer_budget() -> None:
    """层已接近上限时不再涨 K。"""
    cfg = ApproxConfig(k_start=4, delta_k=2, k_max=12, k_max_iters=4, n_margin=6, max_layers=40, pass_score=0.48)
    st = BudgetState(k=4, iteration=1, n_layers=5, score=0.2)
    assert next_k(st, cfg) == 6
    st_full = BudgetState(k=4, iteration=1, n_layers=36, score=0.2)
    assert next_k(st_full, cfg) is None
    st_pass = BudgetState(k=4, iteration=1, n_layers=5, score=0.9)
    assert next_k(st_pass, cfg) is None


def test_region_simplicity_high_for_disk() -> None:
    """实心圆块简洁度应高。"""
    s = region_simplicity(circ=0.9, elong=1.1, fill_ratio=0.85, area_frac=0.2)
    assert s >= 0.65
    s2 = region_simplicity(circ=0.3, elong=4.0, fill_ratio=0.3, area_frac=0.02)
    assert s2 < s


def test_geometric_complexity_low_for_solid_disk() -> None:
    c = geometric_complexity(circularity=0.95, elongation=1.05, n_holes=0, area_frac=0.7)
    assert c < 0.35
    c2 = geometric_complexity(circularity=0.3, elongation=3.0, n_holes=2, area_frac=0.3)
    assert c2 > c


def test_circle_arc_primitive_fits_disk_contour() -> None:
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    circle = np.stack([80 + 40 * np.cos(t), 80 + 40 * np.sin(t)], axis=1)
    circle = np.vstack([circle, circle[:1]])
    ptype, params, res, hard = fit_segment_primitive(circle, eps_arc=0.06)
    assert ptype in {PrimitiveType.circle_arc, PrimitiveType.ellipse_arc}
    assert res < 0.08
    assert not hard
    assert "r" in params or "a" in params


def test_low_complexity_assets_prefer_simple() -> None:
    lib = StampLibrary(STAMPS)
    curves = StampCurveLibrary.build(
        lib,
        ["Circle", "Square", "OpenCircle", "Star", "Line", "Triangle"],
        tex_size=64,
        force_refit=True,
    )
    ids = low_complexity_assets(curves, circ=0.9, k=4, max_complexity=0.45, min_fill=0.1)
    assert len(ids) >= 1
    for aid in ids:
        assert curves.by_id[aid].complexity <= 0.45 + 1e-6


@pytest.mark.timeout(240)
def test_approximate_coarse_to_fine_smoke() -> None:
    """由简入繁路径：小 K 起步，能产出层与预览。"""
    subset = ["Circle", "Square", "OpenCircle", "Drop", "HalfCircle", "Triangle", "Line", "Banner"]
    cfg = ApproxConfig(
        stamps_dir=STAMPS,
        max_layers=12,
        palette_k=4,
        k_start=4,
        k_max=8,
        k_max_iters=2,
        delta_k=2,
        n_margin=4,
        coarse_max_regions=6,
        pass_score=0.2,
        recall_k=8,
        refine=True,
        refine_iters=1,
        seed=0,
        stamp_subset=subset,
        enable_special_fx=False,
        prefer_primitive_seed=True,
    )
    result = approximate_image(EMOJI, cfg, n_particles=64, use_cuda=torch.cuda.is_available())
    assert len(result.document) >= 1
    assert len(result.document) <= 12
    assert result.k_used >= 4
    assert result.preview_rgb is not None
    assert result.log_lines
    # 粗阶段日志应存在
    assert any("K=4" in line or "K=6" in line or "K=8" in line for line in result.log_lines)
