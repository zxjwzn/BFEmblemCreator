"""层预算工具与图章匹配冒烟（ModeRecipe）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bf_emblem_creator.approx.contour_arcs import PrimitiveType, fit_segment_primitive
from bf_emblem_creator.approx.match_curve import low_complexity_assets, region_simplicity
from bf_emblem_creator.approx.models import AbstractionMode
from bf_emblem_creator.approx.pipeline import approximate_image
from bf_emblem_creator.approx.recipe import default_recipe_for_mode
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary, geometric_complexity
from bf_emblem_creator.stamps import StampLibrary

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ROOT / "assets" / "stamps"
EMOJI = ROOT / "examples" / "smile.png"


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


def test_exact_free_primitive_for_disk_contour() -> None:
    """圆轮廓在精确描边路径下为 free hard，不替换为圆基元。"""
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    circle = np.stack([80 + 40 * np.cos(t), 80 + 40 * np.sin(t)], axis=1)
    circle = np.vstack([circle, circle[:1]])
    ptype, params, res, hard = fit_segment_primitive(circle, eps_arc=0.06)
    assert ptype == PrimitiveType.free
    assert hard
    assert res == 0.0
    assert params == {}


def test_low_complexity_assets_exist() -> None:
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
def test_approximate_num_colors_smoke() -> None:
    """严格 num_colors 路径：能产出层与预览。"""
    subset = ["Circle", "Square", "OpenCircle", "Drop", "HalfCircle", "Triangle", "Line", "Banner"]
    recipe = default_recipe_for_mode(AbstractionMode.illustration).override(
        stamps_dir=STAMPS,
        max_layers=40,
        num_colors=4,
        max_faces=12,
        pass_score=0.2,
        asset_allowlist=subset,
        enable_special_fx=False,
        refine=True,
        seed=0,
        use_cuda=torch.cuda.is_available(),
        n_particles=64,
    )
    recipe = recipe.model_copy(
        update={"match": recipe.match.model_copy(update={"recall_k": 8, "refine_iters": 1, "prefer_primitive_seed": True})}
    )
    result = approximate_image(EMOJI, recipe, n_particles=64)
    assert len(result.document) >= 1
    assert len(result.document) <= 40
    assert result.k_used == 4
