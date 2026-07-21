"""同色多图章并集覆盖。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bf_emblem_creator.approx.curves import fit_mask_contour_high_precision
from bf_emblem_creator.approx.regions import Region
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
from bf_emblem_creator.approx.torch_render import TorchStampRenderer
from bf_emblem_creator.approx.union_cover import (
    cover_region_with_union_stamps,
    coverage_stats,
    union_mask_of_layers,
)
from bf_emblem_creator.stamps import StampLibrary

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ROOT / "assets" / "stamps"


def _disk_region(size: int = 64, r: int = 18) -> Region:
    yy, xx = np.ogrid[:size, :size]
    m = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 <= r**2
    outer, _, rs, err = fit_mask_contour_high_precision(m, resample_n=96)
    from bf_emblem_creator.approx.curves import contour_curvature_descriptor, mask_to_sdf

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


def test_coverage_stats_and_union() -> None:
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[:5, :5] = True
    b[3:, 3:] = True
    u = a | b
    cover, leak, iou = coverage_stats(u, a)
    assert cover >= 0.999
    assert leak > 0.0
    assert 0.0 < iou <= 1.0


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
    layers = cover_region_with_union_stamps(
        region,
        curves,
        ren,
        n_particles=64,
        recall_k=8,
        seed=0,
        max_stamps=3,
        min_cover=0.75,
        min_cover_gain=0.03,
        max_leak=0.55,
        refine=False,
        layer_budget=10,
    )
    assert len(layers) >= 1
    assert len(layers) <= 3
    # 同色
    assert all(str(layer.fill).lower() == region.color_hex.lower() for layer in layers)
    uni = union_mask_of_layers(layers, ren)
    cover, _leak, _iou = coverage_stats(uni, np.asarray(region.mask, dtype=bool))
    # 至少有一定覆盖
    assert cover >= 0.35
