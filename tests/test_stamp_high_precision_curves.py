"""图章/概括高精度曲线：无折线简化、面积约束密采样。"""

from __future__ import annotations

import numpy as np

from bf_emblem_creator.approx.contour_arcs import PrimitiveType, fit_segment_primitive
from bf_emblem_creator.approx.curves import (
    fit_mask_contour_area_constrained,
    fit_mask_contour_high_precision,
)


def test_high_precision_contour_no_rdp_collapse() -> None:
    """圆 mask 高精度轮廓应保持足够采样密度，且面积误差可控。"""
    h = w = 64
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - 32) ** 2 + (yy - 32) ** 2 <= 20**2
    outer, holes, rs, err = fit_mask_contour_high_precision(mask, resample_n=128)
    assert len(holes) == 0
    assert len(outer) >= 64
    assert len(rs) >= 64
    assert err <= 0.08
    assert len(outer) > 16


def test_area_constrained_is_high_precision() -> None:
    """面积约束入口即高精度路径。"""
    h = w = 48
    mask = np.zeros((h, w), dtype=bool)
    mask[10:38, 8:40] = True
    cont, rs, err = fit_mask_contour_area_constrained(mask, resample_n=96)
    assert len(cont) >= 32
    assert len(rs) >= 32
    assert err <= 0.1


def test_fit_segment_always_free_exact() -> None:
    """精确描边：fit_segment 始终 free，禁止 line/圆概括替换。"""
    t = np.linspace(0, 1, 20)
    seg = np.stack([t * 30, t * 10 + 2], axis=1)
    ptype, _params, _res, hard = fit_segment_primitive(seg, eps_arc=0.02)
    assert ptype == PrimitiveType.free
    assert hard
    assert ptype != PrimitiveType.line
    assert ptype != PrimitiveType.circle_arc
    assert ptype != PrimitiveType.ellipse_arc


def test_annulus_mask_has_hole() -> None:
    """合成圆环：高精度拟合应检出至少 1 个内孔。"""
    h = w = 96
    yy, xx = np.ogrid[:h, :w]
    mask = ((xx - 48) ** 2 + (yy - 48) ** 2 <= 36**2) & ((xx - 48) ** 2 + (yy - 48) ** 2 >= 16**2)
    outer, holes, _rs, err = fit_mask_contour_high_precision(mask, resample_n=128, use_cc_holes=True)
    assert len(outer) >= 48
    assert len(holes) >= 1
    assert err <= 0.2


def test_solid_disk_no_hole() -> None:
    """实心圆：不应检出内孔。"""
    h = w = 64
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - 32) ** 2 + (yy - 32) ** 2 <= 20**2
    _outer, holes, _rs, err = fit_mask_contour_high_precision(mask, resample_n=96, use_cc_holes=True)
    assert len(holes) == 0
    assert err <= 0.1


def test_stamp_curve_high_precision_build(tmp_path) -> None:
    """图章曲线库：主动预计算写入缓存后可直接命中加载。"""
    from pathlib import Path

    from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
    from bf_emblem_creator.stamps import StampLibrary

    root = Path(__file__).resolve().parents[1]
    stamps = root / "assets" / "stamps"
    if not stamps.is_dir():
        return
    lib = StampLibrary(stamps)
    cache = tmp_path / "curves"
    curves = StampCurveLibrary.build(
        lib,
        ["Circle", "Square", "OpenCircle"],
        tex_size=96,
        cache_dir=cache,
        force_refit=True,
        resample_n=128,
        workers=2,
    )
    assert "Circle" in curves.by_id
    circ = curves.by_id["Circle"]
    assert len(circ.contour_px) >= 64
    assert circ.contour_area_rel_err <= 0.15
    assert circ.n_holes == 0
    if "OpenCircle" in curves.by_id:
        oc = curves.by_id["OpenCircle"]
        assert oc.n_holes >= 1
        assert len(oc.holes_px) >= 1
        assert len(curves.all_rings_normalized("OpenCircle")) >= 2
    # 再次 build 应完整命中缓存（不 force）
    again = StampCurveLibrary.build(
        lib,
        ["Circle", "Square", "OpenCircle"],
        tex_size=96,
        cache_dir=cache,
        force_refit=False,
        resample_n=128,
        workers=2,
    )
    assert set(again.by_id) == set(curves.by_id)
    # 缺一个 npz 时只补算缺失，不因 meta 版本废全库
    (cache / "Square.npz").unlink(missing_ok=True)
    partial = StampCurveLibrary.build(
        lib,
        ["Circle", "Square"],
        tex_size=96,
        cache_dir=cache,
        force_refit=False,
        resample_n=128,
        workers=2,
    )
    assert "Circle" in partial.by_id and "Square" in partial.by_id
    assert (cache / "Square.npz").is_file()
