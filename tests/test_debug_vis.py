"""逐步调试图：主阶段输出、匹配仅通过者、几何变换一致性。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bf_emblem_creator.approx.contour_arcs import ArcPrimitive, PrimitiveType
from bf_emblem_creator.approx.debug_vis import (
    DebugVisualizer,
    stamp_layer_curve_on_canvas,
)
from bf_emblem_creator.approx.depth_order import EdgeRole, OrderedRegion
from bf_emblem_creator.approx.match_curve import transform_stamp_contour_batch
from bf_emblem_creator.approx.models import ApproxConfig
from bf_emblem_creator.approx.pipeline import approximate_image
from bf_emblem_creator.approx.regions import Region
from bf_emblem_creator.models import StampLayer

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ROOT / "assets" / "stamps"
EMOJI = ROOT / "examples" / "😄.png"


def _make_region(mask: np.ndarray) -> Region:
    """测试用区域。"""
    cont = np.array([[10.0, 10], [40, 10], [40, 40], [10, 40], [10, 10]], dtype=np.float64)
    return Region(
        region_id=0,
        color_hex="#C86414",
        color_rgb=(200, 100, 20),
        area_frac=0.2,
        bbox=(10, 10, 40, 40),
        mask=mask,
        contour=cont,
        contour_resampled=cont.copy(),
        descriptor=np.zeros(8, dtype=np.float64),
        sdf=None,
        depth=0,
        centroid=(25.0, 25.0),
    )


def test_stamp_layer_curve_matches_torch_batch() -> None:
    """numpy 调试图变换应与 match_curve GPU 批量变换一致。"""
    t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    local = np.stack([0.4 * np.cos(t), 0.3 * np.sin(t)], axis=1).astype(np.float64)
    left, top, w, h, ang = 160.0, 140.0, 80.0, 60.0, 35.0
    np_xy = stamp_layer_curve_on_canvas(local, left=left, top=top, width=w, height=h, angle_deg=ang)
    device = torch.device("cpu")
    lt = torch.from_numpy(local.astype(np.float32)).to(device)
    batch = (
        transform_stamp_contour_batch(
            lt,
            left=torch.tensor([left], device=device),
            top=torch.tensor([top], device=device),
            width=torch.tensor([w], device=device),
            height=torch.tensor([h], device=device),
            angle_deg=torch.tensor([ang], device=device),
        )[0]
        .cpu()
        .numpy()
    )
    assert np.allclose(np_xy, batch, atol=1e-4)


def test_debug_visualizer_disabled_writes_nothing(tmp_path: Path) -> None:
    """debug_dir=None 时不写盘。"""
    dbg = DebugVisualizer(None)
    assert not dbg.enabled
    assert dbg.save_rgb("x", np.zeros((8, 8, 3), dtype=np.uint8)) is None
    assert dbg.saved == []
    # 启用则写
    dbg2 = DebugVisualizer(tmp_path)
    p = dbg2.save_rgb("hello", np.zeros((8, 8, 3), dtype=np.uint8))
    assert p is not None and p.is_file()
    assert len(dbg2.saved) == 1


def test_debug_visualizer_stages_and_accepted_only(tmp_path: Path) -> None:
    """主阶段各写一张；匹配只调 save_accepted_match 时才有 match 文件。"""
    dbg = DebugVisualizer(tmp_path)
    dbg.set_k(4, 1)
    rgb = np.full((64, 64, 3), 40, dtype=np.uint8)
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[10:40, 10:40] = 0
    labels[20:30, 20:30] = 1
    dbg.save_planarized(rgb, labels, [(200, 100, 20), (20, 100, 200)])
    mask = labels == 0
    reg = _make_region(mask)
    dbg.save_regions(rgb, [reg])
    dbg.save_contours(rgb, [reg])
    dbg.save_depth_order(rgb, [OrderedRegion(region=reg, depth=0, boundary_role_default=EdgeRole.shape_boundary)])
    prim = ArcPrimitive(
        type=PrimitiveType.circle_arc,
        params={"cx": 25.0, "cy": 25.0, "r": 12.0},
        sample_points=reg.contour,
        hard=False,
        region_id=0,
        depth=0,
    )
    dbg.save_primitives(rgb, [prim])
    # 未调用 save_accepted_match → 无 match 文件
    names = [p.name for p in dbg.saved]
    assert any("planarized" in n for n in names)
    assert any("regions" in n for n in names)
    assert any("contours" in n for n in names)
    assert any("primitives" in n for n in names)
    assert not any("match_ok" in n for n in names)

    layer = StampLayer(
        asset="Circle",
        left=25.0,
        top=25.0,
        width=30.0,
        height=30.0,
        angle=0.0,
        fill="#FFCC00",
    )
    local = np.stack(
        [0.5 * np.cos(np.linspace(0, 2 * np.pi, 24)), 0.5 * np.sin(np.linspace(0, 2 * np.pi, 24))],
        axis=1,
    )
    curve = stamp_layer_curve_on_canvas(local, left=25, top=25, width=30, height=30, angle_deg=0)
    dbg.save_accepted_match(
        base_rgb=rgb,
        region=reg,
        layer=layer,
        stamp_curve_canvas=curve,
        layer_index=0,
    )
    assert any("match_ok" in p.name for p in dbg.saved)


@pytest.mark.timeout(240)
def test_approximate_with_debug_dir(tmp_path: Path) -> None:
    """管线启用 debug_dir 时写出主阶段图，且匹配图仅有 accepted。"""
    subset = ["Circle", "Square", "OpenCircle", "HalfCircle", "Line", "Triangle"]
    out_dbg = tmp_path / "dbg"
    cfg = ApproxConfig(
        stamps_dir=STAMPS,
        max_layers=8,
        palette_k=4,
        k_start=4,
        k_max=6,
        k_max_iters=2,
        delta_k=2,
        n_margin=4,
        coarse_max_regions=5,
        pass_score=0.2,
        recall_k=6,
        refine=False,
        seed=0,
        stamp_subset=subset,
        enable_special_fx=False,
        prefer_primitive_seed=True,
        debug_dir=out_dbg,
        n_particles=48,
    )
    result = approximate_image(EMOJI, cfg, n_particles=48, use_cuda=torch.cuda.is_available())
    assert out_dbg.is_dir()
    pngs = list(out_dbg.rglob("*.png"))
    assert len(pngs) >= 4
    names = " ".join(p.name for p in pngs)
    # 主阶段
    assert "planarized" in names or "source" in names
    assert "regions" in names or "contours" in names
    # 不应出现粒子失败 dump 命名
    assert "particle" not in names.lower()
    assert "reject" not in names.lower()
    assert result.debug_images
    assert all(Path(p).suffix == ".png" for p in result.debug_images)
