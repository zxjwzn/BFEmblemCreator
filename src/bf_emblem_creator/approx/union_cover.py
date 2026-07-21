"""同色多图章并集覆盖：多枚相同 fill 的图章交叠，可见形状为其 mask 并集。"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from bf_emblem_creator.approx.contour_arcs import ArcPrimitive
from bf_emblem_creator.approx.curves import (
    contour_curvature_descriptor,
    fit_mask_contour_high_precision,
    mask_to_sdf,
    resample_closed_contour,
)
from bf_emblem_creator.approx.match_curve import match_region_with_particles, refine_layer_particles
from bf_emblem_creator.approx.regions import Region
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
from bf_emblem_creator.approx.torch_render import TorchStampRenderer, stamp_layer_to_dict
from bf_emblem_creator.models import StampLayer

BoolArr = NDArray[np.bool_]
FloatArr = NDArray[np.floating]
U8Arr = NDArray[np.uint8]


def _layer_mask(
    layer: StampLayer,
    renderer: TorchStampRenderer,
    *,
    thr: float = 0.5,
) -> BoolArr:
    """单层图章在画布上的硬 mask。"""
    d = stamp_layer_to_dict(layer)
    left = torch.tensor([d["left"]], dtype=torch.float32, device=renderer.device)
    top = torch.tensor([d["top"]], dtype=torch.float32, device=renderer.device)
    width = torch.tensor([d["width"]], dtype=torch.float32, device=renderer.device)
    height = torch.tensor([d["height"]], dtype=torch.float32, device=renderer.device)
    ang = torch.tensor([d["angle"]], dtype=torch.float32, device=renderer.device)
    m = renderer.render_batch_masks(
        str(d["asset"]),
        left=left,
        top=top,
        width=width,
        height=height,
        angle_deg=ang,
    )[0, 0]
    return (m.detach().cpu().numpy() >= thr).astype(bool)


def union_mask_of_layers(
    layers: list[StampLayer],
    renderer: TorchStampRenderer,
    *,
    thr: float = 0.5,
) -> BoolArr:
    """同色层 mask 并集（忽略 fill 差异时仅几何并集）。"""
    cs = renderer.canvas_size
    out = np.zeros((cs, cs), dtype=bool)
    for layer in layers:
        out |= _layer_mask(layer, renderer, thr=thr)
    return out


def coverage_stats(
    union: BoolArr,
    target: BoolArr,
) -> tuple[float, float, float]:
    """
    返回 (cover, leak, iou)。

    cover = |U∩T| / |T|
    leak  = |U\\T| / |U|
    """
    t = np.asarray(target, dtype=bool)
    u = np.asarray(union, dtype=bool)
    t_sum = float(t.sum()) + 1e-9
    u_sum = float(u.sum()) + 1e-9
    inter = float(np.logical_and(u, t).sum())
    cover = inter / t_sum
    leak = float(np.logical_and(u, ~t).sum()) / u_sum
    iou = inter / (float(np.logical_or(u, t).sum()) + 1e-9)
    return cover, leak, iou


def _region_from_residual(
    residual: BoolArr,
    *,
    color_hex: str,
    color_rgb: tuple[int, int, int],
    region_id: int,
    depth: int,
    canvas_area: float,
    resample_n: int = 128,
) -> Region | None:
    """由残差 mask 构造临时 Region（高精度轮廓，无折线概括）。"""
    m = np.asarray(residual, dtype=bool)
    if not m.any():
        return None
    area = float(m.sum())
    if area < 8:
        return None
    outer, _holes, rs, err = fit_mask_contour_high_precision(m, resample_n=resample_n)
    if len(outer) < 3:
        ys, xs = np.where(m)
        outer = np.array(
            [
                [float(xs.min()), float(ys.min())],
                [float(xs.max()), float(ys.min())],
                [float(xs.max()), float(ys.max())],
                [float(xs.min()), float(ys.max())],
                [float(xs.min()), float(ys.min())],
            ],
            dtype=np.float64,
        )
        rs = resample_closed_contour(outer, resample_n)
        err = 1.0
    ys, xs = np.where(m)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    desc = contour_curvature_descriptor(rs if len(rs) >= 8 else outer)
    return Region(
        region_id=region_id,
        color_hex=color_hex,
        color_rgb=color_rgb,
        area_frac=area / max(canvas_area, 1.0),
        bbox=bbox,
        mask=m,
        contour=outer,
        contour_resampled=rs,
        descriptor=desc,
        sdf=mask_to_sdf(m),
        depth=depth,
        centroid=(float(xs.mean()), float(ys.mean())),
        contour_area_rel_err=float(err),
    )


def cover_region_with_union_stamps(
    region: Region,
    curve_lib: StampCurveLibrary,
    renderer: TorchStampRenderer,
    *,
    primitives: list[ArcPrimitive] | None = None,
    target_curve_pts: FloatArr | None = None,
    n_particles: int = 256,
    recall_k: int = 48,
    seed: int = 0,
    prefer_primitive_seed: bool = True,
    refine: bool = True,
    refine_iters: int = 3,
    max_stamps: int = 4,
    min_cover: float = 0.82,
    min_cover_gain: float = 0.045,
    max_leak: float = 0.42,
    layer_budget: int = 40,
) -> list[StampLayer]:
    """
    用 **相同 fill** 的多枚图章覆盖色块；可见几何为各章 mask **并集**。

    贪心流程：
    1. 对当前残差（初始=区域 mask）做曲线匹配，得到一枚图章；
    2. fill 强制为区域色；并入并集；
    3. 残差 ← 残差 \\ 新章 mask；若 cover 达标或增益不足则停；
    4. 重复至 max_stamps 或层预算用尽。

    不要求一区一章；同色交叠是合法且期望的表达。
    """
    if max_stamps <= 0 or layer_budget <= 0:
        return []
    target = np.asarray(region.mask, dtype=bool)
    if not target.any():
        return []
    cs = renderer.canvas_size
    canvas_area = float(cs * cs)
    fill = region.color_hex
    placed: list[StampLayer] = []
    residual = target.copy()
    union = np.zeros_like(target)
    cover = 0.0

    for k in range(max_stamps):
        if len(placed) >= layer_budget:
            break
        if k == 0:
            match_region = region
            curve_pts = target_curve_pts
        else:
            # 残差岛 → 新目标；曲线用残差高精度轮廓
            rid = region.region_id * 1000 + k
            match_region = _region_from_residual(
                residual,
                color_hex=fill,
                color_rgb=region.color_rgb,
                region_id=rid,
                depth=region.depth,
                canvas_area=canvas_area,
            )
            if match_region is None:
                break
            curve_pts = np.asarray(match_region.contour_resampled, dtype=np.float64)
            if len(curve_pts) < 4:
                curve_pts = None

        layer = match_region_with_particles(
            match_region,
            curve_lib,
            renderer,
            primitives=primitives if k == 0 else None,
            n_particles=max(96, n_particles // (1 + k // 2)),
            recall_k=max(16, recall_k - 8 * k),
            seed=seed + k * 17,
            prefer_primitive_seed=prefer_primitive_seed and k == 0,
            target_curve_pts=curve_pts,
            min_score=0.10 if k == 0 else 0.08,
        )
        if layer is None:
            break
        # 同色：强制区域 fill（并集染色一致）
        layer = StampLayer(
            asset=layer.asset,
            opacity=1.0,
            angle=layer.angle,
            flipX=layer.flipX,
            flipY=layer.flipY,
            top=layer.top,
            left=layer.left,
            height=layer.height,
            width=layer.width,
            fill=fill,
        )
        if refine:
            layer = refine_layer_particles(
                layer,
                match_region,
                curve_lib,
                renderer,
                n=max(24, refine_iters * 12),
                seed=seed + 100 + k,
                target_curve_pts=curve_pts,
            )
            layer = StampLayer(
                asset=layer.asset,
                opacity=1.0,
                angle=layer.angle,
                flipX=layer.flipX,
                flipY=layer.flipY,
                top=layer.top,
                left=layer.left,
                height=layer.height,
                width=layer.width,
                fill=fill,
            )

        m = _layer_mask(layer, renderer)
        # 对目标的贡献
        gain = float(np.logical_and(m, residual).sum()) / (float(target.sum()) + 1e-9)
        if k > 0 and gain < min_cover_gain * 0.5:
            break
        trial_union = union | m
        new_cover, new_leak, _ = coverage_stats(trial_union, target)
        # 泄漏过大且首层以后：拒绝
        if k > 0 and new_leak > max_leak and new_cover < cover + min_cover_gain:
            break
        # 增益不足
        if k > 0 and (new_cover - cover) < min_cover_gain:
            break

        placed.append(layer)
        union = trial_union
        residual = target & ~union
        cover, _leak = new_cover, new_leak
        if cover >= min_cover:
            break
        if float(residual.sum()) < max(12.0, 0.02 * float(target.sum())):
            break

    return placed


def merge_same_fill_groups(layers: list[StampLayer]) -> list[list[int]]:
    """
    按 fill 连续分组（用于调试/日志）：返回层下标分组。

    装配顺序中相邻且同 fill 的层视为一组并集。
    """
    if not layers:
        return []
    groups: list[list[int]] = [[0]]
    for i in range(1, len(layers)):
        if str(layers[i].fill).lower() == str(layers[i - 1].fill).lower():
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups
