"""同色构造式覆盖：并集外轮廓贴合贝塞尔 + 画布裁切 + 约束补缝。

核心机制（必须用满）：
1. 同色多章 mask 并集：相接内边消失，可见外形 = ∂(∪M_i ∩ Canvas)
2. 异色上层遮挡：由装配层序体现（本模块只负责同色组）
3. 画布正方形裁切：章可伸出画布，只露一段弧

设计见 docs/stamp-constructive-matching.md。
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from bf_emblem_creator.approx.contour_arcs import ArcPrimitive
from bf_emblem_creator.approx.curves import (
    contour_curvature_descriptor,
    extract_outer_contour,
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
    """单层图章在画布上的硬 mask（渲染器已做正方形裁切）。"""
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
    """同色层 mask 并集（可见几何）。"""
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
    leak  = |U\\T| / |U|   （同色并集时允许一定溢出，供上层遮挡消化）
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


def chamfer_mean_np(a: FloatArr, b: FloatArr) -> float:
    """双向最近邻距离均值。"""
    pa = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    pb = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if len(pa) < 1 or len(pb) < 1:
        return 1e6
    if len(pa) > 128:
        pa = pa[np.linspace(0, len(pa) - 1, 128).astype(int)]
    if len(pb) > 128:
        pb = pb[np.linspace(0, len(pb) - 1, 128).astype(int)]
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def union_boundary_points(union: BoolArr, *, max_points: int = 192) -> FloatArr:
    """并集 mask 的外轮廓采样点 = 同色可见外形（内边已消失）。"""
    m = np.asarray(union, dtype=bool)
    if not m.any():
        return np.zeros((0, 2), dtype=np.float64)
    cont = extract_outer_contour(m, simplify=0.0)
    if cont is None or len(cont) < 4:
        ys, xs = np.where(m)
        return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)[:: max(1, len(xs) // max_points)]
    pts = np.asarray(cont, dtype=np.float64)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    return pts


def resolve_target_curve(region: Region, target_curve_pts: FloatArr | None) -> FloatArr:
    """曲线真源：外部 Γ_F（after_fit 共享边）优先。"""
    if target_curve_pts is not None and len(np.asarray(target_curve_pts)) >= 4:
        return np.asarray(target_curve_pts, dtype=np.float64)
    rs = np.asarray(region.contour_resampled, dtype=np.float64)
    if len(rs) >= 4:
        return rs
    cont = np.asarray(region.contour, dtype=np.float64)
    return cont if len(cont) >= 4 else np.zeros((0, 2), dtype=np.float64)


def uncovered_curve_pts(
    gamma: FloatArr,
    union_boundary: FloatArr,
    *,
    thr_px: float = 3.0,
    min_keep: int = 6,
) -> FloatArr:
    """Γ_F 上尚未被并集外轮廓解释的点 → 下一枚同色章的贴边目标。"""
    g = np.asarray(gamma, dtype=np.float64).reshape(-1, 2)
    if len(g) < 4:
        return g
    ub = np.asarray(union_boundary, dtype=np.float64).reshape(-1, 2)
    if len(ub) < 2:
        return g
    d = np.linalg.norm(g[:, None, :] - ub[None, :, :], axis=2).min(axis=1)
    keep = g[d > thr_px]
    if len(keep) < min_keep:
        order = np.argsort(-d)
        keep = g[order[: max(min_keep, min(len(g), 24))]]
    return keep


def boundary_loss(union: BoolArr, gamma: FloatArr) -> float:
    """并集可见外轮廓相对 Γ_F 的 Chamfer（越小越好）——构造评分主项。"""
    g = np.asarray(gamma, dtype=np.float64)
    if len(g) < 4:
        return 0.0
    if not np.asarray(union).any():
        return 1e3
    return chamfer_mean_np(union_boundary_points(union), g)


def _force_fill(layer: StampLayer, fill: str) -> StampLayer:
    """强制同色 fill（并集染色一致）。"""
    return StampLayer(
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


def _match_region_for_mask(
    region: Region,
    mask: BoolArr,
    *,
    region_id: int,
) -> Region | None:
    """覆盖用 residual mask；曲线目标仍由外部 Γ_F / uncovered 传入。"""
    m = np.asarray(mask, dtype=bool)
    if float(m.sum()) < 8:
        return None
    ys, xs = np.where(m)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    outer = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)
    rs = resample_closed_contour(outer, 32)
    return Region(
        region_id=region_id,
        color_hex=region.color_hex,
        color_rgb=region.color_rgb,
        area_frac=float(m.sum()) / float(m.size),
        bbox=bbox,
        mask=m,
        contour=outer,
        contour_resampled=rs,
        descriptor=contour_curvature_descriptor(np.asarray(region.contour_resampled, dtype=np.float64)),
        sdf=mask_to_sdf(m),
        depth=region.depth,
        centroid=(float(xs.mean()), float(ys.mean())),
        contour_area_rel_err=0.0,
    )


def propose_canvas_clip_layers(
    region: Region,
    curve_lib: StampCurveLibrary,
    *,
    fill: str,
    seed: int = 0,
    max_proposals: int = 16,
    canvas_size: int = 320,
) -> list[StampLayer]:
    """
    画布正方形裁切提案：大章中心出界 / 尺寸可超过画布，只露一段弧。

    这是合法构造手段，不是噪声；评分阶段用 ∂(M∩Canvas) vs Γ_F 筛选。
    """
    x0, y0, x1, y1 = region.bbox
    cx, cy = region.centroid
    bw, bh = float(max(8, x1 - x0)), float(max(8, y1 - y0))
    cs = float(canvas_size)
    prefer = ["Circle", "Oval", "Square", "HalfCircle", "Drop", "Egg", "Shield"]
    assets = [a for a in prefer if a in curve_lib.by_id] or list(curve_lib.by_id.keys())[:8]
    if not assets:
        return []
    rng = np.random.default_rng(seed + 91)
    # 中心：区内 + 出画布（四边/四角）
    offsets = [
        (cx, cy),
        (cx, -0.15 * cs),
        (cx, cs + 0.15 * cs),
        (-0.15 * cs, cy),
        (cs + 0.15 * cs, cy),
        (-0.2 * cs, -0.2 * cs),
        (cs + 0.2 * cs, -0.2 * cs),
        (-0.2 * cs, cs + 0.2 * cs),
        (cs + 0.2 * cs, cs + 0.2 * cs),
        (cx - bw * 0.8, cy),
        (cx + bw * 0.8, cy),
        (cx, cy - bh * 0.8),
        (cx, cy + bh * 0.8),
    ]
    scales = (1.0, 1.4, 1.9, 2.6, 3.5)
    angles = (0.0, 20.0, 45.0, 90.0, 160.0)
    out: list[StampLayer] = []
    for asset in assets[:5]:
        for ox, oy in offsets:
            for sc in scales:
                for ang in angles:
                    if len(out) >= max_proposals * 4:
                        break
                    if rng.random() > 0.4 and len(out) > 6:
                        continue
                    w, h = max(bw * sc, 16.0), max(bh * sc, 16.0)
                    if asset in {"Circle", "Oval"}:
                        side = max(w, h)
                        w = h = side
                    # 允许超过画布（裁切用），但别无限大
                    w = float(np.clip(w, 12.0, 2.8 * cs))
                    h = float(np.clip(h, 12.0, 2.8 * cs))
                    out.append(
                        StampLayer(
                            asset=asset,
                            opacity=1.0,
                            angle=float(ang),
                            flipX=False,
                            flipY=False,
                            top=float(oy),
                            left=float(ox),
                            height=h,
                            width=w,
                            fill=fill,
                        )
                    )
    if len(out) > max_proposals:
        idx = rng.choice(len(out), size=max_proposals, replace=False)
        out = [out[int(j)] for j in idx]
    return out


def propose_bbox_aligned_layer(
    region: Region,
    curve_lib: StampCurveLibrary,
    *,
    fill: str,
    scale: float = 1.15,
    canvas_size: int = 320,
) -> list[StampLayer]:
    """整环粗提案：多种简单实心章 + 多尺度，供并集首枚候选。"""
    x0, y0, x1, y1 = region.bbox
    cx, cy = region.centroid
    bw0, bh0 = float(max(8, x1 - x0)), float(max(8, y1 - y0))
    cs = float(canvas_size)
    out: list[StampLayer] = []
    for asset in ("Circle", "Oval", "Square", "Egg", "Shield", "Drop", "HalfCircle"):
        if asset not in curve_lib.by_id:
            continue
        for sc in (0.95, 1.1, 1.3, 1.6):
            bw, bh = bw0 * scale * sc, bh0 * scale * sc
            if asset == "Circle":
                side = max(bw, bh)
                bw = bh = side
            bw = float(np.clip(bw, 10.0, 2.2 * cs))
            bh = float(np.clip(bh, 10.0, 2.2 * cs))
            for ang in (0.0, 30.0, 90.0):
                out.append(
                    StampLayer(
                        asset=asset,
                        opacity=1.0,
                        angle=ang,
                        flipX=False,
                        flipY=False,
                        top=float(cy),
                        left=float(cx),
                        height=bh,
                        width=bw,
                        fill=fill,
                    )
                )
    return out


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
    max_stamps: int = 6,
    min_cover: float = 0.90,
    min_cover_gain: float = 0.02,
    max_leak: float = 0.45,
    layer_budget: int = 40,
    enable_canvas_clip: bool = True,
    max_boundary_chamfer: float = 5.0,
) -> list[StampLayer]:
    """
    同色构造覆盖：多枚相同 fill 的 mask **并集** 逼近色块。

    评分对象是加入候选后的 **∪ 可见外轮廓** 对 Γ_F 的贴合，不是单章单独像不像。
    同色内边消失；章可伸出画布（正方形裁切）；溢出可被上层异色消化，故 leak 只作软约束。
    """
    if max_stamps <= 0 or layer_budget <= 0:
        return []
    target = np.asarray(region.mask, dtype=bool)
    if not target.any():
        return []
    gamma_raw = resolve_target_curve(region, target_curve_pts)
    gamma: FloatArr | None = gamma_raw if len(gamma_raw) >= 4 else None
    fill = region.color_hex
    placed: list[StampLayer] = []
    residual = target.copy()
    union = np.zeros_like(target)
    cover = 0.0
    best_b_loss = 1e9
    cs = float(renderer.canvas_size)

    for k in range(max_stamps):
        if len(placed) >= layer_budget:
            break

        # 曲线：始终 Γ_F 或其未覆盖子弧（同色并集语义）
        curve_pts: FloatArr | None
        if gamma is not None:
            if k == 0 or not union.any():
                curve_pts = gamma
            else:
                curve_pts = uncovered_curve_pts(gamma, union_boundary_points(union), thr_px=2.5)
        else:
            curve_pts = None

        if k == 0:
            match_region = region
        else:
            match_region = _match_region_for_mask(region, residual, region_id=region.region_id * 1000 + k)
            if match_region is None:
                break

        candidates: list[StampLayer] = []

        # 1) 粒子（允许出画布）
        layer = match_region_with_particles(
            match_region,
            curve_lib,
            renderer,
            primitives=primitives if k == 0 else None,
            n_particles=max(128, n_particles // (1 + k // 2)),
            recall_k=max(20, recall_k - 4 * k),
            seed=seed + k * 17,
            prefer_primitive_seed=prefer_primitive_seed and k == 0,
            target_curve_pts=curve_pts,
            min_score=0.08 if k == 0 else 0.05,
        )
        if layer is not None:
            candidates.append(_force_fill(layer, fill))

        # 2) 结构化：bbox 整环 + 画布裁切（构造主力）
        if k == 0:
            candidates.extend(propose_bbox_aligned_layer(region, curve_lib, fill=fill, canvas_size=int(cs)))
            if enable_canvas_clip:
                candidates.extend(
                    propose_canvas_clip_layers(
                        region,
                        curve_lib,
                        fill=fill,
                        seed=seed,
                        max_proposals=14,
                        canvas_size=int(cs),
                    )
                )
        elif enable_canvas_clip and curve_pts is not None:
            # 后续章也可用裁切补弧
            candidates.extend(
                propose_canvas_clip_layers(
                    match_region,
                    curve_lib,
                    fill=fill,
                    seed=seed + 50 + k,
                    max_proposals=8,
                    canvas_size=int(cs),
                )
            )

        # 3) 未覆盖子弧再粒子
        if k > 0 and curve_pts is not None and len(curve_pts) >= 4:
            layer2 = match_region_with_particles(
                match_region,
                curve_lib,
                renderer,
                n_particles=max(96, n_particles // 2),
                recall_k=max(16, recall_k // 2),
                seed=seed + 333 + k,
                prefer_primitive_seed=False,
                target_curve_pts=curve_pts,
                min_score=0.04,
            )
            if layer2 is not None:
                candidates.append(_force_fill(layer2, fill))

        if not candidates:
            break

        # 精修前若干候选
        refined: list[StampLayer] = []
        for i, cand in enumerate(candidates):
            layer_i = cand
            if refine and i < 3:
                layer_i = refine_layer_particles(
                    layer_i,
                    match_region,
                    curve_lib,
                    renderer,
                    n=max(20, refine_iters * 12),
                    seed=seed + 100 + k * 5 + i,
                    target_curve_pts=curve_pts,
                )
                layer_i = _force_fill(layer_i, fill)
            refined.append(layer_i)

        # —— 选最优：并入后 ∂(∪) 贴 Γ_F 为主 ——
        best_layer: StampLayer | None = None
        best_key: tuple[float, float, float] | None = None
        best_stats: tuple[BoolArr, float, float, float] | None = None

        for cand in refined:
            m = _layer_mask(cand, renderer)
            if not m.any():
                continue
            gain = float(np.logical_and(m, residual).sum()) / (float(target.sum()) + 1e-9)
            if k == 0 and gain < 0.08:
                continue
            if k > 0 and gain < min_cover_gain * 0.35:
                continue
            trial = union | m
            new_cover, new_leak, _ = coverage_stats(trial, target)
            # 软泄漏：同色可多溢一点（上层可遮）；过疯才拒
            hard_leak = 0.72 if k == 0 else max_leak + 0.25
            if new_leak > hard_leak and new_cover < cover + min_cover_gain:
                continue
            if k > 0 and (new_cover - cover) < min_cover_gain * 0.4 and gain < min_cover_gain * 0.8:
                continue
            b_loss = boundary_loss(trial, gamma) if gamma is not None else (1.0 - new_cover) * 20.0
            # 主：边界；次：覆盖增益；轻罚泄漏（不扼杀裁切大章）
            score = b_loss + 2.2 * (1.0 - new_cover) + 0.8 * max(0.0, new_leak - 0.25)
            key = (score, -(new_cover - cover), new_leak)
            if best_key is None or key < best_key:
                best_key = key
                best_layer = cand
                best_stats = (trial, new_cover, new_leak, b_loss)

        if best_layer is None or best_stats is None:
            break

        trial_u, new_cover, _new_leak, b_loss = best_stats
        if k > 0 and new_cover < cover + 1e-4 and b_loss > best_b_loss * 0.995:
            # 无改进则停
            break

        placed.append(best_layer)
        union = trial_u
        residual = target & ~union
        cover = new_cover
        best_b_loss = b_loss

        if cover >= min_cover and (gamma is None or b_loss <= max_boundary_chamfer):
            break
        if float(residual.sum()) < max(10.0, 0.015 * float(target.sum())):
            break

    return placed


def constrained_gap_fill(
    layers: list[StampLayer],
    regions: list[Region],
    face_curves: dict[int, FloatArr],
    curve_lib: StampCurveLibrary,
    renderer: TorchStampRenderer,
    *,
    alpha: FloatArr,
    target_rgb: U8Arr,
    max_rounds: int = 6,
    max_stamps_per_gap: int = 2,
    layer_budget: int = 40,
    n_particles: int = 128,
    recall_k: int = 24,
    seed: int = 0,
    min_cover: float = 0.8,
    max_leak: float = 0.5,
) -> list[StampLayer]:
    """
    约束补缝：合成 residual 归属最近 Face；曲线仍绑 Γ_F。
    同色并集/裁切规则与主覆盖相同。
    """
    if layer_budget <= 0 or not regions:
        return list(layers)
    from bf_emblem_creator.approx.assemble import composite_layers

    out = list(layers)
    by_id = {r.region_id: r for r in regions}

    for rnd in range(max_rounds):
        if len(out) >= layer_budget:
            break
        pred_rgb, pred_a = composite_layers(out, renderer)
        err = np.linalg.norm(pred_rgb.astype(np.float64) - target_rgb.astype(np.float64), axis=2) / 255.0
        need = (np.asarray(alpha, dtype=np.float64) >= 0.5) & ((pred_a < 0.4) | (err > 0.22))
        if not need.any():
            break

        best_face: int | None = None
        best_overlap = 0
        need_mask: BoolArr | None = None
        for reg in regions:
            if max(reg.color_rgb) < 25:
                continue
            m = np.asarray(reg.mask, dtype=bool) & need
            ov = int(m.sum())
            if ov > best_overlap:
                best_overlap = ov
                best_face = reg.region_id
                need_mask = m
        if best_face is None or need_mask is None or best_overlap < 12:
            ys, xs = np.where(need)
            if len(xs) < 12:
                break
            cy, cx = float(ys.mean()), float(xs.mean())
            best_d = 1e18
            for reg in regions:
                d = (reg.centroid[0] - cx) ** 2 + (reg.centroid[1] - cy) ** 2
                if d < best_d:
                    best_d = d
                    best_face = reg.region_id
            if best_face is None:
                break
            reg0 = by_id[best_face]
            need_mask = np.asarray(reg0.mask, dtype=bool) & need
            if int(need_mask.sum()) < 12:
                need_mask = need

        reg = by_id[best_face]
        gamma = face_curves.get(best_face)
        if gamma is None or len(np.asarray(gamma)) < 4:
            gamma = resolve_target_curve(reg, None)
        m = np.asarray(need_mask, dtype=bool)
        ys, xs = np.where(m)
        if len(xs) < 8:
            break
        gap_region = Region(
            region_id=reg.region_id,
            color_hex=reg.color_hex,
            color_rgb=reg.color_rgb,
            area_frac=float(m.sum()) / float(m.size),
            bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            mask=m,
            contour=np.asarray(reg.contour, dtype=np.float64),
            contour_resampled=np.asarray(reg.contour_resampled, dtype=np.float64),
            descriptor=np.asarray(reg.descriptor, dtype=np.float64),
            sdf=mask_to_sdf(m),
            depth=reg.depth,
            centroid=(float(xs.mean()), float(ys.mean())),
        )
        remain = layer_budget - len(out)
        more = cover_region_with_union_stamps(
            gap_region,
            curve_lib,
            renderer,
            target_curve_pts=np.asarray(gamma, dtype=np.float64),
            n_particles=max(80, n_particles // 2),
            recall_k=recall_k,
            seed=seed + 2000 + rnd * 13,
            prefer_primitive_seed=False,
            refine=True,
            refine_iters=2,
            max_stamps=min(max_stamps_per_gap, remain),
            min_cover=min_cover,
            min_cover_gain=0.02,
            max_leak=max_leak,
            layer_budget=remain,
            enable_canvas_clip=True,
            max_boundary_chamfer=8.0,
        )
        if not more:
            break
        out.extend(more)
    return out


def merge_same_fill_groups(layers: list[StampLayer]) -> list[list[int]]:
    """按 fill 连续分组。"""
    if not layers:
        return []
    groups: list[list[int]] = [[0]]
    for i in range(1, len(layers)):
        if str(layers[i].fill).lower() == str(layers[i - 1].fill).lower():
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups
