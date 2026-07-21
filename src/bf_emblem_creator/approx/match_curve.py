"""曲线匹配：描述子 top-M + GPU 粒子 + Chamfer（有损），无形状路由。"""

from __future__ import annotations

import math

import numpy as np
import torch
from numpy.typing import NDArray

from bf_emblem_creator.approx.contour_arcs import ArcPrimitive, PrimitiveType
from bf_emblem_creator.approx.regions import Region
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
from bf_emblem_creator.approx.torch_render import TorchStampRenderer
from bf_emblem_creator.models import StampLayer

FloatArr = NDArray[np.floating]


def _region_geom(region: Region) -> tuple[float, float, float, float, float, float, float]:
    """cx, cy, bw, bh, angle_deg, circ, elong。"""
    x0, y0, x1, y1 = region.bbox
    mask = np.asarray(region.mask, dtype=bool)
    if mask.any():
        ys, xs = np.where(mask)
        cx, cy = float(xs.mean()), float(ys.mean())
    else:
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    bw, bh = float(max(1, x1 - x0)), float(max(1, y1 - y0))
    pts = np.asarray(region.contour_resampled, dtype=np.float64)
    ang = 0.0
    if len(pts) >= 5:
        c = pts - pts.mean(axis=0, keepdims=True)
        cov = c.T @ c / max(len(c) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main = eigvecs[:, int(np.argmax(eigvals))]
        ang = math.degrees(math.atan2(main[1], main[0])) % 360.0
    area = float(mask.sum())
    cont = np.asarray(region.contour, dtype=np.float64)
    peri = float(np.linalg.norm(np.diff(np.vstack([cont, cont[:1]]), axis=0), axis=1).sum()) if len(cont) >= 2 else 1.0
    circ = float(np.clip(4.0 * math.pi * area / (peri * peri + 1e-6), 0.0, 2.0))
    elong = max(bw / bh, bh / bw)
    return cx, cy, bw, bh, ang, circ, elong


def _init_from_primitive(prim: ArcPrimitive | None) -> tuple[float, float, float, float, float] | None:
    """由圆/椭圆基元得 s,θ,c 初值。"""
    if prim is None:
        return None
    p = prim.params
    if prim.type == PrimitiveType.circle_arc and "r" in p:
        r = float(p["r"]) * 2.2
        return float(p["cx"]), float(p["cy"]), r, r, 0.0
    if prim.type == PrimitiveType.ellipse_arc and "a" in p:
        return (
            float(p["cx"]),
            float(p["cy"]),
            float(p["a"]) * 2.2,
            float(p["b"]) * 2.2,
            math.degrees(float(p.get("angle", 0.0))) % 360.0,
        )
    return None


def chamfer_loss_torch(
    pred_pts: torch.Tensor,
    tgt_pts: torch.Tensor,
    *,
    huber_delta: float = 8.0,
) -> torch.Tensor:
    """
    批量 Chamfer（有损 Huber）。

    pred_pts: (N,M,2)，tgt_pts: (K,2) 或 (N,K,2)
    返回 (N,)
    """
    t = tgt_pts[None, ...].expand(pred_pts.shape[0], -1, -1) if tgt_pts.dim() == 2 else tgt_pts
    # (N,M,K)
    d = torch.cdist(pred_pts, t)
    # pred → tgt
    d_pt = d.min(dim=2).values
    # tgt → pred
    d_tp = d.min(dim=1).values
    d_pt = torch.where(d_pt < huber_delta, 0.5 * d_pt**2 / huber_delta, d_pt - 0.5 * huber_delta)
    d_tp = torch.where(d_tp < huber_delta, 0.5 * d_tp**2 / huber_delta, d_tp - 0.5 * huber_delta)
    return d_pt.mean(dim=1) + d_tp.mean(dim=1)


def transform_stamp_contour_batch(
    local_contour: torch.Tensor,
    *,
    left: torch.Tensor,
    top: torch.Tensor,
    width: torch.Tensor,
    height: torch.Tensor,
    angle_deg: torch.Tensor,
) -> torch.Tensor:
    """
    local_contour: (M,2) in [-0.5,0.5]；参数 (N,) → (N,M,2) 画布坐标。
    """
    # 顺时针角
    th = torch.deg2rad(angle_deg)
    c = torch.cos(th)
    s = torch.sin(th)
    sx = local_contour[None, :, 0] * width[:, None]
    sy = local_contour[None, :, 1] * height[:, None]
    x = c[:, None] * sx + s[:, None] * sy + left[:, None]
    y = -s[:, None] * sx + c[:, None] * sy + top[:, None]
    return torch.stack([x, y], dim=-1)


def _score_curve_and_cover(
    masks: torch.Tensor,
    tgt_mask: torch.Tensor,
    curve_loss: torch.Tensor,
    *,
    width: torch.Tensor,
    height: torch.Tensor,
    cs: float,
) -> torch.Tensor:
    """
    曲线损失为主，IoU/覆盖为辅（通用几何，无场景名）。

    抑制「巨大图章只蹭一截边」：高 outside、低 cover 且面积远大于目标时扣分。
    """
    pred = masks
    inter = (pred * tgt_mask[None]).sum(dim=(1, 2))
    pred_sum = pred.sum(dim=(1, 2)).clamp_min(1e-6)
    tgt_sum = tgt_mask.sum().clamp_min(1e-6)
    iou = inter / (pred_sum + tgt_sum - inter).clamp_min(1e-6)
    cover = inter / tgt_sum
    outside = (pred * (1.0 - tgt_mask[None])).sum(dim=(1, 2)) / pred_sum
    curve_score = torch.exp(-curve_loss / 12.0)
    area_ratio = (width * height / (cs * cs)).clamp(0.02, 50.0)
    # 适度尺度奖励（完整覆盖时），过大且溢出则惩罚
    scale_term = 0.03 * torch.log(area_ratio.clamp(0.2, 4.0))
    giant_penalty = 0.55 * outside * torch.log(area_ratio.clamp_min(1.0)).clamp(0, 4)
    ghost = 0.25 * (1.0 - cover).clamp(0, 1) * outside
    return 0.50 * curve_score + 0.28 * cover + 0.18 * iou - 0.40 * outside + scale_term - giant_penalty - ghost


def region_simplicity(circ: float, elong: float, fill_ratio: float, area_frac: float) -> float:
    """
    区域几何简洁度 0~1（通用）：高圆度、接近 1 的细长度、高填充 → 简洁。
    """
    c = float(np.clip(circ / 1.0, 0.0, 1.2))
    e = float(max(elong, 1.0))
    elong_pen = float(np.clip(np.log(e) / np.log(4.0), 0.0, 1.0))
    fill = float(np.clip(fill_ratio, 0.0, 1.0))
    af = float(np.clip(area_frac * 8.0, 0.0, 1.0))  # 大块略加分
    s = 0.45 * min(c, 1.0) + 0.30 * fill + 0.15 * (1.0 - elong_pen) + 0.10 * af
    return float(np.clip(s, 0.0, 1.0))


def best_primitive_for_region(
    region: Region,
    primitives: list[ArcPrimitive] | None,
) -> ArcPrimitive | None:
    """取该区域残差最低的圆/椭圆/直线基元（通用，非场景）。"""
    if not primitives:
        return None
    cands = [
        p
        for p in primitives
        if p.region_id == region.region_id
        and not p.hard
        and p.type in {PrimitiveType.circle_arc, PrimitiveType.ellipse_arc, PrimitiveType.line}
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: p.residual)
    return cands[0]


def low_complexity_assets(
    curve_lib: StampCurveLibrary,
    *,
    circ: float,
    elong: float = 1.0,
    k: int = 12,
    max_complexity: float = 0.42,
    min_fill: float = 0.0,
) -> list[str]:
    """
    按几何复杂度与圆度/细长/填充接近程度取候选（无形状桶名单、无 asset 白名单）。

    仅使用 complexity / circularity / elongation / area_frac 字段。
    """
    scored: list[tuple[float, str]] = []
    te = max(float(elong), 1.0)
    for e in curve_lib.entries:
        if e.complexity > max_complexity:
            continue
        if e.area_frac < min_fill:
            continue
        ee = max(float(e.elongation), 1.0)
        # 圆度接近 + 细长接近 + 低复杂度 + 高填充（完整形体）
        sc = float(
            np.exp(-2.0 * abs(e.circularity - circ))
            + np.exp(-1.5 * abs(math.log(ee + 1e-6) - math.log(te + 1e-6)))
            + np.exp(-3.0 * e.complexity)
            + e.area_frac * 0.5
            + (0.4 if e.n_holes == 0 else 0.0)
        )
        scored.append((sc, e.asset_id))
    scored.sort(key=lambda t: -t[0])
    out: list[str] = []
    for _, aid in scored:
        if aid not in out:
            out.append(aid)
        if len(out) >= k:
            break
    return out


def match_region_with_particles(
    region: Region,
    curve_lib: StampCurveLibrary,
    renderer: TorchStampRenderer,
    *,
    primitives: list[ArcPrimitive] | None = None,
    n_particles: int = 256,
    recall_k: int = 48,
    seed: int = 0,
    min_score: float = 0.12,
    prefer_primitive_seed: bool = True,
    target_curve_pts: FloatArr | None = None,
) -> StampLayer | None:
    """
    对区域做曲线导向的大尺度粒子搜索。

    target_curve_pts：图章曲线拟合参考，应来自 SharedEdge.polyline
    （curve_fit 模式下即 edges_bezier_after_fit）。
    未提供时回退 region.contour_resampled（Region 轮廓已与共享边同源）。
    """
    device = renderer.device
    cs = float(renderer.canvas_size)
    cx, cy, bw, bh, pang, circ, elong = _region_geom(region)
    desc = np.asarray(region.descriptor, dtype=np.float64)

    # 目标曲线点：优先外部传入的共享边点云（after_fit / dense）
    if target_curve_pts is not None and len(np.asarray(target_curve_pts)) >= 4:
        tgt_pts_np = np.asarray(target_curve_pts, dtype=np.float32)
    else:
        tgt_pts_np = np.asarray(region.contour_resampled, dtype=np.float32)
        if len(tgt_pts_np) < 4:
            tgt_pts_np = np.asarray(region.contour, dtype=np.float32)
    tgt_pts = torch.from_numpy(np.ascontiguousarray(tgt_pts_np)).to(device)
    tgt_mask = torch.from_numpy(np.asarray(region.mask, dtype=np.float32)).to(device)

    # 通用区域几何：填充率 / 简洁度（无场景标签）
    tgt_area = float(tgt_mask.sum().item()) + 1e-6
    bbox_area = max(bw * bh, 1.0)
    fill_ratio = float(np.clip(tgt_area / bbox_area, 0.0, 1.0))
    prefer_holes = fill_ratio < 0.55 and circ < 0.75
    area_frac = float(region.area_frac)
    r_simp = region_simplicity(circ, elong, fill_ratio, area_frac)

    prim = best_primitive_for_region(region, primitives) if prefer_primitive_seed else None
    # 优质圆/椭圆基元或高圆度实心区 → 完整形体：抬高简洁度，抑制复杂/镂空章
    solid_disk = fill_ratio >= 0.55 and circ >= 0.55
    # 实心细长条（如金锭类像素块）：完整矩形/条带形体
    solid_bar = fill_ratio >= 0.45 and elong >= 1.6 and circ < 0.75
    if prim is not None and prim.residual < 0.06 and prim.type in {PrimitiveType.circle_arc, PrimitiveType.ellipse_arc}:
        r_simp = max(r_simp, 0.78)
        prefer_holes = False
        solid_disk = True
    elif solid_disk:
        r_simp = max(r_simp, 0.70)
        prefer_holes = False
    elif solid_bar:
        r_simp = max(r_simp, 0.62)
        prefer_holes = False

    assets = curve_lib.recall(
        desc,
        circ,
        elong,
        k=max(recall_k, 16),
        prefer_holes=prefer_holes,
        region_simplicity=r_simp,
        region_area_frac=area_frac,
    )
    if prefer_primitive_seed and ((prim is not None and prim.residual < 0.08) or solid_disk or solid_bar):
        max_c = 0.32 if solid_disk else (0.38 if solid_bar else 0.40)
        min_f = 0.40 if solid_disk else (0.25 if solid_bar else 0.0)
        seed_assets = low_complexity_assets(
            curve_lib,
            circ=max(circ, 0.35 if solid_bar else 0.75),
            elong=elong if solid_bar else 1.0,
            k=14,
            max_complexity=max_c,
            min_fill=min_f,
        )
        assets = list(dict.fromkeys([*seed_assets, *assets]))
        if (solid_disk or solid_bar) and seed_assets:
            simple_set = set(seed_assets)
            assets = [
                a
                for a in assets
                if a in simple_set or (curve_lib.by_id[a].complexity <= max_c and curve_lib.by_id[a].area_frac >= min_f * 0.85)
            ]
            if not assets:
                assets = seed_assets
    if not assets:
        return None

    init_pose = _init_from_primitive(prim) if prim is not None else None
    # 完整形体播种：若有圆/椭圆参数，用直径覆盖区域
    full_shape_seeds: list[tuple[float, float, float, float, float]] = []
    if prim is not None and init_pose is not None and prim.type in {PrimitiveType.circle_arc, PrimitiveType.ellipse_arc}:
        pcx, pcy, pw, ph, pang0 = init_pose
        for s in (0.85, 0.95, 1.0, 1.08, 1.18, 1.35, 1.6, 2.0):
            full_shape_seeds.append((pcx, pcy, max(pw * s, 8.0), max(ph * s, 8.0), pang0))
            if prim.type == PrimitiveType.circle_arc:
                side = max(bw, bh) * s
                full_shape_seeds.append((cx, cy, side, side, 0.0))
    # 细长实心：bbox 对齐多种纵横比
    if solid_bar:
        for s in (0.9, 1.0, 1.1, 1.25, 1.45):
            full_shape_seeds.append((cx, cy, max(bw * s, 8.0), max(bh * s, 8.0), pang))
            full_shape_seeds.append((cx, cy, max(bw * s, 8.0), max(bh * s, 8.0), (pang + 90.0) % 360.0))

    rng = np.random.default_rng(seed + region.region_id * 17)
    best_score = -1e9
    best_layer: StampLayer | None = None

    for asset_id in assets:
        entry = curve_lib.by_id.get(asset_id)
        if entry is None:
            continue
        # 多环：外轮廓 + 孔洞全部参与 Chamfer（高精度）
        rings = curve_lib.all_rings_normalized(asset_id)
        ring_tensors: list[torch.Tensor] = []
        pts_per_ring = max(64, min(192, 256 // max(len(rings), 1)))
        for ring in rings:
            r = np.asarray(ring, dtype=np.float32)
            if len(r) < 3:
                continue
            rt = torch.from_numpy(np.ascontiguousarray(r)).to(device)
            if rt.shape[0] != pts_per_ring:
                idx = torch.linspace(0, rt.shape[0] - 1e-3, pts_per_ring, device=device).long()
                rt = rt[idx % rt.shape[0]]
            ring_tensors.append(rt)
        if not ring_tensors:
            continue
        local = torch.cat(ring_tensors, dim=0)

        n = n_particles
        left = np.empty(n, dtype=np.float64)
        top = np.empty(n, dtype=np.float64)
        width = np.empty(n, dtype=np.float64)
        height = np.empty(n, dtype=np.float64)
        angles = np.empty(n, dtype=np.float64)

        i = 0
        # 基元完整形体种子优先
        for pose in full_shape_seeds:
            if i >= n:
                break
            left[i], top[i], width[i], height[i], angles[i] = pose
            i += 1
        # 确定性：bbox 对齐
        scales = (0.9, 1.0, 1.15, 1.35, 1.7, 2.2) if r_simp > 0.6 else (0.85, 1.0, 1.2, 1.5, 2.0, 2.8, 4.0)
        angles_det = (0.0, 45.0, 90.0, 180.0, 270.0) if not solid_bar else (0.0, 15.0, 30.0, 45.0, 90.0, 180.0, 270.0)
        for sc in scales:
            for da in angles_det:
                if i >= n:
                    break
                left[i], top[i] = cx, cy
                if r_simp > 0.65 and circ >= 0.5 and not solid_bar:
                    side = max(bw, bh) * sc
                    width[i] = side
                    height[i] = side
                else:
                    width[i] = max(bw, 8.0) * sc
                    height[i] = max(bh, 8.0) * sc
                angles[i] = (pang + da) % 360.0
                i += 1
        if init_pose is not None and i < n:
            left[i], top[i], width[i], height[i], angles[i] = init_pose
            i += 1

        n_rand = n - i
        if n_rand > 0:
            out_frac = 0.08 if r_simp > 0.65 else (0.18 if r_simp > 0.45 else 0.32)
            n_out = max(0, int(n_rand * out_frac))
            n_norm = n_rand - n_out
            if n_norm > 0:
                sc_hi = min(2.2 * cs, 3.5 * max(bw, bh)) if r_simp > 0.6 else min(3.5 * cs, 8 * max(bw, bh))
                sc = np.exp(rng.uniform(math.log(0.6 * max(bw, bh, 8.0)), math.log(max(sc_hi, 16.0)), n_norm))
                if r_simp > 0.65 and circ >= 0.5:
                    width[i : i + n_norm] = sc
                    height[i : i + n_norm] = sc
                else:
                    asp = rng.uniform(0.7, 1.35, n_norm)
                    width[i : i + n_norm] = sc * asp
                    height[i : i + n_norm] = sc / asp
                left[i : i + n_norm] = cx + rng.normal(0, 0.12 * cs, n_norm)
                top[i : i + n_norm] = cy + rng.normal(0, 0.12 * cs, n_norm)
                angles[i : i + n_norm] = rng.uniform(0, 360, n_norm)
            if n_out > 0:
                j = i + n_norm
                wmax = 2.8 * cs if r_simp > 0.6 else 6.0 * cs
                width[j:] = rng.uniform(1.05 * cs, wmax, n_out)
                height[j:] = width[j:] * rng.uniform(0.8, 1.2, n_out)
                left[j:] = rng.uniform(-0.4 * cs, 1.4 * cs, n_out)
                top[j:] = rng.uniform(-0.4 * cs, 1.4 * cs, n_out)
                angles[j:] = rng.uniform(0, 360, n_out)

        w_cap = (5.0 if r_simp > 0.6 else 10.0) * cs
        width = np.clip(width, 4.0, w_cap)
        height = np.clip(height, 4.0, w_cap)

        left_t = torch.tensor(left, dtype=torch.float32, device=device)
        top_t = torch.tensor(top, dtype=torch.float32, device=device)
        width_t = torch.tensor(width, dtype=torch.float32, device=device)
        height_t = torch.tensor(height, dtype=torch.float32, device=device)
        ang_t = torch.tensor(angles, dtype=torch.float32, device=device)

        pred_pts = transform_stamp_contour_batch(local, left=left_t, top=top_t, width=width_t, height=height_t, angle_deg=ang_t)
        closs = chamfer_loss_torch(pred_pts, tgt_pts, huber_delta=8.0)

        masks = renderer.render_batch_masks(
            asset_id,
            left=left_t,
            top=top_t,
            width=width_t,
            height=height_t,
            angle_deg=ang_t,
        )[:, 0]
        score = _score_curve_and_cover(
            masks,
            tgt_mask,
            closs,
            width=width_t,
            height=height_t,
            cs=cs,
        )
        bi = int(torch.argmax(score).item())
        sc = float(score[bi].item())
        cplx = float(entry.complexity)
        sc = sc - 0.28 * r_simp * cplx
        # 高圆度实心区：圆度差的候选再扣分（通用几何）
        if solid_disk:
            sc = sc - 0.30 * abs(float(entry.circularity) - max(circ, 0.8))
            sc = sc - 0.25 * max(0.0, 0.55 - float(entry.area_frac))
            sc = sc + 0.12 * min(float(entry.circularity), 1.1)
            sc = sc + 0.18 * float(entry.area_frac)
        elif solid_bar:
            # 细长实心：奖励细长接近、惩罚过复杂/镂空
            sc = sc - 0.20 * abs(math.log(max(entry.elongation, 1.0)) - math.log(max(elong, 1.0)))
            sc = sc + 0.15 * float(entry.area_frac)
            sc = sc - 0.15 * float(entry.complexity)
        if sc > best_score:
            best_score = sc
            best_layer = StampLayer(
                asset=asset_id,
                opacity=1.0,
                angle=float(angles[bi]),
                flipX=False,
                flipY=False,
                top=float(top[bi]),
                left=float(left[bi]),
                height=float(height[bi]),
                width=float(width[bi]),
                fill=region.color_hex,
            )

    if best_layer is None or best_score < min_score:
        return None
    return best_layer


def refine_layer_particles(
    layer: StampLayer,
    region: Region,
    curve_lib: StampCurveLibrary,
    renderer: TorchStampRenderer,
    *,
    n: int = 48,
    seed: int = 0,
    target_curve_pts: FloatArr | None = None,
) -> StampLayer:
    """局部粒子精修（多环曲线 + 覆盖）；目标曲线可来自共享边去重点。"""
    device = renderer.device
    cs = float(renderer.canvas_size)
    entry = curve_lib.by_id.get(layer.asset)
    if entry is None:
        return layer
    rings = curve_lib.all_rings_normalized(layer.asset)
    ring_tensors: list[torch.Tensor] = []
    pts_per_ring = max(48, 128 // max(len(rings), 1))
    for ring in rings:
        r = np.asarray(ring, dtype=np.float32)
        if len(r) < 3:
            continue
        rt = torch.from_numpy(np.ascontiguousarray(r)).to(device)
        if rt.shape[0] != pts_per_ring:
            idx = torch.linspace(0, rt.shape[0] - 1e-3, pts_per_ring, device=device).long()
            rt = rt[idx % rt.shape[0]]
        ring_tensors.append(rt)
    if not ring_tensors:
        return layer
    local = torch.cat(ring_tensors, dim=0)
    if target_curve_pts is not None and len(np.asarray(target_curve_pts)) >= 4:
        tgt_np = np.asarray(target_curve_pts, dtype=np.float32)
    else:
        tgt_np = np.asarray(region.contour_resampled, dtype=np.float32)
    tgt_pts = torch.from_numpy(np.ascontiguousarray(tgt_np)).to(device)
    tgt_mask = torch.from_numpy(np.asarray(region.mask, dtype=np.float32)).to(device)
    rng = np.random.default_rng(seed)
    left = layer.left + rng.normal(0, 8.0, n)
    top = layer.top + rng.normal(0, 8.0, n)
    width = np.clip(layer.width * rng.uniform(0.88, 1.15, n), 4.0, None)
    height = np.clip(layer.height * rng.uniform(0.88, 1.15, n), 4.0, None)
    angles = (layer.angle + rng.normal(0, 12.0, n)) % 360.0
    left[0], top[0], width[0], height[0], angles[0] = (
        layer.left,
        layer.top,
        layer.width,
        layer.height,
        layer.angle,
    )
    left_t = torch.tensor(left, dtype=torch.float32, device=device)
    top_t = torch.tensor(top, dtype=torch.float32, device=device)
    width_t = torch.tensor(width, dtype=torch.float32, device=device)
    height_t = torch.tensor(height, dtype=torch.float32, device=device)
    ang_t = torch.tensor(angles, dtype=torch.float32, device=device)
    pred_pts = transform_stamp_contour_batch(local, left=left_t, top=top_t, width=width_t, height=height_t, angle_deg=ang_t)
    closs = chamfer_loss_torch(pred_pts, tgt_pts, huber_delta=10.0)
    masks = renderer.render_batch_masks(
        layer.asset,
        left=left_t,
        top=top_t,
        width=width_t,
        height=height_t,
        angle_deg=ang_t,
    )[:, 0]
    score = _score_curve_and_cover(masks, tgt_mask, closs, width=width_t, height=height_t, cs=cs)
    i = int(torch.argmax(score).item())
    return StampLayer(
        asset=layer.asset,
        opacity=1.0,
        angle=float(angles[i]),
        flipX=layer.flipX,
        flipY=layer.flipY,
        top=float(top[i]),
        left=float(left[i]),
        height=float(height[i]),
        width=float(width[i]),
        fill=layer.fill,
    )
