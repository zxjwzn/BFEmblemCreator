"""P4：轮廓分段与圆/椭圆/直线弧逼近（GPU 拟合）。"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.curves import rdp, resample_closed_contour
from bf_emblem_creator.approx.depth_order import DepthOrderResult, EdgeRole
from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.gpu_ops import to_torch
from bf_emblem_creator.approx.regions import Region

if TYPE_CHECKING:
    from bf_emblem_creator.approx.planar_map import PlanarMap, SharedEdge

FloatArr = NDArray[np.floating]


class PrimitiveType(str, Enum):
    """弧基元类型。"""

    line = "line"
    circle_arc = "circle_arc"
    ellipse_arc = "ellipse_arc"
    free = "free"


class ArcPrimitive(BaseModel):
    """轮廓弧段中间表示。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    type: PrimitiveType
    params: dict[str, float] = Field(default_factory=dict)
    sample_points: Any = Field(..., description="(N,2) 画布坐标")
    role: EdgeRole = EdgeRole.shape_boundary
    region_id: int
    depth: int
    residual: float = Field(default=0.0, description="拟合残差（相对弦长）")
    hard: bool = Field(default=False, description="HARD_SEGMENT")


def _polyline_length_t(pts: torch.Tensor) -> float:
    if pts.shape[0] < 2:
        return 0.0
    return float(torch.linalg.norm(pts[1:] - pts[:-1], dim=1).sum().item())


def _fit_line(pts: FloatArr) -> tuple[dict[str, float], float]:
    """直线最小二乘（GPU SVD）。"""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 2:
        return {}, 1.0
    dev = get_device()
    t = to_torch(p.astype(np.float32), device=dev)
    c = t.mean(dim=0)
    q = t - c
    _, _, vh = torch.linalg.svd(q, full_matrices=False)
    direction = vh[0]
    nrm = torch.stack([-direction[1], direction[0]])
    d = (q @ nrm).abs().mean()
    chord = max(_polyline_length_t(t), 1e-6)
    res = float(d.item() / chord)
    return {
        "cx": float(c[0].item()),
        "cy": float(c[1].item()),
        "dx": float(direction[0].item()),
        "dy": float(direction[1].item()),
    }, res


def _fit_circle(pts: FloatArr) -> tuple[dict[str, float], float]:
    """代数圆拟合（GPU lstsq）。"""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 3:
        return {}, 1.0
    dev = get_device()
    t = to_torch(p.astype(np.float32), device=dev)
    x, y = t[:, 0], t[:, 1]
    a = torch.stack([2 * x, 2 * y, torch.ones_like(x)], dim=1)
    b = x**2 + y**2
    sol = torch.linalg.lstsq(a, b.unsqueeze(1)).solution.squeeze(1)
    cx, cy, c0 = float(sol[0].item()), float(sol[1].item()), float(sol[2].item())
    r = float(math.sqrt(max(c0 + cx * cx + cy * cy, 1e-8)))
    dist = (torch.hypot(x - cx, y - cy) - r).abs()
    chord = max(_polyline_length_t(t), 1e-6)
    res = float(dist.mean().item() / chord)
    ang0 = float(torch.atan2(y[0] - cy, x[0] - cx).item())
    ang1 = float(torch.atan2(y[-1] - cy, x[-1] - cx).item())
    return {"cx": cx, "cy": cy, "r": r, "ang0": ang0, "ang1": ang1}, res


def _fit_ellipse(pts: FloatArr) -> tuple[dict[str, float], float]:
    """PCA 椭圆近似（GPU eigh）。"""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 5:
        return {}, 1.0
    dev = get_device()
    t = to_torch(p.astype(np.float32), device=dev)
    c = t.mean(dim=0)
    q = t - c
    cov = (q.T @ q) / max(len(p) - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    aa = float(max(2.0 * float(torch.sqrt(eigvals[0].clamp_min(1e-8)).item()), 1.0))
    bb = float(max(2.0 * float(torch.sqrt(eigvals[1].clamp_min(1e-8)).item()), 1.0))
    ang = float(torch.atan2(eigvecs[1, 0], eigvecs[0, 0]).item())
    ca, sa = math.cos(-ang), math.sin(-ang)
    rot = torch.tensor([[ca, -sa], [sa, ca]], device=dev, dtype=torch.float32)
    local = (rot @ q.T).T
    rn = torch.hypot(local[:, 0] / aa, local[:, 1] / bb)
    res = float((rn - 1.0).abs().mean().item())
    chord = max(_polyline_length_t(t), 1e-6)
    res = float(res * min(aa, bb) / chord)
    return {
        "cx": float(c[0].item()),
        "cy": float(c[1].item()),
        "a": aa,
        "b": bb,
        "angle": ang,
    }, res


def _curvature_corners(pts: FloatArr, *, angle_thr_deg: float = 35.0) -> list[int]:
    """转角超过阈值的分割点索引（GPU）。"""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 5:
        return []
    dev = get_device()
    t = to_torch(p.astype(np.float32), device=dev)
    if torch.linalg.norm(t[0] - t[-1]) < 1e-6:
        t = t[:-1]
    n = int(t.shape[0])
    thr = math.radians(angle_thr_deg)
    cuts: list[int] = []
    for i in range(n):
        p0 = t[(i - 1) % n]
        p1 = t[i]
        p2 = t[(i + 1) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = torch.linalg.norm(v1).clamp_min(1e-9)
        n2 = torch.linalg.norm(v2).clamp_min(1e-9)
        cos_a = float(torch.clamp(torch.dot(v1, v2) / (n1 * n2), -1.0, 1.0).item())
        ang = math.acos(cos_a)
        if ang > thr:
            cuts.append(i)
    return cuts


def segment_contour(
    points: FloatArr,
    *,
    closed: bool = True,
    angle_thr_deg: float = 35.0,
    min_seg_len: int = 4,
) -> list[FloatArr]:
    """按转角分割轮廓为弧段。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < min_seg_len:
        return [pts]
    body = pts[:-1] if closed and float(np.linalg.norm(pts[0] - pts[-1])) < 1e-6 else pts
    cuts = sorted(set(_curvature_corners(body, angle_thr_deg=angle_thr_deg)))
    if len(cuts) < 2:
        n = len(body)
        if n >= 16:
            cuts = [0, n // 4, n // 2, 3 * n // 4]
        else:
            return [np.vstack([body, body[:1]]) if closed else body]

    if 0 not in cuts:
        cuts = [0, *cuts]
    cuts = sorted(set(cuts))
    segs: list[FloatArr] = []
    for i, c0 in enumerate(cuts):
        c1 = cuts[(i + 1) % len(cuts)]
        seg = body[c0 : c1 + 1] if c1 > c0 else np.vstack([body[c0:], body[: c1 + 1]])
        if len(seg) >= min_seg_len:
            segs.append(seg.astype(np.float64))
    return segs if segs else [body.astype(np.float64)]


def fit_segment_primitive(
    seg: FloatArr,
    *,
    eps_arc: float = 0.04,
    eps_hard: float = 0.08,
) -> tuple[PrimitiveType, dict[str, float], float, bool]:
    """对单段尝试 line / circle / ellipse；有损阈值。"""
    line_p, line_r = _fit_line(seg)
    circ_p, circ_r = _fit_circle(seg)
    ell_p, ell_r = _fit_ellipse(seg)

    candidates: list[tuple[PrimitiveType, dict[str, float], float]] = [
        (PrimitiveType.line, line_p, line_r),
        (PrimitiveType.circle_arc, circ_p, circ_r),
        (PrimitiveType.ellipse_arc, ell_p, ell_r),
    ]
    candidates.sort(key=lambda t: t[2])
    best_t, best_p, best_r = candidates[0]
    if best_r <= eps_arc and best_p:
        return best_t, best_p, best_r, False
    if best_r <= eps_hard and best_p:
        return best_t, best_p, best_r, False
    return PrimitiveType.free, {}, float(best_r), True


def extract_primitives_for_region(
    region: Region,
    depth: int,
    *,
    role: EdgeRole = EdgeRole.shape_boundary,
    eps_arc: float = 0.04,
    sample_n: int = 48,
) -> list[ArcPrimitive]:
    """对区域外轮廓分段并拟合基元；优先整环圆/椭圆，避免二次猛 RDP。"""
    if role == EdgeRole.occlusion_cut:
        return []
    # 优先用面积约束后的密重采样轮廓
    cont = np.asarray(region.contour_resampled, dtype=np.float64)
    if len(cont) < 8:
        cont = np.asarray(region.contour, dtype=np.float64)
    if len(cont) < 4:
        return []

    # 整环拟合：圆/椭圆残差好则单基元覆盖（金锭/圆脸等）
    ptype_all, params_all, res_all, hard_all = fit_segment_primitive(cont, eps_arc=eps_arc)
    if (
        not hard_all
        and res_all <= max(eps_arc * 1.5, 0.06)
        and ptype_all in {PrimitiveType.circle_arc, PrimitiveType.ellipse_arc}
        and params_all
    ):
        samples = resample_closed_contour(cont, max(sample_n, 64))
        return [
            ArcPrimitive(
                type=ptype_all,
                params=params_all,
                sample_points=samples,
                role=role,
                region_id=region.region_id,
                depth=depth,
                residual=res_all,
                hard=False,
            )
        ]

    # 轻量简化后再分段（不破坏面积：仅用于切段）
    if len(cont) >= 24:
        cont_seg = rdp(cont, 0.35)
        if len(cont_seg) < 8:
            cont_seg = cont
    else:
        cont_seg = cont
    segs = segment_contour(cont_seg, closed=True)
    out: list[ArcPrimitive] = []
    for seg in segs:
        ptype, params, res, hard = fit_segment_primitive(seg, eps_arc=eps_arc)
        samples = resample_closed_contour(seg, sample_n) if len(seg) >= 2 else seg
        if len(seg) >= 2 and float(np.linalg.norm(seg[0] - seg[-1])) > 1e-3:
            samples = _resample_open(seg, sample_n)
        out.append(
            ArcPrimitive(
                type=ptype,
                params=params,
                sample_points=samples,
                role=role,
                region_id=region.region_id,
                depth=depth,
                residual=res,
                hard=hard,
            )
        )
    return out


def _resample_open(points: FloatArr, n: int) -> FloatArr:
    """开放折线弧长重采样（GPU）。"""
    dev = get_device()
    t = to_torch(np.asarray(points, dtype=np.float32), device=dev)
    if t.shape[0] < 2:
        return np.zeros((n, 2), dtype=np.float64)
    seg = torch.linalg.norm(t[1:] - t[:-1], dim=1)
    u = torch.cat([torch.zeros(1, device=dev), torch.cumsum(seg, dim=0)])
    total = float(u[-1].item())
    if total < 1e-8:
        return np.repeat(np.asarray(points[:1], dtype=np.float64), n, axis=0)
    u = u / total
    samples = torch.linspace(0.0, 1.0, n, device=dev)
    idx = torch.searchsorted(u.contiguous(), samples.contiguous(), right=True).clamp(1, u.numel() - 1)
    x0, x1 = u[idx - 1], u[idx]
    y0x, y1x = t[idx - 1, 0], t[idx, 0]
    y0y, y1y = t[idx - 1, 1], t[idx, 1]
    tt = (samples - x0) / (x1 - x0).clamp_min(1e-12)
    x = y0x + tt * (y1x - y0x)
    y = y0y + tt * (y1y - y0y)
    return torch.stack([x, y], dim=1).detach().cpu().numpy().astype(np.float64)


def extract_primitives_for_shared_edges(
    edges: list[SharedEdge],
    *,
    face_depth: dict[int, int] | None = None,
    eps_arc: float = 0.04,
    sample_n: int = 48,
) -> list[ArcPrimitive]:
    """
    Batch D：在共享边上拟合 line/circle/ellipse（锁端点语义由边几何保证）。
    """
    depth_of = face_depth or {}
    out: list[ArcPrimitive] = []
    for e in edges:
        role = e.role
        if role == EdgeRole.occlusion_cut:
            continue
        poly = np.asarray(e.polyline, dtype=np.float64)
        if len(poly) < 2:
            continue
        # 背景外环可整环拟合
        closed = float(np.linalg.norm(poly[0] - poly[-1])) < 1e-3
        if closed and len(poly) >= 8:
            ptype, params, res, hard = fit_segment_primitive(poly, eps_arc=eps_arc)
            samples = resample_closed_contour(poly, max(sample_n, 64))
            face = int(e.left_face)
            role_e = role if isinstance(role, EdgeRole) else EdgeRole.shape_boundary
            out.append(
                ArcPrimitive(
                    type=ptype,
                    params=params,
                    sample_points=samples,
                    role=role_e,
                    region_id=face if face >= 0 else int(e.id),
                    depth=int(depth_of.get(face, 0)),
                    residual=res,
                    hard=hard,
                )
            )
            continue
        segs = segment_contour(poly, closed=closed)
        face = int(e.left_face)
        role_e = role if isinstance(role, EdgeRole) else EdgeRole.shape_boundary
        for seg in segs:
            ptype, params, res, hard = fit_segment_primitive(seg, eps_arc=eps_arc)
            if len(seg) >= 2 and float(np.linalg.norm(seg[0] - seg[-1])) > 1e-3:
                samples = _resample_open(seg, sample_n)
            else:
                samples = resample_closed_contour(seg, sample_n) if len(seg) >= 2 else seg
            out.append(
                ArcPrimitive(
                    type=ptype,
                    params=params,
                    sample_points=samples,
                    role=role_e,
                    region_id=face if face >= 0 else int(e.id),
                    depth=int(depth_of.get(face, 0)),
                    residual=res,
                    hard=hard,
                )
            )
    return out


def extract_all_primitives(
    depth_result: DepthOrderResult,
    *,
    eps_arc: float = 0.04,
    planar_map: PlanarMap | None = None,
) -> list[ArcPrimitive]:
    """按层序为所有区域提取 SHAPE_BOUNDARY 弧基元；若给 PlanarMap 则优先边上拟合。"""
    if planar_map is not None and planar_map.edges:
        depth_of = {item.region.region_id: item.depth for item in depth_result.ordered}
        return extract_primitives_for_shared_edges(
            list(planar_map.edges),
            face_depth=depth_of,
            eps_arc=eps_arc,
        )
    prims: list[ArcPrimitive] = []
    for item in depth_result.ordered:
        prims.extend(
            extract_primitives_for_region(
                item.region,
                item.depth,
                role=item.boundary_role_default,
                eps_arc=eps_arc,
            )
        )
    return prims


def primitives_to_point_cloud(
    prims: list[ArcPrimitive],
    *,
    only_shape: bool = True,
) -> FloatArr:
    """合并基元采样点，供曲线比对。"""
    pts: list[FloatArr] = []
    for p in prims:
        if only_shape and p.role != EdgeRole.shape_boundary:
            continue
        sp = np.asarray(p.sample_points, dtype=np.float64)
        if len(sp):
            pts.append(sp)
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(pts)
