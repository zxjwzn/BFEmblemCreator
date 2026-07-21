"""色块精确边界 → 弧段表示（默认保留 free 折线，禁止圆/椭圆概括替换）。"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.curves import resample_closed_contour
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
    hard: bool = Field(default=False, description="HARD_SEGMENT / 精确折线")


def _resample_open(points: FloatArr, n: int) -> FloatArr:
    """开放折线弧长重采样（GPU）。"""
    import torch

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


def _exact_free_primitive(
    poly: FloatArr,
    *,
    region_id: int,
    depth: int,
    role: EdgeRole,
    sample_n: int,
    closed: bool,
) -> ArcPrimitive:
    """精确边界 → free 基元。"""
    arr = np.asarray(poly, dtype=np.float64)
    if len(arr) < 2:
        samples = arr
    elif closed:
        samples = resample_closed_contour(arr, max(sample_n, min(len(arr), 256)))
    else:
        samples = _resample_open(arr, max(sample_n, min(len(arr), 256)))
    return ArcPrimitive(
        type=PrimitiveType.free,
        params={},
        sample_points=samples,
        role=role,
        region_id=region_id,
        depth=depth,
        residual=0.0,
        hard=True,
    )


def extract_primitives_for_region(
    region: Region,
    depth: int,
    *,
    role: EdgeRole = EdgeRole.shape_boundary,
    eps_arc: float = 0.04,
    sample_n: int = 64,
) -> list[ArcPrimitive]:
    """
    区域外轮廓 → 精确 free 描边（禁止圆/椭圆/直线概括替换边界）。

    `eps_arc` 仅保留调用签名；不用于几何替换。
    """
    _ = eps_arc
    if role == EdgeRole.occlusion_cut:
        return []
    cont = np.asarray(region.contour, dtype=np.float64)
    if len(cont) < 4:
        cont = np.asarray(region.contour_resampled, dtype=np.float64)
    if len(cont) < 4:
        return []
    return [
        _exact_free_primitive(
            cont,
            region_id=region.region_id,
            depth=depth,
            role=role,
            sample_n=sample_n,
            closed=True,
        )
    ]


def extract_primitives_for_shared_edges(
    edges: list[SharedEdge],
    *,
    face_depth: dict[int, int] | None = None,
    eps_arc: float = 0.04,
    sample_n: int = 64,
) -> list[ArcPrimitive]:
    """共享边精确折线 → free 基元（每条边一份几何）。"""
    _ = eps_arc
    depth_of = face_depth or {}
    out: list[ArcPrimitive] = []
    for e in edges:
        role = e.role
        if role == EdgeRole.occlusion_cut:
            continue
        poly = np.asarray(e.polyline, dtype=np.float64)
        if len(poly) < 2:
            continue
        closed = float(np.linalg.norm(poly[0] - poly[-1])) < 1e-3
        face = int(e.left_face) if e.left_face >= 0 else int(e.right_face)
        role_e = role if isinstance(role, EdgeRole) else EdgeRole.shape_boundary
        out.append(
            _exact_free_primitive(
                poly,
                region_id=face if face >= 0 else int(e.id),
                depth=int(depth_of.get(face, 0)),
                role=role_e,
                sample_n=sample_n,
                closed=closed,
            )
        )
    return out


def extract_all_primitives(
    depth_result: DepthOrderResult,
    *,
    eps_arc: float = 0.04,
    planar_map: PlanarMap | None = None,
) -> list[ArcPrimitive]:
    """按层序提取 SHAPE_BOUNDARY 精确边界基元。"""
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


def _curvature_corners(pts: FloatArr, *, angle_thr_deg: float = 35.0) -> list[int]:
    """转角超过阈值的分割点索引（分析用）。"""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 5:
        return []
    body = p[:-1] if float(np.linalg.norm(p[0] - p[-1])) < 1e-6 else p
    n = len(body)
    thr = math.radians(angle_thr_deg)
    cuts: list[int] = []
    for i in range(n):
        p0 = body[(i - 1) % n]
        p1 = body[i]
        p2 = body[(i + 1) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = float(np.linalg.norm(v1)) + 1e-9
        n2 = float(np.linalg.norm(v2)) + 1e-9
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        if math.acos(cos_a) > thr:
            cuts.append(i)
    return cuts


def segment_contour(
    points: FloatArr,
    *,
    closed: bool = True,
    angle_thr_deg: float = 35.0,
    min_seg_len: int = 4,
) -> list[FloatArr]:
    """按转角分割轮廓（分析用，主路径不替换几何）。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < min_seg_len:
        return [pts]
    body = pts[:-1] if closed and float(np.linalg.norm(pts[0] - pts[-1])) < 1e-6 else pts
    cuts = sorted(set(_curvature_corners(body, angle_thr_deg=angle_thr_deg)))
    if len(cuts) < 2:
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
    """单段诊断：始终 free，不替换精确边界。"""
    _ = eps_arc, eps_hard
    p = np.asarray(seg, dtype=np.float64)
    return PrimitiveType.free, {}, 0.0 if len(p) >= 2 else 1.0, True
