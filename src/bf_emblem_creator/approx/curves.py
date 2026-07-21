"""轮廓提取：Moore 边界跟踪（外轮廓 + 内部闭合孔）+ RDP/重采样/SDF。"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.gpu_ops import (
    curvature_descriptor_torch,
    foreground_cc_masks,
    interior_hole_masks,
    mask_to_sdf_fast,
    rdp_torch,
    resample_closed_contour_torch,
    to_torch,
)

FloatArr = NDArray[np.floating]
BoolArr = NDArray[np.bool_]

# 8 邻域：从「来向」的下一格起顺时针扫（经典 Moore）
# 方向编号 0..7：E, SE, S, SW, W, NW, N, NE
_DX = (1, 1, 0, -1, -1, -1, 0, 1)
_DY = (0, 1, 1, 1, 0, -1, -1, -1)


class ContourPoly(BaseModel):
    """闭合或开放折线。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    points: Any = Field(..., description="(N,2) xy 浮点，图像坐标")
    closed: bool = Field(default=True, description="是否闭合")
    kind: Literal["outer", "hole"] = Field(default="outer", description="外轮廓或孔洞")

    def numpy(self) -> FloatArr:
        """返回点列数组。"""
        return np.asarray(self.points, dtype=np.float64)


def rdp(points: FloatArr, epsilon: float) -> FloatArr:
    """Ramer–Douglas–Peucker（GPU）。"""
    dev = get_device()
    t = to_torch(np.asarray(points, dtype=np.float32), device=dev)
    out = rdp_torch(t, float(epsilon))
    return out.detach().cpu().numpy().astype(np.float64)


def signed_area(points: FloatArr) -> float:
    """多边形有向面积（闭合；>0 为 CCW）。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    if float(np.linalg.norm(pts[0] - pts[-1])) > 1e-6:
        pts = np.vstack([pts, pts[:1]])
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _pad_mask(mask: BoolArr) -> tuple[NDArray[np.uint8], int, int]:
    """外扩 1 像素 0 边，便于边界跟踪。返回 pad、h、w（原尺寸）。"""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), dtype=np.uint8)
    pad[1:-1, 1:-1] = m.astype(np.uint8)
    return pad, h, w


def _find_start(pad: NDArray[np.uint8], visited_border: NDArray[np.bool_]) -> tuple[int, int] | None:
    """
    找下一个边界起点：前景像素且 8 邻有背景，且该边界像素未用过。

    返回 pad 坐标 (y, x)。
    """
    h, w = pad.shape
    # 行优先扫描
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if pad[y, x] == 0:
                continue
            # 是否未访问边界像素
            is_border = (
                pad[y - 1, x] == 0
                or pad[y + 1, x] == 0
                or pad[y, x - 1] == 0
                or pad[y, x + 1] == 0
                or pad[y - 1, x - 1] == 0
                or pad[y - 1, x + 1] == 0
                or pad[y + 1, x - 1] == 0
                or pad[y + 1, x + 1] == 0
            )
            if is_border and not visited_border[y, x]:
                return y, x
    return None


def _moore_trace(
    pad: NDArray[np.uint8],
    start_y: int,
    start_x: int,
    *,
    max_steps: int = 500_000,
) -> list[tuple[int, int]]:
    """
    Moore 邻域跟踪一条闭合边界。

    坐标为 pad 系；返回有序边界像素列表（不含闭合重复首点）。
    """
    # 初始：从起点左侧背景进入（Jacob's stopping 变体）
    # 找起点邻域中一个背景方向作为 backtrack
    back_dir = 4  # W
    for d in range(8):
        ny, nx = start_y + _DY[d], start_x + _DX[d]
        if pad[ny, nx] == 0:
            back_dir = d
            break

    path: list[tuple[int, int]] = [(start_y, start_x)]
    y, x = start_y, start_x
    # 从 back_dir 的下一方向开始顺时针找第一个前景
    enter_dir = back_dir

    for _ in range(max_steps):
        # 从进入方向的逆时针一侧开始扫（标准：backtrack 的下一格）
        start_scan = (enter_dir + 1) % 8
        found = False
        for k in range(8):
            d = (start_scan + k) % 8
            ny, nx = y + _DY[d], x + _DX[d]
            if pad[ny, nx] != 0:
                # 进入新像素时，backtrack 是反方向
                enter_dir = (d + 4) % 8
                y, x = ny, nx
                found = True
                break
        if not found:
            break
        if (y, x) == (start_y, start_x) and len(path) > 2:
            break
        path.append((y, x))
        if len(path) > 2 and path[-1] == path[1] and path[-2] == path[0]:
            # Jacob stop
            path = path[:-1]
            break
    return path


def extract_all_contours(
    mask: BoolArr,
    *,
    simplify: float = 0.0,
    min_points: int = 8,
    min_area: float = 4.0,
    max_rings: int = 32,
) -> list[ContourPoly]:
    """
    提取全部闭合轮廓：外轮廓 + 内部孔洞。

    - Moore 边界跟踪（忠实于掩膜拓扑）
    - 默认 simplify=0（不做 RDP）；仅当显式传入 >0 时降采样
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []

    pad, _, _ = _pad_mask(m)
    visited = np.zeros_like(pad, dtype=bool)
    raw_rings: list[FloatArr] = []

    for _ in range(max_rings * 4):
        start = _find_start(pad, visited)
        if start is None:
            break
        sy, sx = start
        path = _moore_trace(pad, sy, sx)
        if len(path) < min_points:
            visited[sy, sx] = True
            continue
        # 标记本环边界已访问
        for py, px in path:
            visited[py, px] = True
        # pad → 原图像素中心坐标
        pts = np.array([[float(px - 1) + 0.5, float(py - 1) + 0.5] for py, px in path], dtype=np.float64)
        # 闭合
        if float(np.linalg.norm(pts[0] - pts[-1])) > 1e-6:
            pts = np.vstack([pts, pts[:1]])
        area = abs(signed_area(pts))
        if area < min_area:
            continue
        if simplify > 0 and len(pts) > 8:
            simp = rdp(pts, simplify)
            if len(simp) >= 4:
                pts = simp
                if float(np.linalg.norm(pts[0] - pts[-1])) > 1e-6:
                    pts = np.vstack([pts, pts[:1]])
        raw_rings.append(pts)

    if not raw_rings:
        # 兜底 bbox
        ys, xs = np.where(m)
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)
        return [ContourPoly(points=box, closed=True, kind="outer")]

    # 分类 outer / hole：面积最大者为各部件外轮廓；负面积为孔
    classified: list[ContourPoly] = []
    areas = [signed_area(r) for r in raw_rings]
    # 统一：外轮廓 CCW (正)，孔 CW (负) — 若符号反了则翻转
    abs_order = sorted(range(len(raw_rings)), key=lambda i: -abs(areas[i]))
    for i in abs_order[:max_rings]:
        pts = raw_rings[i]
        a = areas[i]
        if abs(a) < min_area:
            continue
        # 最大环强制 outer；其余若在最大外轮廓 bbox 内且面积小 → hole
        kind: Literal["outer", "hole"] = "outer"
        if classified:
            # 相对已有最大 outer
            main = next((c for c in classified if c.kind == "outer"), classified[0])
            main_pts = main.numpy()
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            if _point_in_poly(cx, cy, main_pts) and abs(a) < abs(signed_area(main_pts)) * 0.95:
                kind = "hole"
        # 定向：outer 取 CCW(正面积)，hole 取 CW(负面积)
        if (kind == "outer" and a < 0) or (kind == "hole" and a > 0):
            pts = pts[::-1].copy()
        classified.append(ContourPoly(points=pts, closed=True, kind=kind))

    # 至少一条 outer
    if not any(c.kind == "outer" for c in classified) and classified:
        classified[0] = ContourPoly(points=classified[0].numpy(), closed=True, kind="outer")
    return classified


def _point_in_poly(x: float, y: float, poly: FloatArr) -> bool:
    """射线法点在多边形内。"""
    pts = np.asarray(poly, dtype=np.float64)
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(pts[i, 0]), float(pts[i, 1])
        xj, yj = float(pts[j, 0]), float(pts[j, 1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def extract_outer_contour(mask: BoolArr, *, simplify: float = 0.0) -> FloatArr | None:
    """
    提取主外轮廓（面积最大 outer）。

    主外轮廓；内部孔请用 extract_all_contours。
    """
    rings = extract_all_contours(mask, simplify=simplify)
    if not rings:
        return None
    outers = [r for r in rings if r.kind == "outer"]
    pool = outers if outers else rings
    best = max(pool, key=lambda r: abs(signed_area(r.numpy())))
    return best.numpy()


def extract_contour_bundle(
    mask: BoolArr,
    *,
    simplify: float = 0.0,
    resample_n: int = 128,
) -> tuple[FloatArr, list[FloatArr], FloatArr]:
    """
    返回 (outer, holes, descriptor_source_points)。

    holes 为孔洞点列列表；descriptor 用 outer 重采样点。
    """
    rings = extract_all_contours(mask, simplify=simplify)
    if not rings:
        return np.zeros((0, 2), dtype=np.float64), [], np.zeros((0, 2), dtype=np.float64)
    outers = [r.numpy() for r in rings if r.kind == "outer"]
    holes = [r.numpy() for r in rings if r.kind == "hole"]
    if not outers:
        outers = [rings[0].numpy()]
        holes = [r.numpy() for r in rings[1:]]
    # 主 outer = 面积最大
    outer = max(outers, key=lambda p: abs(signed_area(p)))
    # 其余 outer 作为附加「外环」并入 holes 侧的 all_rings 由调用方处理
    extra_outers = [p for p in outers if p is not outer and abs(signed_area(p)) > 4.0]
    all_holes = holes + extra_outers
    desc_pts = resample_closed_contour(outer, resample_n)
    return outer, all_holes, desc_pts


def contour_curvature_descriptor(points: FloatArr, bins: int = 24) -> FloatArr:
    """基于转角直方图的描述子（GPU，更高 bins）。"""
    dev = get_device()
    t = to_torch(np.asarray(points, dtype=np.float32), device=dev)
    return curvature_descriptor_torch(t, bins=bins).detach().cpu().numpy().astype(np.float64)


def multi_ring_descriptor(outer: FloatArr, holes: list[FloatArr], bins: int = 24) -> FloatArr:
    """
    外轮廓描述子 + 孔洞统计特征，便于区分镂空图章。
    """
    base = contour_curvature_descriptor(outer, bins=bins)
    n_holes = float(len(holes))
    hole_area = float(sum(abs(signed_area(h)) for h in holes))
    outer_area = max(abs(signed_area(outer)), 1e-6)
    extra = np.array(
        [
            n_holes / 8.0,
            min(hole_area / outer_area, 2.0),
            min(len(outer) / 256.0, 2.0),
        ],
        dtype=np.float64,
    )
    return np.concatenate([base, extra])


def resample_closed_contour(points: FloatArr, n: int = 128) -> FloatArr:
    """将闭合轮廓按弧长重采样为 n 点（GPU）。"""
    dev = get_device()
    t = to_torch(np.asarray(points, dtype=np.float32), device=dev)
    out = resample_closed_contour_torch(t, n=n)
    return out.detach().cpu().numpy().astype(np.float64)


def mask_to_sdf(mask: BoolArr, *, iters: int = 24) -> FloatArr:
    """
    SDF：外部为正、内部为负。

    高精度路径：优先多轮 GPU 箱式；分辨率高时 iters 加大。
    """
    m = np.asarray(mask, dtype=bool)
    # 高分辨率用更多模糊迭代近似距离
    h = max(m.shape)
    it = max(iters, min(48, h // 8))
    return mask_to_sdf_fast(m, device=get_device(), iters=it)


def normalize_contour_to_unit(contour_px: FloatArr, tex_size: int) -> FloatArr:
    """像素轮廓 → 中心坐标系 [-0.5,0.5]^2（相对 tex 包围盒）。"""
    pts = np.asarray(contour_px, dtype=np.float64).copy()
    half = (tex_size - 1) / 2.0
    pts[:, 0] = (pts[:, 0] - half) / float(tex_size)
    pts[:, 1] = (pts[:, 1] - half) / float(tex_size)
    return pts


def polygon_area(points: FloatArr) -> float:
    """闭合多边形面积（绝对值）。"""
    return abs(signed_area(points))


def area_relative_error(poly_area: float, mask_area: float) -> float:
    """|A_poly - A_mask| / A_mask；mask_area 过小返回 0。"""
    m = float(max(mask_area, 1e-9))
    return abs(float(poly_area) - m) / m


def _ensure_closed(pts: FloatArr) -> FloatArr:
    """首尾闭合。"""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 2:
        return p
    if float(np.linalg.norm(p[0] - p[-1])) > 1e-6:
        return np.vstack([p, p[:1]])
    return p


def sample_circle_contour(cx: float, cy: float, r: float, n: int = 128) -> FloatArr:
    """均匀采样闭合圆轮廓。"""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)
    return _ensure_closed(pts)


def sample_ellipse_contour(
    cx: float,
    cy: float,
    a: float,
    b: float,
    angle_rad: float = 0.0,
    n: int = 128,
) -> FloatArr:
    """均匀参数采样闭合椭圆轮廓。"""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    xr = ca * x - sa * y + cx
    yr = sa * x + ca * y + cy
    return _ensure_closed(np.stack([xr, yr], axis=1))


def _rasterize_polygon(poly: FloatArr, shape: tuple[int, int]) -> BoolArr:
    """闭合多边形栅格化为 bool 蒙版。"""
    from PIL import Image, ImageDraw

    h, w = shape
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    xy = [(float(x), float(y)) for x, y in np.asarray(poly, dtype=np.float64)]
    if len(xy) < 3:
        return np.zeros(shape, dtype=bool)
    draw.polygon(xy, outline=1, fill=1)
    return np.asarray(img, dtype=bool)


def _mask_fill_iou(mask: BoolArr, poly: FloatArr, shape: tuple[int, int]) -> float:
    """多边形栅格化后与 mask 的 IoU（CPU 扫描，供面积约束校验）。"""
    poly_m = _rasterize_polygon(poly, shape)
    m = np.asarray(mask, dtype=bool)
    inter = float(np.logical_and(poly_m, m).sum())
    union = float(np.logical_or(poly_m, m).sum()) + 1e-9
    return inter / union


def fitted_region_area(
    outer: FloatArr,
    holes: list[FloatArr] | None = None,
    *,
    mask_shape: tuple[int, int] | None = None,
) -> float:
    """
    拟合闭合色块面积。

    提供 mask_shape 时：栅格化 outer 减去 holes，与像素 mask 可比；
    否则：鞋带公式 outer−holes。
    """
    if mask_shape is not None:
        out = _rasterize_polygon(outer, mask_shape)
        if holes:
            for h in holes:
                out = np.logical_and(out, np.logical_not(_rasterize_polygon(h, mask_shape)))
        return float(out.sum())
    a = polygon_area(outer)
    if holes:
        for h in holes:
            a -= polygon_area(h)
    return float(max(a, 0.0))


def _scale_about(pts: FloatArr, cx: float, cy: float, s: float) -> FloatArr:
    """相对 (cx,cy) 等比缩放闭合折线。"""
    p = np.asarray(pts, dtype=np.float64).copy()
    p[:, 0] = (p[:, 0] - cx) * s + cx
    p[:, 1] = (p[:, 1] - cy) * s + cy
    return _ensure_closed(p)


def adjust_contour_area_to_mask(
    outer: FloatArr,
    mask: BoolArr,
    holes: list[FloatArr] | None = None,
    *,
    max_area_rel_err: float = 0.03,
) -> tuple[FloatArr, list[FloatArr], float]:
    """
    绕质心微调轮廓尺度，使栅格面积相对 mask 误差 ≤ max_area_rel_err。

    用于 Moore 密轮廓固有的半像素偏差；返回 (outer', holes', err)。
    """
    m = np.asarray(mask, dtype=bool)
    mask_a = float(m.sum())
    if mask_a < 1.0:
        return outer, list(holes or []), 0.0
    holes_list = [_ensure_closed(h) for h in (holes or [])]
    outer0 = _ensure_closed(outer)
    ys, xs = np.where(m)
    cx, cy = float(xs.mean()), float(ys.mean())

    def err_at(s: float) -> tuple[float, FloatArr, list[FloatArr]]:
        o = _scale_about(outer0, cx, cy, s)
        hs = [_scale_about(h, cx, cy, s) for h in holes_list]
        a = fitted_region_area(o, hs if hs else None, mask_shape=m.shape)
        return area_relative_error(a, mask_a), o, hs

    best_err, best_o, best_h = err_at(1.0)
    if best_err <= max_area_rel_err:
        return best_o, best_h, best_err

    # 在 [0.85, 1.15] 上粗搜 + 细化
    for s in np.linspace(0.85, 1.15, 31):
        e, o, hs = err_at(float(s))
        if e < best_err:
            best_err, best_o, best_h = e, o, hs
    if best_err <= max_area_rel_err:
        return best_o, best_h, best_err
    # 二分：按面积符号调 s
    lo, hi = 0.85, 1.15
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        e, o, hs = err_at(mid)
        a = fitted_region_area(o, hs if hs else None, mask_shape=m.shape)
        if e < best_err:
            best_err, best_o, best_h = e, o, hs
        if a < mask_a:
            lo = mid
        else:
            hi = mid
        if best_err <= max_area_rel_err:
            break
    return best_o, best_h, float(best_err)


def fit_circle_to_mask(mask: BoolArr, *, n: int = 160) -> tuple[FloatArr, float] | None:
    """
    按 mask 面积与质心拟合圆；面积误差为 0（r 由面积决定）。
    仅当与 mask IoU 足够高时返回。
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    ys, xs = np.where(m)
    area = float(m.sum())
    cx, cy = float(xs.mean()), float(ys.mean())
    r = math.sqrt(max(area / math.pi, 1.0))
    poly = sample_circle_contour(cx, cy, r, n=n)
    iou = _mask_fill_iou(m, poly, m.shape)
    if iou < 0.82:
        return None
    # 面积由构造精确匹配
    return poly, 0.0


def fit_ellipse_to_mask(mask: BoolArr, *, n: int = 160) -> tuple[FloatArr, float] | None:
    """
    用图像矩拟合椭圆，再按面积比缩放使多边形面积贴合 mask（相对误差≈0）。
    IoU 不足则放弃。
    """
    m = np.asarray(mask, dtype=bool)
    if float(m.sum()) < 16:
        return None
    ys, xs = np.where(m)
    cx, cy = float(xs.mean()), float(ys.mean())
    x = xs.astype(np.float64) - cx
    y = ys.astype(np.float64) - cy
    cov = np.array([[np.mean(x * x), np.mean(x * y)], [np.mean(x * y), np.mean(y * y)]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-8)
    eigvecs = eigvecs[:, order]
    # 2σ 椭圆轴
    a = 2.0 * math.sqrt(float(eigvals[0]))
    b = 2.0 * math.sqrt(float(eigvals[1]))
    ang = math.atan2(float(eigvecs[1, 0]), float(eigvecs[0, 0]))
    poly0 = sample_ellipse_contour(cx, cy, max(a, 1.0), max(b, 1.0), ang, n=n)
    a0 = polygon_area(poly0)
    mask_a = float(m.sum())
    if a0 < 1e-6:
        return None
    # 等比缩放使面积贴合
    s = math.sqrt(mask_a / a0)
    poly = sample_ellipse_contour(cx, cy, max(a * s, 1.0), max(b * s, 1.0), ang, n=n)
    err = area_relative_error(polygon_area(poly), mask_a)
    iou = _mask_fill_iou(m, poly, m.shape)
    if iou < 0.78 or err > 0.03:
        return None
    return poly, float(err)


def fit_mask_contour_high_precision(
    mask: BoolArr,
    *,
    max_area_rel_err: float = 0.03,
    resample_n: int = 256,
    use_cc_holes: bool = True,
) -> tuple[FloatArr, list[FloatArr], FloatArr, float]:
    """
    高精度闭合轮廓（与概括色块同一路线，**禁止折线/直线概括**）。

    策略：
    1. Moore 密轮廓（`simplify=0`，不做 RDP）+ 可选 GPU 内洞 CC 补全；
    2. 外环 + 孔洞分别弧长密重采样；
    3. 绕质心微调外环/孔，使栅格面积相对 mask 误差 ≤ 阈值；
    4. **不**用 line / polyline 替换几何。

    返回 (outer_dense, holes_dense, outer_resampled, area_rel_err)。
    holes_dense 含内孔；多部件时其余外环也并入 holes 侧（与 extract_contour_bundle 一致，供 all_rings）。
    """
    m = np.asarray(mask, dtype=bool)
    mask_a = float(m.sum())
    empty = np.zeros((0, 2), dtype=np.float64)
    if mask_a < 1.0:
        return empty, [], empty, 0.0

    rings = extract_all_contours(m, simplify=0.0, min_points=6, min_area=1.0)
    outer: FloatArr | None = None
    holes: list[FloatArr] = []
    if rings:
        outers = [r.numpy() for r in rings if r.kind == "outer"]
        holes = [_ensure_closed(r.numpy()) for r in rings if r.kind == "hole"]
        pool = outers if outers else [r.numpy() for r in rings]
        outer_idx = int(np.argmax([polygon_area(p) for p in pool]))
        outer = _ensure_closed(pool[outer_idx])
        # 其余 outer 并入附加环（多部件）；勿用 `is` 比较数组
        for i, p in enumerate(pool):
            if i == outer_idx:
                continue
            if polygon_area(p) > 4.0:
                holes.append(_ensure_closed(p))
    if outer is None or len(outer) < 4:
        ys, xs = np.where(m)
        x0, x1 = float(xs.min()), float(xs.max()) + 1.0
        y0, y1 = float(ys.min()), float(ys.max()) + 1.0
        outer = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)

    if use_cc_holes:
        holes = _merge_cc_holes_and_extra_outers(m, outer, holes)

    # 丢弃相对主外环过小的噪声孔（抗锯齿/阈值噪声）
    outer_a = max(polygon_area(outer), 1.0)
    holes = [h for h in holes if polygon_area(h) >= max(8.0, 0.004 * outer_a)]

    # 孔洞也做弧长密采样（保持曲线精度，非折线化）
    n_outer = max(int(resample_n), 64)
    outer_rs = resample_closed_contour(outer, n_outer)
    holes_rs: list[FloatArr] = []
    for h in holes:
        if len(h) < 4:
            continue
        n_h = max(48, min(n_outer, max(32, int(len(h) * 1.5))))
        holes_rs.append(resample_closed_contour(h, n_h))

    adj_o, adj_h, err = adjust_contour_area_to_mask(
        outer_rs,
        m,
        holes_rs if holes_rs else None,
        max_area_rel_err=max_area_rel_err,
    )
    # 微调后再次弧长均匀，保证匹配/描述子采样稳定
    outer_final = resample_closed_contour(adj_o, n_outer)
    holes_final = [resample_closed_contour(h, max(48, min(n_outer, len(h)))) for h in adj_h if len(h) >= 4]
    return outer_final, holes_final, outer_final, float(err)


def _orient_ring(pts: FloatArr, *, hole: bool) -> FloatArr:
    """outer 取 CCW（正面积），hole 取 CW（负面积）。"""
    p = _ensure_closed(pts)
    a = signed_area(p)
    if hole and a > 0:
        return p[::-1].copy()
    if (not hole) and a < 0:
        return p[::-1].copy()
    return p


def _ring_centroid(pts: FloatArr) -> tuple[float, float]:
    p = np.asarray(pts, dtype=np.float64)
    return float(p[:, 0].mean()), float(p[:, 1].mean())


def _rings_near_duplicate(a: FloatArr, b: FloatArr, *, dist_thr: float = 3.0) -> bool:
    """质心距 + 面积比粗判重复环。"""
    ca = _ring_centroid(a)
    cb = _ring_centroid(b)
    if math.hypot(ca[0] - cb[0], ca[1] - cb[1]) > dist_thr:
        return False
    aa, ab = polygon_area(a), polygon_area(b)
    if aa < 1e-6 or ab < 1e-6:
        return True
    ratio = min(aa, ab) / max(aa, ab)
    return ratio >= 0.7


def _contour_from_binary_component(comp: BoolArr, *, hole: bool, min_points: int = 6) -> FloatArr | None:
    """对单连通 bool 分量提取闭合轮廓并定向。"""
    if not comp.any():
        return None
    rings = extract_all_contours(comp, simplify=0.0, min_points=min_points, min_area=1.0)
    if not rings:
        return None
    # 分量上最大环
    best = max(rings, key=lambda r: polygon_area(r.numpy()))
    return _orient_ring(best.numpy(), hole=hole)


def _merge_cc_holes_and_extra_outers(
    mask: BoolArr,
    outer: FloatArr,
    holes: list[FloatArr],
) -> list[FloatArr]:
    """
    用 GPU 内洞 CC + 多部件前景 CC 补全/去重 holes 列表。

    - 内洞：~mask 连通域中不触边者 → hole 轮廓
    - 附加外环：其余大前景 CC（非主外环覆盖）→ 并入 holes 侧 all_rings
    """
    m = np.asarray(mask, dtype=bool)
    out: list[FloatArr] = list(holes)
    # 1) 内洞 CC
    try:
        hole_ms = interior_hole_masks(m, min_area=4.0, max_holes=32)
    except Exception:
        hole_ms = []
    for hm in hole_ms:
        cont = _contour_from_binary_component(hm, hole=True)
        if cont is None or len(cont) < 4:
            continue
        if any(_rings_near_duplicate(cont, h) for h in out):
            continue
        out.append(cont)
    # 2) 多部件前景：仅保留相对主部件足够大的次级块（避免抗锯齿碎斑）
    try:
        parts = foreground_cc_masks(m, min_area=max(64.0, float(m.sum()) * 0.05), max_parts=8)
    except Exception:
        parts = []
    if len(parts) > 1:
        # 主部件：与 outer 质心最近且面积最大优先
        ocx, ocy = _ring_centroid(outer)
        scored = []
        for pmask in parts:
            ys, xs = np.where(pmask)
            if len(xs) == 0:
                continue
            cx, cy = float(xs.mean()), float(ys.mean())
            scored.append((int(pmask.sum()), math.hypot(cx - ocx, cy - ocy), pmask))
        scored.sort(key=lambda t: (-t[0], t[1]))
        main_area = max(scored[0][0], 1)
        for area, _d, pmask in scored[1:]:
            # 次级部件至少主部件 12% 才记为附加外环
            if area < 0.12 * main_area:
                continue
            cont = _contour_from_binary_component(pmask, hole=False)
            if cont is None or len(cont) < 4:
                continue
            # 不与主 outer 重复
            if _rings_near_duplicate(cont, outer, dist_thr=4.0):
                continue
            if any(_rings_near_duplicate(cont, h) for h in out):
                continue
            out.append(cont)
    return out


def fit_mask_contour_area_constrained(
    mask: BoolArr,
    *,
    max_area_rel_err: float = 0.03,
    resample_n: int = 192,
) -> tuple[FloatArr, FloatArr, float]:
    """
    闭合色块轮廓：Moore 密跟踪 + 面积微调 + 弧长重采样。

    **唯一路径**；不做 RDP / 直线折线概括。
    返回 (contour, contour_resampled, area_rel_err)。
    """
    outer, _holes, rs, err = fit_mask_contour_high_precision(
        mask,
        max_area_rel_err=max_area_rel_err,
        resample_n=resample_n,
    )
    return outer, rs, err
