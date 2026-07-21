"""共享边宏观特征钉扎三次贝塞尔拟合。

dual 精确描边是 1px 轴对齐阶梯，相邻点转角几乎处处 ~90°。
本模块用**弧长窗口**估计宏观切向，只把形状尺度上的尖角钉住；
阶梯状「锯齿起伏」落在锚点之间，由贝塞尔/直线有损吸收。
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.planar_map import PlanarMap, SharedEdge

FloatArr = NDArray[np.floating]


class EdgeCurveFitReport(BaseModel):
    """单次平面图曲线拟合摘要（供调试与日志）。"""

    model_config = ConfigDict(extra="forbid")

    edges_total: int = Field(default=0, description="共享边总数")
    edges_fitted: int = Field(default=0, description="成功贝塞尔替换的边数")
    edges_skipped: int = Field(default=0, description="未触发拟合（点数不足阈值）")
    edges_fallback: int = Field(default=0, description="无法构造有效结果而回退密折线")
    vertices_before: int = Field(default=0, description="拟合前顶点总数")
    vertices_after: int = Field(default=0, description="拟合后顶点总数")


def _dedup_poly(poly: FloatArr) -> FloatArr:
    """去掉连续重复点。"""
    arr = np.asarray(poly, dtype=np.float64)
    if len(arr) < 2:
        return arr.reshape(-1, 2) if arr.size else np.zeros((0, 2), dtype=np.float64)
    keep = [arr[0]]
    for i in range(1, len(arr)):
        if float(np.linalg.norm(arr[i] - keep[-1])) > 1e-9:
            keep.append(arr[i])
    return np.asarray(keep, dtype=np.float64)


def _is_closed(poly: FloatArr, tol: float = 1e-3) -> bool:
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 3:
        return False
    return float(np.linalg.norm(p[0] - p[-1])) <= tol


def _open_body(poly: FloatArr) -> tuple[FloatArr, bool]:
    """返回去闭合重复点后的点列与是否闭合。"""
    p = _dedup_poly(poly)
    closed = _is_closed(p)
    if closed and len(p) >= 2:
        return p[:-1].copy(), True
    return p, False


def _edge_lengths(body: FloatArr, closed: bool) -> FloatArr:
    """相邻顶点弦长；闭合时含末→首。"""
    n = len(body)
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    if closed:
        nxt = np.roll(body, -1, axis=0)
        return np.linalg.norm(nxt - body, axis=1)
    return np.linalg.norm(np.diff(body, axis=0), axis=1)


def _point_at_arclen(
    body: FloatArr,
    edge_len: FloatArr,
    s_query: float,
    *,
    closed: bool,
) -> FloatArr:
    """
    沿折线弧长取点。

    - 开放：s 钳制到 [0, total]
    - 闭合：s 对 total 取模
    """
    n = len(body)
    if n == 0:
        return np.zeros(2, dtype=np.float64)
    if n == 1:
        return body[0].copy()
    total = float(edge_len.sum())
    if total < 1e-12:
        return body[0].copy()
    if closed:
        s = float(s_query) % total
        if s < 0:
            s += total
    else:
        s = float(np.clip(s_query, 0.0, total))
    acc = 0.0
    n_edges = n if closed else n - 1
    for i in range(n_edges):
        el = float(edge_len[i])
        if acc + el >= s - 1e-12:
            t = 0.0 if el < 1e-12 else (s - acc) / el
            j = (i + 1) % n if closed else i + 1
            return (1.0 - t) * body[i] + t * body[j]
        acc += el
    return body[-1].copy() if not closed else body[0].copy()


def _vertex_arclen(edge_len: FloatArr, closed: bool) -> FloatArr:
    """各顶点从起点量起的弧长坐标（闭合时最后一顶点不重复 total）。"""
    if len(edge_len) == 0:
        return np.zeros(1, dtype=np.float64)
    if closed:
        # edge_len[i] = vertex i → i+1；顶点 i 的弧长 = sum(edge_len[:i])
        return np.concatenate([[0.0], np.cumsum(edge_len[:-1])])
    return np.concatenate([[0.0], np.cumsum(edge_len)])


def _macro_corner_anchors(
    body: FloatArr,
    *,
    corner_deg: float,
    closed: bool,
    smooth_radius_px: float,
    min_anchor_spacing_px: float,
) -> list[int]:
    """
    宏观角点锚钉（抑制 dual 1px 阶梯假尖角）。

    在每个顶点用前后各 smooth_radius 弧长处的点估计切向，
    仅当宏观转角 ≥ corner_deg 时钉住；再按弧长间距合并过近锚点。
    端点（开放）始终保留。
    """
    n = len(body)
    if n < 2:
        return list(range(n))
    if n < 4:
        return [0, n - 1] if not closed else [0]

    edge_len = _edge_lengths(body, closed)
    total = float(edge_len.sum())
    if total < 1e-9:
        return [0, n - 1] if not closed else [0]

    # 窗口至少覆盖数个 dual 台阶，且不超过半周
    win = float(np.clip(smooth_radius_px, 2.0, max(2.0, total * 0.25)))
    thr = math.radians(float(corner_deg))
    s_at = _vertex_arclen(edge_len, closed)

    raw_idx: list[int] = []
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            continue
        si = float(s_at[i])
        p_prev = _point_at_arclen(body, edge_len, si - win, closed=closed)
        p_next = _point_at_arclen(body, edge_len, si + win, closed=closed)
        v1 = body[i] - p_prev
        v2 = p_next - body[i]
        n1 = float(np.linalg.norm(v1)) + 1e-12
        n2 = float(np.linalg.norm(v2)) + 1e-12
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        turn = math.acos(cos_a)
        if turn >= thr:
            raw_idx.append(i)

    # 按弧长间距合并：同一簇保留转角最大者
    min_sp = max(1.0, float(min_anchor_spacing_px))
    scored: list[tuple[int, float]] = []
    for i in raw_idx:
        si = float(s_at[i])
        p_prev = _point_at_arclen(body, edge_len, si - win, closed=closed)
        p_next = _point_at_arclen(body, edge_len, si + win, closed=closed)
        v1 = body[i] - p_prev
        v2 = p_next - body[i]
        n1 = float(np.linalg.norm(v1)) + 1e-12
        n2 = float(np.linalg.norm(v2)) + 1e-12
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        scored.append((i, math.acos(cos_a)))
    scored.sort(key=lambda t: -t[1])

    chosen: list[int] = []
    chosen_s: list[float] = []
    for i, _turn in scored:
        si = float(s_at[i])
        ok = True
        for sj in chosen_s:
            ds = abs(si - sj)
            if closed:
                ds = min(ds, total - ds)
            if ds < min_sp:
                ok = False
                break
        if ok:
            chosen.append(i)
            chosen_s.append(si)

    anchors: set[int] = set(chosen)
    if not closed:
        anchors.add(0)
        anchors.add(n - 1)
    elif not anchors:
        # 闭合且无宏观角：钉一个起点便于分段
        anchors.add(0)

    return sorted(anchors)


def _cubic_bezier(p0: FloatArr, p1: FloatArr, p2: FloatArr, p3: FloatArr, t: FloatArr) -> FloatArr:
    """三次贝塞尔批量求值；t shape (M,)。"""
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    u = 1.0 - t
    return (u**3) * p0 + 3 * (u**2) * t * p1 + 3 * u * (t**2) * p2 + (t**3) * p3


def _fit_bezier_segment(seg: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
    """
    对开放折线段拟合三次贝塞尔（端点固定）。

    端点切向用段内前/后若干点的宏观方向（抗 dual 阶梯噪声）。
    """
    s = np.asarray(seg, dtype=np.float64)
    p0 = s[0].copy()
    p3 = s[-1].copy()
    if len(s) == 2:
        p1 = p0 + (p3 - p0) / 3.0
        p2 = p0 + 2.0 * (p3 - p0) / 3.0
        return p0, p1, p2, p3
    d = np.linalg.norm(np.diff(s, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    total = float(u[-1])
    if total < 1e-12:
        p1 = p0 + (p3 - p0) / 3.0
        p2 = p0 + 2.0 * (p3 - p0) / 3.0
        return p0, p1, p2, p3
    u = u / total
    # 宏观切向：从端点沿弧长约 20% 处
    i_fwd = int(np.searchsorted(u, 0.2, side="left"))
    i_fwd = int(np.clip(i_fwd, 1, len(s) - 1))
    i_bwd = int(np.searchsorted(u, 0.8, side="left"))
    i_bwd = int(np.clip(i_bwd, 0, len(s) - 2))
    t0 = s[i_fwd] - s[0]
    t1 = s[-1] - s[i_bwd]
    n0 = float(np.linalg.norm(t0)) + 1e-12
    n1 = float(np.linalg.norm(t1)) + 1e-12
    t0 = t0 / n0
    t1 = t1 / n1
    chord = float(np.linalg.norm(p3 - p0))
    alpha = max(chord / 3.0, total / 3.0)
    p1 = p0 + t0 * alpha
    p2 = p3 - t1 * alpha
    if len(s) >= 4:
        ts = u[1:-1]
        pred = _cubic_bezier(p0, p1, p2, p3, ts)
        err = s[1:-1] - pred
        p1 = p1 + err.mean(axis=0) * 0.5
        p2 = p2 + err.mean(axis=0) * 0.5
    return p0, p1, p2, p3


def _sample_bezier(p0: FloatArr, p1: FloatArr, p2: FloatArr, p3: FloatArr, n: int, *, include_end: bool) -> FloatArr:
    """采样段内点。"""
    n = max(2, int(n))
    t = np.linspace(0.0, 1.0, n) if include_end else np.linspace(0.0, 1.0, n, endpoint=False)
    return _cubic_bezier(p0, p1, p2, p3, t)


def _is_nearly_line(seg: FloatArr, eps: float) -> bool:
    """弦到点偏差均 ≤ eps 则视为直线（允许吸收 dual 小阶梯）。"""
    if len(seg) < 3:
        return True
    a, b = seg[0], seg[-1]
    w = b - a
    ww = float(np.dot(w, w))
    if ww < 1e-18:
        return True
    for p in seg[1:-1]:
        t = float(np.clip(np.dot(p - a, w) / ww, 0.0, 1.0))
        proj = a + t * w
        if float(np.linalg.norm(p - proj)) > eps:
            return False
    return True


def _poly_arc_length(poly: FloatArr) -> float:
    p = _dedup_poly(poly)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def fit_polyline_bezier(
    poly: FloatArr,
    *,
    max_vertices: int = 8,
    min_arc_length_px: float = 6.0,
    line_flat_eps_px: float = 2.0,
    corner_deg: float = 50.0,
    samples_per_seg: int = 6,
    smooth_radius_px: float = 8.0,
    min_anchor_spacing_px: float = 10.0,
) -> tuple[FloatArr, str]:
    """
    dual 密折线 → 宏观锚点钉扎 + 段内有损贝塞尔/直线。

    - **钉住**：形状尺度尖角与开放端点（不省略宏观尖角）；
    - **概括**：锚点之间的像素阶梯/小起伏用曲线或直线吸收；
    - 触发：顶点数 > max_vertices **或** 弧长 > min_arc_length_px（短 dense 边也会拟合）；
    - 无点距/面积硬回退。

    返回 (新折线, status∈{skip,fitted,fallback})。
    """
    raw = _dedup_poly(poly)
    arc = _poly_arc_length(raw)
    # 极短边跳过；其余只要够长或够密就拟合（避免「≤24 点整边 skip」）
    if len(raw) < 3 or arc < 2.0:
        return raw, "skip"
    if len(raw) <= max_vertices and arc <= min_arc_length_px:
        return raw, "skip"
    body, closed = _open_body(raw)
    if len(body) < 3:
        return raw, "skip"

    anchors = _macro_corner_anchors(
        body,
        corner_deg=corner_deg,
        closed=closed,
        smooth_radius_px=smooth_radius_px,
        min_anchor_spacing_px=min_anchor_spacing_px,
    )
    if closed and 0 not in anchors and len(anchors) >= 1:
        # 闭合时保证有稳定分段起点
        anchors = sorted({0, *anchors})
    if len(anchors) < 2:
        anchors = [0, len(body) - 1] if not closed else [0, max(1, len(body) // 2)]

    out_parts: list[FloatArr] = []
    n_anchor = len(anchors)
    pairs = n_anchor if closed else n_anchor - 1
    for k in range(pairs):
        i0 = anchors[k]
        i1 = anchors[(k + 1) % n_anchor] if closed else anchors[k + 1]
        if closed:
            if i1 > i0:
                seg = body[i0 : i1 + 1]
            elif i1 == i0:
                seg = np.vstack([body[i0:], body[: i0 + 1]])
            else:
                seg = np.vstack([body[i0:], body[: i1 + 1]])
        else:
            seg = body[i0 : i1 + 1]
        if len(seg) < 2:
            continue
        is_last = k == pairs - 1
        # 宏观近直（含阶梯锯齿）：压成端点连线
        if _is_nearly_line(seg, line_flat_eps_px):
            if is_last and not closed:
                out_parts.append(seg[[0, -1]])
            else:
                out_parts.append(seg[:1])
            continue
        p0, p1, p2, p3 = _fit_bezier_segment(seg)
        # 采样密度随段长，但远少于 dual 点数
        n_samp = max(samples_per_seg, min(samples_per_seg * 3, max(4, len(seg) // 4)))
        include_end = is_last and not closed
        samples = _sample_bezier(p0, p1, p2, p3, n_samp, include_end=include_end)
        out_parts.append(samples)

    if not out_parts:
        return raw, "fallback"

    fitted = np.vstack(out_parts)
    if closed:
        if float(np.linalg.norm(fitted[0] - fitted[-1])) > 1e-6:
            fitted = np.vstack([fitted, fitted[:1]])
    else:
        end = body[-1]
        if float(np.linalg.norm(fitted[-1] - end)) > 1e-6:
            fitted = np.vstack([fitted, end.reshape(1, 2)])

    fitted = _dedup_poly(fitted)
    if len(fitted) < 2:
        return raw, "fallback"
    return fitted, "fitted"


def _face_pair_key(e: SharedEdge) -> tuple[int, int]:
    a, b = int(e.left_face), int(e.right_face)
    return (min(a, b), max(a, b))


def _endpoint_key(p: FloatArr, tol: float = 0.6) -> tuple[int, int]:
    """端点量化键（像素级），用于链拼接。"""
    return (round(float(p[0]) / tol), round(float(p[1]) / tol))


def _chain_edges_same_pair(edges: list[SharedEdge]) -> list[list[tuple[SharedEdge, bool]]]:
    """
    将同 face 对的边按端点串成链。

    返回每条链：[(edge, reversed), ...]，reversed 表示沿链前进时取 polyline 逆序。
    """
    if not edges:
        return []
    remaining = set(range(len(edges)))
    chains: list[list[tuple[SharedEdge, bool]]] = []

    def ends(e: SharedEdge) -> tuple[FloatArr, FloatArr]:
        poly = _dedup_poly(np.asarray(e.polyline, dtype=np.float64))
        if len(poly) == 0:
            z = np.zeros(2, dtype=np.float64)
            return z, z
        if len(poly) == 1:
            return poly[0], poly[0]
        return poly[0], poly[-1]

    while remaining:
        i0 = min(remaining)
        remaining.remove(i0)
        e0 = edges[i0]
        chain: list[tuple[SharedEdge, bool]] = [(e0, False)]
        _a0, b0 = ends(e0)
        head_k = _endpoint_key(_a0)
        tail_k = _endpoint_key(b0)
        # 向尾延伸
        grew = True
        while grew and remaining:
            grew = False
            for j in list(remaining):
                e = edges[j]
                a, b = ends(e)
                ka, kb = _endpoint_key(a), _endpoint_key(b)
                if ka == tail_k:
                    chain.append((e, False))
                    tail_k = kb
                    remaining.remove(j)
                    grew = True
                    break
                if kb == tail_k:
                    chain.append((e, True))
                    tail_k = ka
                    remaining.remove(j)
                    grew = True
                    break
        # 向头延伸
        grew = True
        while grew and remaining:
            grew = False
            for j in list(remaining):
                e = edges[j]
                a, b = ends(e)
                ka, kb = _endpoint_key(a), _endpoint_key(b)
                if kb == head_k:
                    chain.insert(0, (e, False))
                    head_k = ka
                    remaining.remove(j)
                    grew = True
                    break
                if ka == head_k:
                    chain.insert(0, (e, True))
                    head_k = kb
                    remaining.remove(j)
                    grew = True
                    break
        chains.append(chain)
    return chains


def _merge_chain_polyline(chain: list[tuple[SharedEdge, bool]]) -> tuple[FloatArr, list[float]]:
    """
    合并链为一条折线，并返回每条原边在合并折线上的弧长占比 cum。

    返回 (merged_poly, edge_arc_lengths) 与 chain 等长。
    """
    parts: list[FloatArr] = []
    edge_arcs: list[float] = []
    for e, rev in chain:
        poly = _dedup_poly(np.asarray(e.polyline, dtype=np.float64))
        if rev and len(poly) >= 2:
            poly = poly[::-1].copy()
        if len(poly) == 0:
            edge_arcs.append(0.0)
            continue
        if parts:
            # 去重接点
            if float(np.linalg.norm(parts[-1][-1] - poly[0])) < 1e-6:
                poly = poly[1:]
            if len(poly) == 0:
                edge_arcs.append(0.0)
                continue
        arc = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()) if len(poly) >= 2 else 0.0
        edge_arcs.append(arc)
        parts.append(poly)
    if not parts:
        return np.zeros((0, 2), dtype=np.float64), edge_arcs
    merged = _dedup_poly(np.vstack(parts))
    return merged, edge_arcs


def _split_polyline_by_arc_lengths(poly: FloatArr, edge_arcs: list[float]) -> list[FloatArr]:
    """按各边原弧长比例切开拟合后的折线，保证每段至少 2 点。"""
    p = _dedup_poly(poly)
    if len(p) < 2 or not edge_arcs:
        return [p.copy() for _ in edge_arcs]
    total_w = float(sum(max(a, 1e-6) for a in edge_arcs))
    # 累积弧长表
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total_len = float(cum[-1]) if len(cum) else 0.0
    if total_len < 1e-12:
        return [p.copy() for _ in edge_arcs]

    def point_at(s: float) -> FloatArr:
        s = float(np.clip(s, 0.0, total_len))
        i = int(np.searchsorted(cum, s, side="right") - 1)
        i = int(np.clip(i, 0, len(p) - 2))
        t = 0.0 if seg[i] < 1e-12 else (s - cum[i]) / seg[i]
        return (1.0 - t) * p[i] + t * p[i + 1]

    out: list[FloatArr] = []
    s0 = 0.0
    for k, arc in enumerate(edge_arcs):
        w = max(arc, 1e-6) / total_w
        s1 = total_len if k == len(edge_arcs) - 1 else s0 + w * total_len
        # 取 [s0,s1] 内顶点 + 端点
        pts = [point_at(s0)]
        for i in range(1, len(p) - 1):
            if s0 < cum[i] < s1:
                pts.append(p[i])
        pts.append(point_at(s1))
        seg_poly = _dedup_poly(np.asarray(pts, dtype=np.float64))
        if len(seg_poly) < 2:
            seg_poly = np.vstack([point_at(s0), point_at(s1)])
        out.append(seg_poly)
        s0 = s1
    return out


def simplify_planar_map_curves(
    pmap: PlanarMap,
    *,
    max_vertices: int = 8,
    min_arc_length_px: float = 6.0,
    line_flat_eps_px: float = 2.0,
    corner_deg: float = 50.0,
    samples_per_seg: int = 6,
    smooth_radius_px: float = 8.0,
    min_anchor_spacing_px: float = 10.0,
) -> EdgeCurveFitReport:
    """
    原地简化 PlanarMap 共享边。

    1. 按 face 对分组；
    2. 端点相接的边串成 chain；
    3. 整条 chain 做宏观贝塞尔拟合；
    4. 按原弧长比例切回各 SharedEdge（拓扑 id 不变）。
    """
    report = EdgeCurveFitReport(edges_total=len(pmap.edges))
    if not pmap.edges:
        return report

    for e in pmap.edges:
        report.vertices_before += len(np.asarray(e.polyline))

    # face 对 → edges
    groups: dict[tuple[int, int], list[SharedEdge]] = {}
    for e in pmap.edges:
        groups.setdefault(_face_pair_key(e), []).append(e)

    for _pair, elist in groups.items():
        chains = _chain_edges_same_pair(elist)
        for chain in chains:
            merged, edge_arcs = _merge_chain_polyline(chain)
            new_poly, status = fit_polyline_bezier(
                merged,
                max_vertices=max_vertices,
                min_arc_length_px=min_arc_length_px,
                line_flat_eps_px=line_flat_eps_px,
                corner_deg=corner_deg,
                samples_per_seg=samples_per_seg,
                smooth_radius_px=smooth_radius_px,
                min_anchor_spacing_px=min_anchor_spacing_px,
            )
            if status == "skip":
                report.edges_skipped += len(chain)
                for e, _rev in chain:
                    report.vertices_after += len(np.asarray(e.polyline))
                continue
            if status == "fallback" or len(new_poly) < 2:
                report.edges_fallback += len(chain)
                for e, _rev in chain:
                    report.vertices_after += len(np.asarray(e.polyline))
                continue
            # 切回各边；注意 chain 中 reverse 的边要写回时再翻回 v0→v1 语义
            pieces = _split_polyline_by_arc_lengths(new_poly, edge_arcs)
            for (e, rev), piece in zip(chain, pieces, strict=False):
                poly_out = np.asarray(piece, dtype=np.float64)
                if rev and len(poly_out) >= 2:
                    poly_out = poly_out[::-1].copy()
                e.polyline = poly_out
                e.length = _poly_arc_length(poly_out)
                report.vertices_after += len(poly_out)
            report.edges_fitted += len(chain)
    return report


def should_curve_fit_policy(boundary_policy: str) -> bool:
    """boundary_policy 为 curve_fit 时启用。"""
    return str(boundary_policy) == "curve_fit"


def collect_edge_polylines(pmap: PlanarMap) -> list[tuple[int, FloatArr, bool]]:
    """调试用：[(edge_id, polyline, closed), ...]。"""
    out: list[tuple[int, FloatArr, bool]] = []
    for e in pmap.edges:
        poly = np.asarray(e.polyline, dtype=np.float64)
        out.append((int(e.id), poly, _is_closed(poly)))
    return out
