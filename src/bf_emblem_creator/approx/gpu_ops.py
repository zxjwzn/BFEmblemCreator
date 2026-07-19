"""GPU 张量算子：形态学、连通域、边界、Chamfer、LAB、线条质量。

近似热路径优先 torch（CUDA 可用时在 GPU 执行）。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from bf_emblem_creator.approx.device import get_device

FloatArr = NDArray[np.floating]
U8Arr = NDArray[np.uint8]
BoolArr = NDArray[np.bool_]


def as_device(prefer_cuda: bool = True) -> torch.device:
    """当前计算设备。"""
    return get_device(prefer_cuda=prefer_cuda)


def to_torch(
    arr: NDArray[np.generic] | torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """numpy → torch。"""
    dev = device or as_device()
    if isinstance(arr, torch.Tensor):
        return arr.to(device=dev, dtype=dtype)
    np_arr = np.ascontiguousarray(arr)
    if np_arr.dtype == np.bool_ or np_arr.dtype == np.uint8:
        t = torch.from_numpy(np_arr.astype(np.float32))
    else:
        t = torch.from_numpy(np_arr.astype(np.float32, copy=False))
    return t.to(device=dev, dtype=dtype)


def morph_close_open(
    mask: BoolArr | torch.Tensor,
    *,
    close: int = 2,
    open_: int = 1,
    device: torch.device | None = None,
) -> BoolArr:
    """3×3 最大/最小池化闭开（GPU）。"""
    dev = device or as_device()
    if isinstance(mask, torch.Tensor):
        m = mask.float().to(dev)
        if m.dim() == 2:
            m = m[None, None]
    else:
        m = to_torch(mask.astype(np.float32), device=dev)[None, None]

    def dilate(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def erode(x: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

    for _ in range(close):
        m = dilate(m)
    for _ in range(close):
        m = erode(m)
    for _ in range(open_):
        m = erode(m)
    for _ in range(open_):
        m = dilate(m)
    return (m[0, 0] >= 0.5).detach().cpu().numpy().astype(bool)


def label_connected_components(
    binary: BoolArr | torch.Tensor,
    *,
    device: torch.device | None = None,
    max_labels: int = 512,
) -> list[BoolArr]:
    """四连通域（GPU 标签传播）。"""
    dev = device or as_device()
    fg = binary.to(dev) > 0.5 if isinstance(binary, torch.Tensor) else to_torch(binary.astype(np.float32), device=dev) > 0.5
    if not bool(fg.any().item()):
        return []

    h, w = int(fg.shape[0]), int(fg.shape[1])
    yy = torch.arange(h, device=dev).view(h, 1).expand(h, w)
    xx = torch.arange(w, device=dev).view(1, w).expand(h, w)
    labels = (yy * w + xx + 1).to(torch.float32)
    labels = torch.where(fg, labels, torch.zeros_like(labels))

    for _ in range(max(h, w)):
        prev = labels
        p = F.pad(labels[None, None], (1, 1, 1, 1), value=0.0)[0, 0]
        neigh = torch.stack(
            [
                labels,
                p[1:-1, :-2],
                p[1:-1, 2:],
                p[:-2, 1:-1],
                p[2:, 1:-1],
            ],
            dim=0,
        )
        neigh = torch.where(neigh > 0, neigh, torch.full_like(neigh, float("inf")))
        new_lab = neigh.min(dim=0).values
        new_lab = torch.where(fg, new_lab, torch.zeros_like(new_lab))
        new_lab = torch.where(torch.isfinite(new_lab), new_lab, torch.zeros_like(new_lab))
        labels = new_lab
        if torch.equal(labels, prev):
            break

    uniq = torch.unique(labels)
    uniq = uniq[uniq > 0]
    if uniq.numel() == 0:
        return []
    if int(uniq.numel()) > max_labels:
        areas = torch.stack([(labels == u).sum() for u in uniq])
        top = torch.topk(areas, k=max_labels).indices
        uniq = uniq[top]

    out: list[BoolArr] = []
    for u in uniq.tolist():
        m = (labels == float(u)).detach().cpu().numpy()
        out.append(m.astype(bool))
    return out


def adjacency_edge_lengths(
    region_map: NDArray[np.int32] | torch.Tensor,
    *,
    device: torch.device | None = None,
) -> dict[tuple[int, int], float]:
    """水平/垂直共享边界长度（GPU 筛选 + 聚合）。"""
    dev = device or as_device()
    if isinstance(region_map, torch.Tensor):
        rm = region_map.to(dev).long()
    else:
        rm = torch.from_numpy(np.ascontiguousarray(region_map.astype(np.int64))).to(dev)

    edge_len: dict[tuple[int, int], float] = {}

    def _accumulate(a: torch.Tensor, b: torch.Tensor) -> None:
        mask = (a >= 0) & (b >= 0) & (a != b)
        if not bool(mask.any().item()):
            return
        aa = a[mask]
        bb = b[mask]
        lo = torch.minimum(aa, bb)
        hi = torch.maximum(aa, bb)
        pair = lo * 100000 + hi
        uniq, counts = torch.unique(pair, return_counts=True)
        for p, c in zip(uniq.tolist(), counts.tolist(), strict=False):
            x = int(p) // 100000
            y = int(p) % 100000
            key = (x, y)
            edge_len[key] = edge_len.get(key, 0.0) + float(c)

    _accumulate(rm[:, :-1], rm[:, 1:])
    _accumulate(rm[:-1, :], rm[1:, :])
    return edge_len


def visible_boundary_mask_torch(
    rgb: U8Arr | torch.Tensor,
    alpha: FloatArr | torch.Tensor | None = None,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """可见边界，返回 device 上 bool (H,W)。"""
    dev = device or as_device()
    img = to_torch(rgb, device=dev) if not isinstance(rgb, torch.Tensor) else rgb.float().to(dev)
    h, w = img.shape[0], img.shape[1]
    d = torch.zeros(h, w, device=dev)
    d[:, :-1] = torch.maximum(d[:, :-1], (img[:, :-1] - img[:, 1:]).abs().amax(dim=-1))
    d[:-1, :] = torch.maximum(d[:-1, :], (img[:-1, :] - img[1:, :]).abs().amax(dim=-1))
    color_edge = d > 28.0
    if alpha is None:
        return color_edge
    a = (
        to_torch(np.asarray(alpha, dtype=np.float32), device=dev)
        if not isinstance(alpha, torch.Tensor)
        else alpha.float().to(dev)
    )
    ga = torch.zeros_like(a)
    ga[:, :-1] = torch.maximum(ga[:, :-1], (a[:, :-1] - a[:, 1:]).abs())
    ga[:-1, :] = torch.maximum(ga[:-1, :], (a[:-1, :] - a[1:, :]).abs())
    flat = ga.reshape(-1)
    k = max(1, int(0.1 * flat.numel()))
    thr_t = torch.topk(flat, k=k).values.min()
    thr = max(0.12, float(thr_t.item()) * 0.4)
    return (color_edge | (ga > thr)) & (a > 0.05)


def chamfer_points(
    a: FloatArr | torch.Tensor,
    b: FloatArr | torch.Tensor,
    *,
    device: torch.device | None = None,
    huber_delta: float | None = None,
) -> float:
    """双向 Chamfer 均值（GPU cdist）。"""
    dev = device or as_device()
    ta = to_torch(np.asarray(a, dtype=np.float32), device=dev) if not isinstance(a, torch.Tensor) else a.float().to(dev)
    tb = to_torch(np.asarray(b, dtype=np.float32), device=dev) if not isinstance(b, torch.Tensor) else b.float().to(dev)
    if ta.numel() == 0 or tb.numel() == 0:
        return 50.0
    if ta.dim() != 2:
        ta = ta.reshape(-1, 2)
    if tb.dim() != 2:
        tb = tb.reshape(-1, 2)
    d = torch.cdist(ta, tb)
    d_ab = d.min(dim=1).values
    d_ba = d.min(dim=0).values
    if huber_delta is not None:
        hd = float(huber_delta)
        d_ab = torch.where(d_ab < hd, 0.5 * d_ab**2 / hd, d_ab - 0.5 * hd)
        d_ba = torch.where(d_ba < hd, 0.5 * d_ba**2 / hd, d_ba - 0.5 * hd)
    return float(0.5 * (d_ab.mean() + d_ba.mean()).item())


def mask_to_sdf_fast(
    mask: BoolArr | torch.Tensor,
    *,
    device: torch.device | None = None,
    iters: int = 12,
) -> FloatArr:
    """GPU 快速近似 SDF（箱式模糊差分）。iters 越大距离场越平滑/更远。"""
    dev = device or as_device()
    m = to_torch(mask.astype(np.float32), device=dev) if not isinstance(mask, torch.Tensor) else mask.float().to(dev)
    x = m[None, None]
    n_iter = max(1, int(iters))
    for _ in range(n_iter):
        x = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    soft = x[0, 0]
    # 尺度随迭代略增，保持数值量级可用
    scale = 16.0 * (n_iter / 12.0) ** 0.5
    sdf = (0.5 - soft) * scale
    return sdf.detach().cpu().numpy().astype(np.float64)


def rgb_u8_to_lab_torch(
    rgb: U8Arr | torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """sRGB → 近似 LAB（GPU）。"""
    dev = device or as_device()
    x = to_torch(rgb, device=dev) / 255.0 if not isinstance(rgb, torch.Tensor) else rgb.float().to(dev) / 255.0
    lim = x <= 0.04045
    lin = torch.where(lim, x / 12.92, ((x + 0.055) / 1.055).clamp(min=1e-8) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x_ = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y_ = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z_ = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    xr, yr, zr = x_ / 0.95047, y_, z_ / 1.08883

    def f(t: torch.Tensor) -> torch.Tensor:
        delta = 6.0 / 29.0
        return torch.where(t > delta**3, t.clamp(min=1e-12).pow(1.0 / 3.0), t / (3 * delta * delta) + 4.0 / 29.0)

    fx, fy, fz = f(xr), f(yr), f(zr)
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=-1)


def lab_distance_map(
    rgb_a: U8Arr | torch.Tensor,
    rgb_b: U8Arr | torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """两 RGB 图的 LAB 距离图 (H,W)。"""
    dev = device or as_device()
    la = rgb_u8_to_lab_torch(rgb_a, device=dev)
    lb = rgb_u8_to_lab_torch(rgb_b, device=dev)
    return torch.linalg.norm(la - lb, dim=-1)


def evaluate_line_quality_torch(
    rgb: U8Arr | torch.Tensor,
    alpha: FloatArr | torch.Tensor | None = None,
    *,
    device: torch.device | None = None,
    hard_jagged: float = 0.55,
    hard_corner: float = 0.22,
) -> tuple[float, float, float, float, float, bool]:
    """
    GPU 线条质量。

    返回 score, jaggedness, corner_density, fragment_ratio, edge_frac, hard_fail。
    """
    dev = device or as_device()
    edge = visible_boundary_mask_torch(rgb, alpha, device=dev)
    n_edge = int(edge.sum().item())
    h, w = int(edge.shape[0]), int(edge.shape[1])
    if n_edge < 8:
        return 0.85, 0.0, 0.0, 0.0, 0.0, False

    img = to_torch(rgb, device=dev) if not isinstance(rgb, torch.Tensor) else rgb.float().to(dev)
    gray = img.mean(dim=-1)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    ang = torch.atan2(gy, gx)

    def circ_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dlt = (a - b).abs()
        return torch.minimum(dlt, torch.tensor(2 * math.pi, device=dev) - dlt)

    diffs: list[torch.Tensor] = []
    mask_r = edge[:, :-1] & edge[:, 1:]
    if bool(mask_r.any().item()):
        diffs.append(circ_diff(ang[:, :-1][mask_r], ang[:, 1:][mask_r]))
    mask_d = edge[:-1, :] & edge[1:, :]
    if bool(mask_d.any().item()):
        diffs.append(circ_diff(ang[:-1, :][mask_d], ang[1:, :][mask_d]))
    if diffs:
        arr = torch.cat(diffs)
        jaggedness = float((arr > (math.pi / 4.0)).float().mean().item())
    else:
        jaggedness = 0.0

    lap = -4 * gray + torch.roll(gray, 1, 0) + torch.roll(gray, -1, 0) + torch.roll(gray, 1, 1) + torch.roll(gray, -1, 1)
    corner = edge & (lap.abs() > 40.0)
    corner_density = float(corner.sum().item()) / float(n_edge)
    frags = label_connected_components(edge, device=dev, max_labels=256)
    small = sum(int(f.sum()) for f in frags if int(f.sum()) < 12)
    frag = float(small) / float(n_edge)
    edge_frac = float(n_edge) / float(h * w)

    j_pen = min(1.0, jaggedness / 0.5)
    c_pen = min(1.0, corner_density / 0.25)
    f_pen = min(1.0, frag / 0.5)
    e_pen = min(1.0, max(0.0, (edge_frac - 0.08) / 0.15)) if edge_frac > 0.08 else 0.0
    ugly = 0.4 * j_pen + 0.3 * c_pen + 0.2 * f_pen + 0.1 * e_pen
    score = float(max(0.0, min(1.0, 1.0 - ugly)))
    hard_fail = jaggedness >= hard_jagged or (corner_density >= hard_corner and jaggedness > 0.15) or frag >= 0.65
    if hard_fail:
        score = min(score, 0.35)
    return score, jaggedness, corner_density, frag, edge_frac, hard_fail


def extract_border_points_torch(
    mask: BoolArr | torch.Tensor,
    *,
    device: torch.device | None = None,
    max_points: int = 400,
) -> FloatArr:
    """GPU 提取边界像素并按极角排序，返回 (N,2) numpy。"""
    pts_t = extract_border_points_tensor(mask, device=device, max_points=max_points)
    if pts_t.numel() == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return pts_t.detach().cpu().numpy().astype(np.float64)


def extract_border_points_tensor(
    mask: BoolArr | torch.Tensor,
    *,
    device: torch.device | None = None,
    max_points: int = 400,
) -> torch.Tensor:
    """GPU 边界点 (N,2)，空则 (0,2)。"""
    dev = device or as_device()
    m = to_torch(mask.astype(np.float32), device=dev) if not isinstance(mask, torch.Tensor) else mask.float().to(dev)
    m = m > 0.5
    if not bool(m.any().item()):
        return torch.zeros((0, 2), device=dev)
    p = F.pad(m.float()[None, None], (1, 1, 1, 1), value=0.0)
    full = F.avg_pool2d(p, kernel_size=3, stride=1) >= (1.0 - 1e-5)
    border = m & ~full[0, 0].bool()
    if not bool(border.any().item()):
        border = m
    ys, xs = torch.where(border)
    if xs.numel() == 0:
        return torch.zeros((0, 2), device=dev)
    if int(xs.numel()) > max_points:
        idx = torch.linspace(0, xs.numel() - 1, max_points, device=dev).long()
        xs, ys = xs[idx], ys[idx]
    pts = torch.stack([xs.float(), ys.float()], dim=1)
    center = pts.mean(dim=0)
    ang = torch.atan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[torch.argsort(ang)]


def rdp_torch(points: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    GPU Ramer–Douglas–Peucker（迭代栈实现）。

    points: (N,2)；结果至少保留 3 点（否则退回原点列）。
    """
    if points.shape[0] < 3:
        return points
    keep = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, int(points.shape[0] - 1))]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        seg = points[i0 : i1 + 1]
        start, end = seg[0], seg[-1]
        se = end - start
        norm = torch.linalg.norm(se).clamp_min(1e-12)
        cross = (se[0] * (start[1] - seg[:, 1]) - se[1] * (start[0] - seg[:, 0])).abs()
        dists = cross / norm
        dists[0] = 0.0
        dists[-1] = 0.0
        bi = int(torch.argmax(dists).item())
        if float(dists[bi].item()) > epsilon:
            global_i = i0 + bi
            keep[global_i] = True
            stack.append((i0, global_i))
            stack.append((global_i, i1))
    out = points[keep]
    if out.shape[0] < 3:
        # 保底：均匀抽样
        n = min(16, int(points.shape[0]))
        idx = torch.linspace(0, points.shape[0] - 1, n, device=points.device).long()
        return points[idx]
    return out


def resample_closed_contour_torch(points: torch.Tensor, n: int = 64) -> torch.Tensor:
    """闭合轮廓弧长重采样 (n,2)。"""
    if points.shape[0] < 2:
        return torch.zeros((n, 2), device=points.device)
    pts = points
    if torch.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = torch.cat([pts, pts[:1]], dim=0)
    seg = torch.linalg.norm(pts[1:] - pts[:-1], dim=1)
    u = torch.cat([torch.zeros(1, device=pts.device), torch.cumsum(seg, dim=0)])
    total = float(u[-1].item())
    if total < 1e-8:
        return pts[:1].expand(n, -1).contiguous()
    u = u / total
    samples = torch.linspace(0.0, 1.0, n, device=pts.device, dtype=pts.dtype)
    # 去掉 endpoint 重复：用 [0,1)
    samples = samples * (1.0 - 1e-6)
    # 逐坐标插值
    x = _interp1d(u, pts[:, 0], samples)
    y = _interp1d(u, pts[:, 1], samples)
    return torch.stack([x, y], dim=1)


def _interp1d(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """单调 xp 上的线性插值。"""
    # searchsorted
    idx = torch.searchsorted(xp.contiguous(), x.contiguous(), right=True).clamp(1, xp.numel() - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    t = (x - x0) / (x1 - x0).clamp_min(1e-12)
    return y0 + t * (y1 - y0)


def curvature_descriptor_torch(points: torch.Tensor, bins: int = 16) -> torch.Tensor:
    """转角直方图描述子 (bins,)。"""
    if points.shape[0] < 4:
        return torch.zeros(bins, device=points.device)
    vec = points[1:] - points[:-1]
    ang = torch.atan2(vec[:, 1], vec[:, 0])
    dang = ang[1:] - ang[:-1]
    dang = (dang + math.pi) % (2 * math.pi) - math.pi
    # hist density
    hist = torch.histc(dang, bins=bins, min=-math.pi, max=math.pi)
    s = hist.sum().clamp_min(1e-8)
    return hist / s


def detect_effect_tags_torch(
    mask: BoolArr | torch.Tensor,
    contour: FloatArr | torch.Tensor,
    *,
    circularity: float,
    elongation: float,
    area_frac: float,
    device: torch.device | None = None,
) -> list[str]:
    """效果标签（GPU 几何统计）。"""
    dev = device or as_device()
    tags: list[str] = []
    m = to_torch(mask.astype(np.float32), device=dev) if not isinstance(mask, torch.Tensor) else mask.float().to(dev)
    m = m > 0.5
    if not bool(m.any().item()):
        return tags
    h, w = int(m.shape[0]), int(m.shape[1])
    if area_frac >= 0.35 and circularity >= 0.45 and elongation < 1.8:
        tags.append("solid")
    cy, cx = h // 2, w // 2
    r_in = max(2, min(h, w) // 8)
    yy = torch.arange(h, device=dev).view(h, 1).expand(h, w).float()
    xx = torch.arange(w, device=dev).view(1, w).expand(h, w).float()
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    core = rr2 <= float(r_in**2)
    ring_band = (rr2 <= float((min(h, w) * 0.4) ** 2)) & ~core
    if (
        bool(core.any().item())
        and float(m[core].float().mean().item()) < 0.25
        and bool(ring_band.any().item())
        and float(m[ring_band].float().mean().item()) > 0.2
    ):
        tags.append("ring")
    if isinstance(contour, torch.Tensor):
        pts = contour.float().to(dev)
    else:
        pts = to_torch(np.asarray(contour, dtype=np.float32), device=dev)
    if pts.shape[0] >= 16:
        cxy = pts.mean(dim=0)
        rad = torch.hypot(pts[:, 0] - cxy[0], pts[:, 1] - cxy[1])
        # 循环平滑
        k = 5
        rad_pad = torch.cat([rad[-(k - 1) :], rad, rad[: k - 1]])
        kernel = torch.ones(k, device=dev) / k
        rad_s = F.conv1d(rad_pad[None, None], kernel[None, None], stride=1)[0, 0]
        if rad_s.numel() > rad.numel():
            rad_s = rad_s[: rad.numel()]
        mean_r = float(rad_s.mean().item())
        peaks = 0
        n = int(rad_s.numel())
        for i in range(n):
            left = float(rad_s[i - 1].item())
            mid = float(rad_s[i].item())
            right = float(rad_s[(i + 1) % n].item())
            if mid > left and mid >= right and mid > mean_r:
                peaks += 1
        if peaks >= 5 and circularity < 0.7:
            tags.append("star")
    bys, bxs = torch.where(m)
    if int(bxs.numel()) > 30:
        ang = torch.atan2(bys.float() - cy, bxs.float() - cx)
        hist = torch.histc(ang, bins=16, min=-math.pi, max=math.pi)
        hist = hist / hist.sum().clamp_min(1e-8)
        hs = float(hist.std().item())
        if (hs < 0.04 and 0.08 < area_frac < 0.55 and "star" in tags) or (
            hs < 0.03 and elongation < 1.4 and 0.1 < area_frac < 0.45
        ):
            tags.append("radial")
    mf = m.float()
    gx = (mf[:, 1:] - mf[:, :-1]).abs().mean()
    gy = (mf[1:, :] - mf[:-1, :]).abs().mean()
    gxv, gyv = float(gx.item()), float(gy.item())
    if gxv > 0.02 and gyv > 0.02 and abs(gxv - gyv) / max(gxv + gyv, 1e-6) < 0.35 and area_frac < 0.7:
        row_var = float(mf.mean(dim=1).std().item())
        col_var = float(mf.mean(dim=0).std().item())
        if row_var > 0.08 and col_var > 0.08:
            tags.append("grid")
    if "grid" in tags or "radial" in tags or "star" in tags or (0.15 < area_frac < 0.55 and elongation > 2.5):
        tags.append("gradient_candidate")
    return tags


def depth_order_scores_torch(
    area_fracs: torch.Tensor,
    degrees: torch.Tensor,
    centroids: torch.Tensor,
    masks: torch.Tensor,
    canvas: int,
) -> torch.Tensor:
    """
    批量底层分数（越高越靠下）。

    area_fracs:(R,) degrees:(R,) centroids:(R,2) masks:(R,H,W) bool/float
    """
    r = area_fracs.shape[0]
    max_area = area_fracs.max().clamp_min(1e-8)
    max_deg = degrees.max().clamp_min(1.0)
    s_area = area_fracs / max_area
    s_deg = degrees / max_deg
    cx = centroids[:, 0]
    cy = centroids[:, 1]
    dx = (cx - canvas * 0.5) / max(canvas, 1)
    dy = (cy - canvas * 0.5) / max(canvas, 1)
    s_c = torch.exp(-2.5 * (dx * dx + dy * dy))
    # enclose_count: 质心落在其它 mask 内且面积更小
    enclose = torch.zeros(r, device=area_fracs.device)
    for i in range(r):
        x = int(cx[i].item() + 0.5)
        y = int(cy[i].item() + 0.5)
        h = int(masks.shape[1])
        w = int(masks.shape[2])
        if not (0 <= y < h and 0 <= x < w):
            continue
        for j in range(r):
            if i == j:
                continue
            if bool(masks[j, y, x] > 0.5) and float(area_fracs[i]) < float(area_fracs[j]) * 0.95:
                enclose[i] += 1.0
    s_enc = 1.0 / (1.0 + enclose)
    return 0.45 * s_area + 0.25 * s_deg + 0.15 * s_c + 0.15 * s_enc


def image_diff_mean(
    rgb_a: U8Arr | torch.Tensor,
    alpha_a: FloatArr | torch.Tensor,
    rgb_b: U8Arr | torch.Tensor,
    alpha_b: FloatArr | torch.Tensor,
    *,
    device: torch.device | None = None,
) -> float:
    """两 RGBA 平均可见差异（GPU）。"""
    dev = device or as_device()
    ra = to_torch(rgb_a, device=dev) if not isinstance(rgb_a, torch.Tensor) else rgb_a.float().to(dev)
    rb = to_torch(rgb_b, device=dev) if not isinstance(rgb_b, torch.Tensor) else rgb_b.float().to(dev)
    aa = (
        to_torch(np.asarray(alpha_a, dtype=np.float32), device=dev)
        if not isinstance(alpha_a, torch.Tensor)
        else alpha_a.float().to(dev)
    )
    ab = (
        to_torch(np.asarray(alpha_b, dtype=np.float32), device=dev)
        if not isinstance(alpha_b, torch.Tensor)
        else alpha_b.float().to(dev)
    )
    d = torch.linalg.norm(ra - rb, dim=-1) / 255.0
    da = (aa - ab).abs()
    return float((d + da).mean().item())
