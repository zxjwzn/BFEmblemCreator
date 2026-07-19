"""Batch B：通用标签场概括 — 平坦区调色板、空间正则、细丝/小洞、无洞保证。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bf_emblem_creator.approx.color import lab_to_rgb_u8, rgb_to_hex, rgb_u8_to_lab
from bf_emblem_creator.approx.models import PaletteColor

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]
I32Arr = NDArray[np.int32]
BoolArr = NDArray[np.bool_]


def image_gradient_magnitude(rgb: U8Arr) -> FloatArr:
    """RGB 平均灰度的简易梯度幅。"""
    gray = np.asarray(rgb, dtype=np.float64).mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return np.maximum(gx, gy)


def flat_mask(
    rgb: U8Arr,
    alpha: FloatArr,
    *,
    grad_q: float = 0.45,
) -> BoolArr:
    """主体内低梯度平坦区（用于估调色板）。"""
    subject = np.asarray(alpha, dtype=np.float64) >= 0.5
    g = image_gradient_magnitude(rgb)
    if not subject.any():
        return np.zeros_like(subject, dtype=bool)
    thr = float(np.quantile(g[subject], grad_q))
    thr = max(thr, 1.0)
    return subject & (g <= thr)


def kmeans_lab_points(
    pts: FloatArr,
    k: int,
    *,
    iters: int = 15,
    seed: int = 0,
) -> tuple[I32Arr, FloatArr]:
    """CPU LAB k-means；返回 assign (N,) 与 centers (k,3)。"""
    p = np.asarray(pts, dtype=np.float64)
    n = int(p.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros((k, 3), dtype=np.float64)
    k_eff = min(k, n)
    rng = np.random.default_rng(seed)
    centers = np.empty((k_eff, 3), dtype=np.float64)
    centers[0] = p[int(rng.integers(0, n))]
    closest = np.full(n, np.inf)
    for i in range(1, k_eff):
        d = np.linalg.norm(p - centers[i - 1], axis=1)
        closest = np.minimum(closest, d)
        w = closest**2
        s = float(w.sum())
        if s < 1e-12:
            centers[i] = p[int(rng.integers(0, n))]
        else:
            centers[i] = p[int(rng.choice(n, p=w / s))]
    assign = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d = np.linalg.norm(p[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(d, axis=1).astype(np.int32)
        for ci in range(k_eff):
            sel = assign == ci
            if np.any(sel):
                centers[ci] = p[sel].mean(axis=0)
            else:
                centers[ci] = p[int(rng.integers(0, n))]
    return assign, centers


def estimate_palette_flat(
    rgb: U8Arr,
    alpha: FloatArr,
    k: int,
    *,
    grad_q: float = 0.45,
    lab_merge: float = 10.0,
    seed: int = 0,
) -> list[PaletteColor]:
    """
    仅用平坦区像素估计调色板；近色合并。

    高梯度过渡带不参与中心更新，降低杂色中心。
    """
    subject = np.asarray(alpha, dtype=np.float64) >= 0.5
    if not subject.any():
        return []
    flat = flat_mask(rgb, alpha, grad_q=grad_q)
    lab = rgb_u8_to_lab(rgb)
    # 平坦区过少则退回全体主体
    use = flat if float(flat.sum()) >= max(32.0, 0.05 * float(subject.sum())) else subject
    pts = lab[use]
    k_use = max(2, min(16, int(k)))
    _, centers = kmeans_lab_points(pts, k_use, seed=seed)
    # 近色合并
    parent = list(range(len(centers)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if float(np.linalg.norm(centers[i] - centers[j])) < lab_merge:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pj] = pi
    roots = sorted({find(i) for i in range(len(centers))})
    merged: list[FloatArr] = []
    for r in roots:
        members = [i for i in range(len(centers)) if find(i) == r]
        merged.append(np.mean([centers[i] for i in members], axis=0))
    # 用全体主体算 fraction 排序
    all_pts = lab[subject]
    d = np.linalg.norm(all_pts[:, None, :] - np.stack(merged)[None, :, :], axis=2)
    assign = np.argmin(d, axis=1)
    total = float(len(assign))
    order: list[tuple[float, int, tuple[int, int, int]]] = []
    for ci, c in enumerate(merged):
        frac = float((assign == ci).sum()) / max(total, 1.0)
        rgb_c = lab_to_rgb_u8(np.asarray(c, dtype=np.float64)[None, None, :])[0, 0]
        order.append((frac, ci, (int(rgb_c[0]), int(rgb_c[1]), int(rgb_c[2]))))
    order.sort(key=lambda t: -t[0])
    palette: list[PaletteColor] = []
    for frac, _, rgb_c in order:
        palette.append(PaletteColor(hex=rgb_to_hex(rgb_c), fraction=frac, rgb=rgb_c))
    return palette


def assign_labels_hard(
    rgb: U8Arr,
    alpha: FloatArr,
    palette: list[PaletteColor],
) -> I32Arr:
    """主体像素硬分配到最近调色板色；背景 -1。"""
    h, w = alpha.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    if not palette:
        return labels
    subject = np.asarray(alpha, dtype=np.float64) >= 0.5
    if not subject.any():
        return labels
    lab = rgb_u8_to_lab(rgb)
    centers = rgb_u8_to_lab(np.array([p.rgb for p in palette], dtype=np.uint8).reshape(-1, 1, 3))[:, 0, :]
    pts = lab[subject]
    d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    assign = np.argmin(d, axis=1).astype(np.int32)
    labels[subject] = assign
    return labels


def icm_label_refine(
    rgb: U8Arr,
    alpha: FloatArr,
    labels: I32Arr,
    palette: list[PaletteColor],
    *,
    mrf_lambda: float = 2.0,
    iters: int = 5,
) -> I32Arr:
    """
    边界敏感 Potts ICM：颜色相近邻域惩罚切换，原图梯度大处允许边界。
    """
    if iters <= 0 or mrf_lambda <= 0 or not palette:
        return labels
    subject = np.asarray(alpha, dtype=np.float64) >= 0.5
    lab = rgb_u8_to_lab(rgb)
    centers = rgb_u8_to_lab(np.array([p.rgb for p in palette], dtype=np.uint8).reshape(-1, 1, 3))[:, 0, :]
    g = image_gradient_magnitude(rgb)
    g_n = g / (float(np.percentile(g[subject], 90)) + 1e-6) if subject.any() else g
    g_n = np.clip(g_n, 0.0, 2.0)
    out = labels.copy()
    h, w = out.shape
    k = len(palette)
    ys, xs = np.where(subject)
    # 数据项预计算
    data = np.full((h, w, k), 1e6, dtype=np.float64)
    pts = lab[subject]
    d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    data[subject] = d

    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for _ in range(iters):
        changed = 0
        # 扫描顺序交替
        order = np.arange(len(ys))
        for idx in order:
            y, x = int(ys[idx]), int(xs[idx])
            best_e = 1e18
            best_k = int(out[y, x])
            for cand in range(k):
                e = float(data[y, x, cand])
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if not subject[ny, nx]:
                        continue
                    if int(out[ny, nx]) != cand:
                        # 梯度大 → 切换代价低
                        w_edge = mrf_lambda * (1.0 / (1.0 + 2.5 * float(g_n[y, x])))
                        e += w_edge
                if e < best_e:
                    best_e = e
                    best_k = cand
            if best_k != int(out[y, x]):
                out[y, x] = best_k
                changed += 1
        if changed == 0:
            break
    return out


def _label_ccs(binary: BoolArr) -> list[BoolArr]:
    """四连通域列表。"""
    h, w = binary.shape
    vis = np.zeros((h, w), dtype=bool)
    out: list[BoolArr] = []
    from collections import deque

    for y in range(h):
        for x in range(w):
            if not binary[y, x] or vis[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            vis[y, x] = True
            cells: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not vis[ny, nx]:
                        vis[ny, nx] = True
                        q.append((ny, nx))
            m = np.zeros((h, w), dtype=bool)
            for cy, cx in cells:
                m[cy, cx] = True
            out.append(m)
    return out


def _neighbor_majority_label(labels: I32Arr, region: BoolArr, forbid: int) -> int:
    """区域邻域众数标签。"""
    h, w = labels.shape
    from collections import Counter

    counts: Counter[int] = Counter()
    ys, xs = np.where(region)
    for y, x in zip(ys, xs, strict=False):
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w:
                v = int(labels[ny, nx])
                if v >= 0 and v != forbid and not region[ny, nx]:
                    counts[v] += 1
    if not counts:
        # 退回全局最大面积标签
        vals, cnt = np.unique(labels[labels >= 0], return_counts=True)
        if len(vals) == 0:
            return max(forbid, 0)
        return int(vals[int(np.argmax(cnt))])
    return int(counts.most_common(1)[0][0])


def merge_small_components(
    labels: I32Arr,
    alpha: FloatArr,
    *,
    min_area: float,
) -> I32Arr:
    """过小连通域并入邻接众数（禁止标 -1）。"""
    out = labels.copy()
    subject = np.asarray(alpha, dtype=np.float64) >= 0.5
    max_lab = int(out.max()) if out.size and out.max() >= 0 else -1
    for lab in range(max_lab + 1):
        base = (out == lab) & subject
        if not base.any():
            continue
        for cc in _label_ccs(base):
            if float(cc.sum()) < min_area:
                maj = _neighbor_majority_label(out, cc, forbid=lab)
                out[cc] = maj
    return out


def fill_label_gaps(
    labels: I32Arr,
    alpha: FloatArr,
) -> I32Arr:
    """主体内 label<0 用邻域众数 / 最近标签填充，直至无洞。"""
    out = labels.copy()
    subject = np.asarray(alpha, dtype=np.float64) >= 0.5
    h, w = out.shape
    from collections import deque

    # 多轮邻域填充
    for _ in range(64):
        gap = subject & (out < 0)
        if not gap.any():
            break
        ys, xs = np.where(gap)
        changed = False
        for y, x in zip(ys, xs, strict=False):
            votes: list[int] = []
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx] >= 0:
                    votes.append(int(out[ny, nx]))
            if votes:
                from collections import Counter

                out[y, x] = Counter(votes).most_common(1)[0][0]
                changed = True
        if changed:
            continue
        # BFS 最近标签
        dist = np.full((h, w), -1, dtype=np.int32)
        src = np.full((h, w), -1, dtype=np.int32)
        q: deque[tuple[int, int]] = deque()
        known = subject & (out >= 0)
        for y, x in zip(*np.where(known), strict=False):
            dist[y, x] = 0
            src[y, x] = int(out[y, x])
            q.append((int(y), int(x)))
        while q:
            y, x = q.popleft()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or nx < 0 or ny >= h or nx >= w:
                    continue
                if not subject[ny, nx]:
                    continue
                if dist[ny, nx] >= 0:
                    continue
                dist[ny, nx] = dist[y, x] + 1
                src[ny, nx] = src[y, x]
                q.append((ny, nx))
        gap = subject & (out < 0)
        out[gap] = src[gap]
        out[gap & (out < 0)] = 0
        break
    out[~subject] = -1
    return out


def compact_palette_labels(
    labels: I32Arr,
    palette: list[PaletteColor],
    alpha: FloatArr,
) -> tuple[I32Arr, list[PaletteColor]]:
    """压缩未使用标签并按面积重排。"""
    subject = np.asarray(alpha, dtype=np.float64) >= 0.5
    used = sorted({int(v) for v in np.unique(labels) if int(v) >= 0})
    if not used:
        return labels, []
    # 按面积排序
    areas = [(float((labels == u).sum()), u) for u in used]
    areas.sort(key=lambda t: -t[0])
    remap = {old: new for new, (_, old) in enumerate(areas)}
    out = np.full_like(labels, -1)
    for old, new in remap.items():
        out[labels == old] = new
    total = max(1.0, float(subject.sum()))
    new_pal: list[PaletteColor] = []
    for _, old in areas:
        p = palette[old] if old < len(palette) else palette[0]
        frac = float((out == remap[old]).sum()) / total
        new_pal.append(PaletteColor(hex=p.hex, fraction=frac, rgb=p.rgb))
    return out, new_pal


def gap_fraction(labels: I32Arr, alpha: FloatArr) -> float:
    """主体内无标签像素占比。"""
    subject = np.asarray(alpha, dtype=np.float64) >= 0.5
    if not subject.any():
        return 0.0
    return float((subject & (np.asarray(labels) < 0)).sum()) / float(subject.sum())


def noise_fraction(labels: I32Arr, alpha: FloatArr, *, min_area: float) -> float:
    """面积 < min_area 的连通域像素占主体比例。"""
    subject = np.asarray(alpha, dtype=np.float64) >= 0.5
    if not subject.any():
        return 0.0
    noisy = 0
    max_lab = int(labels.max()) if labels.size and labels.max() >= 0 else -1
    for lab in range(max_lab + 1):
        base = (labels == lab) & subject
        for cc in _label_ccs(base):
            a = float(cc.sum())
            if a < min_area:
                noisy += int(a)
    return float(noisy) / float(subject.sum())


def build_label_field(
    rgb: U8Arr,
    alpha: FloatArr,
    k: int,
    *,
    grad_q: float = 0.45,
    lab_merge: float = 10.0,
    mrf_lambda: float = 2.0,
    mrf_iters: int = 5,
    min_area_frac: float = 0.004,
    enforce_no_gap: bool = True,
    seed: int = 0,
) -> tuple[I32Arr, list[PaletteColor], float, float]:
    """
    完整 Batch B：调色板 → 硬分配 → ICM → 小域合并 → 无洞。

    返回 labels, palette, gap_frac, noise_frac。
    """
    h, w = alpha.shape
    min_area = max(1.0, min_area_frac * h * w)
    palette = estimate_palette_flat(rgb, alpha, k, grad_q=grad_q, lab_merge=lab_merge, seed=seed)
    if not palette:
        labels = np.full((h, w), -1, dtype=np.int32)
        return labels, [], 0.0, 0.0
    labels = assign_labels_hard(rgb, alpha, palette)
    labels = icm_label_refine(rgb, alpha, labels, palette, mrf_lambda=mrf_lambda, iters=mrf_iters)
    labels = merge_small_components(labels, alpha, min_area=min_area)
    if enforce_no_gap:
        labels = fill_label_gaps(labels, alpha)
        labels = merge_small_components(labels, alpha, min_area=min_area)
        labels = fill_label_gaps(labels, alpha)
    labels, palette = compact_palette_labels(labels, palette, alpha)
    if enforce_no_gap:
        labels = fill_label_gaps(labels, alpha)
    gf = gap_fraction(labels, alpha)
    nf = noise_fraction(labels, alpha, min_area=min_area)
    return labels, palette, gf, nf


def labels_to_rgb(labels: I32Arr, palette: list[PaletteColor]) -> U8Arr:
    """标签图着色。"""
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, p in enumerate(palette):
        out[labels == i] = np.array(p.rgb, dtype=np.uint8)
    return out
