"""原始图像概括：对齐画布、抠主体、平滑、色量、规整、权重。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from bf_emblem_creator.approx.color import lab_to_rgb_u8, rgb_to_hex, rgb_u8_to_lab
from bf_emblem_creator.approx.models import (
    AbstractionMode,
    ApproxConfig,
    ApproxMeta,
    ApproxTarget,
    LayerHint,
    PaletteColor,
)

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]
I32Arr = NDArray[np.int32]


def _load_rgba(image: Image.Image | str | Path | U8Arr) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGBA")
    arr = np.asarray(image)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGBA")
    if arr.shape[-1] == 3:
        return Image.fromarray(arr.astype(np.uint8), mode="RGB").convert("RGBA")
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def detect_mode(rgba: Image.Image) -> AbstractionMode:
    """粗分类概括模式。"""
    arr = np.asarray(rgba, dtype=np.uint8)
    alpha = arr[:, :, 3].astype(np.float64) / 255.0
    if float(alpha.mean()) < 0.95 and float((alpha > 0.1).mean()) < 0.85:
        # 大量透明：像 emoji/贴纸
        rgb = arr[:, :, :3]
        mask = alpha > 0.5
        if not mask.any():
            return AbstractionMode.silhouette
        colors = rgb[mask]
        # 粗估计独特色
        q = (colors // 32).astype(np.int32)
        uniq = {tuple(map(int, row)) for row in q}
        if len(uniq) <= 12:
            return AbstractionMode.logo
        return AbstractionMode.illustration
    # 不透明图
    return AbstractionMode.illustration


def fit_to_canvas(
    rgba: Image.Image,
    size: int,
    *,
    how: str,
) -> tuple[U8Arr, ApproxMeta]:
    """将图像 fit 到 size×size，返回 RGBA uint8 与元数据。"""
    src_w, src_h = rgba.size
    if how not in {"contain", "cover"}:
        raise ValueError("how 必须是 contain 或 cover")

    scale = min(size / src_w, size / src_h) if how == "contain" else max(size / src_w, size / src_h)

    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    resized = rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    canvas.paste(resized, (ox, oy), resized)
    meta = ApproxMeta(
        source_width=src_w,
        source_height=src_h,
        canvas_size=size,
        fit=how,
        mode=AbstractionMode.auto,  # 稍后覆盖
        scale=scale,
        offset_x=float(ox),
        offset_y=float(oy),
    )
    return np.asarray(canvas, dtype=np.uint8), meta


def estimate_alpha(rgba: U8Arr, mode: AbstractionMode) -> FloatArr:
    """估计主体蒙版。"""
    a = rgba[:, :, 3].astype(np.float64) / 255.0
    # 已有有效 alpha（如 emoji）则直接用，否则角点洪水估背景
    alpha = (
        a
        if float(a.max()) > 0.05 and float(a.mean()) < 0.98
        else _flood_background_alpha(rgba[:, :, :3])
    )

    alpha = _morph_cleanup_alpha(alpha)
    # 保留最大连通域（避免碎屑）
    binary = alpha >= 0.5
    kept = _keep_largest_cc(binary)
    alpha = np.where(kept, alpha, 0.0)
    if mode == AbstractionMode.silhouette:
        alpha = (alpha >= 0.5).astype(np.float64)
    return alpha


def _flood_background_alpha(rgb: U8Arr, tol: float = 28.0) -> FloatArr:
    """从四角洪水填充近似背景，返回前景 alpha。"""
    h, w = rgb.shape[:2]
    lab = rgb_u8_to_lab(rgb)
    vis = np.zeros((h, w), dtype=bool)
    from collections import deque

    q: deque[tuple[int, int]] = deque()
    seeds = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    # 角点中位色作背景参考
    seed_cols = [lab[y, x] for y, x in seeds]
    bg = np.median(np.stack(seed_cols, axis=0), axis=0)
    for y, x in seeds:
        q.append((y, x))
        vis[y, x] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or nx < 0 or ny >= h or nx >= w or vis[ny, nx]:
                continue
            if float(np.linalg.norm(lab[ny, nx] - bg)) <= tol:
                vis[ny, nx] = True
                q.append((ny, nx))
    # vis=背景
    fg = ~vis
    if fg.mean() < 0.02 or fg.mean() > 0.98:
        return np.ones((h, w), dtype=np.float64)
    return fg.astype(np.float64)


def _morph_cleanup_alpha(alpha: FloatArr) -> FloatArr:
    """对 alpha 做简单开闭。"""
    img = Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.MaxFilter(3))
    img = img.filter(ImageFilter.MinFilter(3))
    img = img.filter(ImageFilter.MinFilter(3))
    img = img.filter(ImageFilter.MaxFilter(3))
    return np.asarray(img, dtype=np.float64) / 255.0


def _keep_largest_cc(binary: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """四连通最大连通域。"""
    h, w = binary.shape
    labels = -np.ones((h, w), dtype=np.int32)
    areas: list[int] = []
    from collections import deque

    cur = 0
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or labels[y, x] >= 0:
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            labels[y, x] = cur
            area = 0
            while q:
                cy, cx = q.popleft()
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if binary[ny, nx] and labels[ny, nx] < 0:
                        labels[ny, nx] = cur
                        q.append((ny, nx))
            areas.append(area)
            cur += 1
    if not areas:
        return binary
    best = int(np.argmax(areas))
    return labels == best


def bilateral_smooth(rgb: U8Arr, *, strength: str) -> U8Arr:
    """
    轻量保边平滑（无 OpenCV 时的近似：中值 + 轻高斯，按 alpha 边界可接受）。

    strength: weak / medium / strong
    """
    img = Image.fromarray(rgb, mode="RGB")
    if strength == "weak":
        return rgb
    if strength == "medium":
        img = img.filter(ImageFilter.MedianFilter(size=3))
        return np.asarray(img, dtype=np.uint8)
    # strong
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SMOOTH_MORE)
    return np.asarray(img, dtype=np.uint8)


def quantize_lab(
    rgb: U8Arr,
    alpha: FloatArr,
    k: int,
    *,
    seed: int = 0,
) -> tuple[I32Arr, list[PaletteColor]]:
    """在 LAB 空间对主体像素 k-means 色量。"""
    h, w = alpha.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    mask = alpha >= 0.5
    if not mask.any():
        return labels, []

    lab = rgb_u8_to_lab(rgb)
    pts = lab[mask]  # N,3
    rng = np.random.default_rng(seed)
    k_eff = min(k, max(1, int(pts.shape[0])))
    # k-means++ 初始化
    centers = _kmeans_pp_init(pts, k_eff, rng)
    assign = np.zeros(pts.shape[0], dtype=np.int32)
    for _ in range(12):
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(d, axis=1).astype(np.int32)
        for ci in range(k_eff):
            sel = assign == ci
            if not np.any(sel):
                centers[ci] = pts[int(rng.integers(0, pts.shape[0]))]
            else:
                centers[ci] = pts[sel].mean(axis=0)

    labels[mask] = assign
    # 调色板
    palette: list[PaletteColor] = []
    total = float(mask.sum())
    order = []
    for ci in range(k_eff):
        frac = float((assign == ci).sum()) / total
        rgb_c = lab_to_rgb_u8(centers[ci][None, None, :])[0, 0]
        order.append((frac, ci, rgb_c))
    order.sort(key=lambda t: -t[0])

    # 重映射标签按面积排序
    remap = {ci: new_i for new_i, (_, ci, _) in enumerate(order)}
    flat = labels.ravel()
    out_flat = flat.copy()
    for old, new in remap.items():
        out_flat[flat == old] = new
    labels = out_flat.reshape(h, w)

    for frac, _, rgb_c in order:
        r, g, b = int(rgb_c[0]), int(rgb_c[1]), int(rgb_c[2])
        palette.append(
            PaletteColor(hex=rgb_to_hex((r, g, b)), fraction=frac, rgb=(r, g, b))
        )
    return labels, palette


def _kmeans_pp_init(pts: FloatArr, k: int, rng: np.random.Generator) -> FloatArr:
    n = pts.shape[0]
    centers = np.empty((k, pts.shape[1]), dtype=np.float64)
    centers[0] = pts[int(rng.integers(0, n))]
    closest = np.full(n, np.inf)
    for i in range(1, k):
        d = np.linalg.norm(pts - centers[i - 1], axis=1)
        closest = np.minimum(closest, d)
        p = closest**2
        s = p.sum()
        if s < 1e-12:
            centers[i] = pts[int(rng.integers(0, n))]
        else:
            centers[i] = pts[int(rng.choice(n, p=p / s))]
    return centers


def merge_tiny_and_close(
    labels: I32Arr,
    palette: list[PaletteColor],
    alpha: FloatArr,
    *,
    min_frac: float = 0.005,
    lab_merge: float = 10.0,
) -> tuple[I32Arr, list[PaletteColor]]:
    """合并过小色与 LAB 过近色。"""
    if not palette:
        return labels, palette
    labs = rgb_u8_to_lab(
        np.array([p.rgb for p in palette], dtype=np.uint8).reshape(-1, 1, 3)
    )[:, 0, :]

    # 近色合并
    parent = list(range(len(palette)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(palette)):
        for j in range(i + 1, len(palette)):
            if float(np.linalg.norm(labs[i] - labs[j])) < lab_merge:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pj] = pi

    # 极小色并入最近大色
    total = max(1.0, float((alpha >= 0.5).sum()))
    for i, p in enumerate(palette):
        if p.fraction < min_frac:
            pi = find(i)
            # 找面积最大的其他根
            best = None
            best_frac = -1.0
            for j, q in enumerate(palette):
                if find(j) == pi:
                    continue
                if q.fraction > best_frac:
                    best_frac = q.fraction
                    best = find(j)
            if best is not None:
                parent[pi] = best

    roots = sorted({find(i) for i in range(len(palette))})
    root_to_new = {r: n for n, r in enumerate(roots)}
    new_labels = np.full_like(labels, -1)
    for old in range(len(palette)):
        new_labels[labels == old] = root_to_new[find(old)]

    new_palette: list[PaletteColor] = []
    for n, r in enumerate(roots):
        m = new_labels == n
        frac = float(m.sum()) / total
        # 代表色：原 palette 中该根下面积最大者
        members = [i for i in range(len(palette)) if find(i) == r]
        members.sort(key=lambda i: -palette[i].fraction)
        rgb = palette[members[0]].rgb
        new_palette.append(
            PaletteColor(hex=rgb_to_hex(rgb), fraction=frac, rgb=rgb)
        )
    # 再按面积排序
    order = sorted(range(len(new_palette)), key=lambda i: -new_palette[i].fraction)
    remap = {old: new for new, old in enumerate(order)}
    out = np.full_like(new_labels, -1)
    for old, new in remap.items():
        out[new_labels == old] = new
    pal_sorted = [new_palette[i] for i in order]
    # 重算 fraction
    total = max(1.0, float((alpha >= 0.5).sum()))
    final: list[PaletteColor] = []
    for i, p in enumerate(pal_sorted):
        frac = float((out == i).sum()) / total
        final.append(PaletteColor(hex=p.hex, fraction=frac, rgb=p.rgb))
    return out, final


def regularize_regions(
    labels: I32Arr,
    alpha: FloatArr,
    *,
    a_min: float,
) -> I32Arr:
    """形态学规整并删除过小连通域。"""
    out = labels.copy()
    max_lab = int(out.max()) if out.size and out.max() >= 0 else -1
    for lab in range(max_lab + 1):
        mask = (out == lab) & (alpha >= 0.5)
        if not mask.any():
            continue
        img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        img = img.filter(ImageFilter.MinFilter(3))
        img = img.filter(ImageFilter.MaxFilter(3))
        img = img.filter(ImageFilter.MaxFilter(3))
        img = img.filter(ImageFilter.MinFilter(3))
        mask2 = np.asarray(img) >= 128
        # 删小 CC：并到邻域众数
        ccs = _label_ccs(mask2)
        for cid in range(int(ccs.max()) + 1):
            cc = ccs == cid
            if int(cc.sum()) < a_min:
                out[cc] = _neighbor_majority(out, cc, forbid=lab)
            else:
                out[cc] = lab
        # 开闭后丢失的原区域像素：若仍标 lab 但不在 mask2，尝试保持或并邻
        lost = mask & ~mask2
        if lost.any():
            out[lost] = _neighbor_majority_batch(out, lost, forbid=lab)
    out[alpha < 0.5] = -1
    return out


def _label_ccs(binary: NDArray[np.bool_]) -> I32Arr:
    h, w = binary.shape
    lab = -np.ones((h, w), dtype=np.int32)
    from collections import deque

    cur = 0
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or lab[y, x] >= 0:
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            lab[y, x] = cur
            while q:
                cy, cx = q.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and lab[ny, nx] < 0:
                        lab[ny, nx] = cur
                        q.append((ny, nx))
            cur += 1
    return lab


def _neighbor_majority(labels: I32Arr, region: NDArray[np.bool_], forbid: int) -> int:
    h, w = labels.shape
    counts: Counter[int] = Counter()
    ys, xs = np.where(region)
    for y, x in zip(ys, xs, strict=False):
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w:
                v = int(labels[ny, nx])
                if v >= 0 and v != forbid and not region[ny, nx]:
                    counts[v] += 1
    if not counts:
        return forbid
    return counts.most_common(1)[0][0]


def _neighbor_majority_batch(
    labels: I32Arr, region: NDArray[np.bool_], forbid: int
) -> I32Arr:
    """为 region 内每个像素找邻域众数（简化：整体一个众数）。"""
    maj = _neighbor_majority(labels, region, forbid)
    return np.full(int(region.sum()), maj, dtype=np.int32)


def build_weight(alpha: FloatArr, image_q: U8Arr, mode: AbstractionMode) -> FloatArr:
    """构造损失权重图。"""
    gray = image_q.astype(np.float64).mean(axis=2) / 255.0
    # sobel
    g = np.pad(gray, 1, mode="edge")
    gx = (
        -g[:-2, :-2]
        + g[:-2, 2:]
        - 2 * g[1:-1, :-2]
        + 2 * g[1:-1, 2:]
        - g[2:, :-2]
        + g[2:, 2:]
    )
    gy = (
        -g[:-2, :-2]
        - 2 * g[:-2, 1:-1]
        - g[:-2, 2:]
        + g[2:, :-2]
        + 2 * g[2:, 1:-1]
        + g[2:, 2:]
    )
    edge = np.hypot(gx, gy)
    if edge.max() > 1e-8:
        edge = edge / edge.max()
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = np.hypot((yy - cy) / max(h, 1), (xx - cx) / max(w, 1))
    center = np.exp(-0.5 * (dist / 0.45) ** 2)

    w_alpha = 1.0
    w_edge = 1.1
    w_center = 0.25 if mode in {AbstractionMode.logo, AbstractionMode.silhouette} else 0.45
    weight = w_alpha * alpha + w_edge * edge * np.maximum(alpha, 0.15) + w_center * center * alpha
    weight = np.maximum(weight, 0.02)
    return weight / (float(weight.mean()) + 1e-8)


def build_layer_hints(
    labels: I32Arr,
    palette: list[PaletteColor],
    alpha: FloatArr,
) -> list[LayerHint]:
    """生成底色/主色块提示。"""
    hints: list[LayerHint] = []
    if palette:
        hints.append(
            LayerHint(
                kind="bottom_color",
                color=palette[0].hex,
                area_fraction=palette[0].fraction,
            )
        )
    bin_a = alpha >= 0.5
    if bin_a.any():
        ys, xs = np.where(bin_a)
        hints.append(
            LayerHint(
                kind="silhouette",
                color=palette[0].hex if palette else None,
                area_fraction=float(bin_a.mean()),
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            )
        )
    max_lab = int(labels.max()) if labels.size and labels.max() >= 0 else -1
    regions: list[tuple[float, LayerHint]] = []
    for lab in range(max_lab + 1):
        m = labels == lab
        frac = float(m.mean())
        if frac < 1e-4:
            continue
        ys, xs = np.where(m)
        regions.append(
            (
                frac,
                LayerHint(
                    kind="region",
                    color=palette[lab].hex if lab < len(palette) else None,
                    area_fraction=frac,
                    bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                ),
            )
        )
    regions.sort(key=lambda t: -t[0])
    hints.extend(h for _, h in regions[:12])
    return hints


def labels_to_image(labels: I32Arr, palette: list[PaletteColor]) -> U8Arr:
    """标签图 → RGB。"""
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, p in enumerate(palette):
        out[labels == i] = np.array(p.rgb, dtype=np.uint8)
    return out


def abstract_image(
    image: Image.Image | str | Path | U8Arr,
    config: ApproxConfig | None = None,
) -> ApproxTarget:
    """执行完整概括管线，返回 ApproxTarget。"""
    cfg = config or ApproxConfig()
    rgba_img = _load_rgba(image)
    mode = detect_mode(rgba_img) if cfg.mode == AbstractionMode.auto else cfg.mode
    fit = "cover" if mode.value.startswith("photo") else "contain"

    rgba, meta = fit_to_canvas(rgba_img, cfg.canvas_size, how=fit)
    meta = meta.model_copy(update={"mode": mode})
    alpha = estimate_alpha(rgba, mode)
    rgb = rgba[:, :, :3]

    if cfg.bilateral and mode != AbstractionMode.logo:
        strength = "strong" if mode.value.startswith("photo") else "medium"
        rgb = bilateral_smooth(rgb, strength=strength)
    elif mode == AbstractionMode.logo:
        rgb = bilateral_smooth(rgb, strength="weak")

    labels, palette = quantize_lab(rgb, alpha, cfg.palette_k, seed=cfg.seed)
    labels, palette = merge_tiny_and_close(labels, palette, alpha)
    a_min = cfg.min_region_area_frac * cfg.canvas_size * cfg.canvas_size
    labels = regularize_regions(labels, alpha, a_min=a_min)
    # 规整后可能清空某些标签：压缩
    labels, palette = _compact_labels(labels, palette, alpha)
    image_q = labels_to_image(labels, palette)
    weight = build_weight(alpha, image_q, mode)
    hints = build_layer_hints(labels, palette, alpha)

    return ApproxTarget(
        image_rgb=image_q,
        alpha=alpha.astype(np.float64),
        weight=weight.astype(np.float64),
        labels=labels,
        palette=palette,
        layers_hint=hints,
        meta=meta,
    )


def _compact_labels(
    labels: I32Arr,
    palette: list[PaletteColor],
    alpha: FloatArr,
) -> tuple[I32Arr, list[PaletteColor]]:
    used = sorted({int(v) for v in np.unique(labels) if int(v) >= 0})
    if not used:
        return labels, []
    remap = {old: new for new, old in enumerate(used)}
    out = np.full_like(labels, -1)
    for old, new in remap.items():
        out[labels == old] = new
    total = max(1.0, float((alpha >= 0.5).sum()))
    new_pal: list[PaletteColor] = []
    for old in used:
        p = palette[old] if old < len(palette) else palette[0]
        frac = float((out == remap[old]).sum()) / total
        new_pal.append(PaletteColor(hex=p.hex, fraction=frac, rgb=p.rgb))
    return out, new_pal


def save_debug_montage(target: ApproxTarget, path: str | Path) -> None:
    """导出概括调试拼图。"""
    rgb = target.numpy_rgb()
    a = (np.clip(target.numpy_alpha(), 0, 1) * 255).astype(np.uint8)
    w = target.numpy_weight()
    w_img = (np.clip(w / (w.max() + 1e-8), 0, 1) * 255).astype(np.uint8)
    h, ww = rgb.shape[:2]
    panels = [
        Image.fromarray(rgb, mode="RGB"),
        Image.fromarray(a, mode="L").convert("RGB"),
        Image.fromarray(w_img, mode="L").convert("RGB"),
    ]
    # 色区边界
    lab = target.numpy_labels()
    edge = np.zeros((h, ww), dtype=np.uint8)
    edge[1:, :] |= (lab[1:, :] != lab[:-1, :]) & (lab[1:, :] >= 0)
    edge[:, 1:] |= (lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] >= 0)
    boundary = rgb.copy()
    boundary[edge > 0] = (255, 0, 0)
    panels.append(Image.fromarray(boundary, mode="RGB"))
    mont = Image.new("RGB", (ww * len(panels), h))
    for i, p in enumerate(panels):
        mont.paste(p, (i * ww, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mont.save(path)
