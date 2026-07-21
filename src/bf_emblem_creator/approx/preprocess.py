"""图像预处理：对齐画布、抠主体、保边平滑（色量在 label_field / planarize）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from bf_emblem_creator.approx.color import rgb_u8_to_lab
from bf_emblem_creator.approx.models import AbstractionMode, ApproxMeta, ResampleMode

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


def estimate_color_stats(rgba: Image.Image | U8Arr) -> dict[str, float]:
    """
    主体颜色/边缘统计，供自适应重采样。

    返回 approx_color_count、peak_ratio、soft_alpha_frac、grad_frac 等。
    """
    arr = np.asarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return {
            "approx_color_count": 0.0,
            "peak_ratio": 1.0,
            "soft_alpha_frac": 0.0,
            "grad_frac": 0.0,
            "subject_frac": 0.0,
        }
    if arr.shape[-1] >= 4:
        a = arr[:, :, 3].astype(np.float64) / 255.0
        subject = a >= 0.5
        soft_alpha_frac = float(((a > 0.05) & (a < 0.95)).mean())
    else:
        subject = np.ones(arr.shape[:2], dtype=bool)
        soft_alpha_frac = 0.0
    if not subject.any():
        return {
            "approx_color_count": 0.0,
            "peak_ratio": 1.0,
            "soft_alpha_frac": soft_alpha_frac,
            "grad_frac": 0.0,
            "subject_frac": 0.0,
        }
    rgb = arr[:, :, :3]
    q = (rgb[subject] // 32).astype(np.int32)
    keys = q[:, 0].astype(np.int64) * 1024 + q[:, 1].astype(np.int64) * 32 + q[:, 2].astype(np.int64)
    uniq, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)
    c0 = float(counts[order[0]]) if len(counts) else 1.0
    c1 = float(counts[order[1]]) if len(counts) > 1 else c0
    peak_ratio = c0 / max(c1, 1.0)
    gray = rgb.astype(np.float64).mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    g = np.maximum(gx, gy)
    g_sub = g[subject]
    thr = float(np.percentile(g_sub, 70)) if g_sub.size else 0.0
    grad_frac = float((g_sub >= max(thr, 4.0)).mean()) if g_sub.size else 0.0
    return {
        "approx_color_count": float(len(uniq)),
        "peak_ratio": float(peak_ratio),
        "soft_alpha_frac": soft_alpha_frac,
        "grad_frac": grad_frac,
        "subject_frac": float(subject.mean()),
    }


def detect_resample_mode(
    rgba: Image.Image | U8Arr,
    *,
    target_size: int = 320,
    configured: ResampleMode | str = ResampleMode.auto,
) -> tuple[str, dict[str, float]]:
    """
    通用重采样决策：色数少/尖峰强 → nearest；否则 lanczos。
    """
    stats = estimate_color_stats(rgba)
    conf = configured.value if isinstance(configured, ResampleMode) else str(configured)
    if conf in {"nearest", "lanczos", "bilinear"}:
        return conf, stats
    n_col = float(stats["approx_color_count"])
    peak = float(stats["peak_ratio"])
    if n_col <= 24.0 and peak >= 1.15:
        return "nearest", stats
    if n_col <= 48.0 and peak >= 1.8:
        return "nearest", stats
    if isinstance(rgba, Image.Image):
        sw, sh = rgba.size
    else:
        sh, sw = np.asarray(rgba).shape[:2]
    if max(sw, sh) <= target_size and n_col <= 32.0:
        return "nearest", stats
    return "lanczos", stats


def _pil_resample(name: str) -> Image.Resampling:
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    return Image.Resampling.LANCZOS


def fit_to_canvas(
    rgba: Image.Image,
    size: int,
    *,
    how: str,
    resample: str | ResampleMode = "lanczos",
    color_stats: dict[str, float] | None = None,
) -> tuple[U8Arr, ApproxMeta]:
    """将图像 fit 到 size×size，返回 RGBA uint8 与元数据。"""
    src_w, src_h = rgba.size
    if how not in {"contain", "cover"}:
        raise ValueError("how 必须是 contain 或 cover")

    rname = resample.value if isinstance(resample, ResampleMode) else str(resample)
    if rname == "auto":
        rname, stats = detect_resample_mode(rgba, target_size=size)
    else:
        stats = color_stats if color_stats is not None else estimate_color_stats(rgba)

    scale = min(size / src_w, size / src_h) if how == "contain" else max(size / src_w, size / src_h)

    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    if rname == "nearest" and how == "contain":
        ix = max(1, round(size / src_w))
        iy = max(1, round(size / src_h))
        im = min(ix, iy)
        if (
            im >= 1
            and im * src_w <= size
            and im * src_h <= size
            and im * min(src_w, src_h) >= round(scale * min(src_w, src_h)) * 0.85
        ):
            new_w, new_h = im * src_w, im * src_h
            scale = float(im)

    resized = rgba.resize((new_w, new_h), _pil_resample(rname))
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    canvas.paste(resized, (ox, oy), resized)
    meta = ApproxMeta(
        source_width=src_w,
        source_height=src_h,
        canvas_size=size,
        fit=how,
        mode=AbstractionMode.illustration,  # fit_to_canvas 占位，调用方覆盖 meta.mode
        scale=scale,
        offset_x=float(ox),
        offset_y=float(oy),
        resample=rname,
        approx_color_count=int(stats.get("approx_color_count", 0)),
    )
    return np.asarray(canvas, dtype=np.uint8), meta


def estimate_alpha(rgba: U8Arr, mode: AbstractionMode) -> FloatArr:
    """估计主体蒙版。"""
    a = rgba[:, :, 3].astype(np.float64) / 255.0
    alpha = a if float(a.max()) > 0.05 and float(a.mean()) < 0.98 else _flood_background_alpha(rgba[:, :, :3])

    alpha = _morph_cleanup_alpha(alpha)
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
    轻量保边平滑（无 OpenCV 时的近似：中值 + 轻高斯）。

    strength: weak / medium / strong
    """
    img = Image.fromarray(rgb, mode="RGB")
    if strength == "weak":
        return rgb
    if strength == "medium":
        img = img.filter(ImageFilter.MedianFilter(size=3))
        return np.asarray(img, dtype=np.uint8)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SMOOTH_MORE)
    return np.asarray(img, dtype=np.uint8)
