"""大块光滑色块概括（v2）：少色、合并碎岛、轮廓曲线。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.curves import (
    contour_curvature_descriptor,
    extract_outer_contour,
    mask_to_sdf,
    resample_closed_contour,
)
from bf_emblem_creator.approx.models import AbstractionMode, ApproxConfig, ApproxMeta
from bf_emblem_creator.approx.preprocess import (
    bilateral_smooth,
    detect_mode,
    detect_resample_mode,
    estimate_alpha,
    fit_to_canvas,
    quantize_lab,
)

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]
BoolArr = NDArray[np.bool_]


class ColorBlock(BaseModel):
    """光滑单色区域 + 曲线。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    color_hex: str = Field(..., description="#RRGGBB")
    color_rgb: tuple[int, int, int] = Field(..., description="RGB")
    area_frac: float = Field(..., ge=0.0, description="占画布面积比")
    bbox: tuple[int, int, int, int] = Field(..., description="x0,y0,x1,y1")
    mask: Any = Field(..., description="bool (H,W)")
    contour: Any = Field(..., description="(N,2) 轮廓点")
    contour_resampled: Any = Field(..., description="固定点数重采样轮廓")
    descriptor: Any = Field(..., description="曲率描述子")
    sdf: Any | None = Field(default=None, description="可选 SDF")
    depth_hint: int = Field(default=0, description="建议层序，小者更靠下")


class BlockTarget(BaseModel):
    """v2 拟合目标：大块列表 + 栅格图。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    image_rgb: Any
    alpha: Any
    weight: Any
    blocks: list[ColorBlock]
    meta: ApproxMeta
    canvas_size: int = 320

    def numpy_rgb(self) -> U8Arr:
        return np.asarray(self.image_rgb, dtype=np.uint8)

    def numpy_alpha(self) -> FloatArr:
        return np.asarray(self.alpha, dtype=np.float64)

    def numpy_weight(self) -> FloatArr:
        return np.asarray(self.weight, dtype=np.float64)


def _morph_close_open(mask: BoolArr, close: int = 3, open_: int = 2) -> BoolArr:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    for _ in range(close):
        img = img.filter(ImageFilter.MaxFilter(3))
    for _ in range(close):
        img = img.filter(ImageFilter.MinFilter(3))
    for _ in range(open_):
        img = img.filter(ImageFilter.MinFilter(3))
    for _ in range(open_):
        img = img.filter(ImageFilter.MaxFilter(3))
    return np.asarray(img) >= 128


def _label_ccs(binary: BoolArr) -> list[BoolArr]:
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    from collections import deque

    out: list[BoolArr] = []
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            m = np.zeros_like(binary)
            for cy, cx in cells:
                m[cy, cx] = True
            out.append(m)
    return out


def abstract_to_blocks(
    image: Image.Image | str | Path | U8Arr,
    config: ApproxConfig | None = None,
    *,
    max_blocks: int = 16,
    min_area_frac: float = 0.008,
) -> BlockTarget:
    """
    v2 概括：少色大块 + 轮廓曲线。

    碎岛被合并/丢弃，边界经形态学与 RDP 光滑。
    """
    cfg = config or ApproxConfig()
    if isinstance(image, Image.Image):
        rgba_img = image.convert("RGBA")
    elif isinstance(image, (str, Path)):
        rgba_img = Image.open(image).convert("RGBA")
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            rgba_img = Image.fromarray(arr, mode="L").convert("RGBA")
        elif arr.shape[-1] == 3:
            rgba_img = Image.fromarray(arr.astype(np.uint8), mode="RGB").convert("RGBA")
        else:
            rgba_img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

    mode = detect_mode(rgba_img) if cfg.mode == AbstractionMode.auto else cfg.mode
    fit = "cover" if mode.value.startswith("photo") else "contain"
    r_mode, _ = detect_resample_mode(rgba_img, target_size=cfg.canvas_size, configured=cfg.resample_mode)
    rgba, meta = fit_to_canvas(rgba_img, cfg.canvas_size, how=fit, resample=r_mode)
    meta = meta.model_copy(update={"mode": mode})
    alpha = estimate_alpha(rgba, mode)
    rgb = rgba[:, :, :3]
    hard_edge = meta.resample == "nearest" and meta.approx_color_count <= 48
    if hard_edge:
        pass
    elif cfg.bilateral and mode != AbstractionMode.logo:
        rgb = bilateral_smooth(rgb, strength="strong" if mode.value.startswith("photo") else "medium")

    # 更少色、更大块
    k = min(cfg.palette_k, 5)
    labels, palette = quantize_lab(rgb, alpha, k, seed=cfg.seed)
    h, w = labels.shape
    canvas_area = float(h * w)
    min_area = min_area_frac * canvas_area

    blocks: list[ColorBlock] = []
    image_q = np.zeros((h, w, 3), dtype=np.uint8)

    for i, pal in enumerate(palette):
        base = (labels == i) & (alpha >= 0.5)
        if not base.any():
            continue
        base = _morph_close_open(base, close=2, open_=1)
        for cc in _label_ccs(base):
            area = float(cc.sum())
            if area < min_area:
                continue
            contour = extract_outer_contour(cc, simplify=0.6)
            if contour is None or len(contour) < 3:
                # 兜底：用 bbox 矩形轮廓，避免丢大块
                ys, xs = np.where(cc)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                contour = np.array(
                    [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
                    dtype=np.float64,
                )
            rs = resample_closed_contour(contour, 64)
            desc = contour_curvature_descriptor(contour)
            ys, xs = np.where(cc)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            sdf = mask_to_sdf(cc)
            blocks.append(
                ColorBlock(
                    color_hex=pal.hex,
                    color_rgb=pal.rgb,
                    area_frac=area / canvas_area,
                    bbox=bbox,
                    mask=cc,
                    contour=contour,
                    contour_resampled=rs,
                    descriptor=desc,
                    sdf=sdf,
                    depth_hint=0,
                )
            )
            image_q[cc] = np.array(pal.rgb, dtype=np.uint8)

    # 按面积从大到小 → 底层优先
    blocks.sort(key=lambda b: -b.area_frac)
    for i, b in enumerate(blocks[:max_blocks]):
        b.depth_hint = i
    blocks = blocks[:max_blocks]

    # 权重：边缘 + alpha
    gray = image_q.astype(np.float64).mean(axis=2) / 255.0
    edge = np.zeros_like(gray)
    edge[:, :-1] = np.maximum(edge[:, :-1], np.abs(gray[:, 1:] - gray[:, :-1]))
    edge[:-1, :] = np.maximum(edge[:-1, :], np.abs(gray[1:, :] - gray[:-1, :]))
    if edge.max() > 1e-8:
        edge = edge / edge.max()
    weight = np.maximum(alpha, 0.05) * (0.6 + 0.8 * edge)
    weight = weight / (float(weight.mean()) + 1e-8)

    return BlockTarget(
        image_rgb=image_q,
        alpha=alpha.astype(np.float64),
        weight=weight.astype(np.float64),
        blocks=blocks,
        meta=meta,
        canvas_size=cfg.canvas_size,
    )
