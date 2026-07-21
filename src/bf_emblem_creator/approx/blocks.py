"""大块光滑色块概括：基于平面化标签场提取区域轮廓。"""

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
from bf_emblem_creator.approx.models import AbstractionMode
from bf_emblem_creator.approx.planarize import planarize_image
from bf_emblem_creator.approx.recipe import ModeRecipe, default_recipe_for_mode

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
    """拟合目标：大块列表 + 栅格图。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    image_rgb: Any = Field(..., description="RGB uint8")
    alpha: Any = Field(..., description="alpha float")
    weight: Any = Field(..., description="权重 float")
    blocks: list[ColorBlock] = Field(default_factory=list, description="色块列表")
    meta: Any = Field(..., description="ApproxMeta")
    canvas_size: int = Field(default=320, description="画布边长")

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
    recipe: ModeRecipe | None = None,
    *,
    max_blocks: int = 16,
    min_area_frac: float = 0.008,
) -> BlockTarget:
    """
    严格色量平面化后提取大块 + 轮廓。

    使用 recipe.image.num_colors；碎岛经形态学与面积门槛丢弃。
    """
    r = recipe if recipe is not None else default_recipe_for_mode(AbstractionMode.illustration)
    labels, palette, alpha, image_q, meta, _src = planarize_image(image, r.image, mode=r.mode)
    h, w = labels.shape
    canvas_area = float(h * w)
    min_area = min_area_frac * canvas_area

    blocks: list[ColorBlock] = []
    for i, pal in enumerate(palette):
        base = (labels == i) & (alpha >= 0.5)
        if not base.any():
            continue
        base = _morph_close_open(base, close=2, open_=1)
        for cc in _label_ccs(base):
            area = float(cc.sum())
            if area < min_area:
                continue
            contour = extract_outer_contour(cc, simplify=0.0)
            if contour is None or len(contour) < 3:
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

    blocks.sort(key=lambda b: -b.area_frac)
    for i, b in enumerate(blocks[:max_blocks]):
        b.depth_hint = i
    blocks = blocks[:max_blocks]

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
        canvas_size=r.image.canvas_size,
    )
