"""SVG → RGBA 光栅化（基于 PyMuPDF）。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from numpy.typing import NDArray
from PIL import Image

Rgba = NDArray[np.uint8]


def rasterize_svg(
    svg_path: str | Path,
    *,
    out_width: int,
    out_height: int,
) -> Rgba:
    """
    将 SVG 光栅化为 shape=(H, W, 4) 的 RGBA uint8 数组。

    SVG 会被缩放到恰好填满 out_width × out_height（允许非等比缩放），
    以匹配编辑器对原生宽高比不同的图章（如 Drop）的 width/height 覆盖。
    """
    if out_width < 1 or out_height < 1:
        raise ValueError("out_width/out_height 必须 >= 1")
    return _rasterize_svg_cached(str(Path(svg_path).resolve()), out_width, out_height)


@lru_cache(maxsize=256)
def _rasterize_svg_cached(svg_path: str, out_width: int, out_height: int) -> Rgba:
    """带缓存的实际光栅化实现。"""
    path = Path(svg_path)
    svg_bytes = path.read_bytes()

    # MuPDF 将 SVG 作为单页文档打开。
    doc = fitz.open(stream=svg_bytes, filetype="svg")
    try:
        if doc.page_count < 1:
            raise RuntimeError(f"SVG 无页面: {path}")
        # PyMuPDF 类型桩对 Document.__getitem__ 较松，这里用 load_page 绑定。
        page: fitz.Page = doc.load_page(0)
        rect = page.rect
        if rect.width <= 0 or rect.height <= 0:
            raise RuntimeError(f"无效的 SVG 页面尺寸: {path}")

        zoom_x = out_width / float(rect.width)
        zoom_y = out_height / float(rect.height)
        matrix = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=matrix, alpha=True)
        samples = np.frombuffer(pix.samples, dtype=np.uint8)
        arr: Rgba = samples.reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            rgba = arr
        elif pix.n == 3:
            alpha = np.full((pix.height, pix.width, 1), 255, dtype=np.uint8)
            rgba = np.concatenate([arr, alpha], axis=2)
        elif pix.n == 1:
            rgb = np.repeat(arr, 3, axis=2)
            alpha = np.full((pix.height, pix.width, 1), 255, dtype=np.uint8)
            rgba = np.concatenate([rgb, alpha], axis=2)
        else:
            raise RuntimeError(f"不支持的 pixmap 通道数: {pix.n}")

        if rgba.shape[0] != out_height or rgba.shape[1] != out_width:
            # 偶发舍入导致尺寸偏差时，用最近邻拉回目标尺寸（mask 友好）。
            img = Image.fromarray(np.asarray(rgba), mode="RGBA")
            img = img.resize((out_width, out_height), Image.Resampling.NEAREST)
            return np.asarray(img, dtype=np.uint8)

        return np.array(rgba, dtype=np.uint8, copy=True)
    finally:
        doc.close()


def clear_raster_cache() -> None:
    """清空 SVG 光栅化缓存。"""
    _rasterize_svg_cached.cache_clear()
