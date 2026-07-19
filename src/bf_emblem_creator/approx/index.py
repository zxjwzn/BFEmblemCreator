"""图章形状索引：mask、几何特征、基础子集。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bf_emblem_creator.approx.color import hex_to_rgb_tuple
from bf_emblem_creator.raster import rasterize_svg
from bf_emblem_creator.stamps import StampLibrary

FloatArr = NDArray[np.floating]
U8Arr = NDArray[np.uint8]

# 文档推荐的基础几何子集（MVP）
DEFAULT_BASIC_STAMPS: tuple[str, ...] = (
    "Square",
    "Circle",
    "Triangle",
    "RightTriangle",
    "HalfCircle",
    "OpenCircle",
    "Drop",
    "Line",
    "Stroke",
    "StrokeBent",
    "BentLine",
    "Arrow",
    "ArrowBent",
    "SquareCorner",
    "CrescentMoon",
    "Banner",
    "Shield",
    "Wingpart",
)


@dataclass
class StampFeatures:
    """单个图章的预计算特征。"""

    asset_id: str
    mask: U8Arr  # HxW 0/255 方形画布上的归一化 mask
    size: int
    area_frac: float
    circularity: float
    elongation: float
    fill_ratio: float  # mask 相对 bbox 填充率
    aspect: float  # bbox w/h


class StampIndex:
    """基础图章特征库。"""

    def __init__(self, features: list[StampFeatures]) -> None:
        self.features = features
        self.by_id = {f.asset_id: f for f in features}

    @classmethod
    def build(
        cls,
        library: StampLibrary,
        asset_ids: list[str] | None = None,
        *,
        size: int = 64,
    ) -> StampIndex:
        """光栅化并提取特征。"""
        ids = asset_ids or list(DEFAULT_BASIC_STAMPS)
        feats: list[StampFeatures] = []
        for asset_id in ids:
            try:
                info = library.resolve(asset_id)
            except FileNotFoundError:
                continue
            rgba = rasterize_svg(info.path, out_width=size, out_height=size)
            mask = rgba[:, :, 3]
            feat = _features_from_mask(asset_id, mask, size)
            feats.append(feat)
        if not feats:
            raise RuntimeError("图章索引为空：请检查 stamps_dir 与 stamp_subset")
        return cls(feats)

    def recall(self, region_feat: dict[str, float], k: int = 10) -> list[str]:
        """按几何特征召回 top-k 图章 id。"""
        scored: list[tuple[float, str]] = []
        for f in self.features:
            score = _shape_match_score(region_feat, f)
            scored.append((score, f.asset_id))
        scored.sort(key=lambda t: -t[0])
        # 去重保持序
        out: list[str] = []
        for _, aid in scored:
            if aid not in out:
                out.append(aid)
            if len(out) >= k:
                break
        return out


def _features_from_mask(asset_id: str, mask: U8Arr, size: int) -> StampFeatures:
    m = mask >= 128
    area = float(m.sum())
    area_frac = area / float(size * size)
    if area < 1:
        return StampFeatures(asset_id, mask, size, 0.0, 0.0, 1.0, 0.0, 1.0)
    ys, xs = np.where(m)
    bw = float(xs.max() - xs.min() + 1)
    bh = float(ys.max() - ys.min() + 1)
    aspect = bw / max(bh, 1.0)
    fill_ratio = area / max(bw * bh, 1.0)
    # 周长近似
    erode = m.copy()
    erode[1:, :] &= m[:-1, :]
    erode[:-1, :] &= m[1:, :]
    erode[:, 1:] &= m[:, :-1]
    erode[:, :-1] &= m[:, 1:]
    peri = float(m.sum() - erode.sum())
    circularity = 0.0 if peri < 1 else float(4.0 * np.pi * area / (peri * peri + 1e-6))
    circularity = float(np.clip(circularity, 0.0, 2.0))
    elongation = max(aspect, 1.0 / max(aspect, 1e-6))
    return StampFeatures(
        asset_id=asset_id,
        mask=mask,
        size=size,
        area_frac=area_frac,
        circularity=circularity,
        elongation=elongation,
        fill_ratio=fill_ratio,
        aspect=aspect,
    )


def region_features_from_mask(mask: NDArray[np.bool_]) -> dict[str, float]:
    """从 ROI 布尔 mask 提取与 StampFeatures 对齐的特征。"""
    m = mask
    area = float(m.sum())
    if area < 1:
        return {
            "circularity": 0.0,
            "elongation": 1.0,
            "fill_ratio": 0.0,
            "aspect": 1.0,
            "area": 0.0,
        }
    ys, xs = np.where(m)
    bw = float(xs.max() - xs.min() + 1)
    bh = float(ys.max() - ys.min() + 1)
    aspect = bw / max(bh, 1.0)
    fill_ratio = area / max(bw * bh, 1.0)
    erode = m.copy()
    erode[1:, :] &= m[:-1, :]
    erode[:-1, :] &= m[1:, :]
    erode[:, 1:] &= m[:, :-1]
    erode[:, :-1] &= m[:, 1:]
    peri = float(m.sum() - erode.sum())
    circularity = 0.0 if peri < 1 else float(4.0 * np.pi * area / (peri * peri + 1e-6))
    elongation = max(aspect, 1.0 / max(aspect, 1e-6))
    return {
        "circularity": float(np.clip(circularity, 0.0, 2.0)),
        "elongation": float(elongation),
        "fill_ratio": float(fill_ratio),
        "aspect": float(aspect),
        "area": area,
    }


def _shape_match_score(region: dict[str, float], stamp: StampFeatures) -> float:
    """特征接近程度（越大越好）。"""
    dc = abs(region["circularity"] - stamp.circularity)
    de = abs(np.log(region["elongation"] + 1e-6) - np.log(stamp.elongation + 1e-6))
    df = abs(region["fill_ratio"] - stamp.fill_ratio)
    da = abs(np.log(region["aspect"] + 1e-6) - np.log(stamp.aspect + 1e-6))
    return float(np.exp(-2.0 * dc) + np.exp(-1.5 * de) + np.exp(-2.0 * df) + np.exp(-1.2 * da))


def median_color_hex(rgb: U8Arr, mask: NDArray[np.bool_]) -> str:
    """mask 内 median 色。"""
    if not mask.any():
        return "#808080"
    pixels = rgb[mask]
    med = np.median(pixels, axis=0)
    return f"#{int(med[0]):02X}{int(med[1]):02X}{int(med[2]):02X}"


def fill_from_palette(
    rgb: U8Arr,
    mask: NDArray[np.bool_],
    palette_hexes: list[str],
) -> str:
    """选与覆盖区 median 最近的调色板色。"""
    if not mask.any():
        return palette_hexes[0] if palette_hexes else "#808080"
    med = np.median(rgb[mask], axis=0)
    if not palette_hexes:
        return f"#{int(med[0]):02X}{int(med[1]):02X}{int(med[2]):02X}"
    best = palette_hexes[0]
    best_d = 1e18
    for hx in palette_hexes:
        r, g, b = hex_to_rgb_tuple(hx)
        d = (r - med[0]) ** 2 + (g - med[1]) ** 2 + (b - med[2]) ** 2
        if d < best_d:
            best_d = d
            best = hx
    return best
