"""短名单内位姿离散搜索与局部精修（偏 mask 快速路径）。"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from bf_emblem_creator.approx.index import StampIndex, fill_from_palette
from bf_emblem_creator.approx.models import ApproxTarget
from bf_emblem_creator.approx.propose import RoiProposal
from bf_emblem_creator.models import CanvasConfig, RenderConfig, StampLayer
from bf_emblem_creator.raster import rasterize_svg
from bf_emblem_creator.render import EmblemRenderer
from bf_emblem_creator.stamps import StampLibrary

U8Arr = NDArray[np.uint8]


def _iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-8)


@lru_cache(maxsize=2048)
def _base_mask(svg_path: str, size: int = 48) -> U8Arr:
    """缓存图章方形 alpha。"""
    rgba = rasterize_svg(svg_path, out_width=size, out_height=size)
    return np.array(rgba[:, :, 3], copy=True)


def _stamp_alpha_sprite(
    library: StampLibrary,
    asset_id: str,
    width: int,
    height: int,
    angle: float,
    flip_x: bool,
    flip_y: bool,
) -> Image.Image:
    """由缓存 mask 缩放/旋转得到 alpha 精灵。"""
    info = library.resolve(asset_id)
    base = _base_mask(str(info.path.resolve()), 48)
    img = Image.fromarray(base, mode="L")
    w = max(1, int(width))
    h = max(1, int(height))
    if img.size != (w, h):
        img = img.resize((w, h), Image.Resampling.BILINEAR)
    if flip_x:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_y:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if abs(angle) > 1e-6:
        img = img.rotate(-angle, expand=True, resample=Image.Resampling.BILINEAR, fillcolor=0)
    return img


def _paste_alpha_on_canvas(
    sprite_l: Image.Image,
    *,
    canvas_size: int,
    left: float,
    top: float,
) -> NDArray[np.bool_]:
    """将 L 模式 alpha 精灵按中心贴到画布，返回 bool mask。"""
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    sw, sh = sprite_l.size
    x = round(left - sw / 2.0)
    y = round(top - sh / 2.0)
    canvas.paste(sprite_l, (x, y), sprite_l)
    arr = np.asarray(canvas)
    return arr >= 128


def discrete_search_layer(
    roi: RoiProposal,
    asset_id: str,
    target: ApproxTarget,
    *,
    library: StampLibrary,
    angle_step: float = 30.0,
    palette_hexes: list[str],
) -> StampLayer | None:
    """在 ROI 上搜索单图章位姿（mask IoU）。"""
    canvas = target.meta.canvas_size
    x0, y0, x1, y1 = roi.bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    ys, xs = np.where(roi.mask)
    if len(xs) >= 5:
        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        pts_c = pts - pts.mean(axis=0, keepdims=True)
        cov = pts_c.T @ pts_c / max(len(pts_c) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main = eigvecs[:, int(np.argmax(eigvals))]
        pca_angle = math.degrees(math.atan2(main[1], main[0])) % 360.0
    else:
        pca_angle = 0.0

    # 紧凑角度集
    angles = {0.0, 90.0, pca_angle % 360.0, (pca_angle + 180.0) % 360.0}
    step = max(30.0, float(angle_step))
    for k in range(0, 180, int(step)):
        angles.add((pca_angle + k) % 360.0)
    angle_list = sorted(angles)

    elong = roi.features.get("elongation", 1.0)
    if elong > 2.2:
        scales_w = (0.9, 1.15, 1.4)
        scales_h = (0.75, 1.0)
        flips = ((False, False), (True, False))
    else:
        scales_w = (0.9, 1.15)
        scales_h = (0.9, 1.15)
        flips = ((False, False),)

    tgt_mask = roi.mask
    best_score = -1.0
    best_layer: StampLayer | None = None
    fill = fill_from_palette(target.numpy_rgb(), tgt_mask, palette_hexes)

    for ang in angle_list:
        for sx in scales_w:
            for sy in scales_h:
                width = min(max(4.0, bw * sx), canvas * 1.45)
                height = min(max(4.0, bh * sy), canvas * 1.45)
                for fx, fy in flips:
                    sprite = _stamp_alpha_sprite(
                        library,
                        asset_id,
                        round(width),
                        round(height),
                        ang,
                        fx,
                        fy,
                    )
                    pred = _paste_alpha_on_canvas(
                        sprite, canvas_size=canvas, left=cx, top=cy
                    )
                    iou = _iou(pred, tgt_mask)
                    cover = float(np.logical_and(pred, tgt_mask).sum()) / (
                        float(tgt_mask.sum()) + 1e-8
                    )
                    outside = float(np.logical_and(pred, ~tgt_mask).sum()) / (
                        float(pred.sum()) + 1e-8
                    )
                    score = 0.65 * iou + 0.35 * cover - 0.3 * outside
                    if score > best_score:
                        best_score = score
                        best_layer = StampLayer(
                            asset=asset_id,
                            opacity=1.0,
                            angle=float(ang),
                            flipX=fx,
                            flipY=fy,
                            top=float(cy),
                            left=float(cx),
                            height=float(height),
                            width=float(width),
                            fill=fill,
                        )

    if best_layer is None or best_score < 0.16:
        return None
    return best_layer


def refine_layer(
    layer: StampLayer,
    roi: RoiProposal,
    target: ApproxTarget,
    *,
    library: StampLibrary,
    iters: int = 4,
) -> StampLayer:
    """小范围爬山精修。"""
    if iters <= 0:
        return layer
    canvas = target.meta.canvas_size
    tgt = roi.mask
    best = layer.model_copy()
    best_score = _layer_mask_score(best, tgt, canvas, library)

    for it in range(iters):
        scale = 0.6**it
        improved = False
        for name, delta in (
            ("left", 3.0 * scale),
            ("left", -3.0 * scale),
            ("top", 3.0 * scale),
            ("top", -3.0 * scale),
            ("width", 5.0 * scale),
            ("width", -5.0 * scale),
            ("height", 5.0 * scale),
            ("height", -5.0 * scale),
            ("angle", 6.0 * scale),
            ("angle", -6.0 * scale),
        ):
            data = best.model_dump()
            data[name] = float(data[name]) + delta
            data["width"] = max(4.0, float(data["width"]))
            data["height"] = max(4.0, float(data["height"]))
            cand = StampLayer.model_validate(data)
            sc = _layer_mask_score(cand, tgt, canvas, library)
            if sc > best_score + 1e-4:
                best = cand
                best_score = sc
                improved = True
        if not improved:
            break
    return best


def _layer_mask_score(
    layer: StampLayer,
    tgt_mask: NDArray[np.bool_],
    canvas: int,
    library: StampLibrary,
) -> float:
    sprite = _stamp_alpha_sprite(
        library,
        layer.asset,
        round(layer.width),
        round(layer.height),
        layer.angle,
        layer.flipX,
        layer.flipY,
    )
    pred = _paste_alpha_on_canvas(
        sprite, canvas_size=canvas, left=layer.left, top=layer.top
    )
    iou = _iou(pred, tgt_mask)
    cover = float(np.logical_and(pred, tgt_mask).sum()) / (float(tgt_mask.sum()) + 1e-8)
    outside = float(np.logical_and(pred, ~tgt_mask).sum()) / (float(pred.sum()) + 1e-8)
    return 0.65 * iou + 0.35 * cover - 0.3 * outside


def make_fast_renderer(
    stamps_dir: str | Path,
    canvas_size: int,
    supersample: float = 1.0,
) -> EmblemRenderer:
    """构造拟合用快速渲染器。"""
    return EmblemRenderer(
        RenderConfig(
            canvas=CanvasConfig(width=canvas_size, height=canvas_size, background=None),
            stamps_dir=Path(stamps_dir),
            supersample=supersample,
            stamp_raster_scale=1.0,
        )
    )


def search_best_for_roi(
    roi: RoiProposal,
    index: StampIndex,
    target: ApproxTarget,
    *,
    library: StampLibrary,
    recall_k: int,
    angle_step: float,
    refine: bool,
    refine_iters: int,
) -> StampLayer | None:
    """召回 + 离散搜 + 精修。"""
    assets = index.recall(roi.features, k=recall_k)
    palette_hexes = [p.hex for p in target.palette]
    best: StampLayer | None = None
    best_sc = -1e9
    for asset_id in assets:
        layer = discrete_search_layer(
            roi,
            asset_id,
            target,
            library=library,
            angle_step=angle_step,
            palette_hexes=palette_hexes,
        )
        if layer is None:
            continue
        if refine:
            layer = refine_layer(
                layer, roi, target, library=library, iters=refine_iters
            )
        sc = _layer_mask_score(layer, roi.mask, target.meta.canvas_size, library)
        if sc > best_sc:
            best_sc = sc
            best = layer
    return best
