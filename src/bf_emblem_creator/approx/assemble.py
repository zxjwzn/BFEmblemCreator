"""P8：层栈装配与遮挡一致性校验（GPU 合成/剪枝/Chamfer）。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bf_emblem_creator.approx.contour_arcs import ArcPrimitive, primitives_to_point_cloud
from bf_emblem_creator.approx.curves import extract_outer_contour, resample_closed_contour
from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.gpu_ops import chamfer_points, image_diff_mean
from bf_emblem_creator.approx.line_quality import evaluate_line_quality, visible_boundary_mask
from bf_emblem_creator.approx.torch_render import TorchStampRenderer, stamp_layer_to_dict
from bf_emblem_creator.models import StampLayer

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]


def composite_layers(
    layers: list[StampLayer],
    renderer: TorchStampRenderer,
) -> tuple[U8Arr, FloatArr]:
    """合成 → RGB uint8 + alpha float（GPU torch_render）。"""
    cs = renderer.canvas_size
    if not layers:
        return np.zeros((cs, cs, 3), dtype=np.uint8), np.zeros((cs, cs), dtype=np.float64)
    rgb_t, a_t = renderer.composite_layers([stamp_layer_to_dict(layer) for layer in layers])
    rgb = (rgb_t.detach().float().cpu().numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    alpha = a_t.detach().float().cpu().numpy().astype(np.float64)
    return rgb, alpha


def prune_invisible_layers(
    layers: list[StampLayer],
    renderer: TorchStampRenderer,
    *,
    eps: float = 0.004,
) -> list[StampLayer]:
    """删除对最终可见贡献过小的层（GPU 差分）。"""
    if len(layers) <= 1:
        return layers
    full_rgb, full_a = composite_layers(layers, renderer)
    kept: list[StampLayer] = []
    dev = renderer.device
    for i, layer in enumerate(layers):
        without = layers[:i] + layers[i + 1 :]
        rgb2, a2 = composite_layers(without, renderer)
        contrib = image_diff_mean(full_rgb, full_a, rgb2, a2, device=dev)
        if contrib >= eps:
            kept.append(layer)
    return kept if kept else layers[:1]


def boundary_consistency_score(
    pred_rgb: U8Arr,
    pred_alpha: FloatArr,
    target_prims: list[ArcPrimitive],
) -> float:
    """预测可见边界 vs 目标 SHAPE_BOUNDARY（GPU Chamfer）→ 0~1。"""
    tgt = primitives_to_point_cloud(target_prims, only_shape=True)
    if len(tgt) < 8:
        return 0.7
    edge = visible_boundary_mask(pred_rgb, pred_alpha)
    ys, xs = np.where(edge)
    if len(xs) < 8:
        cont = extract_outer_contour(pred_alpha >= 0.5, simplify=1.0)
        if cont is None or len(cont) < 4:
            return 0.3
        pred_pts = resample_closed_contour(cont, 96)
    else:
        if len(xs) > 500:
            idx = np.linspace(0, len(xs) - 1, 500).astype(int)
            xs, ys = xs[idx], ys[idx]
        pred_pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

    chamfer = chamfer_points(pred_pts, tgt, device=get_device())
    return float(np.clip(np.exp(-chamfer / 14.0), 0.0, 1.0))


def seam_p95_against_targets(
    pred_rgb: U8Arr,
    pred_alpha: FloatArr,
    target_pts: FloatArr,
) -> float:
    """预测可见边界相对目标共享边点的缝宽 p95（像素）。"""
    from bf_emblem_creator.approx.planar_map import seam_width_p95

    tgt = np.asarray(target_pts, dtype=np.float64)
    if len(tgt) < 4:
        return 0.0
    edge = visible_boundary_mask(pred_rgb, pred_alpha)
    ys, xs = np.where(edge)
    if len(xs) < 4:
        cont = extract_outer_contour(pred_alpha >= 0.5, simplify=1.0)
        if cont is None or len(cont) < 4:
            return 99.0
        pred_pts = resample_closed_contour(cont, 96)
    else:
        if len(xs) > 500:
            idx = np.linspace(0, len(xs) - 1, 500).astype(int)
            xs, ys = xs[idx], ys[idx]
        pred_pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    return seam_width_p95(pred_pts, tgt)


def assemble_layers(
    candidates: list[StampLayer],
    renderer: TorchStampRenderer,
    target_prims: list[ArcPrimitive],
    *,
    max_layers: int = 40,
    target_boundary_pts: FloatArr | None = None,
) -> tuple[list[StampLayer], float, float]:
    """
    按给定顺序装配，剪枝无贡献层。

    返回 (层列表, 边界一致性分, seam_p95)。
    """
    layers = candidates[:max_layers]
    layers = prune_invisible_layers(layers, renderer)
    rgb, a = composite_layers(layers, renderer)
    lq = evaluate_line_quality(rgb, a)
    if lq.hard_fail and len(layers) > 2:
        solid = [layer for layer in layers if layer.opacity >= 0.99]
        if solid:
            layers = solid
            rgb, a = composite_layers(layers, renderer)
    bscore = boundary_consistency_score(rgb, a, target_prims)
    if target_boundary_pts is not None and len(np.asarray(target_boundary_pts)) >= 4:
        seam = seam_p95_against_targets(rgb, a, np.asarray(target_boundary_pts, dtype=np.float64))
    else:
        # 回退：用基元点云
        tgt = primitives_to_point_cloud(target_prims, only_shape=True)
        seam = seam_p95_against_targets(rgb, a, tgt) if len(tgt) >= 4 else 0.0
    return layers, bscore, float(seam)
