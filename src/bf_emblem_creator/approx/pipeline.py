"""图章匹配追踪主循环：概括 → 贪婪残差拟合 → 相似度评估。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from bf_emblem_creator.approx.index import DEFAULT_BASIC_STAMPS, StampIndex
from bf_emblem_creator.approx.metrics import fit_loss, image_to_rgba_arrays, score_fit
from bf_emblem_creator.approx.models import ApproxConfig, ApproxResult, ApproxTarget
from bf_emblem_creator.approx.preprocess import abstract_image
from bf_emblem_creator.approx.propose import propose_rois
from bf_emblem_creator.approx.search import make_fast_renderer, search_best_for_roi
from bf_emblem_creator.models import (
    CanvasConfig,
    EmblemDocument,
    RenderConfig,
    StampLayer,
)
from bf_emblem_creator.render import EmblemRenderer
from bf_emblem_creator.stamps import StampLibrary

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]


def _pred_arrays(
    layers: list[StampLayer],
    renderer: EmblemRenderer,
) -> tuple[U8Arr, FloatArr]:
    doc = EmblemDocument.from_layers(layers)
    img = renderer.render(doc)
    return image_to_rgba_arrays(img)


def _try_bottom_layer(target: ApproxTarget, renderer: EmblemRenderer) -> StampLayer | None:
    """若存在大面积主色，尝试整画布 Square 打底。"""
    if not target.palette:
        return None
    p0 = target.palette[0]
    if p0.fraction < 0.12:
        return None
    size = float(target.meta.canvas_size)
    # 略大于画布确保覆盖
    layer = StampLayer(
        asset="Square",
        opacity=1.0,
        angle=0.0,
        flipX=False,
        flipY=False,
        top=size / 2.0,
        left=size / 2.0,
        height=size * 1.05,
        width=size * 1.05,
        fill=p0.hex,
    )
    # 底色仅在目标主色区域占比够时有意义
    rgb, a = _pred_arrays([layer], renderer)
    loss = fit_loss(rgb, a, target)
    # 空层损失
    empty_rgb = np.zeros_like(rgb)
    empty_a = np.zeros_like(a)
    base = fit_loss(empty_rgb, empty_a, target)
    if loss < base - 1e-4:
        return layer
    return None


def approximate_image(
    image: Image.Image | str | Path | U8Arr,
    config: ApproxConfig | None = None,
    *,
    target: ApproxTarget | None = None,
) -> ApproxResult:
    """
    对输入图像执行概括 + Matching Pursuit 图章近似。

    返回文档、目标、损失与相似度报告。
    """
    cfg = config or ApproxConfig()
    tgt = target if target is not None else abstract_image(image, cfg)

    library = StampLibrary(cfg.stamps_dir)
    subset = cfg.stamp_subset if cfg.stamp_subset is not None else list(DEFAULT_BASIC_STAMPS)
    index = StampIndex.build(library, subset, size=64)

    renderer = make_fast_renderer(
        cfg.stamps_dir,
        cfg.canvas_size,
        supersample=cfg.render_supersample,
    )

    layers: list[StampLayer] = []
    gains: list[float] = []

    if cfg.add_bottom_fill:
        bottom = _try_bottom_layer(tgt, renderer)
        if bottom is not None:
            layers.append(bottom)
            gains.append(0.0)

    pred_rgb, pred_a = _pred_arrays(layers, renderer)
    cur_loss = fit_loss(pred_rgb, pred_a, tgt)
    stall = 0

    # 早期要求更大 ROI
    while len(layers) < cfg.max_layers:
        if cur_loss <= cfg.loss_epsilon:
            break
        min_area = 0.004 if len(layers) < 5 else (0.002 if len(layers) < 20 else 0.0012)
        rois = propose_rois(
            tgt,
            pred_rgb,
            pred_a,
            max_rois=cfg.max_rois_per_iter,
            min_area_frac=min_area,
        )
        if not rois:
            break

        best_layer: StampLayer | None = None
        best_loss = cur_loss
        for roi in rois:
            cand = search_best_for_roi(
                roi,
                index,
                tgt,
                library=library,
                recall_k=cfg.recall_k,
                angle_step=cfg.angle_step_deg,
                refine=cfg.refine,
                refine_iters=cfg.refine_iters,
            )
            if cand is None:
                continue
            trial = [*layers, cand]
            tr_rgb, tr_a = _pred_arrays(trial, renderer)
            L = fit_loss(tr_rgb, tr_a, tgt)
            if best_loss > L:
                best_loss = L
                best_layer = cand

        if best_layer is None:
            stall += 1
            if stall >= cfg.stall_patience:
                break
            continue

        rel_gain = (cur_loss - best_loss) / max(cur_loss, 1e-6)
        abs_gain = cur_loss - best_loss
        # 相对或绝对增益之一达到即可（早期绝对增益更重要）
        min_abs = 0.002
        if rel_gain < cfg.min_gain and abs_gain < min_abs:
            stall += 1
            if stall >= cfg.stall_patience:
                break
            # 拒绝但仍计入一次停滞
            continue

        layers.append(best_layer)
        gains.append(float(abs_gain))
        cur_loss = best_loss
        stall = 0
        pred_rgb, pred_a = _pred_arrays(layers, renderer)

    # 全局轻量：仅检查末尾若干层是否可删
    layers = _prune_useless_layers(layers, tgt, renderer, max_checks=8)

    document = EmblemDocument.from_layers(layers)
    # 预览用稍高 ss
    preview_renderer = EmblemRenderer(
        RenderConfig(
            canvas=CanvasConfig(
                width=cfg.canvas_size,
                height=cfg.canvas_size,
                background=None,
            ),
            stamps_dir=cfg.stamps_dir,
            supersample=max(2.0, cfg.render_supersample),
            stamp_raster_scale=1.25,
        )
    )
    preview = preview_renderer.render(document)
    pr_rgb, pr_a = image_to_rgba_arrays(preview)
    final_loss = fit_loss(pr_rgb, pr_a, tgt)
    sim = score_fit(preview, tgt, pass_score=cfg.pass_score)

    return ApproxResult(
        document=document,
        target=tgt,
        final_loss=float(final_loss),
        per_layer_gains=gains,
        similarity=sim,
        preview_rgb=np.asarray(preview.convert("RGBA"), dtype=np.uint8),
    )


def _prune_useless_layers(
    layers: list[StampLayer],
    target: ApproxTarget,
    renderer: EmblemRenderer,
    *,
    max_checks: int = 8,
) -> list[StampLayer]:
    """若去掉某层几乎不增损则删除（从顶到底，限制检查次数）。"""
    if len(layers) <= 1:
        return layers
    kept = list(layers)
    rgb, a = _pred_arrays(kept, renderer)
    base = fit_loss(rgb, a, target)
    i = len(kept) - 1
    checks = 0
    while i >= 0 and len(kept) > 1 and checks < max_checks:
        checks += 1
        trial = kept[:i] + kept[i + 1 :]
        tr, ta = _pred_arrays(trial, renderer)
        L = fit_loss(tr, ta, target)
        if base + 0.0015 >= L:
            kept = trial
            base = L
            i = min(i, len(kept) - 1)
            continue
        i -= 1
    return kept


def approximate_to_files(
    image: Image.Image | str | Path,
    out_json: str | Path,
    out_preview: str | Path | None = None,
    config: ApproxConfig | None = None,
) -> ApproxResult:
    """近似并写 JSON / 预览图。"""
    result = approximate_image(image, config)
    result.document.save_json(out_json)
    if out_preview is not None and result.preview_rgb is not None:
        Path(out_preview).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.preview_rgb, mode="RGBA").save(out_preview)
    return result
