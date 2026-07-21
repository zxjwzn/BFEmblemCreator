"""色彩平面化：ImageProcessorConfig 驱动的严格 LAB 色量。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from bf_emblem_creator.approx.label_field import build_label_field, labels_to_rgb
from bf_emblem_creator.approx.models import AbstractionMode, ApproxMeta, PaletteColor
from bf_emblem_creator.approx.preprocess import (
    bilateral_smooth,
    detect_resample_mode,
    estimate_alpha,
    fit_to_canvas,
)
from bf_emblem_creator.approx.recipe import BilateralStrength, FitPolicy, ImageProcessorConfig

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]
I32Arr = NDArray[np.int32]


def _load_rgba(image: Image.Image | str | Path | U8Arr) -> Image.Image:
    """加载为 RGBA。"""
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


def planarize_image(
    image: Image.Image | str | Path | U8Arr,
    config: ImageProcessorConfig,
    *,
    mode: AbstractionMode,
) -> tuple[I32Arr, list[PaletteColor], FloatArr, U8Arr, ApproxMeta, U8Arr]:
    """
    色彩平面化：严格按 config.num_colors 做 LAB k-means。

    返回 labels、palette、alpha、image_q、meta、src_rgb（平滑后、量化前）。
    """
    k = max(2, min(64, int(config.num_colors)))
    rgba_img = _load_rgba(image)
    fit = "cover" if config.fit_policy == FitPolicy.cover else "contain"
    if mode.value.startswith("photo") and config.fit_policy == FitPolicy.contain:
        # 配方应已设 cover；此处仍尊重 config.fit_policy
        pass
    r_mode, _stats = detect_resample_mode(rgba_img, target_size=config.canvas_size, configured=config.resample_mode)
    rgba, meta = fit_to_canvas(rgba_img, config.canvas_size, how=fit, resample=r_mode)
    meta = meta.model_copy(update={"mode": mode, "num_colors": k})
    alpha = estimate_alpha(rgba, mode)
    rgb = rgba[:, :, :3]
    alpha_f = alpha.astype(np.float64)
    h, w = alpha_f.shape
    mask = alpha_f >= 0.5

    hard_edge = meta.resample == "nearest" and meta.approx_color_count <= 48
    bil_off = (not config.bilateral) or config.bilateral_strength == BilateralStrength.off
    if mode == AbstractionMode.pixel or hard_edge or bil_off:
        src_rgb = rgb.copy()
    else:
        strength = config.bilateral_strength.value
        if strength not in {"weak", "medium", "strong"}:
            strength = "medium"
        src_rgb = bilateral_smooth(rgb, strength=strength)

    if not mask.any():
        labels = np.full((h, w), -1, dtype=np.int32)
        return labels, [], alpha_f, np.zeros((h, w, 3), dtype=np.uint8), meta, src_rgb

    labels, palette, gap_frac, noise_frac = build_label_field(
        src_rgb,
        alpha_f,
        k,
        mrf_lambda=config.mrf_lambda,
        mrf_iters=config.mrf_iters,
        min_area_frac=config.min_region_area_frac,
        enforce_no_gap=config.enforce_no_gap,
        seed=config.seed,
    )
    meta = meta.model_copy(
        update={
            "gap_frac": float(gap_frac),
            "noise_frac": float(noise_frac),
            "num_colors": len(palette) if palette else k,
        }
    )
    image_q = labels_to_rgb(labels, palette)
    return labels, palette, alpha_f, image_q, meta, src_rgb
