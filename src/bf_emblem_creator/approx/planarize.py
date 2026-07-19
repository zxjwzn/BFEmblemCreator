"""P1：色彩平面化（Batch A 重采样 + Batch B 标签场）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.label_field import build_label_field, labels_to_rgb
from bf_emblem_creator.approx.models import AbstractionMode, ApproxConfig, ApproxMeta, PaletteColor
from bf_emblem_creator.approx.preprocess import (
    bilateral_smooth,
    detect_mode,
    detect_resample_mode,
    estimate_alpha,
    fit_to_canvas,
)

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


def kmeans_lab_torch(
    lab_pts: torch.Tensor,
    k: int,
    *,
    iters: int = 15,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU/CPU LAB k-means（兼容旧接口）。

    lab_pts: (N,3)；返回 assign (N,) int64、centers (k,3)。
    """
    n = int(lab_pts.shape[0])
    if n == 0:
        return (
            torch.zeros(0, dtype=torch.long, device=lab_pts.device),
            torch.zeros((k, 3), dtype=lab_pts.dtype, device=lab_pts.device),
        )
    k_eff = min(k, n)
    g = torch.Generator(device=lab_pts.device)
    g.manual_seed(int(seed))
    idx0 = int(torch.randint(0, n, (1,), generator=g, device=lab_pts.device).item())
    centers = lab_pts[idx0 : idx0 + 1]
    closest = torch.full((n,), float("inf"), device=lab_pts.device, dtype=lab_pts.dtype)
    for _ in range(1, k_eff):
        d = torch.cdist(lab_pts, centers[-1:]).squeeze(1)
        closest = torch.minimum(closest, d)
        probs = closest.clamp_min(0.0) ** 2
        s = float(probs.sum().item())
        if s < 1e-12:
            j = int(torch.randint(0, n, (1,), generator=g, device=lab_pts.device).item())
        else:
            j = int(torch.multinomial(probs, 1, generator=g).item())
        centers = torch.cat([centers, lab_pts[j : j + 1]], dim=0)
    assign = torch.zeros(n, dtype=torch.long, device=lab_pts.device)
    for _ in range(iters):
        d = torch.cdist(lab_pts, centers)
        assign = torch.argmin(d, dim=1)
        new_centers = centers.clone()
        for ci in range(k_eff):
            sel = assign == ci
            if bool(sel.any().item()):
                new_centers[ci] = lab_pts[sel].mean(dim=0)
            else:
                j = int(torch.randint(0, n, (1,), generator=g, device=lab_pts.device).item())
                new_centers[ci] = lab_pts[j]
        centers = new_centers
    return assign, centers


def planarize_image(
    image: Image.Image | str | Path | U8Arr,
    config: ApproxConfig | None = None,
    *,
    k: int | None = None,
    device: torch.device | None = None,
) -> tuple[I32Arr, list[PaletteColor], FloatArr, U8Arr, ApproxMeta, U8Arr]:
    """
    色彩平面化（Batch A + B）。

    返回 labels、palette、alpha、image_q、meta、src_rgb（平滑后、量化前）。
    """
    cfg = config or ApproxConfig()
    _ = device or get_device(prefer_cuda=cfg.use_cuda)
    rgba_img = _load_rgba(image)
    mode = detect_mode(rgba_img) if cfg.mode == AbstractionMode.auto else cfg.mode
    fit = "cover" if mode.value.startswith("photo") else "contain"
    r_mode, _stats = detect_resample_mode(rgba_img, target_size=cfg.canvas_size, configured=cfg.resample_mode)
    rgba, meta = fit_to_canvas(rgba_img, cfg.canvas_size, how=fit, resample=r_mode)
    meta = meta.model_copy(update={"mode": mode})
    alpha = estimate_alpha(rgba, mode)
    rgb = rgba[:, :, :3]

    hard_edge = meta.resample == "nearest" and meta.approx_color_count <= 48
    if hard_edge:
        src_rgb = rgb.copy()
    elif cfg.bilateral and mode != AbstractionMode.logo:
        strength = "strong" if mode.value.startswith("photo") else "medium"
        src_rgb = bilateral_smooth(rgb, strength=strength)
    elif mode == AbstractionMode.logo:
        src_rgb = bilateral_smooth(rgb, strength="weak")
    else:
        src_rgb = rgb.copy()

    k_use = int(k if k is not None else cfg.palette_k)
    k_use = max(2, min(16, k_use))
    mask = alpha >= 0.5
    h, w = alpha.shape
    if not mask.any():
        labels = np.full((h, w), -1, dtype=np.int32)
        return labels, [], alpha.astype(np.float64), np.zeros((h, w, 3), dtype=np.uint8), meta, src_rgb

    labels, palette, gap_frac, noise_frac = build_label_field(
        src_rgb,
        alpha.astype(np.float64),
        k_use,
        grad_q=cfg.flat_grad_q,
        lab_merge=cfg.lab_merge,
        mrf_lambda=cfg.mrf_lambda,
        mrf_iters=cfg.mrf_iters,
        min_area_frac=cfg.min_region_area_frac,
        enforce_no_gap=cfg.enforce_no_gap,
        seed=cfg.seed,
    )
    meta = meta.model_copy(update={"gap_frac": float(gap_frac), "noise_frac": float(noise_frac)})
    image_q = labels_to_rgb(labels, palette)
    return labels, palette, alpha.astype(np.float64), image_q, meta, src_rgb
