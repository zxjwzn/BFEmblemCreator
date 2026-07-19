"""P7：特殊图章通道（星形/放射/网格 → 渐变感）。"""

from __future__ import annotations

import math

import numpy as np
import torch
from numpy.typing import NDArray

from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
from bf_emblem_creator.approx.torch_render import TorchStampRenderer
from bf_emblem_creator.models import StampLayer

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]


def detect_gradient_structure(
    src_rgb: U8Arr,
    alpha: FloatArr,
    *,
    min_std: float = 12.0,
) -> dict[str, float] | None:
    """
    在平面化前原图上检测区域级渐变。

    返回主方向（度）、线性强度；无显著渐变则 None。
    """
    a = np.asarray(alpha >= 0.5, dtype=bool)
    if float(a.mean()) < 0.05:
        return None
    lab_like = src_rgb.astype(np.float64).mean(axis=2)
    vals = lab_like[a]
    if float(vals.std()) < min_std:
        return None
    # 图像梯度主方向
    gy_arr, gx_arr = np.gradient(lab_like)
    gx_m = float(np.asarray(gx_arr, dtype=np.float64)[a].mean())
    gy_m = float(np.asarray(gy_arr, dtype=np.float64)[a].mean())
    mag = math.hypot(gx_m, gy_m)
    if mag < 0.15:
        return None
    angle = math.degrees(math.atan2(gy_m, gx_m)) % 360.0
    # 径向：中心到边缘亮度单调
    h, w = lab_like.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(yy - cy, xx - cx)
    r_vals = np.asarray(rr, dtype=np.float64)[a]
    r_norm = r_vals / (r_vals.max() + 1e-6)
    # 相关
    v = vals - vals.mean()
    r = r_norm - r_norm.mean()
    corr = float((v * r).mean() / (vals.std() * r_norm.std() + 1e-6))
    kind = "radial" if abs(corr) > 0.35 else "linear"
    return {
        "angle": angle,
        "strength": float(min(1.0, vals.std() / 40.0)),
        "radial_corr": corr,
        "kind_radial": 1.0 if kind == "radial" else 0.0,
    }


def try_special_fx_layers(
    src_rgb: U8Arr,
    alpha: FloatArr,
    curve_lib: StampCurveLibrary,
    renderer: TorchStampRenderer,
    *,
    base_fill: str = "#808080",
    max_layers: int = 2,
    seed: int = 0,
) -> list[StampLayer]:
    """
    若残差呈渐变结构，从 radial|star|grid|gradient_candidate 检索并放置。

    通用通道，非场景特判。
    """
    info = detect_gradient_structure(src_rgb, alpha)
    if info is None or info["strength"] < 0.25:
        return []

    tags = ("radial", "star", "grid", "gradient_candidate")
    candidates = curve_lib.by_tag(*tags)
    if not candidates:
        # 描述子 boost 召回
        dummy = np.zeros(16, dtype=np.float64)
        candidates = curve_lib.recall(dummy, 0.5, 1.2, k=12, tag_boost=set(tags))
    if not candidates:
        return []

    device = renderer.device
    cs = float(renderer.canvas_size)
    tgt = torch.from_numpy(np.asarray(alpha >= 0.5, dtype=np.float32)).to(device)
    # 用亮度残差作弱目标：高对比区域
    gray = src_rgb.astype(np.float32).mean(axis=2)
    g = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
    # 目标：中等 alpha 调制
    tgt_soft = torch.from_numpy((g * (alpha >= 0.5)).astype(np.float32)).to(device)

    rng = np.random.default_rng(seed)
    layers: list[StampLayer] = []
    angle0 = float(info["angle"])
    for asset_id in candidates[:8]:
        if len(layers) >= max_layers:
            break
        n = 64
        if info["kind_radial"] > 0.5:
            left = np.full(n, cs * 0.5) + rng.normal(0, 10, n)
            top = np.full(n, cs * 0.5) + rng.normal(0, 10, n)
            width = rng.uniform(0.8 * cs, 2.5 * cs, n)
            height = width.copy()
            angles = rng.uniform(0, 360, n)
        else:
            left = np.full(n, cs * 0.5) + rng.normal(0, 20, n)
            top = np.full(n, cs * 0.5) + rng.normal(0, 20, n)
            width = rng.uniform(1.0 * cs, 3.5 * cs, n)
            height = rng.uniform(0.4 * cs, 1.5 * cs, n)
            angles = (angle0 + rng.uniform(-25, 25, n)) % 360.0

        masks = renderer.render_batch_masks(
            asset_id,
            left=torch.tensor(left, dtype=torch.float32, device=device),
            top=torch.tensor(top, dtype=torch.float32, device=device),
            width=torch.tensor(width, dtype=torch.float32, device=device),
            height=torch.tensor(height, dtype=torch.float32, device=device),
            angle_deg=torch.tensor(angles, dtype=torch.float32, device=device),
        )[:, 0]
        # 与 soft 目标相关 + 主体覆盖
        flat_m = masks.reshape(n, -1)
        flat_t = tgt_soft.reshape(-1)
        # 归一化相关
        m0 = flat_m - flat_m.mean(dim=1, keepdim=True)
        t0 = flat_t - flat_t.mean()
        corr = (m0 * t0[None]).mean(dim=1) / (m0.std(dim=1).clamp_min(1e-6) * (t0.std() + 1e-6))
        cover = (masks * tgt[None]).sum(dim=(1, 2)) / tgt.sum().clamp_min(1e-6)
        score = 0.6 * corr + 0.4 * cover
        bi = int(torch.argmax(score).item())
        if float(score[bi].item()) < 0.15:
            continue
        # 取主体中位色作 fill
        ys, xs = np.where(alpha >= 0.5)
        if len(xs):
            med = np.median(src_rgb[ys, xs], axis=0)
            fill = f"#{int(med[0]):02X}{int(med[1]):02X}{int(med[2]):02X}"
        else:
            fill = base_fill
        layers.append(
            StampLayer(
                asset=asset_id,
                opacity=0.55,
                angle=float(angles[bi]),
                flipX=False,
                flipY=False,
                top=float(top[bi]),
                left=float(left[bi]),
                height=float(height[bi]),
                width=float(width[bi]),
                fill=fill,
            )
        )
    return layers
