"""拟合损失与近似相似度评判。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.color import lab_distance, rgb_u8_to_lab
from bf_emblem_creator.approx.models import ApproxTarget

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]


class SimilarityReport(BaseModel):
    """
    近似相似度报告（越高越好，均在 0~1）。

    设计动机：徽章/emoji 应优先剪影与主色，而非纹理 PSNR。
    """

    model_config = ConfigDict(extra="forbid")

    alpha_iou: float = Field(..., ge=0.0, le=1.0, description="主体蒙版 IoU")
    color_score: float = Field(..., ge=0.0, le=1.0, description="加权 LAB 色差相似度")
    edge_score: float = Field(..., ge=0.0, le=1.0, description="边缘图相似度")
    coverage: float = Field(..., ge=0.0, le=1.0, description="预测对目标主体的覆盖率")
    overall: float = Field(..., ge=0.0, le=1.0, description="加权综合分")
    passed: bool = Field(..., description="是否达到 pass_score")
    pass_score: float = Field(..., description="所用达标阈值")
    notes: str = Field(default="", description="简要说明")

    def summary(self) -> str:
        """单行摘要。"""
        flag = "达标" if self.passed else "未达标"
        return (
            f"[{flag}] overall={self.overall:.3f} "
            f"iou={self.alpha_iou:.3f} color={self.color_score:.3f} "
            f"edge={self.edge_score:.3f} cov={self.coverage:.3f}"
        )


def sobel_magnitude(gray: FloatArr) -> FloatArr:
    """3×3 Sobel 梯度幅度（边界 replicate）。"""
    g = np.pad(gray, 1, mode="edge")
    # gx, gy
    gx = (
        -g[:-2, :-2]
        + g[:-2, 2:]
        - 2 * g[1:-1, :-2]
        + 2 * g[1:-1, 2:]
        - g[2:, :-2]
        + g[2:, 2:]
    )
    gy = (
        -g[:-2, :-2]
        - 2 * g[:-2, 1:-1]
        - g[:-2, 2:]
        + g[2:, :-2]
        + 2 * g[2:, 1:-1]
        + g[2:, 2:]
    )
    mag = np.hypot(gx, gy)
    peak = float(mag.max()) if mag.size else 1.0
    if peak < 1e-8:
        return np.zeros_like(mag)
    return mag / peak


def image_to_rgba_arrays(image: Image.Image | U8Arr) -> tuple[U8Arr, FloatArr]:
    """PIL 或数组 → (rgb uint8 HxWx3, alpha float HxW)。"""
    if isinstance(image, Image.Image):
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            rgb = np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
            return rgb, np.ones(arr.shape[:2], dtype=np.float64)
        if arr.shape[-1] == 3:
            return arr.astype(np.uint8), np.ones(arr.shape[:2], dtype=np.float64)
        rgba = arr.astype(np.uint8)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3].astype(np.float64) / 255.0
    return rgb, alpha


def composite_on_background(
    rgb: U8Arr,
    alpha: FloatArr,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> U8Arr:
    """将直通 alpha 图合成到不透明背景上。"""
    a = alpha[..., None]
    bg_arr = np.array(bg, dtype=np.float64)
    out = rgb.astype(np.float64) * a + bg_arr * (1.0 - a)
    return (np.clip(out, 0, 255) + 0.5).astype(np.uint8)


def fit_loss(
    pred_rgb: U8Arr,
    pred_alpha: FloatArr,
    target: ApproxTarget,
    *,
    lambda_color: float = 1.0,
    lambda_edge: float = 0.85,
    lambda_alpha: float = 0.65,
) -> float:
    """
    复合拟合损失（越低越好）。

    L = λc * 加权 LAB L1 + λe * 边缘 L1 + λa * alpha L1
    """
    tgt_rgb = target.numpy_rgb()
    tgt_a = target.numpy_alpha()
    w = target.numpy_weight()

    # 合成到中性灰底再比颜色，避免透明区干扰
    bg = (128, 128, 128)
    p = composite_on_background(pred_rgb, pred_alpha, bg)
    t = composite_on_background(tgt_rgb, tgt_a, bg)
    lab_p = rgb_u8_to_lab(p)
    lab_t = rgb_u8_to_lab(t)
    # LAB L 约 0-100，a/b 约 ±128 → 归一
    color_l1 = lab_distance(lab_p, lab_t) / 100.0
    color_term = float(np.mean(w * color_l1))

    gray_p = p.astype(np.float64).mean(axis=2) / 255.0
    gray_t = t.astype(np.float64).mean(axis=2) / 255.0
    edge_term = float(np.mean(np.abs(sobel_magnitude(gray_p) - sobel_magnitude(gray_t)) * w))

    alpha_term = float(np.mean(np.abs(pred_alpha - tgt_a) * np.maximum(w, tgt_a)))

    return lambda_color * color_term + lambda_edge * edge_term + lambda_alpha * alpha_term


def compare_images(
    pred: Image.Image | U8Arr,
    target_rgb: U8Arr,
    target_alpha: FloatArr,
    *,
    weight: FloatArr | None = None,
    pass_score: float = 0.52,
) -> SimilarityReport:
    """
    比较预测图与目标（概括图或原图对齐后）。

    综合分：
      overall = 0.35*iou + 0.35*color + 0.20*edge + 0.10*coverage
    """
    pred_rgb, pred_a = image_to_rgba_arrays(pred)
    if pred_rgb.shape[:2] != target_rgb.shape[:2]:
        raise ValueError("预测与目标尺寸不一致")

    if weight is None:
        weight = np.maximum(target_alpha, 0.05)
    weight = weight.astype(np.float64)
    wsum = float(weight.sum()) + 1e-8

    # IoU
    pbin = pred_a >= 0.5
    tbin = target_alpha >= 0.5
    inter = float(np.logical_and(pbin, tbin).sum())
    union = float(np.logical_or(pbin, tbin).sum()) + 1e-8
    iou = inter / union

    coverage = inter / (float(tbin.sum()) + 1e-8)

    bg = (128, 128, 128)
    pc = composite_on_background(pred_rgb, pred_a, bg)
    tc = composite_on_background(target_rgb, target_alpha, bg)
    dist = lab_distance(rgb_u8_to_lab(pc), rgb_u8_to_lab(tc))
    # 典型差 0~80，映射到 0~1 分
    mean_dist = float((dist * weight).sum() / wsum)
    color_score = float(np.clip(1.0 - mean_dist / 55.0, 0.0, 1.0))

    gp = pc.astype(np.float64).mean(axis=2) / 255.0
    gt = tc.astype(np.float64).mean(axis=2) / 255.0
    edge_l1 = float((np.abs(sobel_magnitude(gp) - sobel_magnitude(gt)) * weight).sum() / wsum)
    edge_score = float(np.clip(1.0 - edge_l1 / 0.45, 0.0, 1.0))

    overall = 0.35 * iou + 0.35 * color_score + 0.20 * edge_score + 0.10 * coverage
    overall = float(np.clip(overall, 0.0, 1.0))
    passed = overall >= pass_score
    notes = "剪影与主色优先的综合相似度；非 PSNR。"
    return SimilarityReport(
        alpha_iou=float(iou),
        color_score=color_score,
        edge_score=edge_score,
        coverage=float(np.clip(coverage, 0.0, 1.0)),
        overall=overall,
        passed=passed,
        pass_score=pass_score,
        notes=notes,
    )


def score_fit(
    pred: Image.Image | U8Arr,
    target: ApproxTarget,
    *,
    pass_score: float = 0.52,
) -> SimilarityReport:
    """对 ApproxTarget 评分。"""
    return compare_images(
        pred,
        target.numpy_rgb(),
        target.numpy_alpha(),
        weight=target.numpy_weight(),
        pass_score=pass_score,
    )
