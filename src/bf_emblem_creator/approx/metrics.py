"""综合评分：曲线边界比对 × 色彩 × 线条质量 × 简洁度（GPU Chamfer / LAB）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.blocks import BlockTarget
from bf_emblem_creator.approx.contour_arcs import ArcPrimitive, primitives_to_point_cloud
from bf_emblem_creator.approx.curves import extract_outer_contour, resample_closed_contour
from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.gpu_ops import chamfer_points, lab_distance_map, to_torch, visible_boundary_mask_torch
from bf_emblem_creator.approx.line_quality import LineQualityReport, evaluate_line_quality

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]


class SimilarityParts(BaseModel):
    """内容相似度分项。"""

    model_config = ConfigDict(extra="forbid")

    alpha_iou: float = Field(..., ge=0.0, le=1.0)
    color_score: float = Field(..., ge=0.0, le=1.0)
    edge_score: float = Field(..., ge=0.0, le=1.0, description="可见边界曲线比对分")
    coverage: float = Field(..., ge=0.0, le=1.0)
    score: float = Field(..., ge=0.0, le=1.0, description="S_sim")


class FullScoreReport(BaseModel):
    """综合评分报告。"""

    model_config = ConfigDict(extra="forbid")

    sim: SimilarityParts
    line: LineQualityReport
    simple: float = Field(..., ge=0.0, le=1.0, description="S_simple 简洁度")
    overall: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    n_layers: int = Field(..., ge=0)
    pass_sim: float
    pass_line: float
    pass_overall: float
    curve_chamfer: float = Field(default=0.0, description="边界 Chamfer 像素距离")
    notes: str = ""

    def summary(self) -> str:
        flag = "达标" if self.passed else "未达标"
        return (
            f"[{flag}] overall={self.overall:.3f} "
            f"sim={self.sim.score:.3f} line={self.line.score:.3f} "
            f"simple={self.simple:.3f} curve={self.sim.edge_score:.3f} "
            f"color={self.sim.color_score:.3f} layers={self.n_layers}" + (" HARD_LINE_FAIL" if self.line.hard_fail else "")
        )


def _to_rgb_alpha(image: Image.Image | U8Arr) -> tuple[U8Arr, FloatArr]:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.shape[-1] == 3:
            return arr.astype(np.uint8), np.ones(arr.shape[:2], dtype=np.float64)
        arr = arr.astype(np.uint8)
    return arr[:, :, :3], arr[:, :, 3].astype(np.float64) / 255.0


def _composite_torch(
    rgb: U8Arr,
    alpha: FloatArr,
    bg: tuple[int, int, int] = (128, 128, 128),
    *,
    device: torch.device,
) -> torch.Tensor:
    """合成到不透明 RGB 张量 (H,W,3)。"""
    r = to_torch(rgb, device=device)
    a = to_torch(np.asarray(alpha, dtype=np.float32), device=device)[..., None]
    bg_t = torch.tensor(bg, device=device, dtype=torch.float32)
    out = r * a + bg_t * (1.0 - a)
    return out.clamp(0, 255)


def extract_visible_curve_points(rgb: U8Arr, alpha: FloatArr, n: int = 128) -> FloatArr:
    """从渲染图提取可见边界采样点（GPU 边界）。"""
    dev = get_device()
    edge = visible_boundary_mask_torch(rgb, alpha, device=dev)
    ys, xs = torch.where(edge)
    if int(xs.numel()) >= 8:
        if int(xs.numel()) > 600:
            idx = torch.linspace(0, xs.numel() - 1, 600, device=dev).long()
            xs, ys = xs[idx], ys[idx]
        pts = torch.stack([xs.float(), ys.float()], dim=1).detach().cpu().numpy().astype(np.float64)
        return resample_closed_contour(pts, n) if len(pts) >= 4 else pts
    cont = extract_outer_contour(alpha >= 0.5, simplify=1.0)
    if cont is None or len(cont) < 4:
        return np.zeros((0, 2), dtype=np.float64)
    return resample_closed_contour(cont, n)


def curve_boundary_score(
    pred_rgb: U8Arr,
    pred_alpha: FloatArr,
    target_pts: FloatArr,
) -> tuple[float, float]:
    """预测曲线 vs 目标曲线 Chamfer（GPU）→ (score 0~1, chamfer_px)。"""
    tgt = np.asarray(target_pts, dtype=np.float64)
    if len(tgt) < 4:
        return 0.65, 0.0
    pred = extract_visible_curve_points(pred_rgb, pred_alpha, n=max(64, len(tgt)))
    if len(pred) < 4:
        return 0.2, 40.0
    dev = get_device()
    chamfer = chamfer_points(pred, tgt, device=dev)
    score = float(np.clip(np.exp(-chamfer / 14.0), 0.0, 1.0))
    return score, float(chamfer)


def target_curve_from_blocks(target: BlockTarget) -> FloatArr:
    """从色块轮廓合并目标曲线点。"""
    pts: list[FloatArr] = []
    for b in target.blocks:
        c = np.asarray(b.contour_resampled, dtype=np.float64)
        if len(c):
            pts.append(c)
    if not pts:
        cont = extract_outer_contour(target.numpy_alpha() >= 0.5, simplify=1.2)
        if cont is None:
            return np.zeros((0, 2), dtype=np.float64)
        return resample_closed_contour(cont, 128)
    return np.vstack(pts)


def compute_similarity(
    pred_rgb: U8Arr,
    pred_alpha: FloatArr,
    target_rgb: U8Arr,
    target_alpha: FloatArr,
    weight: FloatArr | None = None,
    *,
    target_curve: FloatArr | None = None,
) -> tuple[SimilarityParts, float]:
    """S_sim：IoU + 色（GPU LAB）+ 曲线边界 + 覆盖。"""
    dev = get_device()
    if weight is None:
        weight = np.maximum(target_alpha, 0.05)
    w = to_torch(np.asarray(weight, dtype=np.float32), device=dev)
    wsum = float(w.sum().item()) + 1e-8

    pa = to_torch(np.asarray(pred_alpha, dtype=np.float32), device=dev)
    ta = to_torch(np.asarray(target_alpha, dtype=np.float32), device=dev)
    pbin = pa >= 0.5
    tbin = ta >= 0.5
    inter = float((pbin & tbin).sum().item())
    union = float((pbin | tbin).sum().item()) + 1e-8
    iou = inter / union
    coverage = inter / (float(tbin.sum().item()) + 1e-8)

    pc = _composite_torch(pred_rgb, pred_alpha, device=dev)
    tc = _composite_torch(target_rgb, target_alpha, device=dev)
    dist = lab_distance_map(pc, tc, device=dev)
    mean_d = float((dist * w).sum().item() / wsum)
    color_score = float(np.clip(1.0 - mean_d / 55.0, 0.0, 1.0))

    if target_curve is None or len(target_curve) < 4:
        gp = pc.mean(dim=-1) / 255.0
        gt = tc.mean(dim=-1) / 255.0
        gx_p = torch.zeros_like(gp)
        gy_p = torch.zeros_like(gp)
        gx_p[:, 1:-1] = (gp[:, 2:] - gp[:, :-2]).abs()
        gy_p[1:-1, :] = (gp[2:, :] - gp[:-2, :]).abs()
        ep = torch.hypot(gx_p, gy_p)
        gx_t = torch.zeros_like(gt)
        gy_t = torch.zeros_like(gt)
        gx_t[:, 1:-1] = (gt[:, 2:] - gt[:, :-2]).abs()
        gy_t[1:-1, :] = (gt[2:, :] - gt[:-2, :]).abs()
        et = torch.hypot(gx_t, gy_t)
        for e in (ep, et):
            m = float(e.max().item())
            if m > 1e-8:
                e /= m
        edge_l1 = float(((ep - et).abs() * w).sum().item() / wsum)
        edge_score = float(np.clip(1.0 - edge_l1 / 0.45, 0.0, 1.0))
        chamfer = 0.0
    else:
        edge_score, chamfer = curve_boundary_score(pred_rgb, pred_alpha, target_curve)

    score = 0.30 * iou + 0.35 * color_score + 0.25 * edge_score + 0.10 * coverage
    parts = SimilarityParts(
        alpha_iou=float(iou),
        color_score=color_score,
        edge_score=edge_score,
        coverage=float(np.clip(coverage, 0, 1)),
        score=float(np.clip(score, 0, 1)),
    )
    return parts, chamfer


def simplicity_score(n_layers: int, max_layers: int = 40) -> float:
    """层数越少越高。"""
    return float(np.exp(-0.8 * n_layers / max(max_layers, 1)))


def score_prediction(
    pred: Image.Image | U8Arr,
    target: BlockTarget,
    *,
    n_layers: int,
    target_primitives: list[ArcPrimitive] | None = None,
    pass_sim: float = 0.55,
    pass_line: float = 0.60,
    pass_overall: float = 0.50,
) -> FullScoreReport:
    """
    综合评分（GPU）。

    - 曲线：预测可见边界 vs 原图/基元 SHAPE_BOUNDARY
    - 色彩：LAB 一致
    - overall = sim * line * (0.55 + 0.45*simple)
    """
    pr, pa = _to_rgb_alpha(pred)
    cs = target.canvas_size
    if pr.shape[0] != cs or pr.shape[1] != cs:
        img = Image.fromarray(np.dstack([pr, (pa * 255).astype(np.uint8)]), mode="RGBA")
        img = img.resize((cs, cs), Image.Resampling.BILINEAR)
        pr, pa = _to_rgb_alpha(img)

    if target_primitives:
        tgt_curve = primitives_to_point_cloud(target_primitives, only_shape=True)
        if len(tgt_curve) < 4:
            tgt_curve = target_curve_from_blocks(target)
    else:
        tgt_curve = target_curve_from_blocks(target)

    sim, chamfer = compute_similarity(
        pr,
        pa,
        target.numpy_rgb(),
        target.numpy_alpha(),
        target.numpy_weight(),
        target_curve=tgt_curve,
    )
    line = evaluate_line_quality(pr, pa)
    simple = simplicity_score(n_layers)

    overall = float(sim.score * line.score * (0.55 + 0.45 * simple))
    if line.hard_fail or line.score < pass_line:
        overall = min(overall, 0.35)
        passed = False
    else:
        passed = sim.score >= pass_sim and line.score >= pass_line and overall >= pass_overall and n_layers <= 40

    notes = "综合分 = sim(色+曲线边界,GPU) × line(GPU) × simple；线条硬门槛优先。"
    return FullScoreReport(
        sim=sim,
        line=line,
        simple=simple,
        overall=overall,
        passed=passed,
        n_layers=n_layers,
        pass_sim=pass_sim,
        pass_line=pass_line,
        pass_overall=pass_overall,
        curve_chamfer=chamfer,
        notes=notes,
    )


def score_fit(pred: Image.Image | U8Arr, target: BlockTarget, **kwargs: Any) -> FullScoreReport:
    """兼容入口。"""
    n_layers = int(kwargs.pop("n_layers", 1))
    return score_prediction(pred, target, n_layers=n_layers, **kwargs)
