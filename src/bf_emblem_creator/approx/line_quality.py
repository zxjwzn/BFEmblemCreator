"""线条质量评分：不规则/锯齿/碎边 → 低分（GPU）。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.gpu_ops import evaluate_line_quality_torch, visible_boundary_mask_torch

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]


class LineQualityReport(BaseModel):
    """线条质量分项。"""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(..., ge=0.0, le=1.0, description="综合线条分，越高越好")
    jaggedness: float = Field(..., ge=0.0, description="锯齿指标（越低越好）")
    corner_density: float = Field(..., ge=0.0, description="角点密度（越低越好）")
    fragment_ratio: float = Field(..., ge=0.0, description="碎边比例（越低越好）")
    edge_pixel_frac: float = Field(..., ge=0.0, le=1.0, description="边缘像素占比")
    hard_fail: bool = Field(..., description="是否触发硬失败（过丑）")


def visible_boundary_mask(rgb: U8Arr, alpha: FloatArr | None = None) -> NDArray[np.bool_]:
    """估计可见边界：颜色跳变或 alpha 边界（GPU）。"""
    dev = get_device()
    edge = visible_boundary_mask_torch(rgb, alpha, device=dev)
    return edge.detach().cpu().numpy().astype(bool)


def evaluate_line_quality(
    rgb: U8Arr,
    alpha: FloatArr | None = None,
    *,
    hard_jagged: float = 0.55,
    hard_corner: float = 0.22,
) -> LineQualityReport:
    """
    评估渲染结果的线条规则度（GPU）。

    角点过密或锯齿过强时 hard_fail，并将 score 封顶到 0.35。
    """
    dev = get_device()
    score, jaggedness, corner_density, frag, edge_frac, hard_fail = evaluate_line_quality_torch(
        rgb,
        alpha,
        device=dev,
        hard_jagged=hard_jagged,
        hard_corner=hard_corner,
    )
    return LineQualityReport(
        score=score,
        jaggedness=float(jaggedness),
        corner_density=float(corner_density),
        fragment_ratio=float(frag),
        edge_pixel_frac=edge_frac,
        hard_fail=hard_fail,
    )
