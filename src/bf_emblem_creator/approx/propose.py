"""残差 ROI 提案。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bf_emblem_creator.approx.index import region_features_from_mask
from bf_emblem_creator.approx.models import ApproxTarget

FloatArr = NDArray[np.floating]
U8Arr = NDArray[np.uint8]


@dataclass
class RoiProposal:
    """一个候选区域。"""

    mask: NDArray[np.bool_]
    bbox: tuple[int, int, int, int]  # x0,y0,x1,y1
    color_rgb: tuple[int, int, int]
    area: float
    score: float
    features: dict[str, float]
    label: int | None = None


def propose_rois(
    target: ApproxTarget,
    pred_rgb: U8Arr,
    pred_alpha: FloatArr,
    *,
    max_rois: int = 6,
    min_area_frac: float = 0.0015,
) -> list[RoiProposal]:
    """
    基于色区标签与残差提出 ROI。

    优先：目标有、预测不足的色块；以及 alpha 空洞。
    """
    tgt = target.numpy_rgb()
    alpha = target.numpy_alpha()
    labels = target.numpy_labels()
    h, w = alpha.shape
    min_area = min_area_frac * h * w
    rois: list[RoiProposal] = []

    # 残差能量：目标覆盖处颜色差 + 覆盖差
    color_diff = np.mean(np.abs(tgt.astype(np.float64) - pred_rgb.astype(np.float64)), axis=2) / 255.0
    cover_gap = np.clip(alpha - pred_alpha, 0.0, 1.0)
    residual = color_diff * np.maximum(alpha, 0.2) + 0.75 * cover_gap

    max_lab = int(labels.max()) if labels.size and labels.max() >= 0 else -1
    for lab in range(max_lab + 1):
        base = (labels == lab) & (alpha >= 0.5)
        if int(base.sum()) < min_area:
            continue
        # 残差加权：该色区里仍差的部分
        need = base & (residual > 0.08)
        if int(need.sum()) < min_area * 0.35:
            # 仍可尝试整块（早期层）
            need = base
        # 拆连通域
        for cc in _iter_ccs(need):
            area = float(cc.sum())
            if area < min_area:
                continue
            ys, xs = np.where(cc)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            mean_res = float(residual[cc].mean())
            feat = region_features_from_mask(cc)
            color = tuple(int(v) for v in np.median(tgt[cc], axis=0))
            # 大面积 + 高残差优先
            score = mean_res * np.sqrt(area) * (1.0 + 0.3 * feat["fill_ratio"])
            rois.append(
                RoiProposal(
                    mask=cc,
                    bbox=bbox,
                    color_rgb=(color[0], color[1], color[2]),
                    area=area,
                    score=float(score),
                    features=feat,
                    label=lab,
                )
            )

    # alpha 空洞（目标有预测无）
    hole = (alpha >= 0.5) & (pred_alpha < 0.25)
    for cc in _iter_ccs(hole):
        area = float(cc.sum())
        if area < min_area:
            continue
        ys, xs = np.where(cc)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        feat = region_features_from_mask(cc)
        color = tuple(int(v) for v in np.median(tgt[cc], axis=0))
        score = 0.9 * np.sqrt(area)
        rois.append(
            RoiProposal(
                mask=cc,
                bbox=bbox,
                color_rgb=(color[0], color[1], color[2]),
                area=area,
                score=float(score),
                features=feat,
                label=None,
            )
        )

    rois.sort(key=lambda r: -r.score)
    # 抑制高度重叠 ROI
    picked: list[RoiProposal] = []
    for r in rois:
        if len(picked) >= max_rois:
            break
        if any(_mask_iou(r.mask, p.mask) > 0.7 for p in picked):
            continue
        picked.append(r)
    return picked


def _mask_iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-8)


def _iter_ccs(binary: NDArray[np.bool_]):
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    from collections import deque

    for y in range(h):
        for x in range(w):
            if not binary[y, x] or seen[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            mask = np.zeros_like(binary)
            for cy, cx in cells:
                mask[cy, cx] = True
            yield mask
