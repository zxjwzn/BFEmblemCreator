"""P2：区域分割规整与邻接图（形态学/连通域/邻接 GPU）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.curves import (
    contour_curvature_descriptor,
    fit_mask_contour_area_constrained,
    mask_to_sdf,
)
from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.gpu_ops import adjacency_edge_lengths, label_connected_components, morph_close_open
from bf_emblem_creator.approx.models import PaletteColor

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]
BoolArr = NDArray[np.bool_]
I32Arr = NDArray[np.int32]


class Region(BaseModel):
    """单色连通区域。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    region_id: int = Field(..., description="区域 id")
    color_hex: str
    color_rgb: tuple[int, int, int]
    area_frac: float = Field(..., ge=0.0)
    bbox: tuple[int, int, int, int]
    mask: Any
    contour: Any
    contour_resampled: Any
    descriptor: Any
    sdf: Any | None = None
    depth: int = Field(default=0, description="层序，小者更靠下")
    centroid: tuple[float, float] = Field(default=(0.0, 0.0))
    contour_area_rel_err: float = Field(
        default=0.0,
        ge=0.0,
        description="闭合轮廓多边形面积相对 mask 像素面积的相对误差",
    )


class AdjacencyEdge(BaseModel):
    """两区域共享边界。"""

    model_config = ConfigDict(extra="forbid")

    a: int
    b: int
    length: float = Field(..., ge=0.0, description="共享边界像素数近似")


class RegionGraph(BaseModel):
    """区域 + 邻接图。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    regions: list[Region]
    edges: list[AdjacencyEdge]
    labels: Any = Field(..., description="规整后标签图")
    image_rgb: Any
    alpha: Any
    canvas_size: int = 320

    def by_id(self) -> dict[int, Region]:
        """region_id → Region。"""
        return {r.region_id: r for r in self.regions}


def _morph_close_open(
    mask: BoolArr,
    close: int = 2,
    open_: int = 1,
    *,
    device: torch.device | None = None,
) -> BoolArr:
    """形态学闭开（GPU）。"""
    return morph_close_open(mask, close=close, open_=open_, device=device)


def _label_ccs(binary: BoolArr, *, device: torch.device | None = None) -> list[BoolArr]:
    """四连通域列表（GPU）。"""
    return label_connected_components(binary, device=device)


def build_regions(
    labels: I32Arr,
    palette: list[PaletteColor],
    alpha: FloatArr,
    *,
    min_area_frac: float = 0.002,
    max_regions: int = 48,
    max_contour_area_rel_err: float = 0.03,
    device: torch.device | None = None,
    enforce_no_gap: bool = True,
) -> RegionGraph:
    """
    从标签图构建连通域区域与邻接图。

    过小域并入邻接众数（禁止主体内 -1 空洞）；邻接权重为共享边界长度。
    轮廓用面积约束拟合（相对 mask 面积误差默认 ≤3%）。
    """
    from bf_emblem_creator.approx.label_field import fill_label_gaps

    dev = device or get_device()
    h, w = labels.shape
    canvas_area = float(h * w)
    min_area = min_area_frac * canvas_area
    work_labels = np.asarray(labels, dtype=np.int32).copy()
    if enforce_no_gap:
        work_labels = fill_label_gaps(work_labels, alpha)

    regions: list[Region] = []
    region_map = np.full((h, w), -1, dtype=np.int32)
    image_q = np.zeros((h, w, 3), dtype=np.uint8)
    rid = 0
    # 暂存过小 CC，稍后并入
    tiny: list[tuple[BoolArr, int]] = []

    for i, pal in enumerate(palette):
        base = (work_labels == i) & (alpha >= 0.5)
        if not base.any():
            continue
        # 轻形态学：nearest 硬边时 open=0；仅 1px close 减噪
        base = _morph_close_open(base, close=1, open_=0, device=dev)
        for cc in _label_ccs(base, device=dev):
            area = float(cc.sum())
            if area < min_area:
                tiny.append((cc, i))
                continue
            contour, rs, area_err = fit_mask_contour_area_constrained(
                cc,
                max_area_rel_err=max_contour_area_rel_err,
                resample_n=192,
            )
            if contour is None or len(contour) < 3:
                ys, xs = np.where(cc)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                contour = np.array(
                    [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
                    dtype=np.float64,
                )
                rs = contour.copy()
                area_err = 1.0
            desc = contour_curvature_descriptor(rs if len(rs) >= 8 else contour)
            ys, xs = np.where(cc)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            cx = float(xs.mean())
            cy = float(ys.mean())
            regions.append(
                Region(
                    region_id=rid,
                    color_hex=pal.hex,
                    color_rgb=pal.rgb,
                    area_frac=area / canvas_area,
                    bbox=bbox,
                    mask=cc,
                    contour=contour,
                    contour_resampled=rs,
                    descriptor=desc,
                    sdf=mask_to_sdf(cc),
                    depth=0,
                    centroid=(cx, cy),
                    contour_area_rel_err=float(area_err),
                )
            )
            region_map[cc] = rid
            image_q[cc] = np.array(pal.rgb, dtype=np.uint8)
            rid += 1

    regions.sort(key=lambda r: -r.area_frac)
    if len(regions) > max_regions:
        keep = regions[:max_regions]
        keep_ids = {r.region_id for r in keep}
        for r in regions[max_regions:]:
            tiny.append((np.asarray(r.mask, dtype=bool), -1))
            region_map[np.asarray(r.mask, dtype=bool)] = -1
        regions = keep
        region_map = np.where(np.isin(region_map, list(keep_ids)), region_map, -1)

    # 过小 / 截断域并入邻接区域（无洞）
    for cc, _src in tiny:
        m = np.asarray(cc, dtype=bool)
        if not m.any():
            continue
        ys, xs = np.where(m)
        votes: dict[int, int] = {}
        for y, x in zip(ys, xs, strict=False):
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w:
                    t = int(region_map[ny, nx])
                    if t >= 0:
                        votes[t] = votes.get(t, 0) + 1
        if not votes and regions:
            target = regions[0].region_id
        elif votes:
            target = max(votes.items(), key=lambda kv: kv[1])[0]
        else:
            continue
        region_map[m] = target
        for r in regions:
            if r.region_id == target:
                new_m = np.asarray(r.mask, dtype=bool) | m
                r.mask = new_m
                image_q[m] = np.array(r.color_rgb, dtype=np.uint8)
                r.area_frac = float(new_m.sum()) / canvas_area
                break

    if enforce_no_gap:
        # 残余 gap：填到最近区域 id
        subject = alpha >= 0.5
        gap = subject & (region_map < 0)
        if gap.any() and regions:
            # 用标签众数回填后映射到 region
            filled = fill_label_gaps(work_labels, alpha)
            # 建立 label→最大区域
            lab_to_reg: dict[int, int] = {}
            for r in regions:
                labs, cnts = np.unique(filled[np.asarray(r.mask, dtype=bool)], return_counts=True)
                for lab, c in zip(labs.tolist(), cnts.tolist(), strict=False):
                    if int(lab) < 0:
                        continue
                    if int(lab) not in lab_to_reg or c > 0:
                        lab_to_reg[int(lab)] = r.region_id
            ys, xs = np.where(gap)
            for y, x in zip(ys, xs, strict=False):
                lab = int(filled[y, x])
                rid_t = lab_to_reg.get(lab, regions[0].region_id)
                region_map[y, x] = rid_t
                for r in regions:
                    if r.region_id == rid_t:
                        mm = np.asarray(r.mask, dtype=bool)
                        mm[y, x] = True
                        r.mask = mm
                        image_q[y, x] = np.array(r.color_rgb, dtype=np.uint8)
                        break

    # 邻接：水平/垂直边界（GPU）
    edge_len = adjacency_edge_lengths(region_map, device=dev)
    edges = [AdjacencyEdge(a=a, b=b, length=ln) for (a, b), ln in edge_len.items()]

    return RegionGraph(
        regions=regions,
        edges=edges,
        labels=region_map,
        image_rgb=image_q,
        alpha=alpha.astype(np.float64),
        canvas_size=h,
    )
