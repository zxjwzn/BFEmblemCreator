"""P3：图层顺序假设 π — 将轮廓解释为形状可见边界（GPU 评分）。"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.gpu_ops import depth_order_scores_torch, to_torch
from bf_emblem_creator.approx.regions import Region, RegionGraph

if TYPE_CHECKING:
    from bf_emblem_creator.approx.planar_map import PlanarMap


class EdgeRole(str, Enum):
    """轮廓边角色。"""

    shape_boundary = "SHAPE_BOUNDARY"
    occlusion_cut = "OCCLUSION_CUT"


class OrderedRegion(BaseModel):
    """带深度与角色的区域视图。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    region: Region
    depth: int = Field(..., description="0=最底")
    boundary_role_default: EdgeRole = EdgeRole.shape_boundary


class DepthOrderResult(BaseModel):
    """层序结果。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    ordered: list[OrderedRegion]
    """底→顶。"""
    edge_roles: dict[tuple[int, int], EdgeRole] = Field(default_factory=dict)
    """邻接边 (min_id,max_id) → 角色。"""
    energy: float = 0.0


def _is_enclosed_torch(
    inner_centroid: tuple[float, float], inner_area: float, outer_mask: torch.Tensor, outer_area: float
) -> bool:
    """质心是否在 outer mask 内且面积更小。"""
    cx, cy = inner_centroid
    x, y = round(cx), round(cy)
    h, w = int(outer_mask.shape[0]), int(outer_mask.shape[1])
    if not (0 <= y < h and 0 <= x < w):
        return False
    if float(outer_mask[y, x].item()) <= 0.5:
        return False
    return inner_area < outer_area * 0.95


def infer_depth_order(graph: RegionGraph, planar_map: PlanarMap | None = None) -> DepthOrderResult:
    """
    通用层序启发式（GPU 批量打分）：

    1. 面积大、邻接多、更居中 → 更靠下；
    2. 被其它色包围 → 更靠上；
    3. 共享边界上完整侧更深。
    若提供 PlanarMap，将角色写回 SharedEdge.role。
    """
    if not graph.regions:
        return DepthOrderResult(ordered=[], edge_roles={}, energy=0.0)

    dev = get_device()
    canvas = graph.canvas_size
    regs = list(graph.regions)
    r = len(regs)
    areas = torch.tensor([reg.area_frac for reg in regs], device=dev, dtype=torch.float32)
    cents = torch.tensor([reg.centroid for reg in regs], device=dev, dtype=torch.float32)
    # masks stack
    h = int(np.asarray(regs[0].mask).shape[0])
    w = int(np.asarray(regs[0].mask).shape[1])
    masks = torch.zeros((r, h, w), device=dev, dtype=torch.float32)
    for i, reg in enumerate(regs):
        masks[i] = to_torch(np.asarray(reg.mask, dtype=np.float32), device=dev)

    deg_map = {reg.region_id: 0.0 for reg in regs}
    for e in graph.edges:
        deg_map[e.a] = deg_map.get(e.a, 0.0) + e.length
        deg_map[e.b] = deg_map.get(e.b, 0.0) + e.length
    degrees = torch.tensor([deg_map[reg.region_id] for reg in regs], device=dev, dtype=torch.float32)

    scores = depth_order_scores_torch(areas, degrees, cents, masks, canvas)
    order_idx = torch.argsort(scores, descending=True).tolist()
    ordered_regs = [regs[i] for i in order_idx]
    depth_of = {reg.region_id: d for d, reg in enumerate(ordered_regs)}
    for reg in ordered_regs:
        reg.depth = depth_of[reg.region_id]

    by_id = graph.by_id()
    edge_roles: dict[tuple[int, int], EdgeRole] = {}
    energy = 0.0
    mask_by_id = {regs[i].region_id: masks[i] for i in range(r)}
    area_by_id = {regs[i].region_id: float(areas[i].item()) for i in range(r)}

    for e in graph.edges:
        da, db = depth_of.get(e.a, 0), depth_of.get(e.b, 0)
        key = (min(e.a, e.b), max(e.a, e.b))
        edge_roles[key] = EdgeRole.shape_boundary
        energy += 0.1 * e.length * abs(da - db) * 0.01
        deep_id = e.a if da < db else e.b
        shallow_id = e.b if da < db else e.a
        deep = by_id.get(deep_id)
        shallow = by_id.get(shallow_id)
        if deep is not None and shallow is not None:
            om = mask_by_id[deep_id]
            if _is_enclosed_torch(shallow.centroid, area_by_id[shallow_id], om, area_by_id[deep_id]):
                edge_roles[key] = EdgeRole.occlusion_cut
                energy += 0.05 * e.length

    # 写回 PlanarMap 共享边角色
    if planar_map is not None:
        for se in planar_map.edges:
            fa = int(se.left_face)
            fb = int(se.right_face)
            if fa < 0 or fb < 0:
                se.role = EdgeRole.shape_boundary
                continue
            key = (min(fa, fb), max(fa, fb))
            se.role = edge_roles.get(key, EdgeRole.shape_boundary)

    ordered = [
        OrderedRegion(region=reg, depth=depth_of[reg.region_id], boundary_role_default=EdgeRole.shape_boundary)
        for reg in ordered_regs
    ]
    return DepthOrderResult(ordered=ordered, edge_roles=edge_roles, energy=float(energy))
