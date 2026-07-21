"""区域与邻接图数据模型（由 PlanarMap 派生）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.gpu_ops import label_connected_components

BoolArr = NDArray[np.bool_]


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


def _label_ccs(binary: BoolArr, *, device: torch.device | None = None) -> list[BoolArr]:
    """四连通域列表（GPU）。"""
    return label_connected_components(binary, device=device)
