"""近似管线的 Pydantic 配置与结果模型。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bf_emblem_creator.models import EmblemDocument, HexColor

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]


class AbstractionMode(str, Enum):
    """图像概括模式。"""

    auto = "auto"
    logo = "logo"
    illustration = "illustration"
    photo_portrait = "photo_portrait"
    photo_general = "photo_general"
    silhouette = "silhouette"


class PaletteColor(BaseModel):
    """调色板中的一种代表色。"""

    model_config = ConfigDict(extra="forbid")

    hex: HexColor = Field(..., description="sRGB 十六进制色")
    fraction: float = Field(..., ge=0.0, le=1.0, description="在主体内的面积占比")
    rgb: tuple[int, int, int] = Field(..., description="RGB 元组")


class ApproxMeta(BaseModel):
    """概括阶段元数据。"""

    model_config = ConfigDict(extra="forbid")

    source_width: int = Field(..., description="原图宽")
    source_height: int = Field(..., description="原图高")
    canvas_size: int = Field(default=320, description="工作画布边长")
    fit: str = Field(..., description="contain 或 cover")
    mode: AbstractionMode = Field(..., description="实际使用的概括模式")
    scale: float = Field(default=1.0, description="缩放比例")
    offset_x: float = Field(default=0.0, description="水平偏移（画布坐标）")
    offset_y: float = Field(default=0.0, description="垂直偏移（画布坐标）")


class LayerHint(BaseModel):
    """给匹配追踪的施工顺序提示。"""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., description="bottom_color / silhouette / region / detail")
    color: HexColor | None = Field(default=None, description="相关颜色")
    area_fraction: float = Field(default=0.0, ge=0.0, description="面积占比")
    bbox: tuple[int, int, int, int] | None = Field(
        default=None,
        description="(x0,y0,x1,y1) 像素包围盒",
    )


class ApproxTarget(BaseModel):
    """
    概括后的拟合目标。

    数组字段允许任意类型以便承载 numpy；序列化时一般不直接 dump 大数组。
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    image_rgb: Any = Field(..., description="概括后 RGB uint8 (H,W,3)")
    alpha: Any = Field(..., description="主体蒙版 float (H,W) 0~1")
    weight: Any = Field(..., description="损失权重 float (H,W)")
    labels: Any = Field(..., description="色区标签 int (H,W)，-1 为背景")
    palette: list[PaletteColor] = Field(default_factory=list, description="调色板")
    layers_hint: list[LayerHint] = Field(default_factory=list, description="分层提示")
    meta: ApproxMeta = Field(..., description="元数据")

    def numpy_rgb(self) -> U8Arr:
        """返回 RGB 数组。"""
        return np.asarray(self.image_rgb, dtype=np.uint8)

    def numpy_alpha(self) -> FloatArr:
        """返回 alpha 数组。"""
        return np.asarray(self.alpha, dtype=np.float64)

    def numpy_weight(self) -> FloatArr:
        """返回权重数组。"""
        return np.asarray(self.weight, dtype=np.float64)

    def numpy_labels(self) -> NDArray[np.int32]:
        """返回标签数组。"""
        return np.asarray(self.labels, dtype=np.int32)


class ApproxConfig(BaseModel):
    """近似算法总配置。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # 画布 / 概括
    canvas_size: int = Field(default=320, gt=0, description="工作画布边长")
    mode: AbstractionMode = Field(default=AbstractionMode.auto, description="概括模式")
    palette_k: int = Field(default=6, ge=2, le=16, description="色量聚类数 K")
    min_region_area_frac: float = Field(
        default=0.002,
        ge=0.0,
        le=0.1,
        description="最小保留色区面积占比",
    )
    bilateral: bool = Field(default=True, description="是否做保边平滑")

    # 匹配追踪
    max_layers: int = Field(default=24, ge=1, le=40, description="最大层数")
    min_gain: float = Field(default=0.004, ge=0.0, description="接受新层的最小相对降损")
    stall_patience: int = Field(default=2, ge=1, description="连续无增益则停止的次数")
    loss_epsilon: float = Field(default=0.03, ge=0.0, description="损失足够低则停止")
    recall_k: int = Field(default=6, ge=1, le=32, description="每 ROI 召回图章数")
    max_rois_per_iter: int = Field(default=3, ge=1, description="每轮最多尝试 ROI 数")
    search_size: int = Field(default=96, ge=32, le=256, description="几何搜索分辨率")
    angle_step_deg: float = Field(default=30.0, gt=0.0, description="粗角度步长（度）")
    refine: bool = Field(default=True, description="是否对接受层做局部连续精修")
    refine_iters: int = Field(default=3, ge=0, description="精修迭代次数")
    add_bottom_fill: bool = Field(default=True, description="是否尝试底色打底层")
    stamp_subset: list[str] | None = Field(
        default=None,
        description="限定图章 id 列表；None 表示使用内置基础几何子集",
    )
    stamps_dir: Path = Field(default=Path("assets/stamps"), description="图章目录")
    render_supersample: float = Field(
        default=1.0,
        ge=1.0,
        le=4.0,
        description="拟合过程渲染超采样（1 最快）",
    )
    seed: int = Field(default=0, description="随机种子（色量初始化等）")

    # 相似度评判阈值（emoji 等）
    pass_score: float = Field(
        default=0.48,
        ge=0.0,
        le=1.0,
        description="综合相似度达标线（越高越严）",
    )

    @field_validator("stamps_dir", mode="before")
    @classmethod
    def _path(cls, value: str | Path) -> Path:
        return Path(value)


class ApproxResult(BaseModel):
    """一次近似的完整结果。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    document: EmblemDocument = Field(..., description="输出图章文档")
    target: ApproxTarget = Field(..., description="概括目标")
    final_loss: float = Field(..., description="最终复合损失")
    per_layer_gains: list[float] = Field(default_factory=list, description="每层降损")
    similarity: Any = Field(default=None, description="SimilarityReport，可选")
    preview_rgb: Any = Field(default=None, description="渲染预览 RGB uint8，可选")
