"""近似管线核心数据模型（无扁平巨石配置）。"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.models import HexColor

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]


class AbstractionMode(str, Enum):
    """图像概括模式（须显式声明；默认 illustration）。"""

    logo = "logo"
    illustration = "illustration"
    photo_portrait = "photo_portrait"
    photo_general = "photo_general"
    silhouette = "silhouette"
    pixel = "pixel"


class ResampleMode(str, Enum):
    """画布对齐重采样模式。"""

    auto = "auto"
    nearest = "nearest"
    lanczos = "lanczos"
    bilinear = "bilinear"


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
    resample: str = Field(default="lanczos", description="实际重采样：nearest/lanczos/bilinear")
    approx_color_count: int = Field(default=0, ge=0, description="主体近似独特色数（检测用）")
    gap_frac: float = Field(default=0.0, ge=0.0, description="主体内标签空洞占比（目标 0）")
    noise_frac: float = Field(default=0.0, ge=0.0, description="过小 Face 像素占比")
    seam_p95: float = Field(default=0.0, ge=0.0, description="共享可见边缝宽 95 分位（像素）")
    num_colors: int = Field(default=0, ge=0, description="平面化实际使用色数（k-means K）")


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
    """概括后的拟合目标（数组字段可持有 numpy）。"""

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
