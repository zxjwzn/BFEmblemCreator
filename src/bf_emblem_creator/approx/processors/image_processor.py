"""图像处理器：ImageProcessorConfig 驱动平面化。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.models import AbstractionMode, ApproxMeta, PaletteColor
from bf_emblem_creator.approx.planarize import planarize_image
from bf_emblem_creator.approx.recipe import ImageProcessorConfig

U8Arr = NDArray[np.uint8]


class ProcessedImage(BaseModel):
    """图像处理器输出。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    src_rgb: Any = Field(..., description="平滑后、量化前 RGB uint8 (H,W,3)")
    image_q: Any = Field(..., description="量化后 RGB uint8 (H,W,3)")
    alpha: Any = Field(..., description="主体 alpha float (H,W)")
    labels: Any = Field(..., description="标签 int32 (H,W)，-1 背景")
    palette: list[PaletteColor] = Field(default_factory=list, description="调色板")
    meta: ApproxMeta = Field(..., description="平面化元数据")


class ImageProcessor:
    """图像处理器。"""

    def __init__(self, config: ImageProcessorConfig, *, mode: AbstractionMode) -> None:
        self.config = config
        self.mode = mode

    def process(self, image: Image.Image | str | Path | U8Arr) -> ProcessedImage:
        """执行平面化色量。"""
        labels, palette, alpha, image_q, meta, src_rgb = planarize_image(
            image,
            self.config,
            mode=self.mode,
        )
        return ProcessedImage(
            src_rgb=src_rgb,
            image_q=image_q,
            alpha=alpha,
            labels=labels,
            palette=palette,
            meta=meta,
        )
