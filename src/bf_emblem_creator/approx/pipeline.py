"""对外结果与对外入口（委托 ApproxEngine）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.blocks import BlockTarget
from bf_emblem_creator.approx.metrics import FullScoreReport
from bf_emblem_creator.approx.recipe import ModeRecipe
from bf_emblem_creator.models import EmblemDocument

U8Arr = NDArray[np.uint8]


class ApproxResult(BaseModel):
    """一次近似的完整结果。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    document: EmblemDocument = Field(..., description="输出图章文档")
    target: BlockTarget = Field(..., description="拟合目标色块")
    score: FullScoreReport = Field(..., description="综合评分")
    preview_rgb: Any = Field(default=None, description="预览 RGBA uint8 (H,W,4)")
    device: str = Field(default="cpu", description="计算设备")
    elapsed_sec: float = Field(default=0.0, ge=0.0, description="耗时秒")
    stop_reason: str = Field(default="", description="停止原因")
    blocks_found: int = Field(default=0, ge=0, description="区域/色块数")
    k_used: int = Field(default=0, ge=0, description="色量 K")
    boundary_score: float = Field(default=0.0, description="边界一致性")
    special_fx_assets: list[str] = Field(default_factory=list, description="特效章")
    log_lines: list[str] = Field(default_factory=list, description="日志")
    debug_images: list[str] = Field(default_factory=list, description="调试图路径")
    mode: str = Field(default="", description="实际使用的 AbstractionMode")


def approximate_image(
    image: Image.Image | str | Path | U8Arr,
    recipe: ModeRecipe | None = None,
    *,
    n_particles: int | None = None,
) -> ApproxResult:
    """
    图像近似：唯一入口，委托 ApproxEngine。

    recipe 为完整 ModeRecipe；None 时用 illustration 默认配方。
    """
    from bf_emblem_creator.approx.engine import ApproxEngine
    from bf_emblem_creator.approx.models import AbstractionMode
    from bf_emblem_creator.approx.recipe import default_recipe_for_mode

    r = recipe if recipe is not None else default_recipe_for_mode(AbstractionMode.illustration)
    return ApproxEngine(r).run(image, n_particles=n_particles)


def approximate_to_files(
    image: Image.Image | str | Path,
    out_json: str | Path,
    out_preview: str | Path | None = None,
    recipe: ModeRecipe | None = None,
) -> ApproxResult:
    """近似并写 JSON / 可选预览 PNG。"""
    result = approximate_image(image, recipe)
    result.document.save_json(out_json)
    if out_preview is not None and result.preview_rgb is not None:
        Path(out_preview).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.preview_rgb, mode="RGBA").save(out_preview)
    return result
