"""战地图章工具包：离线渲染与自动近似摆放。"""

from bf_emblem_creator.approx import (
    ApproxConfig,
    ApproxResultV2,
    BlockTarget,
    FullScoreReport,
    abstract_to_blocks,
    approximate_image,
    evaluate_line_quality,
    score_prediction,
)
from bf_emblem_creator.models import (
    CanvasConfig,
    EmblemDocument,
    HexColor,
    RenderConfig,
    StampLayer,
)
from bf_emblem_creator.render import EmblemRenderer

__all__ = [
    "ApproxConfig",
    "ApproxResultV2",
    "BlockTarget",
    "CanvasConfig",
    "EmblemDocument",
    "EmblemRenderer",
    "FullScoreReport",
    "HexColor",
    "RenderConfig",
    "StampLayer",
    "abstract_to_blocks",
    "approximate_image",
    "evaluate_line_quality",
    "score_prediction",
]

__version__ = "0.2.0"
