"""战地图章工具包：离线渲染与自动近似摆放。"""

from bf_emblem_creator.approx import (
    AbstractionMode,
    ApproxEngine,
    ApproxResult,
    BlockTarget,
    FullScoreReport,
    ModeRecipe,
    abstract_to_blocks,
    approximate_image,
    default_recipe_for_mode,
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
    "AbstractionMode",
    "ApproxEngine",
    "ApproxResult",
    "BlockTarget",
    "CanvasConfig",
    "EmblemDocument",
    "EmblemRenderer",
    "FullScoreReport",
    "HexColor",
    "ModeRecipe",
    "RenderConfig",
    "StampLayer",
    "abstract_to_blocks",
    "approximate_image",
    "default_recipe_for_mode",
    "evaluate_line_quality",
    "score_prediction",
]

__version__ = "0.3.0"
