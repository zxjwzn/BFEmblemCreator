"""战地图章工具包：离线渲染与自动近似摆放。"""

from bf_emblem_creator.approx import (
    ApproxConfig,
    ApproxResult,
    ApproxTarget,
    SimilarityReport,
    abstract_image,
    approximate_image,
    compare_images,
    score_fit,
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
    "ApproxResult",
    "ApproxTarget",
    "CanvasConfig",
    "EmblemDocument",
    "EmblemRenderer",
    "HexColor",
    "RenderConfig",
    "SimilarityReport",
    "StampLayer",
    "abstract_image",
    "approximate_image",
    "compare_images",
    "score_fit",
]

__version__ = "0.1.0"
