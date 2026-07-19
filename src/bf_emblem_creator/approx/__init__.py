"""图像概括与图章匹配追踪（Matching Pursuit）近似模块。"""

from bf_emblem_creator.approx.metrics import SimilarityReport, compare_images, score_fit
from bf_emblem_creator.approx.models import (
    ApproxConfig,
    ApproxResult,
    ApproxTarget,
    PaletteColor,
)
from bf_emblem_creator.approx.pipeline import approximate_image
from bf_emblem_creator.approx.preprocess import abstract_image

__all__ = [
    "ApproxConfig",
    "ApproxResult",
    "ApproxTarget",
    "PaletteColor",
    "SimilarityReport",
    "abstract_image",
    "approximate_image",
    "compare_images",
    "score_fit",
]
