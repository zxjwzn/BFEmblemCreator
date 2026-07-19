"""图像概括与图章近似（v3：可见边界曲线拟合 + GPU）。"""

from bf_emblem_creator.approx.blocks import BlockTarget, ColorBlock, abstract_to_blocks
from bf_emblem_creator.approx.line_quality import LineQualityReport, evaluate_line_quality
from bf_emblem_creator.approx.metrics import FullScoreReport, score_prediction
from bf_emblem_creator.approx.models import ApproxConfig
from bf_emblem_creator.approx.pipeline import ApproxResultV2, approximate_image, approximate_to_files
from bf_emblem_creator.approx.preprocess import abstract_image
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary

__all__ = [
    "ApproxConfig",
    "ApproxResultV2",
    "BlockTarget",
    "ColorBlock",
    "FullScoreReport",
    "LineQualityReport",
    "StampCurveLibrary",
    "abstract_image",
    "abstract_to_blocks",
    "approximate_image",
    "approximate_to_files",
    "evaluate_line_quality",
    "score_prediction",
]
