"""图像概括与图章近似（ModeRecipe + 五大处理器，无兼容层）。"""

from bf_emblem_creator.approx.blocks import BlockTarget, ColorBlock, abstract_to_blocks
from bf_emblem_creator.approx.engine import ApproxEngine
from bf_emblem_creator.approx.line_quality import LineQualityReport, evaluate_line_quality
from bf_emblem_creator.approx.metrics import FullScoreReport, score_prediction
from bf_emblem_creator.approx.models import AbstractionMode
from bf_emblem_creator.approx.pipeline import ApproxResult, approximate_image, approximate_to_files
from bf_emblem_creator.approx.recipe import ModeRecipe, default_recipe_for_mode
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary

__all__ = [
    "AbstractionMode",
    "ApproxEngine",
    "ApproxResult",
    "BlockTarget",
    "ColorBlock",
    "FullScoreReport",
    "LineQualityReport",
    "ModeRecipe",
    "StampCurveLibrary",
    "abstract_to_blocks",
    "approximate_image",
    "approximate_to_files",
    "default_recipe_for_mode",
    "evaluate_line_quality",
    "score_prediction",
]
