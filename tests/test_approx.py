"""approx 管线冒烟测试。"""

from __future__ import annotations

from pathlib import Path

from bf_emblem_creator.approx.blocks import abstract_to_blocks
from bf_emblem_creator.approx.models import AbstractionMode
from bf_emblem_creator.approx.recipe import default_recipe_for_mode
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
from bf_emblem_creator.stamps import StampLibrary

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ROOT / "assets" / "stamps"
EMOJI_SMILE = ROOT / "examples" / "smile.png"


def test_abstract_blocks_still_works() -> None:
    recipe = default_recipe_for_mode(AbstractionMode.illustration).override(stamps_dir=STAMPS, num_colors=4)
    target = abstract_to_blocks(EMOJI_SMILE, recipe)
    assert len(target.blocks) >= 1


def test_curve_lib_basic() -> None:
    lib = StampLibrary(STAMPS)
    curves = StampCurveLibrary.build(lib, ["Circle", "Square"], tex_size=32, force_refit=True)
    assert len(curves.entries) >= 1
