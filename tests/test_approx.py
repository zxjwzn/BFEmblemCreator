"""approx 兼容测试（v3 路径）。"""

from __future__ import annotations

from pathlib import Path

from bf_emblem_creator.approx.blocks import abstract_to_blocks
from bf_emblem_creator.approx.models import ApproxConfig
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
from bf_emblem_creator.stamps import StampLibrary

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ROOT / "assets" / "stamps"
EMOJI_SMILE = ROOT / "examples" / "😄.png"


def test_abstract_blocks_still_works() -> None:
    target = abstract_to_blocks(EMOJI_SMILE, ApproxConfig(stamps_dir=STAMPS, palette_k=4))
    assert len(target.blocks) >= 1


def test_curve_lib_basic() -> None:
    lib = StampLibrary(STAMPS)
    curves = StampCurveLibrary.build(lib, ["Circle", "Square"], tex_size=32, force_refit=True)
    assert len(curves.entries) >= 1
