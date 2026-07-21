"""ModeRecipe 与五大处理器（无 ApproxConfig 兼容）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bf_emblem_creator.approx.engine import ApproxEngine
from bf_emblem_creator.approx.models import AbstractionMode
from bf_emblem_creator.approx.pipeline import approximate_image
from bf_emblem_creator.approx.processors.image_processor import ImageProcessor
from bf_emblem_creator.approx.processors.region_partitioner import RegionPartitioner
from bf_emblem_creator.approx.recipe import (
    AngleMode,
    BoundaryPolicy,
    ImageProcessorConfig,
    RegionPartitionerConfig,
    default_recipe_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ROOT / "assets" / "stamps"
EMOJI = ROOT / "examples" / "smile.png"


def test_pixel_recipe_constraints() -> None:
    r = default_recipe_for_mode(AbstractionMode.pixel)
    assert r.stamp_loader.asset_allowlist == ["Square"]
    assert r.region.boundary_policy == BoundaryPolicy.dense
    assert r.match.angle.mode == AngleMode.discrete
    assert r.match.angle.angles_deg == [0.0, 90.0]
    assert r.match.enable_special_fx is False


def test_illustration_recipe_curve_fit() -> None:
    r = default_recipe_for_mode(AbstractionMode.illustration)
    assert r.region.boundary_policy == BoundaryPolicy.curve_fit
    assert r.match.angle.mode == AngleMode.free


def test_recipe_override() -> None:
    r = default_recipe_for_mode(AbstractionMode.logo).override(
        stamps_dir=STAMPS,
        num_colors=8,
        max_layers=20,
    )
    assert r.image.num_colors == 8
    assert r.match.max_layers == 20
    assert r.stamp_loader.stamps_dir == STAMPS


def test_image_processor_two_color() -> None:
    size = 32
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, : size // 2] = (255, 0, 0, 255)
    arr[:, size // 2 :] = (0, 0, 255, 255)
    proc = ImageProcessor(
        ImageProcessorConfig(canvas_size=64, num_colors=2, mrf_iters=2, bilateral=False, use_cuda=False),
        mode=AbstractionMode.illustration,
    )
    out = proc.process(arr)
    assert len(out.palette) >= 1
    assert np.asarray(out.labels).shape == (64, 64)


def test_region_partitioner_runs() -> None:
    size = 40
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, : size // 2] = (255, 0, 0, 255)
    arr[:, size // 2 :] = (0, 0, 255, 255)
    img = ImageProcessor(
        ImageProcessorConfig(canvas_size=64, num_colors=2, mrf_iters=2, bilateral=False, use_cuda=False),
        mode=AbstractionMode.illustration,
    ).process(arr)
    part = RegionPartitioner(RegionPartitionerConfig(max_faces=8, min_area_frac=0.01)).partition(img)
    assert len(part.region_graph.regions) >= 1


@pytest.mark.timeout(240)
def test_engine_smoke_emoji() -> None:
    if not EMOJI.is_file():
        pytest.skip("缺少 smile.png")
    recipe = default_recipe_for_mode(AbstractionMode.illustration).override(
        stamps_dir=STAMPS,
        num_colors=4,
        max_layers=12,
        max_faces=12,
        asset_allowlist=["Circle", "Square", "OpenCircle", "Triangle", "HalfCircle", "Line"],
        enable_special_fx=False,
        refine=False,
        n_particles=48,
        use_cuda=False,
        seed=0,
    )
    result = ApproxEngine(recipe).run(EMOJI, n_particles=48)
    assert result.k_used == 4
    assert result.mode == "illustration"
    assert result.stop_reason


@pytest.mark.timeout(240)
def test_approximate_image_pixel_square_only() -> None:
    if not EMOJI.is_file():
        pytest.skip("缺少 smile.png")
    recipe = default_recipe_for_mode(AbstractionMode.pixel).override(
        stamps_dir=STAMPS,
        num_colors=4,
        max_layers=16,
        max_faces=12,
        refine=False,
        n_particles=48,
        use_cuda=False,
        seed=0,
    )
    result = approximate_image(EMOJI, recipe, n_particles=48)
    for layer in result.document:
        assert layer.asset == "Square"
        ang = float(layer.angle) % 360.0
        assert abs(ang - 0.0) < 1e-3 or abs(ang - 90.0) < 1e-3
