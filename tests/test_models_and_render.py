from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from bf_emblem_creator.models import CanvasConfig, EmblemDocument, RenderConfig, StampLayer
from bf_emblem_creator.render import EmblemRenderer

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ROOT / "assets" / "stamps"
SIMPLE = ROOT / "examples" / "simple_layers.json"
SAMPLE = ROOT / "examples" / "sample_emblem.json"


@pytest.fixture
def renderer() -> EmblemRenderer:
    return EmblemRenderer(RenderConfig(stamps_dir=STAMPS, supersample=4.0))


def test_load_simple_json() -> None:
    doc = EmblemDocument.load_json(SIMPLE)
    assert len(doc) == 3
    assert doc[0].asset == "Square"
    assert doc[2].fill == "#F7D5B9"


def test_load_editor_export_with_selectable() -> None:
    doc = EmblemDocument.load_json(SAMPLE)
    assert len(doc) >= 1
    assert hasattr(doc[0], "selectable")


def test_roundtrip_json(tmp_path: Path) -> None:
    doc = EmblemDocument.load_json(SIMPLE)
    out = tmp_path / "out.json"
    doc.save_json(out)
    doc2 = EmblemDocument.load_json(out)
    assert doc.model_dump() == doc2.model_dump()


def test_render_simple(renderer: EmblemRenderer, tmp_path: Path) -> None:
    doc = EmblemDocument.load_json(SIMPLE)
    img = renderer.render(doc)
    assert img.size == (320, 320)
    assert img.mode == "RGBA"
    alpha = np.asarray(img)[:, :, 3]
    assert int(alpha.max()) > 0
    path = tmp_path / "sample.png"
    renderer.render_to_path(doc, path)
    assert path.is_file()
    loaded = Image.open(path)
    assert loaded.size == (320, 320)


def test_layer_rejects_bad_color() -> None:
    with pytest.raises(ValidationError):
        StampLayer(
            asset="Square",
            top=0,
            left=0,
            height=10,
            width=10,
            fill="not-a-color",
        )


def test_rotated_edges_stay_fill_colored(renderer: EmblemRenderer) -> None:
    """半透明边缘应接近 fill，不得漂成白边。"""
    doc = EmblemDocument.from_layers(
        [
            StampLayer(
                asset="Circle",
                top=160,
                left=160,
                height=180,
                width=180,
                fill="#3A9FFA",
                angle=33.0,
            )
        ]
    )
    img = renderer.render(doc)
    arr = np.asarray(img)
    alpha = arr[:, :, 3]
    # 只检查有一定覆盖的抗锯齿环，极低 alpha 允许量化误差
    semi = (alpha >= 16) & (alpha < 255)
    assert semi.any()
    rgb = arr[semi][:, :3].astype(np.int16)
    fill = np.array([0x3A, 0x9F, 0xFA], dtype=np.int16)
    # SSAA 浮点降采样后应非常接近 fill
    assert np.all(np.abs(rgb - fill).max(axis=1) <= 3)
    assert not np.any(rgb.min(axis=1) > 200)


def _max_luma_gradient(image: Image.Image) -> float:
    rgb = np.asarray(image).astype(np.float32)[:, :, :3]
    luma = rgb.mean(axis=2)
    gx = float(np.abs(np.diff(luma, axis=1)).max())
    gy = float(np.abs(np.diff(luma, axis=0)).max())
    return max(gx, gy)


def test_supersample_softens_diagonal_edges() -> None:
    """更高超采样应降低斜边的最大亮度跳变（锯齿更软）。"""
    doc = EmblemDocument.from_layers(
        [
            StampLayer(
                asset="Square",
                top=160,
                left=160,
                height=140,
                width=140,
                fill="#FF0000",
                angle=30.0,
            )
        ]
    )
    canvas = CanvasConfig(width=320, height=320, background="#000000")
    low = EmblemRenderer(
        RenderConfig(stamps_dir=STAMPS, canvas=canvas, supersample=1.0)
    ).render(doc)
    high = EmblemRenderer(
        RenderConfig(stamps_dir=STAMPS, canvas=canvas, supersample=4.0)
    ).render(doc)
    low_g = _max_luma_gradient(low)
    high_g = _max_luma_gradient(high)
    assert high_g < low_g
