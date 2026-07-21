"""图章渲染器：封装 TorchStampRenderer。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.processors.stamp_loader import StampCatalog
from bf_emblem_creator.approx.recipe import StampRendererConfig
from bf_emblem_creator.approx.torch_render import TorchStampRenderer, stamp_layer_to_dict
from bf_emblem_creator.models import StampLayer

U8Arr = NDArray[np.uint8]


class StampRenderer:
    """图章渲染器。"""

    def __init__(self, config: StampRendererConfig, catalog: StampCatalog) -> None:
        self.config = config
        device = get_device(prefer_cuda=config.use_cuda)
        self._impl = TorchStampRenderer(
            catalog.get_stamp_lib(),
            canvas_size=config.canvas_size,
            stamp_tex_size=config.stamp_tex_size,
            device=device,
        )
        self.device = device

    @property
    def torch_renderer(self) -> TorchStampRenderer:
        """底层渲染器（兼容现有 assemble/union_cover）。"""
        return self._impl

    def composite_layers(self, layers: list[StampLayer]) -> tuple[Any, Any]:
        """合成层 → (rgb_t, a_t) torch。"""
        return self._impl.composite_layers([stamp_layer_to_dict(layer) for layer in layers])

    def to_numpy_rgba(self, rgb_t: Any, a_t: Any) -> U8Arr:
        """torch → RGBA uint8。"""
        return self._impl.to_numpy_rgba(rgb_t, a_t)
