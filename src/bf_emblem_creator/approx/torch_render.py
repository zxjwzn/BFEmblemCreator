"""GPU/CPU 批量图章栅格化（torch grid_sample）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image

from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.raster import rasterize_svg
from bf_emblem_creator.stamps import StampLibrary

U8Arr = NDArray[np.uint8]


class TorchStampRenderer:
    """
    内环批量渲染器：将多枚图章以仿射粒子画到画布上。

    坐标系：画布像素，原点左上；层参数与编辑器一致（中心 left/top，顺时针角，宽高）。
    """

    def __init__(
        self,
        library: StampLibrary,
        *,
        canvas_size: int = 320,
        stamp_tex_size: int = 256,
        device: torch.device | None = None,
    ) -> None:
        self.library = library
        self.canvas_size = canvas_size
        self.stamp_tex_size = stamp_tex_size
        self.device = device or get_device()
        self._tex_cache: dict[str, torch.Tensor] = {}

    def get_stamp_tex(self, asset_id: str) -> torch.Tensor:
        """返回 (1,1,H,W) float mask 纹理，值域 0~1。"""
        if asset_id in self._tex_cache:
            return self._tex_cache[asset_id]
        info = self.library.resolve(asset_id)
        rgba = rasterize_svg(
            info.path,
            out_width=self.stamp_tex_size,
            out_height=self.stamp_tex_size,
        )
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        tex = torch.from_numpy(alpha)[None, None, ...].to(self.device)
        self._tex_cache[asset_id] = tex
        return tex

    def render_batch_masks(
        self,
        asset_id: str,
        *,
        left: torch.Tensor,
        top: torch.Tensor,
        width: torch.Tensor,
        height: torch.Tensor,
        angle_deg: torch.Tensor,
        flip_x: torch.Tensor | None = None,
        canvas_size: int | None = None,
    ) -> torch.Tensor:
        """
        批量渲染 alpha mask。

        参数均为 (N,)；返回 (N,1,H,W)。
        angle_deg：顺时针角度（与编辑器一致）。
        """
        n = int(left.shape[0])
        cs = canvas_size or self.canvas_size
        tex = self.get_stamp_tex(asset_id).expand(n, -1, -1, -1)
        if flip_x is not None:
            # flip_x：水平镜像纹理
            flip = flip_x.bool()
            if flip.any():
                tex = tex.clone()
                tex[flip] = torch.flip(tex[flip], dims=[-1])

        # 构造从输出像素到 stamp 本地 [-1,1] 的 grid
        # 输出坐标 (xs,ys) 像素中心
        ys = torch.linspace(0.5, cs - 0.5, cs, device=self.device)
        xs = torch.linspace(0.5, cs - 0.5, cs, device=self.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # (H,W)
        # 平移到以章中心为原点
        # 顺时针角 θ → 旋转矩阵 [[c,s],[-s,c]] 作用于向量（y 向下）
        theta = torch.deg2rad(angle_deg)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        # (N,H,W)
        dx = grid_x[None, ...] - left[:, None, None]
        dy = grid_y[None, ...] - top[:, None, None]
        # 逆旋转（逆时针 -θ 的反向：用顺时针逆 = 逆时针 θ）
        # 点在图章局部：R(-θ_cw) = R(+θ_ccw) …
        # 编辑器：点绕中心顺时针转 θ 后绘制；等价局部坐标 = R_ccw(θ) * (p-c) 再轴对齐缩放
        lx = cos_t[:, None, None] * dx + sin_t[:, None, None] * dy
        ly = -sin_t[:, None, None] * dx + cos_t[:, None, None] * dy
        # 归一到 [-1,1] 对应 stamp 宽高
        gx = lx / (width[:, None, None] * 0.5 + 1e-6)
        gy = ly / (height[:, None, None] * 0.5 + 1e-6)
        grid = torch.stack([gx, gy], dim=-1)  # N,H,W,2
        masks = F.grid_sample(
            tex,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return masks.clamp(0.0, 1.0)

    def composite_layers(
        self,
        layers: list[dict[str, object]],
        *,
        canvas_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        按从底到顶顺序合成。

        layers 元素：asset,left,top,width,height,angle,fill_rgb(tuple),flip_x
        返回 rgb (3,H,W)、alpha (H,W)，值域 0~1。
        """
        cs = canvas_size or self.canvas_size
        rgb = torch.zeros(3, cs, cs, device=self.device)
        alpha = torch.zeros(cs, cs, device=self.device)
        for layer in layers:
            asset = str(layer["asset"])
            left = torch.tensor([float(layer["left"])], device=self.device)  # type: ignore[arg-type]
            top = torch.tensor([float(layer["top"])], device=self.device)  # type: ignore[arg-type]
            width = torch.tensor([float(layer["width"])], device=self.device)  # type: ignore[arg-type]
            height = torch.tensor([float(layer["height"])], device=self.device)  # type: ignore[arg-type]
            angle = torch.tensor([float(layer["angle"])], device=self.device)  # type: ignore[arg-type]
            flip = torch.tensor([bool(layer.get("flip_x", False))], device=self.device)
            m = self.render_batch_masks(
                asset,
                left=left,
                top=top,
                width=width,
                height=height,
                angle_deg=angle,
                flip_x=flip,
                canvas_size=cs,
            )[0, 0]
            fill = layer["fill_rgb"]
            assert isinstance(fill, tuple)
            fr, fg, fb = (float(fill[0]), float(fill[1]), float(fill[2]))  # type: ignore[index]
            color = torch.tensor(
                [fr / 255.0, fg / 255.0, fb / 255.0],
                device=self.device,
            )
            # over: out = src*a + dst*(1-a)
            a = m
            rgb = color[:, None, None] * a[None, ...] + rgb * (1.0 - a[None, ...])
            alpha = a + alpha * (1.0 - a)
        return rgb.clamp(0, 1), alpha.clamp(0, 1)

    def to_numpy_rgba(self, rgb: torch.Tensor, alpha: torch.Tensor) -> U8Arr:
        """torch rgb/alpha → uint8 RGBA。"""
        r = (rgb.detach().float().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        a = (alpha.detach().float().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        return np.dstack([r.transpose(1, 2, 0), a])

    def preview_pil(self, layers: list[dict[str, object]]) -> Image.Image:
        """合成并返回 PIL 图。"""
        rgb, a = self.composite_layers(layers)
        arr = self.to_numpy_rgba(rgb, a)
        return Image.fromarray(arr, mode="RGBA")


def stamp_layer_to_dict(layer: object) -> dict[str, object]:
    """StampLayer → 批量渲染字典。"""
    from bf_emblem_creator.models import StampLayer

    assert isinstance(layer, StampLayer)
    r, g, b = layer.fill_rgb()
    return {
        "asset": layer.asset,
        "left": layer.left,
        "top": layer.top,
        "width": layer.width,
        "height": layer.height,
        "angle": layer.angle,
        "flip_x": layer.flipX,
        "fill_rgb": (r, g, b),
    }


def ensure_stamps_dir(path: str | Path) -> Path:
    return Path(path)
