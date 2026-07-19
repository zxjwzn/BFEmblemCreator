"""将图章图层合成为徽章图像。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from bf_emblem_creator.models import (
    CanvasConfig,
    EmblemDocument,
    HexColor,
    RenderConfig,
    StampLayer,
    hex_to_rgb,
)
from bf_emblem_creator.raster import rasterize_svg
from bf_emblem_creator.stamps import StampLibrary

Rgba = NDArray[np.uint8]


def _px(value: float) -> int:
    """将浮点长度/尺寸四舍五入为至少为 1 的像素整数。"""
    return max(1, round(value))


def _pos(value: float) -> int:
    """将浮点坐标四舍五入为像素整数（可为 0 或负，用于贴边合成）。"""
    return round(value)


class EmblemRenderer:
    """徽章编辑器合成器的离线数字孪生。"""

    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig()
        self.library = StampLibrary(self.config.stamps_dir)

    def render(self, document: EmblemDocument) -> Image.Image:
        """
        按从底到顶顺序合成全部图层，返回目标分辨率 RGBA 图像。

        内部在 supersample 倍率画布上合成，最后用高质量滤波降采样，
        以减轻 320×320 等低分辨率下的几何锯齿。
        """
        scale = float(self.config.supersample)
        hi_canvas = self._scaled_canvas(self.config.canvas, scale)
        canvas = self._blank_canvas(hi_canvas)
        for layer in document:
            sprite = self._render_layer(layer, scale=scale)
            self._alpha_over(canvas, sprite, layer, scale=scale)
        return self._downsample(canvas, self.config.canvas.width, self.config.canvas.height)

    def render_to_path(
        self,
        document: EmblemDocument,
        output: str | Path,
        *,
        image_format: str | None = None,
    ) -> Path:
        """渲染并保存到文件；返回输出路径。"""
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = self.render(document)
        save_kwargs: dict[str, Any] = {}
        fmt = (image_format or path.suffix.lstrip(".") or "PNG").upper()
        if fmt == "JPG":
            fmt = "JPEG"
        if fmt == "JPEG" and image.mode == "RGBA":
            # JPEG 无 alpha，铺白底后导出。
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.getchannel("A"))
            image = bg
        image.save(path, format=fmt, **save_kwargs)
        return path

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    @staticmethod
    def _scaled_canvas(canvas: CanvasConfig, scale: float) -> CanvasConfig:
        """返回按超采样倍率放大的画布配置（背景色不变）。"""
        if scale == 1.0:
            return canvas
        return CanvasConfig(
            width=_px(canvas.width * scale),
            height=_px(canvas.height * scale),
            background=canvas.background,
        )

    def _blank_canvas(self, canvas: CanvasConfig) -> Image.Image:
        """创建空白画布（透明或纯色底）。"""
        if canvas.background is None:
            return Image.new("RGBA", (canvas.width, canvas.height), (0, 0, 0, 0))
        r, g, b = hex_to_rgb(canvas.background)
        return Image.new("RGBA", (canvas.width, canvas.height), (r, g, b, 255))

    def _render_layer(self, layer: StampLayer, *, scale: float) -> Image.Image:
        """
        将单层图章光栅化、染色、翻转并旋转为精灵图。

        `scale` 为全画布超采样倍率：图章在高分辨率下完成全部几何变换，
        避免「先降到 320 再旋转」造成的台阶锯齿。
        """
        info = self.library.resolve(layer.asset)
        # 额外的源光栅倍率：在全画布 ss 之上再略提高矢量采样密度。
        src_scale = max(scale, 1.0) * float(self.config.stamp_raster_scale)
        rw = _px(layer.width * src_scale)
        rh = _px(layer.height * src_scale)
        rgba = rasterize_svg(info.path, out_width=rw, out_height=rh)
        tinted = self._tint_white_stamp(rgba, layer.fill, layer.opacity)
        # 几何变换在预乘 alpha 空间进行，避免直通 RGBA 插值把透明区的
        # 杂色/白色带进边缘（典型白描边/暗描边）。
        img = self._to_premultiplied_image(tinted)

        if layer.flipX:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if layer.flipY:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        target_w = _px(layer.width * scale)
        target_h = _px(layer.height * scale)
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        if abs(layer.angle) > 1e-6:
            # 编辑器角度为顺时针；PIL 的 rotate 为逆时针，故取负。
            img = img.rotate(
                -layer.angle,
                expand=True,
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0, 0),
            )

        # 反预乘，并把实心染色图章的 RGB 钳回 fill（消除插值过冲白边）。
        return self._from_premultiplied_image(img, fill=layer.fill)

    def _alpha_over(
        self,
        canvas: Image.Image,
        sprite: Image.Image,
        layer: StampLayer,
        *,
        scale: float,
    ) -> None:
        """将精灵按中心原点 alpha 叠到（可能已放大的）画布上。"""
        # 中心原点：left/top 是未旋转包围盒中心；
        # expand=True 旋转后 PIL 仍保持视觉中心稳定。
        sw, sh = sprite.size
        x = _pos(layer.left * scale - sw / 2.0)
        y = _pos(layer.top * scale - sh / 2.0)
        canvas.alpha_composite(sprite, dest=(x, y))

    @staticmethod
    def _downsample(image: Image.Image, width: int, height: int) -> Image.Image:
        """
        将高分辨率合成结果降到目标画布尺寸（SSAA）。

        在 **float 预乘 alpha** 空间做盒式/面积平均，避免 uint8 预乘在
        极低 alpha 处反预乘爆炸（会冒出白/彩边）。
        """
        if image.size == (width, height):
            return image

        src = np.asarray(image).astype(np.float32) / 255.0
        alpha = src[:, :, 3:4]
        pre = np.empty_like(src)
        pre[:, :, :3] = src[:, :, :3] * alpha
        pre[:, :, 3:4] = alpha

        src_h, src_w = pre.shape[:2]
        # 面积平均（整数倍时为标准盒滤 SSAA；非整数倍时用分箱近似）
        if (
            src_w % width == 0
            and src_h % height == 0
            and src_w >= width
            and src_h >= height
        ):
            factor_x = src_w // width
            factor_y = src_h // height
            reshaped = pre.reshape(height, factor_y, width, factor_x, 4)
            down = reshaped.mean(axis=(1, 3))
        else:
            # 非整数倍率：先用 PIL 在预乘 float 上按通道 resize（转 16-bit 近似）
            down = np.empty((height, width, 4), dtype=np.float32)
            # 映射到 0..65535 再 LANCZOS，最后归一化
            u16 = np.clip(pre * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
            for c in range(4):
                ch = Image.fromarray(u16[:, :, c], mode="I;16")
                ch = ch.resize((width, height), Image.Resampling.LANCZOS)
                down[:, :, c] = np.asarray(ch, dtype=np.float32) / 65535.0

        out_a = np.clip(down[:, :, 3], 0.0, 1.0)
        out_rgb = np.zeros((height, width, 3), dtype=np.float32)
        mask = out_a > 1e-6
        out_rgb[mask] = np.clip(down[:, :, :3][mask] / out_a[mask, np.newaxis], 0.0, 1.0)

        out = np.empty((height, width, 4), dtype=np.uint8)
        out[:, :, :3] = (out_rgb * 255.0 + 0.5).astype(np.uint8)
        out[:, :, 3] = (out_a * 255.0 + 0.5).astype(np.uint8)
        # 全透明像素清零 RGB，避免查看器/后续处理读到脏数据
        clear = out[:, :, 3] == 0
        out[clear, :3] = 0
        return Image.fromarray(out, mode="RGBA")

    @staticmethod
    def _tint_white_stamp(
        rgba: Rgba,
        fill: HexColor,
        opacity: float,
    ) -> Rgba:
        """
        将图章视为形状蒙版并染色（直通 alpha）。

        资产 SVG 为白模；覆盖率只信任 alpha（MuPDF 边缘常为预乘灰，
        不能再用亮度去加权，否则会吃掉抗锯齿）。
        - 有覆盖：RGB = fill
        - 无覆盖：RGB = 0（避免后续滤波把残留白/彩色带进描边）
        """
        r, g, b = hex_to_rgb(fill)
        out = np.zeros_like(rgba)
        alpha = rgba[:, :, 3].astype(np.float32) * float(opacity)
        alpha = np.clip(alpha, 0.0, 255.0)
        coverage = alpha > 0.0
        out[coverage, 0] = r
        out[coverage, 1] = g
        out[coverage, 2] = b
        out[:, :, 3] = (alpha + 0.5).astype(np.uint8)
        return out

    @staticmethod
    def _to_premultiplied_image(rgba: Rgba) -> Image.Image:
        """直通 RGBA → 预乘 RGBA（uint8 近似）。"""
        src = rgba.astype(np.float32)
        a = src[:, :, 3:4] / 255.0
        out = src.copy()
        out[:, :, :3] = src[:, :, :3] * a
        return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), mode="RGBA")

    @staticmethod
    def _from_premultiplied_image(
        image: Image.Image,
        *,
        fill: HexColor,
    ) -> Image.Image:
        """
        预乘 RGBA → 直通 RGBA。

        图章为单色填充：凡 alpha>0 处 RGB 一律写 fill，
        既消除滤波过冲白边，也避免半透明边缘 RGB=0 造成发黑描边。
        """
        r, g, b = hex_to_rgb(fill)
        pre = np.asarray(image)
        alpha = pre[:, :, 3]
        out = np.zeros_like(pre)
        out[:, :, 3] = alpha
        solid = alpha > 0
        out[solid, 0] = r
        out[solid, 1] = g
        out[solid, 2] = b
        return Image.fromarray(out, mode="RGBA")


def render_emblem(
    document: EmblemDocument,
    *,
    stamps_dir: str | Path = "assets/stamps",
    canvas: CanvasConfig | None = None,
    supersample: float = 4.0,
) -> Image.Image:
    """便捷封装：按默认配置渲染文档。"""
    cfg = RenderConfig(
        canvas=canvas or CanvasConfig(),
        stamps_dir=Path(stamps_dir),
        supersample=supersample,
    )
    return EmblemRenderer(cfg).render(document)
