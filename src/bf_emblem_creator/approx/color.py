"""颜色空间工具：sRGB ↔ CIELAB、调色板距离。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.floating]
U8Arr = NDArray[np.uint8]


def srgb_to_linear(u8: U8Arr) -> FloatArr:
    """将 0–255 sRGB 转为线性 RGB（约 0–1）。"""
    x = u8.astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_u8(lin: FloatArr) -> U8Arr:
    """线性 RGB（0–1）→ sRGB uint8。"""
    x = np.clip(lin, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def rgb_u8_to_lab(rgb: U8Arr) -> FloatArr:
    """
    sRGB uint8 (...,3) → CIELAB (...,3)。

    使用 D65 与标准 sRGB 矩阵；足够用于色量与色差，非 ICC 级精确。
    """
    lin = srgb_to_linear(rgb)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = lin @ m.T
    # D65 白点
    xyz_n = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

    def f(t: FloatArr) -> FloatArr:
        delta = 6.0 / 29.0
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta * delta) + 4.0 / 29.0)

    fxyz = f(xyz_n)
    lightness = 116.0 * fxyz[..., 1] - 16.0
    a_ch = 500.0 * (fxyz[..., 0] - fxyz[..., 1])
    b_ch = 200.0 * (fxyz[..., 1] - fxyz[..., 2])
    return np.stack([lightness, a_ch, b_ch], axis=-1)


def lab_to_rgb_u8(lab: FloatArr) -> U8Arr:
    """CIELAB → sRGB uint8。"""
    lightness = lab[..., 0]
    a_ch = lab[..., 1]
    b_ch = lab[..., 2]
    fy = (lightness + 16.0) / 116.0
    fx = fy + a_ch / 500.0
    fz = fy - b_ch / 200.0

    def finv(t: FloatArr) -> FloatArr:
        delta = 6.0 / 29.0
        return np.where(t > delta, t**3, 3 * delta * delta * (t - 4.0 / 29.0))

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1)
    xyz = xyz * np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float64,
    )
    lin = xyz @ m_inv.T
    return linear_to_srgb_u8(lin)


def rgb_to_hex(rgb: tuple[int, int, int] | U8Arr) -> str:
    """RGB 元组或长度为 3 的数组 → #RRGGBB。"""
    r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb_tuple(color: str) -> tuple[int, int, int]:
    """#RRGGBB → (R,G,B)。"""
    body = color[1:] if color.startswith("#") else color
    return int(body[0:2], 16), int(body[2:4], 16), int(body[4:6], 16)


def lab_distance(a: FloatArr, b: FloatArr) -> FloatArr:
    """逐点欧氏 LAB 距离。"""
    return np.linalg.norm(a - b, axis=-1)
