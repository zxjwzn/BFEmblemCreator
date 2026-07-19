"""兼容常量：v3 默认全库检索；此列表仅作测试/调试子集。"""

from __future__ import annotations

# 小几何子集（测试用）；生产路径 stamp_subset=None 时用全库
DEFAULT_BASIC_STAMPS: tuple[str, ...] = (
    "Square",
    "Circle",
    "Triangle",
    "RightTriangle",
    "HalfCircle",
    "OpenCircle",
    "Drop",
    "Line",
    "Stroke",
    "StrokeBent",
    "BentLine",
    "Arrow",
    "ArrowBent",
    "SquareCorner",
    "CrescentMoon",
    "Banner",
    "Shield",
    "Wingpart",
)
