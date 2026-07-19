"""图章库：将 asset id 映射到 SVG 路径。"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from bf_emblem_creator.models import StampAssetInfo

_VIEWBOX_RE = re.compile(
    r"viewBox\s*=\s*['\"]\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*['\"]",
    re.IGNORECASE,
)


class StampLibrary:
    """基于文件系统的图章目录（`{id}.svg`）。"""

    def __init__(self, stamps_dir: str | Path) -> None:
        self.stamps_dir = Path(stamps_dir)
        if not self.stamps_dir.is_dir():
            raise FileNotFoundError(f"图章目录不存在: {self.stamps_dir}")

    def list_ids(self) -> list[str]:
        """列出全部图章 id（按文件名排序）。"""
        return sorted(p.stem for p in self.stamps_dir.glob("*.svg"))

    def resolve(self, asset_id: str) -> StampAssetInfo:
        """解析 asset id 为磁盘上的 SVG 资源信息。"""
        name = asset_id[:-4] if asset_id.lower().endswith(".svg") else asset_id
        path = self.stamps_dir / f"{name}.svg"
        if not path.is_file():
            # 大小写不敏感回退（Windows 通常不区分，Linux 可能区分）。
            match = self._find_case_insensitive(name)
            if match is None:
                raise FileNotFoundError(f"未找到图章 SVG: asset={asset_id!r}，目录={self.stamps_dir}")
            path = match
            name = path.stem
        nw, nh = _read_svg_native_size(str(path.resolve()))
        return StampAssetInfo(id=name, path=path, native_width=nw, native_height=nh)

    def _find_case_insensitive(self, name: str) -> Path | None:
        """按不区分大小写的 stem 查找 SVG。"""
        target = name.lower()
        for path in self.stamps_dir.glob("*.svg"):
            if path.stem.lower() == target:
                return path
        return None


@lru_cache(maxsize=512)
def _read_svg_native_size(path_str: str) -> tuple[float | None, float | None]:
    """读取 SVG 的原生宽高（优先 viewBox，其次 width/height 属性）。"""
    path = Path(path_str)
    text = path.read_text(encoding="utf-8", errors="replace")
    match = _VIEWBOX_RE.search(text)
    if match:
        return float(match.group(3)), float(match.group(4))
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None, None
    width = root.attrib.get("width")
    height = root.attrib.get("height")
    return _parse_svg_length(width), _parse_svg_length(height)


def _parse_svg_length(value: str | None) -> float | None:
    """解析 SVG 长度属性（支持可选 px 后缀）。"""
    if value is None:
        return None
    cleaned = value.strip().removesuffix("px").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None
