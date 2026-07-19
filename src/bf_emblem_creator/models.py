"""徽章图层、文档与渲染配置的 Pydantic 模型。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
)

# ---------------------------------------------------------------------------
# 基础类型
# ---------------------------------------------------------------------------


def _normalize_hex_color(value: str) -> str:
    """将颜色规范为 #RRGGBB（大写）。"""
    raw = value.strip()
    if not raw.startswith("#"):
        raw = f"#{raw}"
    body = raw[1:]
    if len(body) == 3 and all(c in "0123456789abcdefABCDEF" for c in body):
        body = "".join(ch * 2 for ch in body)
    if len(body) != 6 or any(c not in "0123456789abcdefABCDEF" for c in body):
        raise ValueError(f"无效的十六进制颜色: {value!r}")
    return f"#{body.upper()}"


HexColor = Annotated[str, AfterValidator(_normalize_hex_color)]


def hex_to_rgb(color: HexColor) -> tuple[int, int, int]:
    """将 #RRGGBB 转为 (R, G, B) 整数元组。"""
    body = color[1:]
    return int(body[0:2], 16), int(body[2:4], 16), int(body[4:6], 16)


# ---------------------------------------------------------------------------
# 图层 / 文档（与编辑器导出 JSON 对齐）
# ---------------------------------------------------------------------------


class StampLayer(BaseModel):
    """单个图章摆放，字段与徽章编辑器导出格式一致。"""

    # ignore：编辑器可能附带未知 UI 字段；已知字段显式建模以便 round-trip。
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    asset: str = Field(..., description="图章 id，对应 assets/stamps/{asset}.svg")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="不透明度，0–1")
    angle: float = Field(default=0.0, description="旋转角（度，顺时针）")
    flipX: bool = Field(default=False, description="水平镜像")
    flipY: bool = Field(default=False, description="垂直镜像")
    top: float = Field(..., description="图层原点纵坐标（中心点）")
    left: float = Field(..., description="图层原点横坐标（中心点）")
    height: float = Field(..., gt=0.0, description="旋转前包围盒高度")
    width: float = Field(..., gt=0.0, description="旋转前包围盒宽度")
    fill: HexColor = Field(..., description="染色颜色，例如 #3A9FFA")
    selectable: bool = Field(
        default=False,
        description="编辑器 UI 字段；渲染忽略，导出时保留",
    )

    @field_validator("asset")
    @classmethod
    def _asset_non_empty(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("asset 不能为空")
        if name.lower().endswith(".svg"):
            name = name[:-4]
        return name

    def svg_filename(self) -> str:
        """返回对应的 SVG 文件名。"""
        return f"{self.asset}.svg"

    def fill_rgb(self) -> tuple[int, int, int]:
        """返回填充色的 RGB 元组。"""
        return hex_to_rgb(self.fill)


class EmblemDocument(RootModel[list[StampLayer]]):
    """有序图章栈；下标 0 为最底层。"""

    root: list[StampLayer]

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self) -> Iterator[StampLayer]:  # type: ignore[override]
        return iter(self.root)

    def __getitem__(self, index: int) -> StampLayer:
        return self.root[index]

    @classmethod
    def from_layers(cls, layers: Iterable[StampLayer]) -> Self:
        """由图层序列构造文档。"""
        return cls(root=list(layers))

    @classmethod
    def load_json(cls, path: str | Path) -> Self:
        """从 JSON 文件加载并校验。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    @classmethod
    def model_validate_json_file(cls, path: str | Path) -> Self:
        """`load_json` 的别名。"""
        return cls.load_json(path)

    def save_json(self, path: str | Path, *, indent: int = 2) -> None:
        """将文档规范化后写入 JSON 文件。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        target.write_text(
            json.dumps(payload, indent=indent, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def model_dump_list(self) -> list[dict[str, Any]]:
        """导出为图层字典列表（JSON 兼容）。"""
        dumped = self.model_dump(mode="json")
        if not isinstance(dumped, list):
            raise TypeError("EmblemDocument 的 dump 结果必须是 list")
        return dumped


# ---------------------------------------------------------------------------
# 渲染 / 图章库配置
# ---------------------------------------------------------------------------


class CanvasConfig(BaseModel):
    """输出画布。战地图章默认 viewBox 多为 320×320。"""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=320, gt=0, description="画布宽度（像素）")
    height: int = Field(default=320, gt=0, description="画布高度（像素）")
    background: HexColor | None = Field(
        default=None,
        description="纯色背景十六进制色；为 null 时表示透明",
    )


class RenderConfig(BaseModel):
    """离线渲染器配置。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    canvas: CanvasConfig = Field(default_factory=CanvasConfig, description="画布配置")
    stamps_dir: Path = Field(
        default=Path("assets/stamps"),
        description="存放 {Asset}.svg 的目录",
    )
    # 导出格式采用类 fabric 的中心原点 + 顺时针角度。
    origin: str = Field(default="center", description="图层原点约定，目前仅支持 center")
    supersample: float = Field(
        default=4.0,
        ge=1.0,
        le=8.0,
        description=(
            "全画布超采样倍率：在 width*ss × height*ss 上合成后再降采样。"
            "4 表示 320 画布内部按 1280 渲染，可明显减轻低分辨率锯齿。"
        ),
    )
    stamp_raster_scale: float = Field(
        default=1.5,
        ge=1.0,
        le=4.0,
        description="单枚图章 SVG 相对当前合成分辨率的额外光栅倍率",
    )

    @field_validator("stamps_dir", mode="before")
    @classmethod
    def _as_path(cls, value: str | Path) -> Path:
        return Path(value)

    @field_validator("origin")
    @classmethod
    def _origin_center_only(cls, value: str) -> str:
        if value != "center":
            raise ValueError("仅支持 origin='center'（与编辑器导出格式一致）")
        return value


class StampAssetInfo(BaseModel):
    """磁盘上已解析的图章资源信息。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str = Field(..., description="图章 id")
    path: Path = Field(..., description="SVG 文件路径")
    native_width: float | None = Field(default=None, description="SVG 原生宽度")
    native_height: float | None = Field(default=None, description="SVG 原生高度")
