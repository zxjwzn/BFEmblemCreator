"""图章加载器：StampLibrary + 曲线缓存 → StampCatalog。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.recipe import StampLoaderConfig
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
from bf_emblem_creator.stamps import StampLibrary


class StampCatalog(BaseModel):
    """只读图章曲线目录（供匹配/渲染）。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    library: Any = Field(..., description="StampLibrary 实例")
    curves: Any = Field(..., description="StampCurveLibrary 实例")
    allowed_ids: list[str] = Field(default_factory=list, description="允许检索的 asset id")
    cache_dir: Path | None = Field(default=None, description="实际缓存目录")

    def get_curve_lib(self) -> StampCurveLibrary:
        """返回曲线库。"""
        return self.curves  # type: ignore[return-value]

    def get_stamp_lib(self) -> StampLibrary:
        """返回图章库。"""
        return self.library  # type: ignore[return-value]


class StampLoader:
    """图章加载器。"""

    def __init__(self, config: StampLoaderConfig) -> None:
        self.config = config

    def _resolve_cache(self) -> Path:
        if self.config.cache_dir is not None:
            return Path(self.config.cache_dir)
        return Path(self.config.stamps_dir).parent / ".cache" / "stamp_curves"

    def load(self, *, progress_cb: Any | None = None) -> StampCatalog:
        """扫描图章并构建/加载曲线缓存。"""
        lib = StampLibrary(self.config.stamps_dir)
        allow = self.config.asset_allowlist
        block = set(self.config.asset_blocklist)
        if allow is not None:
            subset = [a for a in allow if a not in block]
        else:
            # 全库 id
            subset = None
            if block:
                all_ids = [p.stem for p in Path(self.config.stamps_dir).glob("*.svg")]
                subset = [a for a in all_ids if a not in block]
        cache = self._resolve_cache()
        curves = StampCurveLibrary.build(
            lib,
            subset,
            tex_size=int(self.config.tex_size),
            cache_dir=cache,
            force_refit=bool(self.config.force_rebuild),
            progress_cb=progress_cb,
        )
        allowed = list(curves.by_id.keys()) if subset is None else [a for a in subset if a in curves.by_id]
        return StampCatalog(library=lib, curves=curves, allowed_ids=allowed, cache_dir=cache)
