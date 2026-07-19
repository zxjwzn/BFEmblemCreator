"""图章边缘曲线库 + 效果标签 + 磁盘缓存（高精度多环轮廓，全库可检索）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from bf_emblem_creator.approx.curves import (
    extract_contour_bundle,
    multi_ring_descriptor,
    normalize_contour_to_unit,
    resample_closed_contour,
    signed_area,
)
from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.gpu_ops import mask_to_sdf_fast, to_torch
from bf_emblem_creator.raster import rasterize_svg
from bf_emblem_creator.stamps import StampLibrary

FloatArr = NDArray[np.floating]
U8Arr = NDArray[np.uint8]

# 缓存格式：多环 + 高分辨率；变更时递增强制重拟合
CACHE_VERSION = 4
DEFAULT_TEX_SIZE = 256
DEFAULT_RESAMPLE_N = 192
DEFAULT_SIMPLIFY = 0.35


def geometric_complexity(
    *,
    circularity: float,
    elongation: float,
    n_holes: int,
    descriptor: FloatArr | None = None,
    area_frac: float = 0.5,
) -> float:
    """
    通用图章/区域几何复杂度（0~1），与场景名、asset 名无关。

    高复杂度：低圆度、多孔、极细长、描述子起伏大、填充率中等偏低的剪影。
    """
    c = float(np.clip(circularity, 0.0, 2.0))
    e = float(max(elongation, 1.0))
    # 圆度惩罚：circ 越低越复杂
    p_circ = float(np.clip(1.0 - min(c, 1.0), 0.0, 1.0))
    # 孔洞
    p_hole = float(np.clip(n_holes / 4.0, 0.0, 1.0))
    # 细长
    p_elong = float(np.clip(np.log(e) / np.log(6.0), 0.0, 1.0))
    # 描述子 L2 起伏
    p_desc = 0.0
    if descriptor is not None and len(descriptor) > 2:
        d = np.asarray(descriptor, dtype=np.float64)
        p_desc = float(np.clip(d.std() * 4.0, 0.0, 1.0))
    # 中等填充剪影往往轮廓更碎
    af = float(np.clip(area_frac, 0.0, 1.0))
    p_fill = float(np.clip(1.0 - abs(af - 0.55) * 2.0, 0.0, 0.5)) * 0.5
    raw = 0.40 * p_circ + 0.25 * p_hole + 0.15 * p_elong + 0.15 * p_desc + 0.05 * p_fill
    return float(np.clip(raw, 0.0, 1.0))


@dataclass
class StampCurveEntry:
    """单个图章的多环曲线 / SDF / 效果标签。"""

    asset_id: str
    contour: FloatArr  # 主外轮廓，归一化到 [-0.5,0.5]^2
    contour_px: FloatArr  # 主外轮廓像素坐标
    holes: list[FloatArr] = field(default_factory=list)  # 孔洞（归一化）
    holes_px: list[FloatArr] = field(default_factory=list)  # 孔洞像素
    descriptor: FloatArr = field(default_factory=lambda: np.zeros(27, dtype=np.float64))
    sdf: FloatArr = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    tex_size: int = DEFAULT_TEX_SIZE
    circularity: float = 0.0
    elongation: float = 1.0
    area_frac: float = 0.0
    tags: list[str] = field(default_factory=list)
    n_holes: int = 0
    complexity: float = 0.0
    """通用几何复杂度 0~1（非场景）：低圆度、多孔、高细长、描述子起伏 → 更高。"""


def detect_effect_tags(
    mask: NDArray[np.bool_],
    contour: FloatArr,
    *,
    circularity: float,
    elongation: float,
    area_frac: float,
    n_holes: int = 0,
) -> list[str]:
    """从图章几何自动打效果标签（GPU）。"""
    from bf_emblem_creator.approx.gpu_ops import detect_effect_tags_torch

    tags = detect_effect_tags_torch(
        mask,
        contour,
        circularity=circularity,
        elongation=elongation,
        area_frac=area_frac,
        device=get_device(),
    )
    if n_holes > 0 and "ring" not in tags:
        tags.append("ring")
    return tags


class StampCurveLibrary:
    """全库图章曲线库（描述子检索，多环高精度）。"""

    def __init__(self, entries: list[StampCurveEntry]) -> None:
        self.entries = entries
        self.by_id = {e.asset_id: e for e in entries}

    @classmethod
    def build(
        cls,
        library: StampLibrary,
        asset_ids: list[str] | None = None,
        *,
        tex_size: int = DEFAULT_TEX_SIZE,
        cache_dir: str | Path | None = None,
        force_refit: bool = False,
        simplify: float = DEFAULT_SIMPLIFY,
        resample_n: int = DEFAULT_RESAMPLE_N,
    ) -> StampCurveLibrary:
        """
        构建曲线库。

        asset_ids=None → 目录下全部图章。
        force_refit=True → 忽略缓存并重写。
        tex_size 默认 256，保证内部闭合边精度。
        """
        ids = list(asset_ids) if asset_ids is not None else list(library.list_ids())
        ids = sorted(ids)
        cache_path = Path(cache_dir) if cache_dir else None
        if cache_path is not None and not force_refit:
            cache_path.mkdir(parents=True, exist_ok=True)
            meta_file = cache_path / "stamp_curves_meta.json"
            if meta_file.is_file():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if (
                        meta.get("version") == CACHE_VERSION
                        and meta.get("tex_size") == tex_size
                        and meta.get("ids") == ids
                        and meta.get("resample_n", DEFAULT_RESAMPLE_N) == resample_n
                    ):
                        return cls._load_cache(cache_path, ids, tex_size)
                except Exception:
                    pass

        dev = get_device()
        entries: list[StampCurveEntry] = []
        for asset_id in ids:
            try:
                info = library.resolve(asset_id)
            except FileNotFoundError:
                continue
            # SVG→栅格（PyMuPDF）；高分辨率后掩膜全程 GPU/高精度跟踪
            rgba = rasterize_svg(info.path, out_width=tex_size, out_height=tex_size)
            alpha_t = to_torch(rgba[:, :, 3].astype(np.float32), device=dev)
            # 略软阈值：保留抗锯齿内侧，减少收缩
            mask_t = alpha_t >= 96.0
            if not bool(mask_t.any().item()):
                continue
            mask = mask_t.detach().cpu().numpy().astype(bool)

            outer_px, holes_px, _ = extract_contour_bundle(
                mask,
                simplify=simplify,
                resample_n=resample_n,
            )
            if len(outer_px) < 3:
                ys, xs = torch.where(mask_t)
                x0, x1 = float(xs.min().item()), float(xs.max().item())
                y0, y1 = float(ys.min().item()), float(ys.max().item())
                outer_px = np.array(
                    [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
                    dtype=np.float64,
                )
                holes_px = []

            # 弧长重采样到高密度，保证内部闭合边与复杂外轮廓精度
            outer_px = resample_closed_contour(outer_px, resample_n)
            holes_px = [resample_closed_contour(h, resample_n) for h in holes_px if len(h) >= 3]

            outer_n = normalize_contour_to_unit(outer_px, tex_size)
            holes_n = [normalize_contour_to_unit(h, tex_size) for h in holes_px]
            desc = multi_ring_descriptor(outer_px, holes_px, bins=24)

            # 高精度 SDF
            sdf_iters = max(16, min(64, tex_size // 8))
            sdf = mask_to_sdf_fast(mask, device=dev, iters=sdf_iters).astype(np.float32)

            area = float(mask_t.float().mean().item())
            ys, xs = torch.where(mask_t)
            bw = float((xs.max() - xs.min() + 1).item())
            bh = float((ys.max() - ys.min() + 1).item())
            aspect = bw / max(bh, 1.0)
            elong = max(aspect, 1.0 / max(aspect, 1e-6))

            cont_t = to_torch(np.asarray(outer_px, dtype=np.float32), device=dev)
            closed = cont_t
            if torch.linalg.norm(closed[0] - closed[-1]) > 1e-6:
                closed = torch.cat([closed, closed[:1]], dim=0)
            peri = float(torch.linalg.norm(closed[1:] - closed[:-1], dim=1).sum().item())
            area_px = float(mask_t.sum().item())
            # 孔洞面积从外轮廓面积中扣除后的填充率影响圆度
            hole_a = float(sum(abs(signed_area(h)) for h in holes_px))
            solid_a = max(area_px - hole_a, 1.0)
            circ = float(4.0 * np.pi * solid_a / (peri * peri + 1e-6))
            circ = float(np.clip(circ, 0, 2))

            n_holes = len(holes_px)
            tags = detect_effect_tags(
                mask,
                outer_px,
                circularity=circ,
                elongation=float(elong),
                area_frac=area,
                n_holes=n_holes,
            )
            cx = geometric_complexity(
                circularity=circ,
                elongation=float(elong),
                n_holes=n_holes,
                descriptor=desc,
                area_frac=area,
            )
            entries.append(
                StampCurveEntry(
                    asset_id=asset_id,
                    contour=outer_n,
                    contour_px=outer_px,
                    holes=holes_n,
                    holes_px=holes_px,
                    descriptor=desc,
                    sdf=sdf,
                    tex_size=tex_size,
                    circularity=circ,
                    elongation=float(elong),
                    area_frac=area,
                    tags=tags,
                    n_holes=n_holes,
                    complexity=cx,
                )
            )
        if not entries:
            raise RuntimeError("图章曲线库为空")
        if cache_path is not None:
            cls._save_cache(cache_path, entries, ids, tex_size, resample_n)
        return cls(entries)

    @classmethod
    def prefit_directory(
        cls,
        stamps_dir: str | Path,
        *,
        cache_dir: str | Path | None = None,
        tex_size: int = DEFAULT_TEX_SIZE,
        simplify: float = DEFAULT_SIMPLIFY,
        resample_n: int = DEFAULT_RESAMPLE_N,
    ) -> StampCurveLibrary:
        """强制重拟合目录下全部图章并持久化（高精度多环）。"""
        library = StampLibrary(stamps_dir)
        stamps_path = Path(stamps_dir)
        cache = Path(cache_dir) if cache_dir else stamps_path.parent / ".cache" / "stamp_curves"
        return cls.build(
            library,
            asset_ids=None,
            tex_size=tex_size,
            cache_dir=cache,
            force_refit=True,
            simplify=simplify,
            resample_n=resample_n,
        )

    @staticmethod
    def _save_cache(
        cache_path: Path,
        entries: list[StampCurveEntry],
        ids: list[str],
        tex_size: int,
        resample_n: int,
    ) -> None:
        cache_path.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": CACHE_VERSION,
            "tex_size": tex_size,
            "resample_n": resample_n,
            "ids": ids,
            "assets": [e.asset_id for e in entries],
            "tags": {e.asset_id: e.tags for e in entries},
            "n_holes": {e.asset_id: e.n_holes for e in entries},
        }
        (cache_path / "stamp_curves_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for e in entries:
            # 孔洞打包为 object 数组
            holes_arr = np.empty(len(e.holes), dtype=object)
            holes_px_arr = np.empty(len(e.holes_px), dtype=object)
            for i, h in enumerate(e.holes):
                holes_arr[i] = np.asarray(h, dtype=np.float64)
            for i, h in enumerate(e.holes_px):
                holes_px_arr[i] = np.asarray(h, dtype=np.float64)
            np.savez_compressed(
                cache_path / f"{e.asset_id}.npz",
                contour=e.contour,
                contour_px=e.contour_px,
                holes=holes_arr,
                holes_px=holes_px_arr,
                descriptor=e.descriptor,
                sdf=e.sdf,
                circularity=e.circularity,
                elongation=e.elongation,
                area_frac=e.area_frac,
                tex_size=e.tex_size,
                n_holes=e.n_holes,
                complexity=e.complexity,
                tags=np.array(e.tags, dtype=object),
            )

    @classmethod
    def _load_cache(cls, cache_path: Path, ids: list[str], tex_size: int) -> StampCurveLibrary:
        """从磁盘缓存加载。"""
        _ = tex_size
        entries: list[StampCurveEntry] = []
        for asset_id in ids:
            f = cache_path / f"{asset_id}.npz"
            if not f.is_file():
                continue
            data = np.load(f, allow_pickle=True)
            tags_raw = data["tags"] if "tags" in data.files else np.array([], dtype=object)
            tags = [str(t) for t in tags_raw.tolist()] if getattr(tags_raw, "size", 0) else []
            holes: list[FloatArr] = []
            holes_px: list[FloatArr] = []
            if "holes" in data.files:
                for h in data["holes"]:
                    holes.append(np.asarray(h, dtype=np.float64))
            if "holes_px" in data.files:
                for h in data["holes_px"]:
                    holes_px.append(np.asarray(h, dtype=np.float64))
            n_holes = int(data["n_holes"]) if "n_holes" in data.files else len(holes)
            circ = float(data["circularity"])
            elong = float(data["elongation"])
            afrac = float(data["area_frac"])
            desc = data["descriptor"]
            if "complexity" in data.files:
                cplx = float(data["complexity"])
            else:
                cplx = geometric_complexity(
                    circularity=circ,
                    elongation=elong,
                    n_holes=n_holes,
                    descriptor=desc,
                    area_frac=afrac,
                )
            entries.append(
                StampCurveEntry(
                    asset_id=asset_id,
                    contour=data["contour"],
                    contour_px=data["contour_px"],
                    holes=holes,
                    holes_px=holes_px,
                    descriptor=desc,
                    sdf=data["sdf"],
                    tex_size=int(data["tex_size"]),
                    circularity=circ,
                    elongation=elong,
                    area_frac=afrac,
                    tags=tags,
                    n_holes=n_holes,
                    complexity=cplx,
                )
            )
        if not entries:
            raise RuntimeError("缓存曲线库为空")
        return cls(entries)

    def recall(
        self,
        block_descriptor: FloatArr,
        circularity: float = 1.0,
        elongation: float = 1.0,
        k: int = 48,
        *,
        tag_boost: set[str] | None = None,
        prefer_holes: bool | None = None,
        region_simplicity: float | None = None,
        region_area_frac: float = 0.1,
    ) -> list[str]:
        """
        描述子 top-k 召回（全库，无形状桶、无场景名）。

        通用几何策略：
        - prefer_holes：区域填充率暗示镂空时，对带孔图章加权
        - region_simplicity 高时，抑制高 complexity 图章（避免复杂剪影蹭简单色块边）
        """
        scored: list[tuple[float, str]] = []
        desc = np.asarray(block_descriptor, dtype=np.float64)
        r_simp = 0.5 if region_simplicity is None else float(np.clip(region_simplicity, 0.0, 1.0))
        for e in self.entries:
            ed = np.asarray(e.descriptor, dtype=np.float64)
            n = min(len(ed), len(desc))
            d = float(np.linalg.norm(ed[:n] - desc[:n]))
            if len(ed) != len(desc):
                d += 0.15 * abs(len(ed) - len(desc))
            dc = abs(e.circularity - circularity) * 0.35
            de = abs(np.log(e.elongation + 1e-6) - np.log(elongation + 1e-6)) * 0.25
            score = float(np.exp(-d) + np.exp(-dc) + np.exp(-de))
            if tag_boost and any(t in tag_boost for t in e.tags):
                score *= 1.35
            if prefer_holes is True and e.n_holes > 0:
                score *= 1.25
            if prefer_holes is False and e.n_holes > 0:
                score *= 0.75
            # 简洁区域 ↔ 复杂图章 不匹配：指数惩罚（与 asset 名无关）
            mismatch = r_simp * e.complexity
            score *= float(np.exp(-2.2 * mismatch))
            # 简单大块优先高填充率图章
            if r_simp > 0.55 and region_area_frac > 0.04:
                score *= float(np.exp(-1.0 * abs(e.area_frac - 0.75) * r_simp))
            scored.append((score, e.asset_id))
        scored.sort(key=lambda t: -t[0])
        out: list[str] = []
        for _, aid in scored:
            if aid not in out:
                out.append(aid)
            if len(out) >= k:
                break
        return out

    def by_tag(self, *tags: str) -> list[str]:
        """返回含任一标签的图章 id。"""
        want = set(tags)
        return [e.asset_id for e in self.entries if want.intersection(e.tags)]

    def all_rings_normalized(self, asset_id: str) -> list[FloatArr]:
        """主外轮廓 + 全部孔洞（归一化坐标）。"""
        e = self.by_id[asset_id]
        return [e.contour, *e.holes]

    def contour_on_canvas(
        self,
        asset_id: str,
        *,
        left: float,
        top: float,
        width: float,
        height: float,
        angle_deg: float,
        flip_x: bool = False,
        n: int = 128,
        include_holes: bool = True,
    ) -> FloatArr:
        """
        将图章轮廓（可选含孔）变换到画布坐标。

        多环时拼接为 (M,2)，孔与外轮廓之间不插值连接点由调用方分环处理更佳；
        此处提供合并点云供 Chamfer。
        """
        rings = self.all_rings_normalized(asset_id) if include_holes else [self.by_id[asset_id].contour]
        chunks: list[FloatArr] = []
        th = np.deg2rad(angle_deg)
        c, s = np.cos(th), np.sin(th)
        for ring in rings:
            pts = resample_closed_contour(ring, n)
            if flip_x:
                pts = pts.copy()
                pts[:, 0] = -pts[:, 0]
            sx = pts[:, 0] * width
            sy = pts[:, 1] * height
            x = c * sx + s * sy + left
            y = -s * sx + c * sy + top
            chunks.append(np.stack([x, y], axis=1))
        if not chunks:
            return np.zeros((0, 2), dtype=np.float64)
        return np.vstack(chunks)
