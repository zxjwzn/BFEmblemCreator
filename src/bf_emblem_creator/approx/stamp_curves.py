"""图章边缘曲线库 + 效果标签 + 磁盘缓存（多环轮廓，全库可检索）。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from bf_emblem_creator.approx.curves import (
    fit_mask_contour_high_precision,
    multi_ring_descriptor,
    normalize_contour_to_unit,
    resample_closed_contour,
    signed_area,
)
from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.gpu_ops import batch_mask_to_sdf, mask_to_sdf_fast, to_torch
from bf_emblem_creator.raster import rasterize_svg
from bf_emblem_creator.stamps import StampLibrary

FloatArr = NDArray[np.floating]
U8Arr = NDArray[np.uint8]

DEFAULT_TEX_SIZE = 256
DEFAULT_RESAMPLE_N = 256
DEFAULT_AREA_REL_ERR = 0.03
DEFAULT_WORKERS = 8
DEFAULT_SDF_BATCH = 32

# 单图章缓存文件必须具备的字段（缺一则视为无缓存，重新计算）
_NPZ_REQUIRED_KEYS = (
    "contour",
    "contour_px",
    "holes",
    "holes_px",
    "descriptor",
    "sdf",
    "circularity",
    "elongation",
    "area_frac",
    "tex_size",
    "n_holes",
    "complexity",
    "contour_area_rel_err",
    "tags",
)


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
    p_circ = float(np.clip(1.0 - min(c, 1.0), 0.0, 1.0))
    p_hole = float(np.clip(n_holes / 4.0, 0.0, 1.0))
    p_elong = float(np.clip(np.log(e) / np.log(6.0), 0.0, 1.0))
    p_desc = 0.0
    if descriptor is not None and len(descriptor) > 2:
        d = np.asarray(descriptor, dtype=np.float64)
        p_desc = float(np.clip(d.std() * 4.0, 0.0, 1.0))
    af = float(np.clip(area_frac, 0.0, 1.0))
    p_fill = float(np.clip(1.0 - abs(af - 0.55) * 2.0, 0.0, 0.5)) * 0.5
    raw = 0.40 * p_circ + 0.25 * p_hole + 0.15 * p_elong + 0.15 * p_desc + 0.05 * p_fill
    return float(np.clip(raw, 0.0, 1.0))


@dataclass
class StampCurveEntry:
    """单个图章的多环曲线 / SDF / 效果标签。"""

    asset_id: str
    contour: FloatArr
    contour_px: FloatArr
    holes: list[FloatArr] = field(default_factory=list)
    holes_px: list[FloatArr] = field(default_factory=list)
    descriptor: FloatArr = field(default_factory=lambda: np.zeros(27, dtype=np.float64))
    sdf: FloatArr = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    tex_size: int = DEFAULT_TEX_SIZE
    circularity: float = 0.0
    elongation: float = 1.0
    area_frac: float = 0.0
    tags: list[str] = field(default_factory=list)
    n_holes: int = 0
    complexity: float = 0.0
    contour_area_rel_err: float = 0.0


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


def _raster_one(args: tuple[str, Path, int]) -> tuple[str, U8Arr | None, str | None]:
    """线程池任务：单图章 SVG→RGBA。"""
    asset_id, path, tex_size = args
    try:
        rgba = rasterize_svg(path, out_width=tex_size, out_height=tex_size)
    except Exception as exc:  # noqa: BLE001
        return asset_id, None, str(exc)
    else:
        return asset_id, rgba, None


class StampCurveLibrary:
    """全库图章曲线库（描述子检索，多环轮廓）。"""

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
        resample_n: int = DEFAULT_RESAMPLE_N,
        max_area_rel_err: float = DEFAULT_AREA_REL_ERR,
        workers: int = DEFAULT_WORKERS,
        progress_cb: Callable[[str], None] | None = None,
        sdf_batch: int = DEFAULT_SDF_BATCH,
    ) -> StampCurveLibrary:
        """
        构建曲线库。

        - 默认：磁盘上已有完整缓存则直接加载；缺哪些图章只补算哪些，并写回缓存。
        - force_refit=True：忽略已有缓存，全量重算并覆盖（供 prefit-stamps 主动预计算）。
        - asset_ids=None → 目录下全部图章。
        """
        ids = list(asset_ids) if asset_ids is not None else list(library.list_ids())
        ids = sorted(ids)
        cache_path = Path(cache_dir) if cache_dir else None
        log = progress_cb or (lambda _msg: None)

        cached: dict[str, StampCurveEntry] = {}
        if cache_path is not None and not force_refit:
            cache_path.mkdir(parents=True, exist_ok=True)
            for aid in ids:
                entry = cls._try_load_entry(cache_path, aid)
                if entry is not None:
                    cached[aid] = entry
            if len(cached) == len(ids) and ids:
                log(f"加载图章曲线缓存（{len(ids)} 个，完整命中）…")
                return cls([cached[aid] for aid in ids])
            if cached:
                log(f"缓存部分命中 {len(cached)}/{len(ids)}，补算缺失项…")

        need_ids = [aid for aid in ids if aid not in cached]
        if not need_ids and not ids:
            raise RuntimeError("图章曲线库为空（无可用 SVG）")
        if not need_ids:
            return cls([cached[aid] for aid in ids])

        computed = cls._compute_entries(
            library,
            need_ids,
            tex_size=tex_size,
            resample_n=resample_n,
            max_area_rel_err=max_area_rel_err,
            workers=workers,
            sdf_batch=sdf_batch,
            log=log,
        )
        if cache_path is not None:
            cache_path.mkdir(parents=True, exist_ok=True)
            log(f"写入缓存 {cache_path}（{len(computed)} 个）…")
            for e in computed:
                cls._save_entry(cache_path, e)
            cls._write_index(
                cache_path,
                assets=sorted({*cached.keys(), *[e.asset_id for e in computed]}),
                tex_size=tex_size,
                resample_n=resample_n,
                max_area_rel_err=max_area_rel_err,
            )

        by_id = {**cached, **{e.asset_id: e for e in computed}}
        entries = [by_id[aid] for aid in ids if aid in by_id]
        if not entries:
            raise RuntimeError("图章曲线库为空")
        return cls(entries)

    @classmethod
    def prefit_directory(
        cls,
        stamps_dir: str | Path,
        *,
        cache_dir: str | Path | None = None,
        tex_size: int = DEFAULT_TEX_SIZE,
        resample_n: int = DEFAULT_RESAMPLE_N,
        max_area_rel_err: float = DEFAULT_AREA_REL_ERR,
        workers: int = DEFAULT_WORKERS,
        progress_cb: Callable[[str], None] | None = None,
    ) -> StampCurveLibrary:
        """主动预计算：全量重算目录下全部图章并覆盖写入缓存。"""
        library = StampLibrary(stamps_dir)
        stamps_path = Path(stamps_dir)
        cache = Path(cache_dir) if cache_dir else stamps_path.parent / ".cache" / "stamp_curves"
        return cls.build(
            library,
            asset_ids=None,
            tex_size=tex_size,
            cache_dir=cache,
            force_refit=True,
            resample_n=resample_n,
            max_area_rel_err=max_area_rel_err,
            workers=workers,
            progress_cb=progress_cb,
        )

    @classmethod
    def _compute_entries(
        cls,
        library: StampLibrary,
        ids: list[str],
        *,
        tex_size: int,
        resample_n: int,
        max_area_rel_err: float,
        workers: int,
        sdf_batch: int,
        log: Callable[[str], None],
    ) -> list[StampCurveEntry]:
        """对给定 id 列表做栅格 + SDF + 轮廓拟合（不读缓存）。"""
        dev = get_device()
        log(f"拟合图章曲线：device={dev} workers={workers} tex={tex_size} n={len(ids)}")
        t0 = time.perf_counter()

        jobs: list[tuple[str, Path, int]] = []
        for asset_id in ids:
            try:
                info = library.resolve(asset_id)
                jobs.append((asset_id, info.path, tex_size))
            except FileNotFoundError:
                continue

        rgba_by_id: dict[str, U8Arr] = {}
        n_jobs = len(jobs)
        n_workers = max(1, min(int(workers), n_jobs or 1))
        if n_jobs == 0:
            return []
        if n_workers == 1:
            for i, job in enumerate(jobs):
                aid, rgba, _err = _raster_one(job)
                if rgba is not None:
                    rgba_by_id[aid] = rgba
                if (i + 1) % 16 == 0 or i + 1 == n_jobs:
                    log(f"  栅格 {i + 1}/{n_jobs}")
        else:
            done = 0
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(_raster_one, j) for j in jobs]
                for fut in as_completed(futs):
                    aid, rgba, err = fut.result()
                    done += 1
                    if rgba is not None:
                        rgba_by_id[aid] = rgba
                    elif err:
                        log(f"  跳过 {aid}: {err}")
                    if done % 16 == 0 or done == n_jobs:
                        log(f"  栅格 {done}/{n_jobs}")

        ordered_ids = [aid for aid in ids if aid in rgba_by_id]
        if not ordered_ids:
            return []

        sdf_iters = max(16, min(64, tex_size // 8))
        masks_list: list[NDArray[np.bool_]] = []
        for aid in ordered_ids:
            a = rgba_by_id[aid][:, :, 3]
            masks_list.append(a >= 96)
        sdfs_by_id: dict[str, FloatArr] = {}
        chunk = max(1, int(sdf_batch))
        for start in range(0, len(ordered_ids), chunk):
            batch_ids = ordered_ids[start : start + chunk]
            batch_masks = np.stack(
                [masks_list[start + j].astype(np.float32) for j in range(len(batch_ids))],
                axis=0,
            )
            batch_sdf = batch_mask_to_sdf(batch_masks, device=dev, iters=sdf_iters)
            for j, aid in enumerate(batch_ids):
                sdfs_by_id[aid] = batch_sdf[j].astype(np.float32)
            log(f"  SDF {min(start + chunk, len(ordered_ids))}/{len(ordered_ids)}")

        entries: list[StampCurveEntry] = []
        for i, asset_id in enumerate(ordered_ids):
            mask = masks_list[i]
            if not mask.any():
                continue
            outer_px, holes_px, _, area_err = fit_mask_contour_high_precision(
                mask,
                max_area_rel_err=max_area_rel_err,
                resample_n=resample_n,
                use_cc_holes=True,
            )
            if len(outer_px) < 3:
                ys, xs = np.where(mask)
                x0, x1 = float(xs.min()), float(xs.max())
                y0, y1 = float(ys.min()), float(ys.max())
                outer_px = np.array(
                    [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
                    dtype=np.float64,
                )
                holes_px = []
                area_err = 1.0

            outer_px = resample_closed_contour(outer_px, resample_n)
            holes_px = [resample_closed_contour(h, max(48, min(resample_n, max(32, len(h))))) for h in holes_px if len(h) >= 3]

            outer_n = normalize_contour_to_unit(outer_px, tex_size)
            holes_n = [normalize_contour_to_unit(h, tex_size) for h in holes_px]
            desc = multi_ring_descriptor(outer_px, holes_px, bins=24)
            sdf = sdfs_by_id.get(asset_id)
            if sdf is None:
                sdf = mask_to_sdf_fast(mask, device=dev, iters=sdf_iters).astype(np.float32)

            area = float(mask.mean())
            ys, xs = np.where(mask)
            bw = float(xs.max() - xs.min() + 1)
            bh = float(ys.max() - ys.min() + 1)
            aspect = bw / max(bh, 1.0)
            elong = max(aspect, 1.0 / max(aspect, 1e-6))

            cont_t = to_torch(np.asarray(outer_px, dtype=np.float32), device=dev)
            closed = cont_t
            if torch.linalg.norm(closed[0] - closed[-1]) > 1e-6:
                closed = torch.cat([closed, closed[:1]], dim=0)
            peri = float(torch.linalg.norm(closed[1:] - closed[:-1], dim=1).sum().item())
            area_px = float(mask.sum())
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
                    contour_area_rel_err=float(area_err),
                )
            )
            if (i + 1) % 16 == 0 or i + 1 == len(ordered_ids):
                log(f"  轮廓 {i + 1}/{len(ordered_ids)}")

        hole_n = sum(1 for e in entries if e.n_holes > 0)
        log(f"完成：{len(entries)} 章，含孔 {hole_n}，耗时 {time.perf_counter() - t0:.1f}s，device={dev}")
        return entries

    @staticmethod
    def _save_entry(cache_path: Path, e: StampCurveEntry) -> None:
        """写入单个图章 npz（当前字段全集，无条件分支）。"""
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
            contour_area_rel_err=e.contour_area_rel_err,
            tags=np.array(e.tags, dtype=object),
        )

    @staticmethod
    def _write_index(
        cache_path: Path,
        *,
        assets: list[str],
        tex_size: int,
        resample_n: int,
        max_area_rel_err: float,
    ) -> None:
        """写入缓存索引（仅供查阅，不参与命中判定）。"""
        meta = {
            "tex_size": tex_size,
            "resample_n": resample_n,
            "max_area_rel_err": max_area_rel_err,
            "assets": sorted(assets),
        }
        (cache_path / "stamp_curves_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def _try_load_entry(cls, cache_path: Path, asset_id: str) -> StampCurveEntry | None:
        """
        尝试加载单个图章缓存。

        文件不存在、字段不全或读取失败 → 返回 None（视为无缓存，由调用方重算）。
        不做字段缺省回填，不做旧格式兼容。
        """
        f = cache_path / f"{asset_id}.npz"
        if not f.is_file():
            return None
        try:
            data = np.load(f, allow_pickle=True)
            files = set(data.files)
            if not all(k in files for k in _NPZ_REQUIRED_KEYS):
                return None
            tags_raw = data["tags"]
            tags = [str(t) for t in tags_raw.tolist()] if getattr(tags_raw, "size", 0) else []
            holes: list[FloatArr] = [np.asarray(h, dtype=np.float64) for h in data["holes"]]
            holes_px: list[FloatArr] = [np.asarray(h, dtype=np.float64) for h in data["holes_px"]]
            return StampCurveEntry(
                asset_id=asset_id,
                contour=np.asarray(data["contour"], dtype=np.float64),
                contour_px=np.asarray(data["contour_px"], dtype=np.float64),
                holes=holes,
                holes_px=holes_px,
                descriptor=np.asarray(data["descriptor"], dtype=np.float64),
                sdf=np.asarray(data["sdf"], dtype=np.float32),
                tex_size=int(data["tex_size"]),
                circularity=float(data["circularity"]),
                elongation=float(data["elongation"]),
                area_frac=float(data["area_frac"]),
                tags=tags,
                n_holes=int(data["n_holes"]),
                complexity=float(data["complexity"]),
                contour_area_rel_err=float(data["contour_area_rel_err"]),
            )
        except Exception:
            return None

    def recall(
        self,
        block_descriptor: FloatArr,
        circularity: float,
        elongation: float,
        k: int = 32,
        *,
        prefer_holes: bool | None = None,
        tag_boost: set[str] | None = None,
        region_simplicity: float | None = None,
        region_area_frac: float = 0.0,
        max_complexity: float | None = None,
        min_fill: float = 0.0,
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
            if max_complexity is not None and e.complexity > max_complexity:
                continue
            if e.area_frac < min_fill:
                continue
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
        """主外轮廓 + 全部闭合内环/附加环（归一化坐标）。"""
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
        """将图章轮廓（可选含孔）变换到画布坐标。多环拼接供 Chamfer。"""
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
