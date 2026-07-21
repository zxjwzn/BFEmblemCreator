"""图章匹配装配器：区域覆盖 + 残差 + 特效 + 装配。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.assemble import assemble_layers, composite_layers
from bf_emblem_creator.approx.color import rgb_to_hex
from bf_emblem_creator.approx.contour_arcs import ArcPrimitive
from bf_emblem_creator.approx.curves import (
    contour_curvature_descriptor,
    fit_mask_contour_high_precision,
    mask_to_sdf,
)
from bf_emblem_creator.approx.debug_vis import DebugVisualizer, stamp_layer_rings_on_canvas
from bf_emblem_creator.approx.planar_map import face_shape_boundary_points
from bf_emblem_creator.approx.processors.image_processor import ProcessedImage
from bf_emblem_creator.approx.processors.region_partitioner import RegionPartition
from bf_emblem_creator.approx.processors.stamp_loader import StampCatalog
from bf_emblem_creator.approx.processors.stamp_renderer import StampRenderer
from bf_emblem_creator.approx.recipe import AngleMode, StampMatchAssemblerConfig
from bf_emblem_creator.approx.regions import Region, _label_ccs
from bf_emblem_creator.approx.special_fx import try_special_fx_layers
from bf_emblem_creator.approx.union_cover import cover_region_with_union_stamps
from bf_emblem_creator.models import EmblemDocument, StampLayer

FloatArr = NDArray[np.floating]
U8Arr = NDArray[np.uint8]


class AssemblyResult(BaseModel):
    """匹配装配输出。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    layers: list[StampLayer] = Field(default_factory=list, description="装配后图章层")
    document: EmblemDocument = Field(..., description="徽章文档")
    boundary_score: float = Field(default=0.0, description="边界一致性")
    seam_p95: float = Field(default=0.0, description="缝宽 95 分位")
    special_fx_assets: list[str] = Field(default_factory=list, description="特效章")
    log_lines: list[str] = Field(default_factory=list, description="日志")


def _accepted_match_curve(catalog: StampCatalog, layer: StampLayer) -> list[FloatArr]:
    """已接受层轮廓到画布。"""
    curve_lib = catalog.get_curve_lib()
    entry = curve_lib.by_id.get(layer.asset)
    if entry is None:
        return []
    rings = curve_lib.all_rings_normalized(layer.asset)
    return stamp_layer_rings_on_canvas(
        rings,
        left=float(layer.left),
        top=float(layer.top),
        width=float(layer.width),
        height=float(layer.height),
        angle_deg=float(layer.angle),
    )


def _snap_layer_angle(layer: StampLayer, cfg: StampMatchAssemblerConfig) -> StampLayer:
    """discrete 角度约束：投影到最近允许角。"""
    if cfg.angle.mode != AngleMode.discrete or not cfg.angle.angles_deg:
        return layer
    allowed = [float(a) % 360.0 for a in cfg.angle.angles_deg]
    ang = float(layer.angle) % 360.0
    # 轴对齐矩形：180≈0、270≈90（等价），先折叠到 [0,180)
    if set(allowed) <= {0.0, 90.0}:
        ang = ang % 180.0
        if ang >= 135.0:
            ang = 0.0
        elif ang >= 45.0:
            ang = 90.0
        else:
            ang = 0.0
        best = ang
    else:
        best = min(allowed, key=lambda a: min(abs(ang - a), 360.0 - abs(ang - a)))
    if abs(best - float(layer.angle) % 360.0) < 1e-6:
        return layer
    return layer.model_copy(update={"angle": best})


class StampMatchAssembler:
    """图章匹配装配器。"""

    def __init__(
        self,
        config: StampMatchAssemblerConfig,
        catalog: StampCatalog,
        renderer: StampRenderer,
    ) -> None:
        self.config = config
        self.catalog = catalog
        self.renderer = renderer

    def assemble(
        self,
        partition: RegionPartition,
        image: ProcessedImage,
        *,
        particles: int | None = None,
        dbg: DebugVisualizer | None = None,
    ) -> AssemblyResult:
        """区域匹配 → 残差补层 → 特效 → 装配截断。"""
        cfg = self.config
        n_particles = int(particles if particles is not None else cfg.n_particles)
        curve_lib = self.catalog.get_curve_lib()
        ren = self.renderer.torch_renderer
        pmap = partition.planar_map
        graph = partition.region_graph
        depth = partition.depth_order
        prims: list[ArcPrimitive] = list(partition.primitives)
        boundary_pts = partition.boundary_points
        image_q = np.asarray(image.image_q, dtype=np.uint8)
        alpha = np.asarray(image.alpha, dtype=np.float64)
        src_rgb = np.asarray(image.src_rgb, dtype=np.uint8)

        # 资产过滤：仅保留 catalog 允许的 id
        allowed = set(cfg.allowed_assets) if cfg.allowed_assets is not None else set(self.catalog.allowed_ids)
        if allowed:
            # StampCurveLibrary 无运行时 filter；靠 subset 已在 loader 限制
            pass

        layers: list[StampLayer] = []
        special_assets: list[str] = []
        logs: list[str] = []
        place_budget = int(cfg.max_layers)
        ordered_items = list(depth.ordered)
        uc = cfg.union_cover

        for item in ordered_items:
            if len(layers) >= place_budget:
                break
            region = item.region
            if max(region.color_rgb) < 25:
                continue
            edge_tgt = face_shape_boundary_points(pmap, region.region_id, only_shape=True)
            if len(edge_tgt) < 4:
                edge_tgt = None
            remain = place_budget - len(layers)
            if remain <= 0:
                break
            n_stamps_cap = min(int(uc.max_stamps_per_region), remain)
            new_layers = cover_region_with_union_stamps(
                region,
                curve_lib,
                ren,
                primitives=prims,
                target_curve_pts=edge_tgt,
                n_particles=n_particles,
                recall_k=cfg.recall_k,
                seed=cfg.seed + region.region_id * 3,
                prefer_primitive_seed=cfg.prefer_primitive_seed,
                refine=cfg.refine,
                refine_iters=cfg.refine_iters,
                max_stamps=n_stamps_cap,
                min_cover=uc.min_cover,
                min_cover_gain=uc.min_cover_gain,
                max_leak=uc.max_leak,
                layer_budget=remain,
            )
            for layer in new_layers:
                if allowed and layer.asset not in allowed:
                    continue
                layer = _snap_layer_angle(layer, cfg)
                if cfg.force_uniform_scale:
                    s = max(float(layer.width), float(layer.height))
                    layer = layer.model_copy(update={"width": s, "height": s})
                layers.append(layer)
                if dbg is not None and dbg.enabled:
                    curve_xy = _accepted_match_curve(self.catalog, layer)
                    dbg.save_accepted_match(
                        base_rgb=image_q,
                        region=region,
                        layer=layer,
                        stamp_curve_canvas=curve_xy,
                        layer_index=len(layers) - 1,
                    )

        residual_cap = place_budget
        residual_rounds = max(2, min(20, residual_cap - len(layers) + 2))
        stall = 0
        rnd = 0
        residual_dumped = False
        while len(layers) < residual_cap and rnd < residual_rounds:
            rnd += 1
            pred_rgb, pred_a = composite_layers(layers, ren)
            tgt_rgb = np.asarray(graph.image_rgb, dtype=np.float64)
            err = np.linalg.norm(pred_rgb.astype(np.float64) - tgt_rgb, axis=2) / 255.0
            need = (alpha >= 0.5) & ((pred_a < 0.4) | (err > 0.22))
            if dbg is not None and dbg.enabled and not residual_dumped:
                dbg.save_residual(err, need)
                residual_dumped = True
            if not need.any():
                break
            ccs = _label_ccs(need.astype(bool), device=ren.device)
            ccs.sort(key=lambda m: -int(m.sum()))
            gained = False
            min_pix = image.meta.canvas_size**2 * 0.0015 * 0.5
            for m in ccs[:4]:
                if len(layers) >= residual_cap:
                    break
                if float(m.sum()) < min_pix:
                    continue
                med = np.median(tgt_rgb[m], axis=0)
                rgb = (int(med[0]), int(med[1]), int(med[2]))
                if max(rgb) < 25:
                    continue
                outer, _holes, rs, _err = fit_mask_contour_high_precision(m, resample_n=96)
                if len(outer) < 3:
                    ys, xs = np.where(m)
                    outer = np.array(
                        [
                            [float(xs.min()), float(ys.min())],
                            [float(xs.max()), float(ys.min())],
                            [float(xs.max()), float(ys.max())],
                            [float(xs.min()), float(ys.max())],
                            [float(xs.min()), float(ys.min())],
                        ],
                        dtype=np.float64,
                    )
                    rs = outer
                ys, xs = np.where(m)
                region = Region(
                    region_id=9000 + rnd,
                    color_hex=rgb_to_hex(rgb),
                    color_rgb=rgb,
                    area_frac=float(m.sum()) / float(m.size),
                    bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                    mask=m,
                    contour=outer,
                    contour_resampled=rs,
                    descriptor=contour_curvature_descriptor(rs if len(rs) >= 8 else outer),
                    sdf=mask_to_sdf(m),
                    depth=len(layers),
                    centroid=(float(xs.mean()), float(ys.mean())),
                )
                remain = residual_cap - len(layers)
                more = cover_region_with_union_stamps(
                    region,
                    curve_lib,
                    ren,
                    primitives=prims,
                    n_particles=max(96, n_particles // 2),
                    recall_k=min(cfg.recall_k, 32),
                    seed=cfg.seed + 1000 + rnd,
                    prefer_primitive_seed=cfg.prefer_primitive_seed,
                    refine=cfg.refine,
                    refine_iters=max(1, cfg.refine_iters),
                    max_stamps=min(2, int(uc.max_stamps_per_region), remain),
                    min_cover=max(0.55, uc.min_cover - 0.15),
                    min_cover_gain=uc.min_cover_gain,
                    max_leak=min(0.55, uc.max_leak + 0.1),
                    layer_budget=remain,
                )
                if not more:
                    continue
                for cand in more:
                    if allowed and cand.asset not in allowed:
                        continue
                    cand = _snap_layer_angle(cand, cfg)
                    layers.append(cand)
                    if dbg is not None and dbg.enabled:
                        curve_xy = _accepted_match_curve(self.catalog, cand)
                        dbg.save_accepted_match(
                            base_rgb=image_q,
                            region=region,
                            layer=cand,
                            stamp_curve_canvas=curve_xy,
                            layer_index=len(layers) - 1,
                        )
                gained = True
                break
            if not gained:
                stall += 1
                if stall >= cfg.stall_patience:
                    break
            else:
                stall = 0

        if cfg.enable_special_fx and len(layers) < cfg.max_layers:
            fx = try_special_fx_layers(
                src_rgb,
                alpha,
                curve_lib,
                ren,
                max_layers=min(2, cfg.max_layers - len(layers)),
                seed=cfg.seed + 99,
            )
            for layer in fx:
                if allowed and layer.asset not in allowed:
                    continue
                layers.append(layer)
                special_assets.append(layer.asset)

        layers, bscore, seam = assemble_layers(
            layers,
            ren,
            prims,
            max_layers=cfg.max_layers,
            target_boundary_pts=boundary_pts,
        )
        # 再投影角度（装配可能未改）
        layers = [_snap_layer_angle(layer, cfg) for layer in layers]
        if dbg is not None and dbg.enabled:
            pred_rgb, pred_a = composite_layers(layers, ren)
            rgba = np.dstack([pred_rgb, (np.clip(pred_a, 0, 1) * 255).astype(np.uint8)])
            dbg.save_composite("09_assembled", rgba)

        logs.append(
            f"assemble layers={len(layers)} boundary={float(bscore):.3f} seam_p95={float(seam):.2f} fx={special_assets}"
        )
        return AssemblyResult(
            layers=layers,
            document=EmblemDocument.from_layers(layers),
            boundary_score=float(bscore),
            seam_p95=float(seam),
            special_fx_assets=special_assets,
            log_lines=logs,
        )
