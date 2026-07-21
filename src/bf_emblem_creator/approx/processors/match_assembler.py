"""图章匹配装配器：构造式同色覆盖 + 约束补缝 + 特效 + 装配。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.assemble import assemble_layers, composite_layers
from bf_emblem_creator.approx.contour_arcs import ArcPrimitive
from bf_emblem_creator.approx.debug_vis import DebugVisualizer, stamp_layer_rings_on_canvas
from bf_emblem_creator.approx.depth_order import EdgeRole
from bf_emblem_creator.approx.planar_map import face_shape_boundary_points
from bf_emblem_creator.approx.processors.image_processor import ProcessedImage
from bf_emblem_creator.approx.processors.region_partitioner import RegionPartition
from bf_emblem_creator.approx.processors.stamp_loader import StampCatalog
from bf_emblem_creator.approx.processors.stamp_renderer import StampRenderer
from bf_emblem_creator.approx.recipe import AngleMode, StampMatchAssemblerConfig
from bf_emblem_creator.approx.regions import Region
from bf_emblem_creator.approx.special_fx import try_special_fx_layers
from bf_emblem_creator.approx.union_cover import (
    constrained_gap_fill,
    cover_region_with_union_stamps,
    resolve_target_curve,
)
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


def _face_gamma(pmap: object, region: Region) -> FloatArr:
    """Face 曲线真源 Γ_F：SharedEdge after_fit / dense。"""
    from bf_emblem_creator.approx.planar_map import PlanarMap

    if isinstance(pmap, PlanarMap):
        edge_tgt = face_shape_boundary_points(pmap, region.region_id, only_shape=True)
        if len(edge_tgt) >= 4:
            return np.asarray(edge_tgt, dtype=np.float64)
    return resolve_target_curve(region, None)


def _occlusion_carve_layers(
    layers: list[StampLayer],
    pmap: object,
    regions: list[Region],
    curve_lib: object,
    renderer: object,
    *,
    layer_budget: int,
    seed: int,
    uc_max_stamps: int,
) -> list[StampLayer]:
    """
    异色切边轻量修补：对 OCCLUSION_CUT 边，在上层 Face 再补 0～1 枚贴边章。

    首版只对「上层」Face 用 cut 边点云做一次构造覆盖追加。
    """
    from bf_emblem_creator.approx.planar_map import PlanarMap
    from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
    from bf_emblem_creator.approx.torch_render import TorchStampRenderer

    if not isinstance(pmap, PlanarMap):
        return layers
    if not isinstance(curve_lib, StampCurveLibrary) or not isinstance(renderer, TorchStampRenderer):
        return layers
    if len(layers) >= layer_budget:
        return layers

    by_id = {r.region_id: r for r in regions}
    depth_of = {r.region_id: int(r.depth) for r in regions}
    out = list(layers)
    carved = 0
    for e in pmap.edges:
        if carved >= 4 or len(out) >= layer_budget:
            break
        if e.role != EdgeRole.occlusion_cut:
            continue
        fa, fb = int(e.left_face), int(e.right_face)
        if fa < 0 or fb < 0:
            continue
        # 上层 = depth 更大
        up = fa if depth_of.get(fa, 0) >= depth_of.get(fb, 0) else fb
        reg = by_id.get(up)
        if reg is None or max(reg.color_rgb) < 25:
            continue
        poly = np.asarray(e.polyline, dtype=np.float64)
        if len(poly) < 4:
            continue
        remain = layer_budget - len(out)
        more = cover_region_with_union_stamps(
            reg,
            curve_lib,
            renderer,
            target_curve_pts=poly,
            n_particles=96,
            recall_k=16,
            seed=seed + 5000 + e.id,
            prefer_primitive_seed=False,
            refine=True,
            refine_iters=1,
            max_stamps=min(1, uc_max_stamps, remain),
            min_cover=0.55,
            min_cover_gain=0.02,
            max_leak=0.4,
            layer_budget=remain,
            enable_canvas_clip=True,
            max_boundary_chamfer=10.0,
        )
        if more:
            out.append(more[0])
            carved += 1
    return out


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
        """构造式覆盖 → 异色切边 → 约束补缝 → 特效 → 装配截断。"""
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

        allowed = set(cfg.allowed_assets) if cfg.allowed_assets is not None else set(self.catalog.allowed_ids)

        layers: list[StampLayer] = []
        special_assets: list[str] = []
        logs: list[str] = []
        place_budget = int(cfg.max_layers)
        ordered_items = list(depth.ordered)
        uc = cfg.union_cover

        # Face → Γ_F 缓存（约束补缝复用）
        face_curves: dict[int, FloatArr] = {}
        regions_list: list[Region] = [item.region for item in ordered_items]

        # —— Phase 1：同色构造覆盖（全程 after_fit 曲线）——
        n_union_stamps = 0
        for item in ordered_items:
            if len(layers) >= place_budget:
                break
            region = item.region
            if max(region.color_rgb) < 25:
                continue
            gamma = _face_gamma(pmap, region)
            face_curves[region.region_id] = gamma
            remain = place_budget - len(layers)
            if remain <= 0:
                break
            n_stamps_cap = min(int(uc.max_stamps_per_region), remain)
            new_layers = cover_region_with_union_stamps(
                region,
                curve_lib,
                ren,
                primitives=prims,
                target_curve_pts=gamma if len(gamma) >= 4 else None,
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
                enable_canvas_clip=bool(uc.enable_canvas_clip),
                max_boundary_chamfer=float(uc.max_boundary_chamfer),
            )
            for layer in new_layers:
                if allowed and layer.asset not in allowed:
                    continue
                layer = _snap_layer_angle(layer, cfg)
                if cfg.force_uniform_scale:
                    s = max(float(layer.width), float(layer.height))
                    layer = layer.model_copy(update={"width": s, "height": s})
                layers.append(layer)
                n_union_stamps += 1
                if dbg is not None and dbg.enabled:
                    curve_xy = _accepted_match_curve(self.catalog, layer)
                    dbg.save_accepted_match(
                        base_rgb=image_q,
                        region=region,
                        layer=layer,
                        stamp_curve_canvas=curve_xy,
                        layer_index=len(layers) - 1,
                    )

        # —— Phase 2：异色切边 ——
        if uc.enable_occlusion_carve and len(layers) < place_budget:
            before = len(layers)
            layers = _occlusion_carve_layers(
                layers,
                pmap,
                regions_list,
                curve_lib,
                ren,
                layer_budget=place_budget,
                seed=cfg.seed,
                uc_max_stamps=int(uc.max_stamps_per_region),
            )
            layers = [_snap_layer_angle(layer, cfg) for layer in layers]
            if len(layers) > before:
                logs.append(f"occlusion_carve +{len(layers) - before}")

        # —— Phase 3：约束补缝（曲线回绑 Face，禁止自由 r9xxx 轮廓）——
        if uc.enable_constrained_gap_fill and len(layers) < place_budget:
            # 确保每个 region 有曲线缓存
            for reg in regions_list:
                if reg.region_id not in face_curves:
                    face_curves[reg.region_id] = _face_gamma(pmap, reg)
            before = len(layers)
            if dbg is not None and dbg.enabled:
                pred_rgb, pred_a = composite_layers(layers, ren)
                err = (
                    np.linalg.norm(pred_rgb.astype(np.float64) - np.asarray(graph.image_rgb, dtype=np.float64), axis=2) / 255.0
                )
                need = (alpha >= 0.5) & ((pred_a < 0.4) | (err > 0.22))
                dbg.save_residual(err, need)
            layers = constrained_gap_fill(
                layers,
                regions_list,
                face_curves,
                curve_lib,
                ren,
                alpha=alpha,
                target_rgb=np.asarray(graph.image_rgb, dtype=np.uint8),
                max_rounds=max(2, min(8, place_budget - len(layers))),
                max_stamps_per_gap=2,
                layer_budget=place_budget,
                n_particles=max(96, n_particles // 2),
                recall_k=min(cfg.recall_k, 32),
                seed=cfg.seed + 1000,
                min_cover=max(0.78, uc.min_cover - 0.1),
                max_leak=max(0.45, uc.max_leak),
            )
            layers = [_snap_layer_angle(layer, cfg) for layer in layers]
            if len(layers) > before:
                logs.append(f"constrained_gap_fill +{len(layers) - before}")
                if dbg is not None and dbg.enabled:
                    # 仅记录补缝新增层（用归属 region 轮廓可视化）
                    for i, layer in enumerate(layers[before:], start=before):
                        # 找颜色最接近的 region 作 debug 底
                        reg_dbg = regions_list[0] if regions_list else None
                        for reg in regions_list:
                            if str(reg.color_hex).lower() == str(layer.fill).lower():
                                reg_dbg = reg
                                break
                        if reg_dbg is None:
                            continue
                        curve_xy = _accepted_match_curve(self.catalog, layer)
                        dbg.save_accepted_match(
                            base_rgb=image_q,
                            region=reg_dbg,
                            layer=layer,
                            stamp_curve_canvas=curve_xy,
                            layer_index=i,
                        )

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
        layers = [_snap_layer_angle(layer, cfg) for layer in layers]
        if dbg is not None and dbg.enabled:
            pred_rgb, pred_a = composite_layers(layers, ren)
            rgba = np.dstack([pred_rgb, (np.clip(pred_a, 0, 1) * 255).astype(np.uint8)])
            dbg.save_composite("09_assembled", rgba)

        logs.append(
            f"assemble layers={len(layers)} union_phase_stamps={n_union_stamps} "
            f"boundary={float(bscore):.3f} seam_p95={float(seam):.2f} fx={special_assets}"
        )
        return AssemblyResult(
            layers=layers,
            document=EmblemDocument.from_layers(layers),
            boundary_score=float(bscore),
            seam_p95=float(seam),
            special_fx_assets=special_assets,
            log_lines=logs,
        )
