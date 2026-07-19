"""v3 主循环：平面化 → 层序 → 弧基元 → 曲线匹配 → 特效 → 预算环。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.assemble import assemble_layers, composite_layers
from bf_emblem_creator.approx.blocks import BlockTarget, ColorBlock, abstract_to_blocks
from bf_emblem_creator.approx.budget_loop import BudgetState, initial_k, is_coarse_phase, next_k
from bf_emblem_creator.approx.contour_arcs import ArcPrimitive, extract_all_primitives
from bf_emblem_creator.approx.debug_vis import DebugVisualizer, stamp_layer_curve_on_canvas
from bf_emblem_creator.approx.depth_order import DepthOrderResult, infer_depth_order
from bf_emblem_creator.approx.device import get_device
from bf_emblem_creator.approx.label_field import gap_fraction
from bf_emblem_creator.approx.match_curve import match_region_with_particles, refine_layer_particles
from bf_emblem_creator.approx.metrics import FullScoreReport, score_prediction
from bf_emblem_creator.approx.models import ApproxConfig, ApproxMeta
from bf_emblem_creator.approx.planar_map import (
    PlanarMap,
    assert_planar_map_valid,
    build_planar_map,
    face_shape_boundary_points,
    planar_map_to_region_graph,
    shared_shape_points,
)
from bf_emblem_creator.approx.planarize import planarize_image
from bf_emblem_creator.approx.regions import Region, RegionGraph, _label_ccs, build_regions
from bf_emblem_creator.approx.special_fx import try_special_fx_layers
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
from bf_emblem_creator.approx.torch_render import TorchStampRenderer, stamp_layer_to_dict
from bf_emblem_creator.models import EmblemDocument, StampLayer
from bf_emblem_creator.stamps import StampLibrary

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]


class ApproxResultV2(BaseModel):
    """v3 近似结果（保留 V2 类名以兼容导入）。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    document: EmblemDocument
    target: BlockTarget
    score: FullScoreReport
    preview_rgb: Any = None
    device: str = "cpu"
    elapsed_sec: float = 0.0
    stop_reason: str = Field(default="", description="停止原因")
    blocks_found: int = Field(default=0, description="区域/色块数")
    k_used: int = Field(default=0, description="最终色量 K")
    boundary_score: float = Field(default=0.0, description="可见边界一致性")
    special_fx_assets: list[str] = Field(default_factory=list, description="特效章列表")
    log_lines: list[str] = Field(default_factory=list, description="迭代日志")
    debug_images: list[str] = Field(default_factory=list, description="已写入的调试图路径")


def _curve_cache_dir(cfg: ApproxConfig) -> Path:
    if cfg.stamp_curve_cache is not None:
        return Path(cfg.stamp_curve_cache)
    return Path(cfg.stamps_dir).parent / ".cache" / "stamp_curves"


def _accepted_match_curve(
    curve_lib: StampCurveLibrary,
    layer: StampLayer,
) -> FloatArr | None:
    """已接受层的外轮廓变换到画布（供调试图；失败则 None）。"""
    entry = curve_lib.by_id.get(layer.asset)
    if entry is None:
        return None
    local = np.asarray(entry.contour, dtype=np.float64)
    if local.ndim != 2 or len(local) < 2:
        return None
    return stamp_layer_curve_on_canvas(
        local,
        left=float(layer.left),
        top=float(layer.top),
        width=float(layer.width),
        height=float(layer.height),
        angle_deg=float(layer.angle),
    )


def _run_once(
    *,
    k: int,
    image: Image.Image | str | Path | U8Arr,
    cfg: ApproxConfig,
    curve_lib: StampCurveLibrary,
    renderer: TorchStampRenderer,
    particles: int,
    dbg: DebugVisualizer | None = None,
) -> tuple[
    list[StampLayer],
    list[ArcPrimitive],
    RegionGraph,
    DepthOrderResult,
    U8Arr,
    FloatArr,
    U8Arr,
    ApproxMeta,
    float,
    list[str],
    float,
]:
    """
    单次 K：平面化→标签场→平面图/区域→层序→弧→匹配→特效→装配。

    粗阶段（小 K）：只放面积最大的若干区域，优先完整形体；细阶段再残差补层。
    若启用 dbg：写各主阶段图；匹配仅写通过并入层的结果。
    """
    labels, palette, alpha, image_q, meta, src_rgb = planarize_image(image, cfg, k=k, device=renderer.device)
    if dbg is not None and dbg.enabled:
        pal_rgb = [(int(p.rgb[0]), int(p.rgb[1]), int(p.rgb[2])) for p in palette]
        dbg.save_source_and_fit(source_rgb=src_rgb, fitted_rgb=image_q, alpha=alpha)
        dbg.save_planarized(image_q, labels, pal_rgb)

    coarse = is_coarse_phase(k, cfg)
    max_reg = cfg.coarse_max_regions if coarse else min(40, max(12, k * 4))
    min_area = max(0.004 if coarse else 0.0015, cfg.min_region_area_frac)

    pmap: PlanarMap | None = None
    if cfg.use_planar_map:
        pmap = build_planar_map(
            labels,
            palette,
            alpha,
            min_area_frac=min_area,
            max_faces=max_reg,
            max_contour_area_rel_err=cfg.max_contour_area_rel_err,
            gap_frac=float(meta.gap_frac),
            edge_subpixel=bool(cfg.edge_subpixel),
        )
        assert_planar_map_valid(pmap)
        graph = planar_map_to_region_graph(pmap, palette)
        gf = gap_fraction(np.asarray(graph.labels, dtype=np.int32), alpha)
        meta = meta.model_copy(update={"gap_frac": float(gf)})
    else:
        graph = build_regions(
            labels,
            palette,
            alpha,
            min_area_frac=min_area,
            max_regions=max_reg,
            max_contour_area_rel_err=cfg.max_contour_area_rel_err,
            device=renderer.device,
            enforce_no_gap=cfg.enforce_no_gap,
        )
        gf = gap_fraction(np.asarray(graph.labels, dtype=np.int32), alpha)
        meta = meta.model_copy(update={"gap_frac": float(gf)})

    if dbg is not None and dbg.enabled:
        dbg.save_regions(image_q, graph.regions)
        dbg.save_contours(image_q, graph.regions)

    depth = infer_depth_order(graph, planar_map=pmap)
    if dbg is not None and dbg.enabled:
        dbg.save_depth_order(image_q, depth.ordered)

    prims = extract_all_primitives(depth, eps_arc=cfg.eps_arc, planar_map=pmap)
    if dbg is not None and dbg.enabled:
        dbg.save_primitives(image_q, prims)

    boundary_pts = shared_shape_points(pmap) if pmap is not None else None

    layers: list[StampLayer] = []
    special_assets: list[str] = []
    # 放置顺序：底→顶，但粗阶段先按面积从大到小（主形体优先）
    ordered_items = list(depth.ordered)
    if coarse:
        ordered_items = sorted(ordered_items, key=lambda it: -it.region.area_frac)

    place_budget = min(cfg.max_layers, cfg.coarse_max_regions if coarse else cfg.max_layers)
    for item in ordered_items:
        if len(layers) >= place_budget:
            break
        region = item.region
        if max(region.color_rgb) < 25:
            continue
        # 粗阶段跳过过小碎片
        if coarse and region.area_frac < max(0.015, cfg.min_region_area_frac * 3):
            continue
        # Batch E：匹配目标曲线 = 该 Face 关联 SharedEdge 去重点（每 edge_id 一次）
        edge_tgt = None
        if pmap is not None:
            edge_tgt = face_shape_boundary_points(pmap, region.region_id, only_shape=True)
            if len(edge_tgt) < 4:
                edge_tgt = None
        layer = match_region_with_particles(
            region,
            curve_lib,
            renderer,
            primitives=prims,
            n_particles=particles,
            recall_k=cfg.recall_k,
            seed=cfg.seed + region.region_id * 3,
            prefer_primitive_seed=cfg.prefer_primitive_seed,
            target_curve_pts=edge_tgt,
        )
        if layer is None:
            continue
        if cfg.refine:
            layer = refine_layer_particles(
                layer,
                region,
                curve_lib,
                renderer,
                n=max(24, cfg.refine_iters * 12),
                seed=cfg.seed + 7 + region.region_id,
                target_curve_pts=edge_tgt,
            )
        layers.append(layer)
        # 仅输出通过的匹配（不 dump 粒子搜索）
        if dbg is not None and dbg.enabled:
            curve_xy = _accepted_match_curve(curve_lib, layer)
            dbg.save_accepted_match(
                base_rgb=image_q,
                region=region,
                layer=layer,
                stamp_curve_canvas=curve_xy,
                layer_index=len(layers) - 1,
            )

    # 细阶段残差补层；粗阶段只做极少补层（保主结构）
    residual_cap = cfg.max_layers if not coarse else min(cfg.max_layers, len(layers) + 2)
    residual_rounds = max(2, min(20, residual_cap - len(layers) + 2)) if not coarse else min(3, residual_cap - len(layers))
    stall = 0
    rnd = 0
    residual_dumped = False
    while len(layers) < residual_cap and rnd < residual_rounds:
        rnd += 1
        pred_rgb, pred_a = composite_layers(layers, renderer)
        tgt_rgb = np.asarray(graph.image_rgb, dtype=np.float64)
        err = np.linalg.norm(pred_rgb.astype(np.float64) - tgt_rgb, axis=2) / 255.0
        need = (alpha >= 0.5) & ((pred_a < 0.4) | (err > 0.22))
        if dbg is not None and dbg.enabled and not residual_dumped:
            dbg.save_residual(err, need)
            residual_dumped = True
        if not need.any():
            break
        ccs = _label_ccs(need.astype(bool), device=renderer.device)
        ccs.sort(key=lambda m: -int(m.sum()))
        gained = False
        for m in ccs[:4]:
            if len(layers) >= residual_cap:
                break
            if float(m.sum()) < cfg.min_region_area_frac * cfg.canvas_size**2 * (0.8 if coarse else 0.5):
                continue
            med = np.median(tgt_rgb[m], axis=0)
            rgb = (int(med[0]), int(med[1]), int(med[2]))
            if max(rgb) < 25:
                continue
            from bf_emblem_creator.approx.color import rgb_to_hex
            from bf_emblem_creator.approx.curves import (
                contour_curvature_descriptor,
                extract_outer_contour,
                mask_to_sdf,
                resample_closed_contour,
            )

            cont = extract_outer_contour(m, simplify=0.6)
            if cont is None or len(cont) < 3:
                ys, xs = np.where(m)
                cont = np.array(
                    [
                        [float(xs.min()), float(ys.min())],
                        [float(xs.max()), float(ys.min())],
                        [float(xs.max()), float(ys.max())],
                        [float(xs.min()), float(ys.max())],
                        [float(xs.min()), float(ys.min())],
                    ],
                    dtype=np.float64,
                )
            ys, xs = np.where(m)
            region = Region(
                region_id=9000 + rnd,
                color_hex=rgb_to_hex(rgb),
                color_rgb=rgb,
                area_frac=float(m.sum()) / float(m.size),
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                mask=m,
                contour=cont,
                contour_resampled=resample_closed_contour(cont, 64),
                descriptor=contour_curvature_descriptor(cont),
                sdf=mask_to_sdf(m),
                depth=len(layers),
                centroid=(float(xs.mean()), float(ys.mean())),
            )
            cand = match_region_with_particles(
                region,
                curve_lib,
                renderer,
                primitives=prims,
                n_particles=max(96, particles // 2),
                recall_k=min(cfg.recall_k, 32),
                seed=cfg.seed + 1000 + rnd,
                prefer_primitive_seed=cfg.prefer_primitive_seed,
            )
            if cand is None:
                continue
            layers.append(cand)
            if dbg is not None and dbg.enabled:
                curve_xy = _accepted_match_curve(curve_lib, cand)
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

    # 特效章仅细阶段 + 有空余层
    if cfg.enable_special_fx and (not coarse) and len(layers) < cfg.max_layers:
        fx = try_special_fx_layers(
            src_rgb,
            alpha,
            curve_lib,
            renderer,
            max_layers=min(2, cfg.max_layers - len(layers)),
            seed=cfg.seed + 99,
        )
        for layer in fx:
            layers.append(layer)
            special_assets.append(layer.asset)
            if dbg is not None and dbg.enabled:
                # 特效无真实区域：用占位 Region 仅供标注位置
                empty_mask = np.zeros((cfg.canvas_size, cfg.canvas_size), dtype=bool)
                fx_region = Region(
                    region_id=-1,
                    color_hex=str(layer.fill),
                    color_rgb=(0, 0, 0),
                    area_frac=0.0,
                    bbox=(0, 0, 1, 1),
                    mask=empty_mask,
                    contour=np.zeros((0, 2), dtype=np.float64),
                    contour_resampled=np.zeros((0, 2), dtype=np.float64),
                    descriptor=np.zeros(8, dtype=np.float64),
                    sdf=None,
                    depth=len(layers) - 1,
                    centroid=(float(layer.left), float(layer.top)),
                )
                dbg.save_accepted_match(
                    base_rgb=image_q,
                    region=fx_region,
                    layer=layer,
                    stamp_curve_canvas=_accepted_match_curve(curve_lib, layer),
                    layer_index=len(layers) - 1,
                )

    layers, bscore, seam = assemble_layers(
        layers,
        renderer,
        prims,
        max_layers=cfg.max_layers,
        target_boundary_pts=boundary_pts,
    )
    meta = meta.model_copy(update={"seam_p95": float(seam)})
    if dbg is not None and dbg.enabled:
        pred_rgb, pred_a = composite_layers(layers, renderer)
        rgba = np.dstack([pred_rgb, (np.clip(pred_a, 0, 1) * 255).astype(np.uint8)])
        dbg.save_composite("09_assembled", rgba)
    return layers, prims, graph, depth, image_q, alpha, src_rgb, meta, bscore, special_assets, float(seam)


def _to_block_target(
    graph: RegionGraph,
    image_q: U8Arr,
    alpha: FloatArr,
    meta: ApproxMeta,
    canvas_size: int,
) -> BlockTarget:
    """RegionGraph → 兼容评分的 BlockTarget。"""
    blocks: list[ColorBlock] = []
    for r in graph.regions:
        blocks.append(
            ColorBlock(
                color_hex=r.color_hex,
                color_rgb=r.color_rgb,
                area_frac=r.area_frac,
                bbox=r.bbox,
                mask=r.mask,
                contour=r.contour,
                contour_resampled=r.contour_resampled,
                descriptor=r.descriptor,
                sdf=r.sdf,
                depth_hint=r.depth,
            )
        )
    gray = image_q.astype(np.float64).mean(axis=2) / 255.0
    edge = np.zeros_like(gray)
    edge[:, :-1] = np.maximum(edge[:, :-1], np.abs(gray[:, 1:] - gray[:, :-1]))
    edge[:-1, :] = np.maximum(edge[:-1, :], np.abs(gray[1:, :] - gray[:-1, :]))
    if edge.max() > 1e-8:
        edge = edge / edge.max()
    weight = np.maximum(alpha, 0.05) * (0.6 + 0.8 * edge)
    weight = weight / (float(weight.mean()) + 1e-8)
    return BlockTarget(
        image_rgb=image_q,
        alpha=alpha.astype(np.float64),
        weight=weight.astype(np.float64),
        blocks=blocks,
        meta=meta,
        canvas_size=canvas_size,
    )


def approximate_image(
    image: Image.Image | str | Path | U8Arr,
    config: ApproxConfig | None = None,
    *,
    n_particles: int | None = None,
    use_cuda: bool | None = None,
) -> ApproxResultV2:
    """
    v3 近似：

    1. 多级 K 平面化
    2. 区域邻接 + 层序 π
    3. 弧基元 + 曲线匹配（放大/旋转/出画布）
    4. 特殊图章渐变通道
    5. 空余层加密 K 重跑
    """
    import time

    t0 = time.perf_counter()
    cfg = config or ApproxConfig()
    particles = int(n_particles if n_particles is not None else cfg.n_particles)
    prefer_cuda = bool(cfg.use_cuda if use_cuda is None else use_cuda)
    device = get_device(prefer_cuda=prefer_cuda)

    library = StampLibrary(cfg.stamps_dir)
    cache_dir = _curve_cache_dir(cfg)
    curve_lib = StampCurveLibrary.build(
        library,
        cfg.stamp_subset,
        tex_size=256,
        cache_dir=cache_dir,
        force_refit=False,
    )
    renderer = TorchStampRenderer(
        library,
        canvas_size=cfg.canvas_size,
        stamp_tex_size=256,
        device=device,
    )

    dbg = DebugVisualizer(cfg.debug_dir)
    logs: list[str] = []
    k = initial_k(cfg)
    best_layers: list[StampLayer] = []
    best_score_val = -1.0
    best_pack: tuple[list[ArcPrimitive], RegionGraph, U8Arr, FloatArr, ApproxMeta, float, list[str]] | None = None
    stop_reason = "未知"
    iteration = 0

    while True:
        iteration += 1
        if dbg.enabled:
            dbg.set_k(k, iteration)
        layers, prims, graph, _, image_q, alpha, _, meta, bscore, special, seam = _run_once(
            k=k,
            image=image,
            cfg=cfg,
            curve_lib=curve_lib,
            renderer=renderer,
            particles=particles,
            dbg=dbg if dbg.enabled else None,
        )
        target = _to_block_target(graph, image_q, alpha, meta, cfg.canvas_size)
        # 快速预览打分
        prev_rgb, prev_a = composite_layers(layers, renderer)
        preview_tmp = np.dstack([prev_rgb, (np.clip(prev_a, 0, 1) * 255).astype(np.uint8)])
        sc = score_prediction(
            preview_tmp,
            target,
            n_layers=len(layers),
            target_primitives=prims,
            pass_sim=cfg.pass_sim,
            pass_line=cfg.pass_line,
            pass_overall=cfg.pass_score,
        )
        # 通用选优：sim + 边界一致性为主；gap/seam 作惩罚
        b = float(np.clip(bscore, 0.0, 1.0))
        rank = 0.38 * sc.sim.score + 0.22 * sc.sim.edge_score + 0.28 * b + 0.08 * sc.line.score + 0.04 * sc.overall
        if len(layers) < max(4, cfg.max_layers // 6) and sc.sim.score < 0.72:
            rank -= 0.06
        if float(meta.gap_frac) > 1e-6:
            rank -= 0.25
        if seam > 6.0:
            rank -= 0.04 * min((seam - 6.0) / 10.0, 1.0)
        logs.append(
            f"K={k} iter={iteration} layers={len(layers)} overall={sc.overall:.3f} "
            f"rank={rank:.3f} boundary={b:.3f} gap={meta.gap_frac:.4f} seam_p95={seam:.2f} "
            f"resample={meta.resample} regions={len(graph.regions)} fx={special}"
        )
        if dbg.enabled:
            dbg.save_k_preview(preview_tmp, overall=sc.overall, rank=rank, layers=len(layers))
        if rank > best_score_val:
            best_score_val = rank
            best_layers = layers
            best_pack = (prims, graph, image_q, alpha, meta, bscore, special)

        state = BudgetState(k=k, iteration=iteration, n_layers=len(layers), score=sc.overall)
        nk = next_k(state, cfg)
        if nk is None:
            if sc.overall >= cfg.pass_score:
                stop_reason = f"评分达标（K={k}，{len(layers)} 层）"
            elif len(layers) >= cfg.max_layers:
                stop_reason = f"达到 max_layers={cfg.max_layers}"
            elif iteration >= cfg.k_max_iters:
                stop_reason = f"K 外环用尽（{iteration} 次）"
            else:
                stop_reason = f"无空余层可加密（{len(layers)} 层，margin={cfg.n_margin}）"
            break
        k = nk

    debug_paths = [str(p) for p in dbg.saved]

    if best_pack is None:
        stop_reason = "未能放置任何图章"
        empty_target = abstract_to_blocks(image, cfg)
        document = EmblemDocument.from_layers([])
        # 空文档：GPU 空白画布
        cs = cfg.canvas_size
        preview_arr = np.zeros((cs, cs, 4), dtype=np.uint8)
        score = score_prediction(
            preview_arr,
            empty_target,
            n_layers=0,
            pass_sim=cfg.pass_sim,
            pass_line=cfg.pass_line,
            pass_overall=cfg.pass_score,
        )
        return ApproxResultV2(
            document=document,
            target=empty_target,
            score=score,
            preview_rgb=preview_arr,
            device=str(device),
            elapsed_sec=float(time.perf_counter() - t0),
            stop_reason=stop_reason,
            blocks_found=0,
            k_used=k,
            log_lines=logs,
            debug_images=debug_paths,
        )

    prims, graph, image_q, alpha, meta, bscore, special = best_pack
    target = _to_block_target(graph, image_q, alpha, meta, cfg.canvas_size)
    document = EmblemDocument.from_layers(best_layers)
    # 终局预览 + 评分：全程 GPU torch 合成（与内环一致）
    rgb_t, a_t = renderer.composite_layers([stamp_layer_to_dict(layer) for layer in best_layers])
    preview_arr = renderer.to_numpy_rgba(rgb_t, a_t)
    if dbg.enabled:
        dbg.save_final(preview_arr)
        debug_paths = [str(p) for p in dbg.saved]
    score = score_prediction(
        preview_arr,
        target,
        n_layers=len(best_layers),
        target_primitives=prims,
        pass_sim=cfg.pass_sim,
        pass_line=cfg.pass_line,
        pass_overall=cfg.pass_score,
    )
    return ApproxResultV2(
        document=document,
        target=target,
        score=score,
        preview_rgb=preview_arr,
        device=str(device),
        elapsed_sec=float(time.perf_counter() - t0),
        stop_reason=stop_reason,
        blocks_found=len(graph.regions),
        k_used=k,
        boundary_score=float(bscore),
        special_fx_assets=special,
        log_lines=logs,
        debug_images=debug_paths,
    )


def approximate_to_files(
    image: Image.Image | str | Path,
    out_json: str | Path,
    out_preview: str | Path | None = None,
    config: ApproxConfig | None = None,
) -> ApproxResultV2:
    """近似并写文件。"""
    result = approximate_image(image, config)
    result.document.save_json(out_json)
    if out_preview is not None and result.preview_rgb is not None:
        Path(out_preview).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.preview_rgb, mode="RGBA").save(out_preview)
    return result
