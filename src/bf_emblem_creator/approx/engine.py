"""ApproxEngine：ModeRecipe 编排五大处理器（唯一管线）。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from bf_emblem_creator.approx.assemble import composite_layers
from bf_emblem_creator.approx.blocks import BlockTarget, ColorBlock, abstract_to_blocks
from bf_emblem_creator.approx.debug_vis import DebugVisualizer
from bf_emblem_creator.approx.label_field import gap_fraction
from bf_emblem_creator.approx.metrics import score_prediction
from bf_emblem_creator.approx.pipeline import ApproxResult
from bf_emblem_creator.approx.processors.image_processor import ImageProcessor
from bf_emblem_creator.approx.processors.match_assembler import StampMatchAssembler
from bf_emblem_creator.approx.processors.region_partitioner import RegionPartitioner
from bf_emblem_creator.approx.processors.stamp_loader import StampLoader
from bf_emblem_creator.approx.processors.stamp_renderer import StampRenderer
from bf_emblem_creator.approx.recipe import ModeRecipe
from bf_emblem_creator.models import EmblemDocument

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]


def _to_block_target(graph: Any, image_q: U8Arr, alpha: FloatArr, meta: Any, canvas_size: int) -> BlockTarget:
    """RegionGraph → BlockTarget。"""
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


class ApproxEngine:
    """按 ModeRecipe 运行完整近似管线。"""

    def __init__(self, recipe: ModeRecipe) -> None:
        self.recipe = recipe

    def run(
        self,
        image: Image.Image | str | Path | U8Arr,
        *,
        n_particles: int | None = None,
        progress_cb: Any | None = None,
    ) -> ApproxResult:
        """执行：Loader → Image → Partition → Assemble。"""
        t0 = time.perf_counter()
        recipe = self.recipe
        dbg = DebugVisualizer(recipe.debug_dir)
        logs: list[str] = []

        def _prog(msg: str) -> None:
            if progress_cb is not None:
                progress_cb(msg)
            else:
                import sys

                print(msg, file=sys.stderr, flush=True)

        loader = StampLoader(recipe.stamp_loader)
        catalog = loader.load(progress_cb=_prog)
        renderer = StampRenderer(recipe.renderer, catalog)
        image_proc = ImageProcessor(recipe.image, mode=recipe.mode)
        processed = image_proc.process(image)

        k = int(recipe.image.num_colors)
        if dbg.enabled:
            dbg.set_k(k, 1)
            pal_rgb = [(int(p.rgb[0]), int(p.rgb[1]), int(p.rgb[2])) for p in processed.palette]
            src_rgb = np.asarray(processed.src_rgb, dtype=np.uint8)
            image_q = np.asarray(processed.image_q, dtype=np.uint8)
            alpha_dbg = np.asarray(processed.alpha, dtype=np.float64)
            labels_dbg = np.asarray(processed.labels, dtype=np.int32)
            dbg.save_source_and_fit(
                source_rgb=src_rgb,
                fitted_rgb=image_q,
                alpha=alpha_dbg,
            )
            dbg.save_planarized(image_q, labels_dbg, pal_rgb)

        partitioner = RegionPartitioner(recipe.region)
        partition = partitioner.partition(processed)

        graph = partition.region_graph
        alpha = np.asarray(processed.alpha, dtype=np.float64)
        gf = gap_fraction(np.asarray(graph.labels, dtype=np.int32), alpha)
        meta = processed.meta.model_copy(update={"gap_frac": float(gf), "seam_p95": 0.0})
        if dbg.enabled:
            image_q = np.asarray(processed.image_q, dtype=np.uint8)
            if partition.edges_before_fit:
                dbg.save_curve_fit_compare(
                    image_q,
                    before_edges=partition.edges_before_fit,
                    after_edges=partition.edges_after_fit,
                )
            dbg.save_regions(image_q, graph.regions)
            dbg.save_contours(image_q, graph.regions)
            dbg.save_depth_order(image_q, partition.depth_order.ordered)
            dbg.save_primitives(image_q, partition.primitives)

        assembler = StampMatchAssembler(recipe.match, catalog, renderer)
        assembly = assembler.assemble(
            partition,
            processed,
            particles=n_particles,
            dbg=dbg if dbg.enabled else None,
        )
        logs.extend(assembly.log_lines)
        meta = meta.model_copy(update={"seam_p95": float(assembly.seam_p95)})

        layers = assembly.layers
        n_layers = len(layers)
        n_regions = len(graph.regions)
        if n_layers > recipe.match.max_layers:
            logs.append(f"装配后层数 {n_layers} 超过 max_layers={recipe.match.max_layers}，已截断")

        image_q = np.asarray(processed.image_q, dtype=np.uint8)
        target = _to_block_target(graph, image_q, alpha, meta, recipe.image.canvas_size)
        prev_rgb, prev_a = composite_layers(layers, renderer.torch_renderer)
        preview_tmp = np.dstack([prev_rgb, (np.clip(prev_a, 0, 1) * 255).astype(np.uint8)])
        sc = score_prediction(
            preview_tmp,
            target,
            n_layers=n_layers,
            target_primitives=partition.primitives,
            pass_sim=recipe.match.pass_sim,
            pass_line=recipe.match.pass_line,
            pass_overall=recipe.match.pass_score,
        )
        b = float(np.clip(assembly.boundary_score, 0.0, 1.0))
        logs.append(
            f"mode={recipe.mode.value} num_colors={k} layers={n_layers} regions={n_regions} "
            f"overall={sc.overall:.3f} boundary={b:.3f} gap={meta.gap_frac:.4f} "
            f"seam_p95={assembly.seam_p95:.2f} resample={meta.resample} "
            f"used_colors={meta.num_colors} fx={assembly.special_fx_assets}"
        )
        if dbg.enabled:
            dbg.save_k_preview(preview_tmp, overall=sc.overall, rank=sc.overall, layers=n_layers)

        debug_paths = [str(p) for p in dbg.saved]
        stop_reason = f"完成（mode={recipe.mode.value}，num_colors={k}，{n_layers} 层，色块={n_regions}）"
        device = str(renderer.device)

        if n_layers == 0:
            stop_reason = "未能放置任何图章"
            empty_target = abstract_to_blocks(image, recipe)
            cs = recipe.image.canvas_size
            preview_arr = np.zeros((cs, cs, 4), dtype=np.uint8)
            score = score_prediction(
                preview_arr,
                empty_target,
                n_layers=0,
                pass_sim=recipe.match.pass_sim,
                pass_line=recipe.match.pass_line,
                pass_overall=recipe.match.pass_score,
            )
            return ApproxResult(
                document=EmblemDocument.from_layers([]),
                target=empty_target,
                score=score,
                preview_rgb=preview_arr,
                device=device,
                elapsed_sec=float(time.perf_counter() - t0),
                stop_reason=stop_reason,
                blocks_found=0,
                k_used=k,
                log_lines=logs,
                debug_images=debug_paths,
                mode=recipe.mode.value,
            )

        rgb_t, a_t = renderer.composite_layers(layers)
        preview_arr = renderer.to_numpy_rgba(rgb_t, a_t)
        if dbg.enabled:
            dbg.save_final(preview_arr)
            debug_paths = [str(p) for p in dbg.saved]
        score = score_prediction(
            preview_arr,
            target,
            n_layers=len(layers),
            target_primitives=partition.primitives,
            pass_sim=recipe.match.pass_sim,
            pass_line=recipe.match.pass_line,
            pass_overall=recipe.match.pass_score,
        )
        return ApproxResult(
            document=assembly.document,
            target=target,
            score=score,
            preview_rgb=preview_arr,
            device=device,
            elapsed_sec=float(time.perf_counter() - t0),
            stop_reason=stop_reason,
            blocks_found=n_regions,
            k_used=k,
            boundary_score=float(assembly.boundary_score),
            special_fx_assets=assembly.special_fx_assets,
            log_lines=logs,
            debug_images=debug_paths,
            mode=recipe.mode.value,
        )
