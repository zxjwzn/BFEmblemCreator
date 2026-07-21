"""区域划分器：平面图 + 可选贝塞尔 + 层序 + 基元。"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.contour_arcs import extract_all_primitives
from bf_emblem_creator.approx.depth_order import infer_depth_order
from bf_emblem_creator.approx.edge_curve_fit import (
    EdgeCurveFitReport,
    collect_edge_polylines,
    simplify_planar_map_curves,
)
from bf_emblem_creator.approx.label_field import gap_fraction
from bf_emblem_creator.approx.planar_map import (
    assert_planar_map_valid,
    build_planar_map,
    planar_map_to_region_graph,
    shared_shape_points,
)
from bf_emblem_creator.approx.processors.image_processor import ProcessedImage
from bf_emblem_creator.approx.recipe import BoundaryPolicy, RegionPartitionerConfig


class RegionPartition(BaseModel):
    """区域划分输出。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    planar_map: Any = Field(..., description="PlanarMap")
    region_graph: Any = Field(..., description="RegionGraph")
    depth_order: Any = Field(..., description="DepthOrderResult")
    primitives: list[Any] = Field(default_factory=list, description="弧/折线基元列表")
    boundary_points: Any = Field(default=None, description="共享可见边界点云")
    curve_fit_report: EdgeCurveFitReport | None = Field(default=None, description="贝塞尔拟合摘要")
    edges_before_fit: list[Any] = Field(default_factory=list, description="拟合前边折线调试")
    edges_after_fit: list[Any] = Field(default_factory=list, description="拟合后边折线调试")


class RegionPartitioner:
    """区域划分器。"""

    def __init__(self, config: RegionPartitionerConfig) -> None:
        self.config = config

    def partition(self, image: ProcessedImage) -> RegionPartition:
        """标签场 → 平面图 → 可选曲线拟合 → 层序 → 基元。"""
        labels = np.asarray(image.labels, dtype=np.int32)
        alpha = np.asarray(image.alpha, dtype=np.float64)
        max_faces = min(40, max(self.config.max_faces, image.meta.num_colors * 3 if image.meta.num_colors else 8))
        min_area = max(0.0015, self.config.min_area_frac)
        pmap = build_planar_map(
            labels,
            image.palette,
            alpha,
            min_area_frac=min_area,
            max_faces=max_faces,
            gap_frac=float(image.meta.gap_frac),
            edge_subpixel=bool(self.config.edge_subpixel),
        )
        assert_planar_map_valid(pmap)

        edges_before = collect_edge_polylines(pmap)
        fit_report: EdgeCurveFitReport | None = None
        if self.config.boundary_policy == BoundaryPolicy.curve_fit:
            cf = self.config.curve_fit
            fit_report = simplify_planar_map_curves(
                pmap,
                max_vertices=int(cf.max_vertices),
                min_arc_length_px=float(cf.min_arc_length_px),
                line_flat_eps_px=float(cf.line_flat_eps_px),
                corner_deg=float(cf.corner_deg),
                samples_per_seg=int(cf.samples_per_seg),
                smooth_radius_px=float(cf.smooth_radius_px),
                min_anchor_spacing_px=float(cf.min_anchor_spacing_px),
            )
            assert_planar_map_valid(pmap)
        edges_after = collect_edge_polylines(pmap)

        graph = planar_map_to_region_graph(pmap, image.palette)
        gf = gap_fraction(np.asarray(graph.labels, dtype=np.int32), alpha)
        # 写回 meta 由 engine 处理；此处返回 partition
        depth = infer_depth_order(graph, planar_map=pmap)
        prims = extract_all_primitives(depth, eps_arc=self.config.eps_arc, planar_map=pmap)
        bpts = shared_shape_points(pmap)
        _ = gf
        return RegionPartition(
            planar_map=pmap,
            region_graph=graph,
            depth_order=depth,
            primitives=list(prims),
            boundary_points=bpts,
            curve_fit_report=fit_report,
            edges_before_fit=edges_before,
            edges_after_fit=edges_after,
        )
