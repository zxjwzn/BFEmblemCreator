"""近似管线逐步调试可视化：各阶段写 PNG，重复粒子匹配仅输出通过者。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from bf_emblem_creator.approx.contour_arcs import ArcPrimitive, PrimitiveType
from bf_emblem_creator.approx.depth_order import OrderedRegion
from bf_emblem_creator.approx.regions import Region
from bf_emblem_creator.models import StampLayer

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]
BoolArr = NDArray[np.bool_]

# 区域/深度/基元调色板（循环使用）
_PALETTE: list[tuple[int, int, int]] = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
]

_PRIM_COLORS: dict[str, tuple[int, int, int]] = {
    PrimitiveType.line.value: (255, 80, 80),
    PrimitiveType.circle_arc.value: (80, 200, 120),
    PrimitiveType.ellipse_arc.value: (80, 160, 255),
    PrimitiveType.free.value: (200, 200, 80),
}


def _as_rgb_u8(arr: NDArray[np.generic] | U8Arr, size: int | None = None) -> U8Arr:
    """图像数组 → RGB uint8。"""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    out = np.clip(a, 0, 255).astype(np.uint8)
    if size is not None and (out.shape[0] != size or out.shape[1] != size):
        img = Image.fromarray(out, mode="RGB").resize((size, size), Image.Resampling.NEAREST)
        out = np.asarray(img, dtype=np.uint8)
    return out


def _as_rgba_u8(arr: NDArray[np.generic] | U8Arr) -> U8Arr:
    """图像数组 → RGBA uint8。"""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a, a, a, np.full_like(a, 255)], axis=-1)
    elif a.shape[-1] == 3:
        a = np.dstack([a, np.full(a.shape[:2], 255, dtype=a.dtype)])
    return np.clip(a, 0, 255).astype(np.uint8)


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    pts: NDArray[np.floating] | Sequence[Sequence[float]],
    color: tuple[int, int, int],
    *,
    width: int = 2,
    closed: bool = False,
) -> None:
    """在 PIL Draw 上画折线。"""
    p = np.asarray(pts, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 2:
        return
    xy = [(float(x), float(y)) for x, y in p]
    if closed and xy[0] != xy[-1]:
        xy = [*xy, xy[0]]
    draw.line(xy, fill=color, width=width)


def _blend_mask(base: U8Arr, mask: BoolArr, color: tuple[int, int, int], alpha: float = 0.45) -> U8Arr:
    """半透明色块叠到底图。"""
    out = base.astype(np.float64)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return base
    for c in range(3):
        ch = out[..., c]
        ch[m] = ch[m] * (1.0 - alpha) + float(color[c]) * alpha
        out[..., c] = ch
    return np.clip(out, 0, 255).astype(np.uint8)


class DebugVisualizer:
    """
    逐步调试图写盘器。

    - 主阶段（平面化、区域、轮廓、基元、合成）各写一张；
    - 图章匹配只写**通过并入层**的结果（曲线叠加），不 dump 粒子搜索过程。
    """

    def __init__(self, root: str | Path | None) -> None:
        self.root: Path | None = Path(root) if root is not None else None
        self._seq = 0
        self._k_prefix = ""
        self.saved: list[Path] = []
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        """是否启用写盘。"""
        return self.root is not None

    def set_k(self, k: int, iteration: int) -> None:
        """进入某次 K 外环迭代时的子目录前缀。"""
        if not self.enabled:
            return
        self._k_prefix = f"k{k:02d}_iter{iteration:02d}"
        if self.root is None:
            return
        (self.root / self._k_prefix).mkdir(parents=True, exist_ok=True)

    def _next_path(self, name: str) -> Path | None:
        if not self.enabled or self.root is None:
            return None
        self._seq += 1
        safe = name.replace("/", "_").replace("\\", "_")
        rel = f"{self._seq:03d}_{safe}.png"
        if self._k_prefix:
            return self.root / self._k_prefix / rel
        return self.root / rel

    def save_rgb(self, name: str, rgb: NDArray[np.generic] | U8Arr) -> Path | None:
        """保存 RGB 图。"""
        path = self._next_path(name)
        if path is None:
            return None
        arr = _as_rgb_u8(rgb)
        Image.fromarray(arr, mode="RGB").save(path)
        self.saved.append(path)
        return path

    def save_rgba(self, name: str, rgba: NDArray[np.generic] | U8Arr) -> Path | None:
        """保存 RGBA 图。"""
        path = self._next_path(name)
        if path is None:
            return None
        arr = _as_rgba_u8(rgba)
        Image.fromarray(arr, mode="RGBA").save(path)
        self.saved.append(path)
        return path

    def save_source_and_fit(
        self,
        *,
        source_rgb: NDArray[np.generic] | U8Arr,
        fitted_rgb: NDArray[np.generic] | U8Arr,
        alpha: NDArray[np.floating] | None = None,
    ) -> None:
        """原图概括：源画布与 alpha 蒙版。"""
        self.save_rgb("00_source_fitted", fitted_rgb)
        if alpha is not None:
            a = np.asarray(alpha, dtype=np.float64)
            if a.max() <= 1.0 + 1e-6:
                a = a * 255.0
            self.save_rgb("00_alpha_mask", np.clip(a, 0, 255).astype(np.uint8))
        src = _as_rgb_u8(source_rgb)
        self.save_rgb("00_source_raw", src)

    def save_planarized(
        self,
        image_q: NDArray[np.generic] | U8Arr,
        labels: NDArray[np.integer],
        palette_rgb: list[tuple[int, int, int]],
    ) -> None:
        """平面化结果 + 标签伪彩。"""
        self.save_rgb("01_planarized", image_q)
        lab = np.asarray(labels)
        h, w = lab.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        uniq = [int(u) for u in np.unique(lab) if int(u) >= 0]
        for i, lid in enumerate(uniq):
            c = palette_rgb[i] if i < len(palette_rgb) else _PALETTE[i % len(_PALETTE)]
            vis[lab == lid] = c
        self.save_rgb("01_labels_falsecolor", vis)

    def save_regions(self, base_rgb: NDArray[np.generic] | U8Arr, regions: Iterable[Region]) -> None:
        """区域蒙版叠加 + 外轮廓（与 SharedEdge / after_fit 同源的 Region.contour）。"""
        base = _as_rgb_u8(base_rgb)
        overlay = base.copy()
        items = list(regions)
        for i, reg in enumerate(items):
            color = _PALETTE[i % len(_PALETTE)]
            mask = np.asarray(reg.mask, dtype=bool)
            overlay = _blend_mask(overlay, mask, color, alpha=0.4)
        img2 = Image.fromarray(overlay, mode="RGB")
        draw2 = ImageDraw.Draw(img2)
        for i, reg in enumerate(items):
            color = _PALETTE[i % len(_PALETTE)]
            # 完整共享边轮廓（curve_fit 后即贝塞尔环）；过密时等弧抽稀仅影响显示
            raw = reg.contour if reg.contour is not None else reg.contour_resampled
            pts = None
            if raw is not None:
                arr = np.asarray(raw, dtype=np.float64)
                if len(arr) >= 2:
                    if len(arr) > 160:
                        idx = np.linspace(0, len(arr) - 1, 128).astype(int)
                        pts = arr[idx]
                    else:
                        pts = arr
            if pts is not None and len(pts) >= 2:
                _draw_polyline(draw2, pts, color, width=2, closed=True)
            cx, cy = reg.centroid
            err = reg.contour_area_rel_err
            draw2.text(
                (float(cx) + 2, float(cy) + 2),
                f"r{reg.region_id} e={err:.1%}",
                fill=(255, 255, 255),
            )
        path = self._next_path("02_regions")
        if path is None:
            return
        img2.save(path)
        self.saved.append(path)

    def save_depth_order(
        self,
        base_rgb: NDArray[np.generic] | U8Arr,
        ordered: Iterable[OrderedRegion],
    ) -> None:
        """层序：深度编码色 + 标注 depth。"""
        base = _as_rgb_u8(base_rgb)
        items = list(ordered)
        n = max(len(items), 1)
        overlay = base.copy()
        for item in items:
            reg = item.region
            depth = int(item.depth)
            # 底→顶：蓝→红
            t = depth / max(n - 1, 1)
            color = (int(40 + 200 * t), int(80 + 40 * (1 - t)), int(220 * (1 - t)))
            mask = np.asarray(reg.mask, dtype=bool)
            overlay = _blend_mask(overlay, mask, color, alpha=0.5)
        img = Image.fromarray(overlay, mode="RGB")
        draw = ImageDraw.Draw(img)
        for item in items:
            reg = item.region
            depth = int(item.depth)
            cx, cy = reg.centroid
            draw.text((float(cx), float(cy)), f"d{depth}/r{reg.region_id}", fill=(255, 255, 255))
        path = self._next_path("03_depth_order")
        if path is None:
            return
        img.save(path)
        self.saved.append(path)

    def save_edge_polylines(
        self,
        base_rgb: NDArray[np.generic] | U8Arr,
        edges: Sequence[tuple[int, NDArray[np.floating], bool]],
        *,
        name: str,
        width: int = 2,
    ) -> Path | None:
        """
        绘制共享边折线集合。

        edges: [(edge_id, polyline Nx2, closed), ...]
        用于对比贝塞尔拟合前 dense 与拟合后几何。
        """
        base = _as_rgb_u8(base_rgb)
        img = Image.fromarray(base.copy(), mode="RGB")
        draw = ImageDraw.Draw(img)
        for i, (_eid, poly, closed) in enumerate(edges):
            color = _PALETTE[i % len(_PALETTE)]
            pts = np.asarray(poly, dtype=np.float64)
            if len(pts) < 2:
                continue
            _draw_polyline(draw, pts, color, width=width, closed=bool(closed))
        path = self._next_path(name)
        if path is None:
            return None
        img.save(path)
        self.saved.append(path)
        return path

    def save_curve_fit_compare(
        self,
        base_rgb: NDArray[np.generic] | U8Arr,
        *,
        before_edges: Sequence[tuple[int, NDArray[np.floating], bool]],
        after_edges: Sequence[tuple[int, NDArray[np.floating], bool]],
    ) -> None:
        """贝塞尔拟合前 dense 边界 + 拟合后边界（两张图）。"""
        # 同线宽便于对比；拟合后应用更疏的几何
        self.save_edge_polylines(base_rgb, before_edges, name="01c_edges_dense_before_fit", width=1)
        self.save_edge_polylines(base_rgb, after_edges, name="01d_edges_bezier_after_fit", width=1)

    def save_contours(self, base_rgb: NDArray[np.generic] | U8Arr, regions: Iterable[Region]) -> None:
        """区域外轮廓：彩色=完整共享边环，白线=描述子重采样（均与 after_fit 同源）。"""
        base = _as_rgb_u8(base_rgb)
        img = Image.fromarray(base.copy(), mode="RGB")
        draw = ImageDraw.Draw(img)
        for i, reg in enumerate(regions):
            color = _PALETTE[i % len(_PALETTE)]
            if reg.contour is not None and len(np.asarray(reg.contour)) >= 2:
                _draw_polyline(draw, reg.contour, color, width=1, closed=True)
            if reg.contour_resampled is not None and len(np.asarray(reg.contour_resampled)) >= 2:
                _draw_polyline(draw, reg.contour_resampled, (255, 255, 255), width=2, closed=True)
        path = self._next_path("04_contours")
        if path is None:
            return
        img.save(path)
        self.saved.append(path)

    def save_primitives(
        self,
        base_rgb: NDArray[np.generic] | U8Arr,
        prims: Iterable[ArcPrimitive],
    ) -> None:
        """弧基元：按类型着色（line/circle/ellipse/free）。"""
        base = _as_rgb_u8(base_rgb)
        img = Image.fromarray(base.copy(), mode="RGB")
        draw = ImageDraw.Draw(img)
        for p in prims:
            key = p.type.value
            color = _PRIM_COLORS.get(key, (180, 180, 180))
            w = 3 if p.hard else 2
            _draw_polyline(draw, p.sample_points, color, width=w, closed=False)
            params = p.params
            if key in {PrimitiveType.circle_arc.value, PrimitiveType.ellipse_arc.value} and "cx" in params and "cy" in params:
                cx, cy = float(params["cx"]), float(params["cy"])
                if "r" in params:
                    r = float(params["r"])
                elif "a" in params:
                    r = float(params["a"])
                else:
                    r = 8.0
                draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline=color, width=1)
                draw.text((cx + 4, cy + 4), f"{key[:4]} r={r:.0f}", fill=color)
        path = self._next_path("05_primitives")
        if path is None:
            return
        img.save(path)
        self.saved.append(path)

    def save_accepted_match(
        self,
        *,
        base_rgb: NDArray[np.generic] | U8Arr,
        region: Region,
        layer: StampLayer,
        stamp_curve_canvas: FloatArr | Sequence[FloatArr] | None,
        layer_index: int,
        score_hint: float | None = None,
    ) -> None:
        """
        仅输出**已接受**的匹配：区域轮廓（绿）+ 图章曲线含孔（品红）+ 层摘要。

        ``stamp_curve_canvas`` 可为单条折线，或外环+内孔的多环列表。
        粒子搜索过程不写盘。
        """
        base = _as_rgb_u8(base_rgb)
        img = Image.fromarray(base.copy(), mode="RGB")
        draw = ImageDraw.Draw(img)
        if region.contour is not None:
            _draw_polyline(draw, region.contour, (40, 220, 80), width=2, closed=True)
        for ring in _as_curve_rings(stamp_curve_canvas):
            # 外轮廓与内孔同色（品红）
            _draw_polyline(draw, ring, (240, 40, 200), width=2, closed=True)
        asset = layer.asset
        rid = region.region_id
        left = float(layer.left)
        top = float(layer.top)
        w = float(layer.width)
        h = float(layer.height)
        ang = float(layer.angle)
        sc = f" sc={score_hint:.3f}" if score_hint is not None else ""
        label = f"L{layer_index} r{rid} {asset} {w:.0f}x{h:.0f} @{ang:.0f}°{sc}"
        draw.text((max(2, left - w * 0.2), max(2, top - h * 0.2)), label, fill=(255, 255, 255))
        # 中心十字
        draw.line((left - 6, top, left + 6, top), fill=(255, 255, 0), width=1)
        draw.line((left, top - 6, left, top + 6), fill=(255, 255, 0), width=1)
        name = f"06_match_ok_L{layer_index:02d}_r{rid}_{asset}"
        path = self._next_path(name)
        if path is None:
            return
        img.save(path)
        self.saved.append(path)

    def save_composite(self, name: str, rgba: NDArray[np.generic] | U8Arr) -> None:
        """当前合成结果（阶段快照，非逐粒子）。"""
        self.save_rgba(name, rgba)

    def save_residual(
        self,
        err_map: NDArray[np.floating],
        need_mask: BoolArr | None = None,
    ) -> None:
        """残差热力 + 待补区域。"""
        e = np.asarray(err_map, dtype=np.float64)
        if e.ndim == 3:
            e = np.linalg.norm(e, axis=2)
        e = e / (float(e.max()) + 1e-8)
        heat = np.zeros((*e.shape, 3), dtype=np.uint8)
        heat[..., 0] = np.clip(e * 255, 0, 255).astype(np.uint8)
        heat[..., 1] = np.clip((1.0 - e) * 80, 0, 255).astype(np.uint8)
        heat[..., 2] = np.clip((1.0 - e) * 200, 0, 255).astype(np.uint8)
        self.save_rgb("07_residual_heat", heat)
        if need_mask is not None:
            m = np.asarray(need_mask, dtype=bool)
            vis = heat.copy()
            vis[m] = (255, 255, 0)
            self.save_rgb("07_residual_need", vis)

    def save_k_preview(
        self,
        rgba: NDArray[np.generic] | U8Arr,
        *,
        overall: float,
        rank: float,
        layers: int,
    ) -> None:
        """单次 K 迭代预览（带分数文件名）。"""
        name = f"08_preview_L{layers}_ov{overall:.3f}_rk{rank:.3f}"
        self.save_rgba(name, rgba)

    def save_final(self, rgba: NDArray[np.generic] | U8Arr) -> None:
        """全局最优终局预览。"""
        # 跳出 K 子目录
        old = self._k_prefix
        self._k_prefix = ""
        self.save_rgba("99_final_preview", rgba)
        self._k_prefix = old


def stamp_layer_curve_on_canvas(
    local_contour: FloatArr,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    angle_deg: float,
) -> FloatArr:
    """
    将归一化局部轮廓（约 [-0.5,0.5]）按层参数变换到画布坐标。

    与 match_curve.transform_stamp_contour_batch 几何一致，供调试图叠加。
    """
    pts = np.asarray(local_contour, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    # 缩放
    xy = pts.copy()
    xy[:, 0] = xy[:, 0] * width
    xy[:, 1] = xy[:, 1] * height
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    # 顺时针角（与 match_curve 一致）
    x, y = xy[:, 0], xy[:, 1]
    xr = c * x + s * y
    yr = -s * x + c * y
    return np.stack([xr + left, yr + top], axis=1)


def stamp_layer_rings_on_canvas(
    local_rings: Sequence[FloatArr],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    angle_deg: float,
) -> list[FloatArr]:
    """将多环（外轮廓 + 内孔）变换到画布坐标，跳过过短环。"""
    out: list[FloatArr] = []
    for ring in local_rings:
        pts = stamp_layer_curve_on_canvas(
            np.asarray(ring, dtype=np.float64),
            left=left,
            top=top,
            width=width,
            height=height,
            angle_deg=angle_deg,
        )
        if len(pts) >= 2:
            out.append(pts)
    return out


def _as_curve_rings(curves: FloatArr | Sequence[FloatArr] | None) -> list[FloatArr]:
    """将单环或环列表规范为有效折线列表。"""
    if curves is None:
        return []
    # 单条 (N,2) ndarray
    if isinstance(curves, np.ndarray):
        arr = np.asarray(curves, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
            return [arr]
        return []
    rings: list[FloatArr] = []
    for item in curves:
        arr = np.asarray(item, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
            rings.append(arr)
    return rings
