"""Batch C：共享边缘平面图（Vertex / SharedEdge / HalfEdge / Face）。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.color import rgb_to_hex
from bf_emblem_creator.approx.curves import (
    contour_curvature_descriptor,
    fit_mask_contour_area_constrained,
    mask_to_sdf,
    rdp,
    resample_closed_contour,
)
from bf_emblem_creator.approx.depth_order import EdgeRole
from bf_emblem_creator.approx.models import PaletteColor
from bf_emblem_creator.approx.regions import AdjacencyEdge, Region, RegionGraph

FloatArr = NDArray[np.floating]
BoolArr = NDArray[np.bool_]
I32Arr = NDArray[np.int32]
U8Arr = NDArray[np.uint8]


class MapVertex(BaseModel):
    """平面图结点。"""

    model_config = ConfigDict(extra="forbid")

    id: int
    x: float
    y: float


class SharedEdge(BaseModel):
    """两 Face（或 Face 与背景）之间的唯一几何边。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: int
    v0: int
    v1: int
    left_face: int = Field(..., description="左侧 Face id；背景 -1")
    right_face: int = Field(..., description="右侧 Face id；背景 -1")
    polyline: Any = Field(..., description="(N,2) v0→v1")
    length: float = Field(default=0.0, ge=0.0)
    role: EdgeRole = EdgeRole.shape_boundary


class HalfEdge(BaseModel):
    """有向半边。"""

    model_config = ConfigDict(extra="forbid")

    id: int
    edge_id: int
    direction: int = Field(..., description="+1 沿 polyline，-1 逆序")
    face_id: int
    next_id: int = -1


class MapFace(BaseModel):
    """色区 Face。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: int
    region_id: int
    label: int
    color_hex: str
    color_rgb: tuple[int, int, int]
    halfedge_start: int = -1
    hole_starts: list[int] = Field(default_factory=list)
    mask: Any
    area_frac: float = 0.0
    centroid: tuple[float, float] = (0.0, 0.0)
    bbox: tuple[int, int, int, int] = (0, 0, 1, 1)


class PlanarMap(BaseModel):
    """共享边平面图。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    vertices: list[MapVertex] = Field(default_factory=list)
    edges: list[SharedEdge] = Field(default_factory=list)
    halfedges: list[HalfEdge] = Field(default_factory=list)
    faces: list[MapFace] = Field(default_factory=list)
    labels: Any
    image_rgb: Any
    alpha: Any
    canvas_size: int = 320
    gap_frac: float = 0.0

    def edge_by_id(self) -> dict[int, SharedEdge]:
        """edge id → SharedEdge。"""
        return {e.id: e for e in self.edges}

    def face_by_id(self) -> dict[int, MapFace]:
        """face id → MapFace。"""
        return {f.id: f for f in self.faces}


def _poly_length(pts: FloatArr) -> float:
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _face_boundary_polylines(mask: BoolArr) -> list[FloatArr]:
    """
    Face 边界折线（像素中心 Moore 外轮廓 + 孔）。

    MVP：用面积约束轮廓作为外环几何源；孔暂用 extract。
    """
    from bf_emblem_creator.approx.curves import extract_all_contours

    rings = extract_all_contours(mask, simplify=0.0, min_points=4, min_area=1.0)
    if not rings:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return []
        x0, x1 = float(xs.min()), float(xs.max()) + 1.0
        y0, y1 = float(ys.min()), float(ys.max()) + 1.0
        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)
        return [box]
    return [r.numpy() for r in rings]


def _link_halfedge_cycles(pmap: PlanarMap) -> None:
    """
    为每条半边设置 next_id，使同一 face 的半边形成闭合环（C4）。

    策略：按 face 收集半边；在边端点处用「端点最近」连接出/入弧，
    若失败则按半边起点极角排序串成单环。
    """
    if not pmap.halfedges:
        return
    by_id = {he.id: he for he in pmap.halfedges}
    e_by = pmap.edge_by_id()
    v_by = {v.id: v for v in pmap.vertices}

    def _he_endpoints(he: HalfEdge) -> tuple[int, int]:
        e = e_by[he.edge_id]
        if he.direction >= 0:
            return e.v0, e.v1
        return e.v1, e.v0

    def _pt(vid: int) -> tuple[float, float]:
        v = v_by[vid]
        return float(v.x), float(v.y)

    # face → halfedge ids
    face_hes: dict[int, list[int]] = {}
    for he in pmap.halfedges:
        face_hes.setdefault(he.face_id, []).append(he.id)

    for face_id, he_ids in face_hes.items():
        if not he_ids:
            continue
        if len(he_ids) == 1:
            only = by_id[he_ids[0]]
            only.next_id = only.id
            f = pmap.face_by_id().get(face_id)
            if f is not None:
                f.halfedge_start = only.id
            continue

        # 端点 → 以该点为起点的半边
        out_at: dict[int, list[int]] = {}
        for hid in he_ids:
            s, _t = _he_endpoints(by_id[hid])
            out_at.setdefault(s, []).append(hid)

        used_next: set[int] = set()
        for hid in he_ids:
            he = by_id[hid]
            _s, t = _he_endpoints(he)
            cands = [c for c in out_at.get(t, []) if c != hid]
            if len(cands) == 1:
                he.next_id = cands[0]
                used_next.add(cands[0])
            elif len(cands) > 1:
                # 多候选：取与当前边出射方向转角最小者
                e = e_by[he.edge_id]
                poly = np.asarray(e.polyline, dtype=np.float64)
                if he.direction < 0:
                    poly = poly[::-1]
                dcur = poly[-1] - poly[-2] if len(poly) >= 2 else np.array([1.0, 0.0])
                best_c, best_ang = cands[0], 1e9
                for c in cands:
                    e2 = e_by[by_id[c].edge_id]
                    poly2 = np.asarray(e2.polyline, dtype=np.float64)
                    if by_id[c].direction < 0:
                        poly2 = poly2[::-1]
                    d2 = poly2[1] - poly2[0] if len(poly2) >= 2 else np.array([1.0, 0.0])
                    n1 = float(np.linalg.norm(dcur)) + 1e-9
                    n2 = float(np.linalg.norm(d2)) + 1e-9
                    cos_a = float(np.clip(np.dot(dcur, d2) / (n1 * n2), -1.0, 1.0))
                    ang = float(np.arccos(cos_a))
                    if ang < best_ang:
                        best_ang, best_c = ang, c
                he.next_id = best_c
                used_next.add(best_c)
            else:
                he.next_id = -1

        # 未接上的用极角环兜底
        orphan = [hid for hid in he_ids if by_id[hid].next_id < 0]
        if orphan or len(used_next) < len(he_ids) - 1:
            # 按起点极角排序串环
            fobj = pmap.face_by_id().get(face_id)
            cx_f, cy_f = fobj.centroid if fobj is not None else (0.0, 0.0)

            def _ang(hid: int, cx0: float = cx_f, cy0: float = cy_f) -> float:
                s, _ = _he_endpoints(by_id[hid])
                x, y = _pt(s)
                return float(np.arctan2(y - cy0, x - cx0))

            ordered = sorted(he_ids, key=_ang)
            for i, hid in enumerate(ordered):
                by_id[hid].next_id = ordered[(i + 1) % len(ordered)]

        f = pmap.face_by_id().get(face_id)
        if f is not None:
            f.halfedge_start = he_ids[0]


def refine_edges_subpixel(pmap: PlanarMap, *, iters: int = 2) -> None:
    """
    Batch D1：沿标签 SDF 法向微调 SharedEdge 折线（拓扑 id / 端点 Vertex 不变）。

    端点锁定；内部点沿近似法向移向 |SDF| 更小处。
    """
    face_masks = {f.id: np.asarray(f.mask, dtype=bool) for f in pmap.faces if np.asarray(f.mask).any()}
    if not face_masks:
        return
    # 合并所有 face 为「主体」SDF：边界在标签突变处
    h = int(pmap.canvas_size)
    # 用 face_id 标签图
    lab = np.asarray(pmap.labels, dtype=np.int32)
    # 对每条内部边，用两侧 mask 的对称差分边界
    for e in pmap.edges:
        poly = np.asarray(e.polyline, dtype=np.float64).copy()
        if len(poly) < 3:
            continue
        # 选参考 mask：优先 left face
        ref_id = e.left_face if e.left_face >= 0 else e.right_face
        if ref_id < 0 or ref_id not in face_masks:
            continue
        sdf = mask_to_sdf(face_masks[ref_id])
        for _ in range(max(1, iters)):
            for i in range(1, len(poly) - 1):
                # 切向 → 法向
                t = poly[i + 1] - poly[i - 1]
                nrm = float(np.linalg.norm(t))
                if nrm < 1e-9:
                    continue
                nx, ny = -t[1] / nrm, t[0] / nrm
                x0, y0 = float(poly[i, 0]), float(poly[i, 1])
                best = (x0, y0)
                best_abs = 1e18
                for s in (-1.25, -0.75, -0.35, 0.0, 0.35, 0.75, 1.25):
                    x = x0 + s * nx
                    y = y0 + s * ny
                    ix = int(np.clip(round(x), 0, h - 1))
                    iy = int(np.clip(round(y), 0, lab.shape[0] - 1))
                    val = abs(float(sdf[iy, ix]))
                    if val < best_abs:
                        best_abs = val
                        best = (x, y)
                poly[i, 0], poly[i, 1] = best
        # 锁端点到 Vertex
        v0 = next(v for v in pmap.vertices if v.id == e.v0)
        v1 = next(v for v in pmap.vertices if v.id == e.v1)
        poly[0, 0], poly[0, 1] = v0.x, v0.y
        poly[-1, 0], poly[-1, 1] = v1.x, v1.y
        e.polyline = poly
        e.length = _poly_length(poly)


def build_planar_map(
    labels: I32Arr,
    palette: list[PaletteColor],
    alpha: FloatArr,
    *,
    min_area_frac: float = 0.002,
    max_faces: int = 48,
    max_contour_area_rel_err: float = 0.03,
    gap_frac: float = 0.0,
    edge_subpixel: bool = False,
) -> PlanarMap:
    """
    从无洞标签场构建 PlanarMap。

    MVP 策略：
    - 每连通域一个 Face；
    - 外轮廓作为与背景的 SharedEdge（或简化为单环边）；
    - 邻接 Face 对之间用边界像素链估计共享折线（唯一几何）；
    - Region 轮廓由 face 环派生，避免双侧独立 RDP。
    """
    lab = np.asarray(labels, dtype=np.int32)
    a = np.asarray(alpha, dtype=np.float64)
    h, w = lab.shape
    subject = a >= 0.5
    canvas_area = float(h * w)
    min_area = min_area_frac * canvas_area

    # Face = 每标签连通域（小域应在 label_field 已并入；此处再兜底合并）
    faces: list[MapFace] = []
    face_id_map = np.full((h, w), -1, dtype=np.int32)
    fid = 0
    image_q = np.zeros((h, w, 3), dtype=np.uint8)
    max_lab = int(lab.max()) if lab.size and lab.max() >= 0 else -1
    from bf_emblem_creator.approx.label_field import _label_ccs

    for li in range(max_lab + 1):
        base = (lab == li) & subject
        if not base.any():
            continue
        rgb = palette[li].rgb if li < len(palette) else (128, 128, 128)
        hex_c = palette[li].hex if li < len(palette) else rgb_to_hex(rgb)
        for cc in _label_ccs(base):
            area = float(cc.sum())
            if area < min_area:
                continue
            ys, xs = np.where(cc)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            cx, cy = float(xs.mean()), float(ys.mean())
            faces.append(
                MapFace(
                    id=fid,
                    region_id=fid,
                    label=li,
                    color_hex=hex_c,
                    color_rgb=(int(rgb[0]), int(rgb[1]), int(rgb[2])),
                    mask=cc,
                    area_frac=area / canvas_area,
                    centroid=(cx, cy),
                    bbox=bbox,
                )
            )
            face_id_map[cc] = fid
            image_q[cc] = np.array(rgb, dtype=np.uint8)
            fid += 1

    faces.sort(key=lambda f: -f.area_frac)
    if len(faces) > max_faces:
        # 超出：小 Face 像素并入邻接大 Face（保持无洞）
        keep = faces[:max_faces]
        keep_ids = {f.id for f in keep}
        for f in faces[max_faces:]:
            m = np.asarray(f.mask, dtype=bool)
            # 找邻接 keep face
            ys, xs = np.where(m)
            votes: dict[int, int] = {}
            for y, x in zip(ys, xs, strict=False):
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w:
                        t = int(face_id_map[ny, nx])
                        if t in keep_ids:
                            votes[t] = votes.get(t, 0) + 1
            target = max(votes.items(), key=lambda kv: kv[1])[0] if votes else keep[0].id
            face_id_map[m] = target
            tgt = next(ff for ff in keep if ff.id == target)
            new_mask = np.asarray(tgt.mask, dtype=bool) | m
            tgt.mask = new_mask
            image_q[m] = np.array(tgt.color_rgb, dtype=np.uint8)
        faces = keep
        # 重算面积
        for f in faces:
            m = np.asarray(f.mask, dtype=bool)
            f.area_frac = float(m.sum()) / canvas_area
            if m.any():
                ys, xs = np.where(m)
                f.centroid = (float(xs.mean()), float(ys.mean()))
                f.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    # 共享边：扫描像素对偶，聚合 face 对边界中点
    pair_pts: dict[tuple[int, int], list[list[float]]] = {}

    def _add_pair(a: int, b: int, x: float, y: float) -> None:
        if a == b:
            return
        key = (min(a, b), max(a, b))
        pair_pts.setdefault(key, []).append([x, y])

    for y in range(h):
        for x in range(w - 1):
            la, lb = int(face_id_map[y, x]), int(face_id_map[y, x + 1])
            if la != lb:
                _add_pair(la, lb, x + 1.0, y + 0.5)
    for y in range(h - 1):
        for x in range(w):
            la, lb = int(face_id_map[y, x]), int(face_id_map[y + 1, x])
            if la != lb:
                _add_pair(la, lb, x + 0.5, y + 1.0)

    vertices: list[MapVertex] = []
    edges: list[SharedEdge] = []
    halfedges: list[HalfEdge] = []
    vid = 0
    eid = 0
    hid = 0

    def _vertex(x: float, y: float) -> int:
        nonlocal vid
        vertices.append(MapVertex(id=vid, x=x, y=y))
        i = vid
        vid += 1
        return i

    # 为每个 face 建外轮廓边（背景）+ 内部共享边
    face_outer: dict[int, FloatArr] = {}
    for f in faces:
        cont, _, _ = fit_mask_contour_area_constrained(
            np.asarray(f.mask, dtype=bool),
            max_area_rel_err=max_contour_area_rel_err,
            resample_n=128,
        )
        if cont is None or len(cont) < 3:
            rings = _face_boundary_polylines(np.asarray(f.mask, dtype=bool))
            cont = rings[0] if rings else np.zeros((0, 2), dtype=np.float64)
        face_outer[f.id] = np.asarray(cont, dtype=np.float64)

    for (fa, fb), pts in pair_pts.items():
        arr = np.asarray(pts, dtype=np.float64)
        if len(arr) < 2:
            continue
        # 沿主方向排序成折线
        c = arr.mean(axis=0)
        q = arr - c
        cov = q.T @ q
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, int(np.argmax(eigvals))]
        t = q @ direction
        order = np.argsort(t)
        poly = arr[order]
        # 端点
        v0 = _vertex(float(poly[0, 0]), float(poly[0, 1]))
        v1 = _vertex(float(poly[-1, 0]), float(poly[-1, 1]))
        # 轻微 RDP（锁端点）
        if len(poly) > 8:
            simp = rdp(poly, 0.75)
            if len(simp) >= 2:
                simp[0] = poly[0]
                simp[-1] = poly[-1]
                poly = simp
        edges.append(
            SharedEdge(
                id=eid,
                v0=v0,
                v1=v1,
                left_face=fa,
                right_face=fb,
                polyline=poly,
                length=_poly_length(poly),
                role=EdgeRole.shape_boundary,
            )
        )
        halfedges.append(HalfEdge(id=hid, edge_id=eid, direction=1, face_id=fa, next_id=-1))
        halfedges.append(HalfEdge(id=hid + 1, edge_id=eid, direction=-1, face_id=fb, next_id=-1))
        hid += 2
        eid += 1

    # 每个 face 与背景的外轮廓边（若无内部邻接覆盖全部边界，仍保留 outer 作匹配目标）
    for f in faces:
        outer = face_outer.get(f.id)
        if outer is None or len(outer) < 3:
            continue
        v0 = _vertex(float(outer[0, 0]), float(outer[0, 1]))
        v1 = _vertex(float(outer[-1, 0]), float(outer[-1, 1]))
        edges.append(
            SharedEdge(
                id=eid,
                v0=v0,
                v1=v1,
                left_face=f.id,
                right_face=-1,
                polyline=outer,
                length=_poly_length(outer),
                role=EdgeRole.shape_boundary,
            )
        )
        he = HalfEdge(id=hid, edge_id=eid, direction=1, face_id=f.id, next_id=-1)
        halfedges.append(he)
        f.halfedge_start = hid
        hid += 1
        eid += 1

    pmap = PlanarMap(
        vertices=vertices,
        edges=edges,
        halfedges=halfedges,
        faces=faces,
        labels=face_id_map,
        image_rgb=image_q,
        alpha=a,
        canvas_size=h,
        gap_frac=float(gap_frac),
    )
    _link_halfedge_cycles(pmap)
    if edge_subpixel:
        refine_edges_subpixel(pmap)
    return pmap


def assert_planar_map_valid(pmap: PlanarMap) -> None:
    """校验基本不变量；失败抛 ValueError。"""
    eids = {e.id for e in pmap.edges}
    he_ids = {he.id for he in pmap.halfedges}
    if len(eids) != len(pmap.edges):
        raise ValueError("edge id 重复")
    if len(he_ids) != len(pmap.halfedges):
        raise ValueError("halfedge id 重复")
    for he in pmap.halfedges:
        if he.edge_id not in eids:
            raise ValueError(f"半边引用不存在边 {he.edge_id}")
        if he.next_id not in he_ids:
            raise ValueError(f"半边 {he.id} 的 next_id={he.next_id} 无效")
    for e in pmap.edges:
        poly = np.asarray(e.polyline, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 2:
            raise ValueError(f"边 {e.id} polyline 非法")
    # 每个 face 从 halfedge_start 走环应回到起点，且覆盖该 face 全部半边
    he_by = {he.id: he for he in pmap.halfedges}
    for f in pmap.faces:
        if f.halfedge_start < 0:
            continue
        seen: list[int] = []
        cur = f.halfedge_start
        for _ in range(len(pmap.halfedges) + 2):
            if cur in seen:
                break
            seen.append(cur)
            cur = he_by[cur].next_id
            if cur == f.halfedge_start:
                break
        if cur != f.halfedge_start:
            raise ValueError(f"face {f.id} 半边环未闭合")
        face_hes = [he.id for he in pmap.halfedges if he.face_id == f.id]
        # 允许多环（孔）时 seen 可能是外环子集；至少外环非空
        if set(seen) != set(face_hes) and not seen:
            raise ValueError(f"face {f.id} 半边环为空")


def walk_face_halfedges(pmap: PlanarMap, face_id: int) -> list[HalfEdge]:
    """沿 next_id 遍历 face 外环半边（闭合一圈）。"""
    f = pmap.face_by_id().get(face_id)
    if f is None or f.halfedge_start < 0:
        return []
    he_by = {he.id: he for he in pmap.halfedges}
    out: list[HalfEdge] = []
    cur = f.halfedge_start
    for _ in range(len(pmap.halfedges) + 2):
        he = he_by[cur]
        out.append(he)
        cur = he.next_id
        if cur == f.halfedge_start:
            break
    return out


def face_contour(pmap: PlanarMap, face_id: int) -> FloatArr:
    """
    派生 Face 外轮廓：优先半边环拼接共享边几何（共边唯一）；

    否则回退背景 outer 边 / mask 拟合。
    """
    f = pmap.face_by_id().get(face_id)
    if f is None:
        return np.zeros((0, 2), dtype=np.float64)
    e_by = pmap.edge_by_id()
    ring = walk_face_halfedges(pmap, face_id)
    if ring:
        chunks: list[FloatArr] = []
        for he in ring:
            e = e_by[he.edge_id]
            poly = np.asarray(e.polyline, dtype=np.float64)
            if he.direction < 0:
                poly = poly[::-1].copy()
            if chunks and len(poly) > 0 and float(np.linalg.norm(chunks[-1][-1] - poly[0])) < 1e-6:
                poly = poly[1:]
            if len(poly):
                chunks.append(poly)
        if chunks:
            cont = np.vstack(chunks)
            if len(cont) >= 2 and float(np.linalg.norm(cont[0] - cont[-1])) > 1e-6:
                cont = np.vstack([cont, cont[:1]])
            return cont
    # 回退：背景 outer
    for e in pmap.edges:
        if e.left_face == face_id and e.right_face < 0:
            return np.asarray(e.polyline, dtype=np.float64)
        if e.right_face == face_id and e.left_face < 0:
            return np.asarray(e.polyline, dtype=np.float64)[::-1].copy()
    cont, _, _ = fit_mask_contour_area_constrained(np.asarray(f.mask, dtype=bool), resample_n=128)
    return np.asarray(cont, dtype=np.float64)


def face_shape_boundary_points(
    pmap: PlanarMap,
    face_id: int,
    *,
    only_shape: bool = True,
    max_points: int = 192,
) -> FloatArr:
    """
    某 Face 的 SHAPE_BOUNDARY 目标点：来自关联 SharedEdge，**每条 edge_id 只采一次**。

    用于匹配损失，避免与邻 Face 双重计同一共边。
    """
    pts: list[FloatArr] = []
    seen_e: set[int] = set()
    for e in pmap.edges:
        if e.id in seen_e:
            continue
        if e.left_face != face_id and e.right_face != face_id:
            continue
        if only_shape and e.role == EdgeRole.occlusion_cut:
            continue
        seen_e.add(e.id)
        poly = np.asarray(e.polyline, dtype=np.float64)
        if len(poly) >= 2:
            pts.append(poly)
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    all_p = np.vstack(pts)
    if len(all_p) > max_points:
        idx = np.linspace(0, len(all_p) - 1, max_points).astype(int)
        all_p = all_p[idx]
    return all_p


def planar_map_to_region_graph(
    pmap: PlanarMap,
    palette: list[PaletteColor] | None = None,
) -> RegionGraph:
    """PlanarMap → 兼容旧 RegionGraph（轮廓从共享边派生）。"""
    _ = palette
    regions: list[Region] = []
    for f in pmap.faces:
        cont = face_contour(pmap, f.id)
        if len(cont) < 3:
            cont = np.array(
                [
                    [f.bbox[0], f.bbox[1]],
                    [f.bbox[2], f.bbox[1]],
                    [f.bbox[2], f.bbox[3]],
                    [f.bbox[0], f.bbox[3]],
                    [f.bbox[0], f.bbox[1]],
                ],
                dtype=np.float64,
            )
        rs = resample_closed_contour(cont, 192) if len(cont) >= 3 else cont
        desc = contour_curvature_descriptor(rs if len(rs) >= 8 else cont)
        regions.append(
            Region(
                region_id=f.region_id,
                color_hex=f.color_hex,
                color_rgb=f.color_rgb,
                area_frac=f.area_frac,
                bbox=f.bbox,
                mask=np.asarray(f.mask, dtype=bool),
                contour=cont,
                contour_resampled=rs,
                descriptor=desc,
                sdf=mask_to_sdf(np.asarray(f.mask, dtype=bool)),
                depth=0,
                centroid=f.centroid,
                contour_area_rel_err=0.0,
            )
        )
    # 邻接：聚合内部 shared edges
    adj: dict[tuple[int, int], float] = {}
    for e in pmap.edges:
        if e.left_face < 0 or e.right_face < 0:
            continue
        key = (min(e.left_face, e.right_face), max(e.left_face, e.right_face))
        adj[key] = adj.get(key, 0.0) + float(e.length)
    edges = [AdjacencyEdge(a=a, b=b, length=ln) for (a, b), ln in adj.items()]
    return RegionGraph(
        regions=regions,
        edges=edges,
        labels=pmap.labels,
        image_rgb=pmap.image_rgb,
        alpha=np.asarray(pmap.alpha, dtype=np.float64),
        canvas_size=pmap.canvas_size,
    )


def shared_shape_points(pmap: PlanarMap, *, only_shape: bool = True) -> FloatArr:
    """合并 SHAPE_BOUNDARY 共享边采样点。"""
    pts: list[FloatArr] = []
    for e in pmap.edges:
        if only_shape and e.role == EdgeRole.occlusion_cut:
            continue
        # 背景外轮廓与内部共边都可参与
        poly = np.asarray(e.polyline, dtype=np.float64)
        if len(poly) >= 2:
            pts.append(poly)
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(pts)


def seam_width_p95(pred_boundary_pts: FloatArr, target_pts: FloatArr) -> float:
    """目标点到预测边界的距离 95 分位（像素）。"""
    t = np.asarray(target_pts, dtype=np.float64)
    p = np.asarray(pred_boundary_pts, dtype=np.float64)
    if len(t) < 2 or len(p) < 2:
        return 0.0
    # 子采样
    if len(t) > 400:
        t = t[np.linspace(0, len(t) - 1, 400).astype(int)]
    if len(p) > 400:
        p = p[np.linspace(0, len(p) - 1, 400).astype(int)]
    d = np.linalg.norm(t[:, None, :] - p[None, :, :], axis=2).min(axis=1)
    return float(np.percentile(d, 95))
