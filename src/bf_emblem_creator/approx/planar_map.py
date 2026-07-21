"""共享边缘平面图（Vertex / SharedEdge / HalfEdge / Face）。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from bf_emblem_creator.approx.color import rgb_to_hex
from bf_emblem_creator.approx.curves import (
    area_relative_error,
    contour_curvature_descriptor,
    extract_outer_contour,
    mask_to_sdf,
    polygon_area,
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


def _pair_key(fa: int, fb: int) -> tuple[int, int]:
    """Face 对键（无序，背景为 -1）。"""
    return (min(int(fa), int(fb)), max(int(fa), int(fb)))


def _collect_dual_segments(
    face_id_map: I32Arr,
) -> list[tuple[tuple[int, int], tuple[int, int], int, int]]:
    """
    像素 dual 边列表。

    每项 ((x0,y0), (x1,y1), face_a, face_b)：端点为网格角点（整数），
    face_a/face_b 为两侧 Face id（主体外 / 画布外为 -1）。
    含相邻像素界面与画布外框（全铺色块也有闭合周长）。
    """
    lab = np.asarray(face_id_map, dtype=np.int32)
    h, w = lab.shape
    segs: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
    for y in range(h):
        for x in range(w - 1):
            fa = int(lab[y, x])
            fb = int(lab[y, x + 1])
            if fa != fb:
                # 竖直 dual：角点 (x+1,y)→(x+1,y+1)
                segs.append(((x + 1, y), (x + 1, y + 1), fa, fb))
    for y in range(h - 1):
        for x in range(w):
            fa = int(lab[y, x])
            fb = int(lab[y + 1, x])
            if fa != fb:
                # 水平 dual：角点 (x,y+1)→(x+1,y+1)
                segs.append(((x, y + 1), (x + 1, y + 1), fa, fb))
    # 画布外框：外侧视为背景 -1
    for x in range(w):
        top = int(lab[0, x])
        if top >= 0:
            segs.append(((x, 0), (x + 1, 0), -1, top))
        bot = int(lab[h - 1, x])
        if bot >= 0:
            segs.append(((x, h), (x + 1, h), bot, -1))
    for y in range(h):
        left = int(lab[y, 0])
        if left >= 0:
            segs.append(((0, y), (0, y + 1), left, -1))
        right = int(lab[y, w - 1])
        if right >= 0:
            segs.append(((w, y), (w, y + 1), -1, right))
    return segs


def _chain_dual_segments(
    segs: list[tuple[tuple[int, int], tuple[int, int], int, int]],
) -> list[tuple[int, int, FloatArr, bool]]:
    """
    将 dual 边按角点拓扑串成折线链。

    返回 list[(left_face, right_face, polyline(N,2), is_closed)]。
    left/right 与链的首段方向一致：沿 polyline 前进时 left_face 在左侧语义
    （此处沿用 dual 两侧标签，供半边 direction 绑定，不作几何左右硬约束）。
    """
    if not segs:
        return []
    from collections import defaultdict

    at: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (p0, p1, _fa, _fb) in enumerate(segs):
        at[p0].append(i)
        at[p1].append(i)

    def _other(seg_i: int, corner: tuple[int, int]) -> tuple[int, int]:
        p0, p1, _, _ = segs[seg_i]
        return p1 if corner == p0 else p0

    used = [False] * len(segs)
    out: list[tuple[int, int, FloatArr, bool]] = []

    for start in range(len(segs)):
        if used[start]:
            continue
        p0, p1, fa, fb = segs[start]
        pk = _pair_key(fa, fb)
        used[start] = True

        forward: list[tuple[int, int]] = []
        cur = p1
        while True:
            cands = [j for j in at[cur] if (not used[j]) and _pair_key(segs[j][2], segs[j][3]) == pk]
            if not cands:
                break
            j = cands[0]
            used[j] = True
            nxt = _other(j, cur)
            forward.append(nxt)
            cur = nxt

        backward: list[tuple[int, int]] = []
        cur = p0
        while True:
            cands = [j for j in at[cur] if (not used[j]) and _pair_key(segs[j][2], segs[j][3]) == pk]
            if not cands:
                break
            j = cands[0]
            used[j] = True
            nxt = _other(j, cur)
            backward.append(nxt)
            cur = nxt

        corners = [*list(reversed(backward)), p0, p1, *forward]
        is_closed = len(corners) >= 3 and corners[0] == corners[-1]
        poly = np.asarray(corners, dtype=np.float64)
        if len(poly) < 2:
            continue
        # 链方向固定为 p0→p1，与 start 段 fa/fb 一致
        out.append((int(fa), int(fb), poly, bool(is_closed)))
    return out


def _contour_has_interior_chords(
    cont: FloatArr,
    mask: BoolArr,
    *,
    min_chord: float = 12.0,
) -> bool:
    """
    检测轮廓是否含「穿心长弦」：长线段中点深陷 mask 内部。

    用于 halfedge 拼接失败时的质量回退判定。
    """
    p = np.asarray(cont, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if len(p) < 3 or not m.any():
        return False
    h, w = m.shape
    bad = 0
    total = 0
    nseg = len(p) - 1
    for i in range(nseg):
        d = float(np.linalg.norm(p[i + 1] - p[i]))
        if d < min_chord:
            continue
        total += 1
        mid = 0.5 * (p[i] + p[i + 1])
        ix = round(float(mid[0]))
        iy = round(float(mid[1]))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        if not m[iy, ix]:
            continue
        y0, y1 = max(0, iy - 2), min(h, iy + 3)
        x0, x1 = max(0, ix - 2), min(w, ix + 3)
        if bool(m[y0:y1, x0:x1].all()):
            bad += 1
    if total == 0:
        return False
    return bad >= max(1, (total + 2) // 3)


def _exact_mask_outer_contour(mask: BoolArr) -> FloatArr:
    """
    色块外轮廓：对 mask 做 Moore 边界跟踪（simplify=0）。

    **不做**圆/椭圆/RDP 几何概括；忠实于标签色块像素边界。
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.zeros((0, 2), dtype=np.float64)
    outer = extract_outer_contour(m, simplify=0.0)
    if outer is None or len(outer) < 3:
        ys, xs = np.where(m)
        x0, x1 = float(xs.min()), float(xs.max()) + 1.0
        y0, y1 = float(ys.min()), float(ys.max()) + 1.0
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)
    cont = np.asarray(outer, dtype=np.float64)
    if float(np.linalg.norm(cont[0] - cont[-1])) > 1e-6:
        cont = np.vstack([cont, cont[:1]])
    return cont


def _keep_boundary_polyline(pts: FloatArr) -> FloatArr:
    """
    保留 dual 边界点序（精确描边）。

    不圆拟合、不抽稀概括；仅去掉连续重复点。
    """
    arr = np.asarray(pts, dtype=np.float64)
    if len(arr) < 2:
        return arr
    keep = [arr[0]]
    for i in range(1, len(arr)):
        if float(np.linalg.norm(arr[i] - keep[-1])) > 1e-9:
            keep.append(arr[i])
    if len(keep) < 2:
        return arr.copy()
    return np.asarray(keep, dtype=np.float64)


def _link_halfedge_cycles(pmap: PlanarMap) -> None:
    """
    为每条半边设置 next_id，使同一 face 的半边形成闭合环。

    - 自闭合边（v0==v1，内孔）自成单环；
    - 开链半边在共享 Vertex 处连接；若 dual 方向不一致导致双入/双出，
      则翻转部分半边 direction 以恢复欧拉环；
    - **禁止**把孔与外环按极角混串（穿心弦主因）。
    """
    if not pmap.halfedges:
        return
    by_id = {he.id: he for he in pmap.halfedges}
    e_by = pmap.edge_by_id()

    def _he_endpoints(he: HalfEdge) -> tuple[int, int]:
        e = e_by[he.edge_id]
        if he.direction >= 0:
            return e.v0, e.v1
        return e.v1, e.v0

    def _he_poly(he: HalfEdge) -> FloatArr:
        e = e_by[he.edge_id]
        poly = np.asarray(e.polyline, dtype=np.float64)
        if he.direction < 0:
            poly = poly[::-1]
        return poly

    def _flip(he: HalfEdge) -> None:
        he.direction = -1 if he.direction >= 0 else 1

    face_hes: dict[int, list[int]] = {}
    for he in pmap.halfedges:
        face_hes.setdefault(he.face_id, []).append(he.id)

    for face_id, he_ids in face_hes.items():
        if not he_ids:
            continue
        open_ids: list[int] = []
        for hid in he_ids:
            s, t = _he_endpoints(by_id[hid])
            if s == t:
                by_id[hid].next_id = hid
            else:
                open_ids.append(hid)

        if not open_ids:
            f = pmap.face_by_id().get(face_id)
            if f is not None:
                f.halfedge_start = he_ids[0]
            continue

        if len(open_ids) == 1:
            only = by_id[open_ids[0]]
            # 单条开链无法自洽：保持原样并自环（轮廓走 mask 回退）
            only.next_id = only.id
            f = pmap.face_by_id().get(face_id)
            if f is not None:
                f.halfedge_start = only.id
            continue

        # 修正方向：每个顶点 in 度应等于 out 度
        for _ in range(len(open_ids) + 2):
            outs: dict[int, list[int]] = {}
            ins: dict[int, list[int]] = {}
            for hid in open_ids:
                s, t = _he_endpoints(by_id[hid])
                outs.setdefault(s, []).append(hid)
                ins.setdefault(t, []).append(hid)
            flipped = False
            verts = set(outs) | set(ins)
            for v in verts:
                n_out = len(outs.get(v, []))
                n_in = len(ins.get(v, []))
                if n_out == n_in:
                    continue
                if n_in > n_out:
                    # 翻转一条入边 → 变为出边
                    hid = ins[v][0]
                    _flip(by_id[hid])
                    flipped = True
                    break
                if n_out > n_in:
                    hid = outs[v][0]
                    _flip(by_id[hid])
                    flipped = True
                    break
            if not flipped:
                break

        out_at: dict[int, list[int]] = {}
        for hid in open_ids:
            s, _t = _he_endpoints(by_id[hid])
            out_at.setdefault(s, []).append(hid)

        used_next: set[int] = set()
        for hid in open_ids:
            he = by_id[hid]
            _s, t = _he_endpoints(he)
            cands = [c for c in out_at.get(t, []) if c != hid and c not in used_next]
            if not cands:
                cands = [c for c in out_at.get(t, []) if c != hid]
            if not cands:
                he.next_id = -1
                continue
            poly = _he_poly(he)
            dcur = poly[-1] - poly[-2] if len(poly) >= 2 else np.array([1.0, 0.0])
            best_c, best_score = cands[0], 1e18
            for c in cands:
                poly2 = _he_poly(by_id[c])
                d2 = poly2[1] - poly2[0] if len(poly2) >= 2 else np.array([1.0, 0.0])
                n1 = float(np.linalg.norm(dcur)) + 1e-9
                n2 = float(np.linalg.norm(d2)) + 1e-9
                cross = float(dcur[0] * d2[1] - dcur[1] * d2[0]) / (n1 * n2)
                dot = float(np.clip(np.dot(dcur, d2) / (n1 * n2), -1.0, 1.0))
                score = (1.0 - dot) + (0.0 if cross >= -1e-9 else 2.0)
                if score < best_score:
                    best_score, best_c = score, c
            he.next_id = best_c
            used_next.add(best_c)

        for hid in open_ids:
            if by_id[hid].next_id < 0:
                by_id[hid].next_id = hid

        f = pmap.face_by_id().get(face_id)
        if f is not None:
            best_start = open_ids[0]
            best_len = -1.0
            for hid in open_ids:
                ln = _poly_length(_he_poly(by_id[hid]))
                if ln > best_len:
                    best_len = ln
                    best_start = hid
            f.halfedge_start = best_start


def refine_edges_subpixel(pmap: PlanarMap, *, iters: int = 2) -> None:
    """
    沿标签 SDF 法向微调 SharedEdge 折线（拓扑 id / 端点 Vertex 不变）。

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
    gap_frac: float = 0.0,
    edge_subpixel: bool = False,
) -> PlanarMap:
    """
    从无洞标签场构建 PlanarMap（精确边界描边）。

    - 每连通域一个 Face；
    - SharedEdge = dual 角点拓扑链的**原始边界折线**（不圆/椭圆概括、不 RDP）；
    - Region 轮廓优先半边环拼接（共边唯一），失败则 Moore 精确外轮廓。
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
        for f in faces:
            m = np.asarray(f.mask, dtype=bool)
            f.area_frac = float(m.sum()) / canvas_area
            if m.any():
                ys, xs = np.where(m)
                f.centroid = (float(xs.mean()), float(ys.mean()))
                f.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    # 共享边：dual 角点拓扑串链，折线 = 精确像素边界
    dual_segs = _collect_dual_segments(face_id_map)
    chains = _chain_dual_segments(dual_segs)

    vertices: list[MapVertex] = []
    edges: list[SharedEdge] = []
    halfedges: list[HalfEdge] = []
    vid = 0
    eid = 0
    hid = 0
    vtx_index: dict[tuple[int, int], int] = {}

    def _vertex(x: float, y: float) -> int:
        nonlocal vid
        key = (round(float(x) * 2.0), round(float(y) * 2.0))
        found = vtx_index.get(key)
        if found is not None:
            return found
        vertices.append(MapVertex(id=vid, x=float(x), y=float(y)))
        vtx_index[key] = vid
        i = vid
        vid += 1
        return i

    face_ids = {f.id for f in faces}

    for fa, fb, poly0, is_closed in chains:
        arr = np.asarray(poly0, dtype=np.float64)
        if len(arr) < 2:
            continue
        if fa not in face_ids and fb not in face_ids:
            continue
        # 精确描边：保留 dual 点序，不做几何概括
        poly = _keep_boundary_polyline(arr)
        closed = bool(is_closed) or (len(poly) >= 3 and float(np.linalg.norm(poly[0] - poly[-1])) < 1e-6)
        if closed:
            if float(np.linalg.norm(poly[0] - poly[-1])) > 1e-6:
                poly = np.vstack([poly, poly[:1]])
            else:
                poly = poly.copy()
                poly[-1] = poly[0]
        if len(poly) < 2:
            continue
        v0 = _vertex(float(poly[0, 0]), float(poly[0, 1]))
        v1 = v0 if closed else _vertex(float(poly[-1, 0]), float(poly[-1, 1]))
        edges.append(
            SharedEdge(
                id=eid,
                v0=v0,
                v1=v1,
                left_face=fa if fa in face_ids else -1,
                right_face=fb if fb in face_ids else -1,
                polyline=poly,
                length=_poly_length(poly),
                role=EdgeRole.shape_boundary,
            )
        )
        if fa in face_ids:
            halfedges.append(HalfEdge(id=hid, edge_id=eid, direction=1, face_id=fa, next_id=-1))
            hid += 1
        if fb in face_ids:
            halfedges.append(HalfEdge(id=hid, edge_id=eid, direction=-1, face_id=fb, next_id=-1))
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
    he_by_face: dict[int, int] = {}
    for he in pmap.halfedges:
        he_by_face.setdefault(he.face_id, he.id)
    for f in pmap.faces:
        if f.halfedge_start < 0 and f.id in he_by_face:
            f.halfedge_start = he_by_face[f.id]
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
    # 每个 halfedge 的 next 形成有限局部环（允许外环与孔分环；允许退化自环）
    he_by = {he.id: he for he in pmap.halfedges}
    for he in pmap.halfedges:
        seen: list[int] = []
        cur = he.id
        for _ in range(len(pmap.halfedges) + 2):
            if cur in seen:
                break
            seen.append(cur)
            cur = he_by[cur].next_id
        if cur not in seen and cur != he.id:
            raise ValueError(f"半边 {he.id} 的 next 链未形成环")
    for f in pmap.faces:
        if f.halfedge_start < 0:
            # 允许无边极小 face；有边则必须有起点
            if any(he.face_id == f.id for he in pmap.halfedges):
                raise ValueError(f"face {f.id} 有半边但 halfedge_start 未设置")
            continue
        if f.halfedge_start not in he_ids:
            raise ValueError(f"face {f.id} halfedge_start 无效")
        ring = walk_face_halfedges(pmap, f.id)
        if not ring:
            raise ValueError(f"face {f.id} 半边环为空")


def walk_face_halfedges(pmap: PlanarMap, face_id: int) -> list[HalfEdge]:
    """沿 next_id 遍历 face 外环半边（闭合一圈；遇重复节点停止）。"""
    f = pmap.face_by_id().get(face_id)
    if f is None or f.halfedge_start < 0:
        return []
    he_by = {he.id: he for he in pmap.halfedges}
    out: list[HalfEdge] = []
    cur = f.halfedge_start
    seen: set[int] = set()
    for _ in range(len(pmap.halfedges) + 2):
        if cur in seen:
            break
        if cur not in he_by:
            break
        he = he_by[cur]
        out.append(he)
        seen.add(cur)
        cur = he.next_id
        if cur == f.halfedge_start:
            break
    return out


def _assemble_face_ring_polyline(
    pmap: PlanarMap,
    face_id: int,
    *,
    only_shape: bool = False,
    max_join_gap: float = 6.0,
) -> FloatArr | None:
    """
    沿半边环拼接 SharedEdge.polyline 为闭合轮廓。

    几何源唯一：边的 polyline（dense 或 curve_fit 后的贝塞尔采样）。
    仅当环为空或接缝过大时返回 None；**不因穿心弦丢弃**（拟合后宏观弦合法）。
    """
    f = pmap.face_by_id().get(face_id)
    if f is None:
        return None
    e_by = pmap.edge_by_id()
    ring = walk_face_halfedges(pmap, face_id)
    if not ring:
        return None

    # 优先开放边；若环上仅有自环（整圈闭合外边界）则保留自环
    open_he: list[HalfEdge] = []
    loop_he: list[HalfEdge] = []
    for he in ring:
        e = e_by[he.edge_id]
        if only_shape and e.role == EdgeRole.occlusion_cut:
            continue
        poly = np.asarray(e.polyline, dtype=np.float64)
        if len(poly) < 2:
            continue
        if e.v0 == e.v1 and len(poly) > 4:
            loop_he.append(he)
        else:
            open_he.append(he)
    use = open_he if open_he else (loop_he if loop_he else list(ring))

    chunks: list[FloatArr] = []
    max_gap = 0.0
    for he in use:
        e = e_by[he.edge_id]
        poly = np.asarray(e.polyline, dtype=np.float64)
        if he.direction < 0:
            poly = poly[::-1].copy()
        if chunks and len(poly) > 0:
            gap = float(np.linalg.norm(chunks[-1][-1] - poly[0]))
            max_gap = max(max_gap, gap)
            if gap < max_join_gap:
                poly = poly.copy()
                poly[0] = chunks[-1][-1]
                if len(poly) > 1:
                    poly = poly[1:]
        if len(poly):
            chunks.append(poly)
    if not chunks:
        return None
    cont = np.vstack(chunks)
    if len(cont) < 2:
        return None
    close_gap = float(np.linalg.norm(cont[0] - cont[-1]))
    max_gap = max(max_gap, close_gap)
    if close_gap > 1e-6:
        cont = np.vstack([cont, cont[:1]])
    # 拟合后端点可能有亚像素缝；放宽阈值，优先保留共享边环
    if max_gap > max_join_gap:
        return None
    if len(cont) < 3:
        return None
    return cont


def face_contour(pmap: PlanarMap, face_id: int) -> FloatArr:
    """
    Face 外轮廓：优先半边环拼接 SharedEdge.polyline（dense / 贝塞尔拟合后）。

    仅在环拼接失败时回退 mask Moore 精确跟踪（**禁止**圆椭圆概括）。
    拟合后的宏观弦不再触发回退，保证与 edges_bezier_after_fit 同源。
    """
    f = pmap.face_by_id().get(face_id)
    if f is None:
        return np.zeros((0, 2), dtype=np.float64)
    mask = np.asarray(f.mask, dtype=bool)
    cont = _assemble_face_ring_polyline(pmap, face_id, only_shape=False)
    if cont is not None and len(cont) >= 3:
        return cont
    return _exact_mask_outer_contour(mask)


def face_shape_boundary_points(
    pmap: PlanarMap,
    face_id: int,
    *,
    only_shape: bool = True,
    max_points: int = 512,
) -> FloatArr:
    """
    某 Face 的形状目标曲线点：与 after_fit / dense 共享边同源。

    优先半边环有序拼接（匹配 Chamfer 与 Region.contour 一致）；
    环失败时再按 edge_id 去重堆叠各边 polyline。
    """
    ordered = _assemble_face_ring_polyline(pmap, face_id, only_shape=only_shape)
    if ordered is not None and len(ordered) >= 4:
        all_p = ordered
        if len(all_p) > 1 and float(np.linalg.norm(all_p[0] - all_p[-1])) < 1e-6:
            all_p = all_p[:-1]
    else:
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
            # 最后回退：完整 face 轮廓（仍优先共享边环，失败才 Moore）
            cont = face_contour(pmap, face_id)
            if len(cont) < 4:
                return np.zeros((0, 2), dtype=np.float64)
            all_p = cont[:-1] if len(cont) > 1 and float(np.linalg.norm(cont[0] - cont[-1])) < 1e-6 else cont
        else:
            all_p = np.vstack(pts)
    if len(all_p) > max_points:
        idx = np.linspace(0, len(all_p) - 1, max_points).astype(int)
        all_p = all_p[idx]
    return np.asarray(all_p, dtype=np.float64)


def planar_map_to_region_graph(
    pmap: PlanarMap,
    palette: list[PaletteColor] | None = None,
) -> RegionGraph:
    """PlanarMap → RegionGraph：轮廓与 SharedEdge.polyline 同源（拟合后即 after_fit）。"""
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
        # 重采样仅用于描述子；完整轮廓保留共享边几何
        if len(cont) >= 3:
            n_rs = min(256, max(64, len(cont)))
            rs = cont if len(cont) <= n_rs else resample_closed_contour(cont, n_rs)
        else:
            rs = cont
        desc = contour_curvature_descriptor(rs if len(rs) >= 8 else cont)
        mask = np.asarray(f.mask, dtype=bool)
        mask_a = float(mask.sum())
        area_err = area_relative_error(polygon_area(cont), mask_a)
        # 拟合宏观弦可能穿区：仅抬高误差标记，不改轮廓源
        if _contour_has_interior_chords(cont, mask):
            area_err = max(area_err, 0.5)
        regions.append(
            Region(
                region_id=f.region_id,
                color_hex=f.color_hex,
                color_rgb=f.color_rgb,
                area_frac=f.area_frac,
                bbox=f.bbox,
                mask=mask,
                contour=cont,
                contour_resampled=rs,
                descriptor=desc,
                sdf=mask_to_sdf(mask),
                depth=0,
                centroid=f.centroid,
                contour_area_rel_err=float(area_err),
            )
        )
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
