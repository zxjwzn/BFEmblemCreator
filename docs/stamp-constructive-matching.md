# 图章构造式匹配：同色并集 · 异色遮挡 · 画布裁切

> 状态：**主路径已落地（P0–P2 轻量）** — `union_cover.py` + `match_assembler.py`；测试 `tests/test_constructive_cover.py`  
> 目标：用预计算图章曲线 + 搜索，把平面化后的 **贝塞尔区域边界** 高保真重建为 ≤40 层 `StampLayer`  
> 父文档：[`visible-boundary-fitting.md`](./visible-boundary-fitting.md)、[`shared-edge-and-planar-field.md`](./shared-edge-and-planar-field.md)  
> 约束：画布默认 320×320 正方形裁切；图章曲线已预计算（`StampCurveLibrary` / `prefit-stamps`）；注释与文档中文  
> 非目标：照片纹理 PSNR、场景白名单、形状路由桶

---

## 0. 问题重述（以用户机制为准）

战地图章的**可见形状**不是「一区一章贴纸」，而是三种**构造算子**的复合结果：

| 机制 | 几何语义 | 游戏/渲染表现 |
|------|----------|----------------|
| **同色并集** | 相同 `fill` 的多枚图章 mask **∪** | 相接/重叠处**内边消失**，只剩并集外轮廓 |
| **异色遮挡** | 更高层不同色章 **挖掉/盖住** 下层可见部分 | 下层被切出新外轮廓（假边 = 上层边界） |
| **画布裁切** | \(\mathrm{Canvas}\cap \cdot\)，Canvas 为正方形 | 章可移出画布，只露一段弧/一条边 |

最终像素颜色：

\[
I = \mathrm{over}(L_n,\ldots,L_1),\quad
V_i = \mathrm{Canvas}\cap M_i \setminus \bigcup_{j>i} M_j
\]

其中 \(M_i = T_i(S_{t_i})\) 为第 \(i\) 层图章变换后的支撑（alpha>½）。

**匹配目标（硬要求）**：合成后的**可见边界**高程度贴合图像处理后得到的 **贝塞尔拟合曲线**（`SharedEdge.polyline` / `edges_bezier_after_fit` 拼环），而不是仅贴像素 mask 的阶梯边。

图章侧：曲线已预计算，算法只做 **资产选择 + 位姿搜索 + 层序装配**。

---

## 1. 现状诊断（为何观感像「一区一章」）

代码里**已有半套**并集，但**没有按构造式语义做主搜索**。

| 能力 | 代码位置 | 现状问题 |
|------|----------|----------|
| 同色多章 | `union_cover.cover_region_with_union_stamps` | 存在，但默认 `min_cover=0.82`、增益阈值高；实践中常 **1 枚就停** 或第 2 枚被 leak/gain 拒掉 |
| 目标曲线 | `face_shape_boundary_points` → `target_curve_pts` | **仅首枚**用 after_fit；同色第 2+ 枚改用 residual mask 轮廓，**偏离贝塞尔目标** |
| 评分 | `match_curve` Chamfer + mask cover | **双目标**：曲线与像素 mask 拉扯；mask 不随贝塞尔更新 |
| 异色遮挡造型 | 层序合成有，搜索无 | **从不**主动搜「大底章 + 上层异色切边」 |
| 画布裁切 | 粒子允许 `left/top` 出界 | 有随机出界样本，**无**「用正方形窗切出目标弧」的专用提案 |
| 全局残差 `r9xxx` | `match_assembler` 后半 | 绕开平面图/贝塞尔，堆 Square 补洞，**破坏**「贴合贝塞尔」目标 |
| 层序 | `infer_depth_order` 启发式 | 只影响放置顺序，**不参与**可见弧解释与搜索目标 |

因此用户感知「一个色块只能匹配一个图章、没用上三种机制」——**机制在渲染端合法，在搜索端几乎没当一等公民**。

---

## 2. 设计原则

1. **唯一曲线真源**：平面化 + `curve_fit` 后的 `SharedEdge.polyline`（及拼成的 Face 环）。像素 mask 只作辅助（主体内/外、粗覆盖），**不得**在主循环里替换贝塞尔目标。  
2. **构造优先于贴图**：先问「这组章的可见并集/差集边界是否贴合目标环」，再问「单章像不像该色块」。  
3. **三种算子都要有提案器**（proposal），不能只靠统一粒子碰运气。  
4. **补缝可留，但不得另立几何王国**：全局 residual 只能在**原 Face 贝塞尔环约束下**补，禁止 `r9xxx` 自由轮廓当主目标。  
5. **无版本分叉**：改主路径；删掉「一区一章成功即停」的隐式行为。  
6. **预算显式**：`max_layers≤40`；每色/每 Face 有章数预算，优先少章高贴边。

---

## 3. 目标形式化

### 3.1 输入

| 符号 | 来源 |
|------|------|
| 标签场 \(L\)、调色板 | `ImageProcessor` |
| `PlanarMap`：Face、SharedEdge（**已 curve_fit**） | `RegionPartitioner` |
| 层序假设 \(\pi\) 与边角色 SHAPE/OCCLUSION | `infer_depth_order`（可迭代 refinement） |
| 图章曲线库 \(\{\gamma_t\}\)、mask 纹理 | `StampCurveLibrary` + renderer |
| 层预算 \(N\le 40\) | `ModeRecipe` |

### 3.2 每个 Face \(F\) 的目标可见边界

对 Face \(F\)，在层序 \(\pi\) 下：

- **形状真边** \(B_F^{\mathrm{shape}}\)：关联 `role=SHAPE_BOUNDARY` 的 SharedEdge 折线（after_fit），按半边环有序拼接。  
- **遮挡假边** \(B_F^{\mathrm{cut}}\)：`OCCLUSION_CUT`（被上层解释裁出的边）——**不要求**本色章去贴合，应由上层异色章的边界解释。

主匹配目标点云：

\[
\Gamma_F = \mathrm{sample}(B_F^{\mathrm{shape}})
\]

（与当前 `face_shape_boundary_points(..., only_shape=True)` 对齐，但必须**贯穿**同色每一枚，而不是仅首枚。）

### 3.3 同色组的可见外形

同色、且在装配中视为一组的图章集合 \(G=\{L_{i_1},\ldots,L_{i_m}\}\)（同 `fill`）：

\[
U_G = \mathrm{Canvas}\cap \bigcup_{k} M_{i_k}
\]

在未被更上层异色遮挡时，**可见外轮廓** \(\partial U_G\) 应贴合 \(\Gamma_F\)（允许有损：Huber Chamfer / 弧长采样）。

同色内部交线在渲染中消失 → 搜索**奖励**并集外轮廓贴合，**不惩罚**同色内部重叠（只惩罚浪费层数）。

### 3.4 异色遮挡

上层异色组 \(H\) 对下层的可见部分：

\[
V_F = U_{G_F} \setminus \bigcup_{H \succ F} U_H
\]

若某段目标边是 OCCLUSION_CUT，则该段应由**上层**边界对齐，而不是强迫下层章凹进去。

### 3.5 画布裁切

正方形 \(\mathrm{Canvas}=[0,S]^2\)。提案显式包含：

- 中心可在 \([-0.5S, 1.5S]^2\)  
- 尺寸可 \(>S\)（大章只露一角/一边）  
- 目标：\(\partial(\mathrm{Canvas}\cap M)\) 的可见弧贴合 \(\Gamma_F\) 的一段或全周

---

## 4. 算法总览（推荐主路径）

```text
输入图
  → A. 平面化 + 标签场 + PlanarMap
  → B. curve_fit：SharedEdge → 贝塞尔折线（唯一曲线真源）
  → C. 层序假设 π + 边角色
  → D. 构造式图章搜索（本设计核心）
        对 depth 从底→顶 每个 Face F：
          D1. 取 Γ_F = after_fit 形状边点云
          D2. 同色构造覆盖 Cover-Union(F, Γ_F)
          D3. （可选）记录「需由上层解释的 cut 边」
        层序 refinement（可选 1～2 轮）
        异色切边修补：对未解释的 cut，用上层章边界对齐
  → E. 约束补缝（仍绑定 Face/Γ_F，禁止自由 residual 轮廓）
  → F. 不可见层剪枝 + 导出 JSON + 渲染
```

与现状最大差异：

| 步骤 | 现状 | 本设计 |
|------|------|--------|
| 同色第 k 枚目标 | residual mask 轮廓 | **始终 Γ_F**（或 Γ_F 上未贴合的子弧） |
| 成功标准 | cover≥0.82 可停 | **边界 Chamfer + 覆盖** 双达标，默认 cover≥0.92 |
| 出画布 | 随机粒子 | **专用裁切提案** |
| 异色 | 合成才体现 | **搜索提案 + 目标边角色** |
| 全局 residual | 自由 need 轮廓 | **按 Face 回绑 Γ_F** 或仅填 gap 像素且曲线仍评 Γ_F |

---

## 5. 模块 D：构造式搜索（详细）

### 5.1 数据结构

```text
StampPlacement:
  asset, left, top, width, height, angle, flipX/Y, fill
  curve_on_canvas: 变换后预计算环（外+孔）
  mask_on_canvas: 可选，评分时 GPU 批渲染

FaceCoverState:
  face_id, fill, Γ_F          # 贝塞尔目标点云（固定）
  placements: list[StampPlacement]
  union_mask                  # Canvas ∩ ∪ M_i
  boundary_score              # ∂U 对 Γ_F 的有损 Chamfer
  cover, leak                 # 相对 Face.mask 或相对「Γ_F 多边形近似」
  uncovered_arcs              # Γ_F 上尚未被 ∂U 解释的子弧
```

**预计算图章曲线**：只读 `StampCurveLibrary`；变换用现有 `transform_stamp_contour_batch`。

### 5.2 评分（单 Face 同色组）

对状态 \(s\)：

\[
\begin{aligned}
\mathcal{L}_{\partial}
  &= \mathrm{Chamfer}_{Huber}(\mathrm{sample}(\partial U),\; \Gamma_F)
   + \mathrm{Chamfer}_{Huber}(\Gamma_F,\; \mathrm{sample}(\partial U)) \\
\mathcal{L}_{\mathrm{cov}}
  &= (1-\mathrm{cover}) + \lambda_{\ell}\,\mathrm{leak} \\
\mathcal{L}
  &= \mathcal{L}_{\partial} + \lambda_c \mathcal{L}_{\mathrm{cov}} + \lambda_n \frac{m}{m_{\max}}
\end{aligned}
\]

| 项 | 含义 | 建议 |
|----|------|------|
| \(\mathcal{L}_{\partial}\) | **主项**：并集外轮廓 ↔ 贝塞尔目标 | 权重大；Huber 允许平缓有损 |
| cover | \(U\) 覆盖 Face 主体 | 默认 ≥0.92 才可「完成」 |
| leak | \(U\) 溢出到**异色 Face** / 背景 | 同色邻接可放宽；背景严 |
| \(m\) | 章数 | 同等贴边下少章优先 |

**关键**：评分对象是 **∪ 后的可见外轮廓**，不是最后一枚章单独的轮廓。  
第 k 枚加入后应重算 \(\partial U\)，而不是只优化第 k 枚对 residual 轮廓。

可选加速：不每步精确 ∂U，用「各章曲线点 ∪ 后做 alpha-shape / 对 Γ_F 的覆盖半径」近似；精修阶段再精确。

### 5.3 同色并集覆盖算法 `Cover-Union(F, Γ_F)`

```text
s ← 空状态
loop m = 1..m_max（默认 6，受全局层预算限制）:
  若 s 已达标（L_∂ 小 且 cover≥τ）: break

  提案集合 P ← ∅
  P ∪= ProposeFullShape(F, Γ_F)      # 单章尽量包住整环
  P ∪= ProposeArcStamp(F, s)         # 专贴 uncovered_arcs
  P ∪= ProposeCanvasClip(F, s)       # 大章 + 出画布，只露一段
  P ∪= ProposeParticle(F, s)         # 现有 GPU 粒子（目标=Γ_F 或 uncovered）

  对每个 p∈P：
    s' ← s 并入 p（同 fill）
    拒绝：leak 过大且 L_∂ 无改进
    记录 best s'
  s ← best
  若增益 < ε：break

return s.placements
```

#### 5.3.1 `ProposeFullShape`（整环一章）

- 召回：用 \(\Gamma_F\) 的曲率描述子 / 圆度 / 细长度 → `curve_lib.recall`  
- 位姿：bbox 对齐、主轴角、多尺度；**允许** width/height 非均匀  
- 目标：单章曲线 Chamfer(·, Γ_F) + cover  
- 适用：近圆、近矩形、简单色块 → 真正的「一区一章」

#### 5.3.2 `ProposeArcStamp`（子弧补章 = 同色融合核心）

当 \(m\ge 2\) 或首枚后仍有 uncovered_arcs：

1. 把 \(\Gamma_F\) 按弧长/转角切成子弧 \(\{a_r\}\)  
2. 计算每段被当前 \(\partial U\) 的解释度（到 ∂U 的距离中位数）  
3. 取最差子弧 \(a^\star\)（长且远）  
4. 在库中召回「局部曲率像 \(a^\star\)」的图章段（可用预计算环上滑动窗口描述子）  
5. 搜索 \(T\) 使 \(T(\gamma_t)\) 贴合 \(a^\star\)，且 \(T(S_t)\) 落在 Face 主体内为主  
6. 并入后同色内边消失 → 外轮廓应吞掉 \(a^\star\) 附近缺口  

这才是「多枚同色章拼出新形状」的搜索表达，而不是「对 residual 像素再贴一张 Square」。

#### 5.3.3 `ProposeCanvasClip`（正方形裁切专用）

显式采样：

| 模式 | 意图 |
|------|------|
| 大圆/大方心在画布外 | 只露一段凸弧作金锭外缘 |
| 章中心在角外、尺寸 ~1.5S–4S | 用画布直角切出目标直角/外轮廓 |
| 条带章横跨画布 | 用对边裁切得到近似平行边 |

评分时曲线与 mask **必须**先与 Canvas 求交再比 \(\Gamma_F\)。  
现状粒子虽有出界样本，但无上述结构化提案 → 应升格为独立 proposer。

#### 5.3.4 `ProposeParticle`（通用精修）

保留现有 GPU 粒子，但修改：

1. `target_curve_pts` **永远**来自 \(\Gamma_F\) 或 uncovered 子弧采样，**禁止** residual Moore 轮廓  
2. 多章时：loss 可加「并入后 ∂U 的 Chamfer」，至少精修阶段如此  
3. `min_score` 与 cover 阈值提高，避免「放了但几乎没用」

### 5.4 异色遮挡造型 `Occlusion-Carve`

层序底→顶放置后，对仍存在的问题边：

```text
对每条 OCCLUSION_CUT 边 e（或合成后「应用上层才出现」的目标转折）:
  负责层 = 更深/更上的 Face 侧（按 π）
  在上层 Face 的 Cover-Union 中：
    增加约束：上层 ∂U 应覆盖 e 的采样点
  或：追加一枚上层同色小章，使其边界贴 e
```

实现分两阶段即可（避免一开始全局组合爆炸）：

1. **Phase 1**：每 Face 只按 \(\Gamma_F^{\mathrm{shape}}\) 做 Cover-Union（忽略 cut）  
2. **Phase 2**：扫描 cut 边与合成缝；对责任上层做「边界贴 e」的追加/微调  

这对应用户说的：**更高图层不同颜色覆盖，使下层展现新形状**——搜索端要**主动**让上层边界对齐 cut，而不是期望下层章自己凹成 cut。

### 5.5 层序与放置顺序

- 仍用启发式 \(\pi\) 底→顶放置（大块/背景色偏下）。  
- 放置完 Phase 1 后，用合成结果微调：若大量 SHAPE 边被误标为 cut 或反之，可重标 `SharedEdge.role` 再跑 Phase 2。  
- **不**在首版做层序全排列搜索（40 层不可行）；角色局部修正即可。

### 5.6 约束补缝（保留，但改制）

允许在主 Cover-Union 后仍有小 gap，但：

```text
need 像素 → 归属到最近 Face F（标签场）
补章 fill = F.color
target_curve = Γ_F（或 F 上 uncovered 子弧）
禁止 region_id=9000+ 自由轮廓作为曲线真源
```

调试图命名：`match_ok_L*_f{face_id}_u{union_idx}`，不再用 `r9xxx` 表示「另一套几何」。

若 need 无法归属（标签空洞），先修标签场，而不是硬补章。

---

## 6. 与贝塞尔高保真对齐的验收

| 指标 | 定义 | 建议门槛 |
|------|------|----------|
| 边 Chamfer | 合成可见边界 vs 全部 SHAPE after_fit 点 | 中位距离 < 1.5–2.5 px（320 画布） |
| Face 环贴合 | 各 Face 的可见外轮廓 vs \(\Gamma_F\) | 同色并集后 \(\mathcal{L}_{\partial}\) 低于阈值 |
| 覆盖 | 主体内量化色一致率 | >0.9 |
| 层数 | \(n\le 40\) | 硬约束 |
| 无自由残差轮廓 | 匹配目标均能追溯到 SharedEdge | debug 可检查 |

Debug 建议新增：

- `06_union_boundary_f*`：同色并集外轮廓（应贴近 after_fit 环）  
- `06_uncovered_arcs_f*`：尚未解释的贝塞尔子弧  
- `06_clip_proposals*`：画布裁切提案可视化  

---

## 7. 代码改造清单（映射到仓库）

### 7.1 保留

| 模块 | 用途 |
|------|------|
| `edge_curve_fit.simplify_planar_map_curves` | 贝塞尔真源 |
| `planar_map.face_shape_boundary_points` / 半边拼环 | \(\Gamma_F\) |
| `stamp_curves.StampCurveLibrary` | 预计算曲线 |
| `match_curve.transform_stamp_contour_batch` / Chamfer GPU | 变换与距离 |
| `torch_render` 批 mask | cover/leak |
| `infer_depth_order` | 初始 π 与角色 |

### 7.2 重写 / 大改

| 文件 | 改动 |
|------|------|
| `union_cover.py` | 升级为 `Cover-Union`：多 proposer；**全程 Γ_F**；并集外轮廓评分；提高默认 cover；同色内重叠不罚 |
| `match_curve.py` | 支持「目标=子弧」；可选 score 并入后的 ∂U；裁切后曲线再 Chamfer |
| `match_assembler.py` | 编排 Phase1 Cover-Union → Phase2 Occlusion-Carve → 约束补缝；删除自由 `9000+` 轮廓主路径 |
| `depth_order.py` | 导出 cut 边列表供 Phase2；可选 role 回写 refinement |
| `recipe.py` | `UnionCoverConfig` 扩展：`min_cover` 默认 0.92、`max_stamps_per_region` 默认 6、`enable_canvas_clip_proposals`、`enable_occlusion_carve`、`boundary_loss_weight` |
| `debug_vis.py` | 并集边界 / 未覆盖弧 / 裁切提案 |

### 7.3 删除或降级的行为

1. 同色第 2+ 枚使用 `match_region.contour_resampled`（residual Moore）作为曲线目标。  
2. 全局 `need` → `fit_mask_contour_high_precision` → 当 Chamfer 真源。  
3. 「cover≥0.82 且单枚分够就停」导致复杂块从不并集。  
4. 以像素 mask IoU 为主、曲线为辅的隐式权重（应改为曲线主、覆盖辅）。

### 7.4 建议 API 草图

```python
# approx/constructive_cover.py（新模块，或重写 union_cover）

def cover_face_constructive(
    face_id: int,
    planar_map: PlanarMap,
    curve_lib: StampCurveLibrary,
    renderer: TorchStampRenderer,
    *,
    fill: str,
    max_stamps: int,
    min_cover: float,
    boundary_weight: float,
    enable_canvas_clip: bool,
) -> list[StampLayer]:
    """同色构造覆盖：提案并集，使 ∂U 贴合 after_fit Γ_F。"""
    ...

def carve_occlusion_edges(
    layers: list[StampLayer],
    planar_map: PlanarMap,
    depth: DepthOrderResult,
    ...
) -> list[StampLayer]:
    """异色切边：让上层边界解释 OCCLUSION_CUT。"""
    ...

def constrained_gap_fill(
    layers: list[StampLayer],
    planar_map: PlanarMap,
    ...
) -> list[StampLayer]:
    """补缝：need 归属 Face，曲线仍绑 Γ_F。"""
    ...
```

`StampMatchAssembler.assemble` 缩为薄编排层。

---

## 8. 分阶段落地（避免一次改爆）

### P0 — 纠偏（小改，立刻对齐语义）

1. 同色每一枚 `target_curve_pts` 均用 `face_shape_boundary_points` / 未覆盖子弧。  
2. `min_cover` 默认 → 0.92；降低「过早停止」；提高 `max_stamps_per_region` → 6。  
3. 并集后增加 ∂U 对 Γ_F 的验收，不达标继续加章。  
4. 全局 residual 补章必须带上归属 `face_id` 与 Γ_F。  

**验收**：复杂色块 debug 出现多枚同 fill；并集轮廓靠近 after_fit。

### P1 — 构造提案

1. 实现 `ProposeArcStamp`（子弧）。  
2. 实现 `ProposeCanvasClip`。  
3. debug 画 uncovered_arcs 与 union boundary。  

**验收**：金锭类外轮廓可用「大章出画布 + 同色补弧」接近贝塞尔环，而非碎 Square 堆砌。

### P2 — 异色切边

1. Phase2 `Occlusion-Carve`。  
2. cut 边与上层边界 Chamfer 纳入边界分。  

**验收**：假边由上层解释；下层不再为 cut 边畸形内凹。

### P3 — 评分与层序 refinement

1. 统一边界主 loss。  
2. 可选 1 轮 role/层序回写。  
3. 砍不可见层。  

---

## 9. 参数默认（建议）

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `min_cover` | 0.92 | 完成门槛 |
| `max_stamps_per_region` | 6 | 同色并集上限 |
| `min_cover_gain` | 0.02 | 允许小弧补章 |
| `max_leak` | 0.28（异色）/ 0.45（同色邻接放宽） | 分类 leak |
| `boundary_weight` | 1.0（主） | 相对 cover |
| `canvas_clip_fraction` | 提案中 ≥15% | 结构化出界 |
| `max_layers` | 40 | 硬顶 |

---

## 10. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 并集 ∂U 每步太贵 | 粗评用点到 Γ_F 覆盖半径；精修再提轮廓 |
| 章数爆炸 | 少章优先 + 全局 40 硬顶 + 同色 m_max |
| 只盖像素不贴贝塞尔 | 不达标 ∂ 不得 break；P0 强制 Γ_F |
| 异色 Phase2 抖动 | 只动 cut 邻域小章/微调，不动整底 |
| 与旧 residual 习惯冲突 | 文档与 debug 明确；删除 r9xxx 自由轮廓 |

---

## 11. 一句话对照

| 用户机制 | 本设计中的算法对应 |
|----------|-------------------|
| 同色拼贴、内边消失 | `Cover-Union`：多枚同 fill，**评 ∂(∪M)** vs 贝塞尔环 |
| 上层异色覆盖出新形 | `Occlusion-Carve`：上层边界对齐 CUT 边 |
| 画布正方形裁切 | `ProposeCanvasClip` + 评分前 Canvas∩ |
| 高保真贝塞尔 | \(\Gamma_F\) 唯一曲线真源，贯穿每枚同色章与补缝 |
| 预计算图章曲线 | 只搜索 asset + \(T\)；曲线库只读 |
| 补缝保留 | `constrained_gap_fill`，禁止第二套轮廓几何 |

---

## 12. 与现有文档关系

- [`visible-boundary-fitting.md`](./visible-boundary-fitting.md)：可见边界总论；本文是其 **构造搜索落地设计**。  
- [`shared-edge-and-planar-field.md`](./shared-edge-and-planar-field.md)：标签场与 SharedEdge；本文 **消费** after_fit 边，不重做平面化。  
- [`architecture-refactor.md`](./architecture-refactor.md)：Engine/Recipe 骨架；落地时在 `StampMatchAssembler` 换芯，Recipe 增参。  

**落地后**应更新 `architecture-refactor.md` 调用链为：

```text
RegionPartitioner（after_fit）
  → ConstructiveCover（同色并集 + 裁切提案）
  → OcclusionCarve（异色）
  → ConstrainedGapFill
  → Assemble / JSON / 预览
```

---

## 13. 结论

当前管线**渲染语义**支持同色并集、异色遮挡、出画裁切，但**搜索目标与停止条件**仍接近「单区单章贴 mask」，故效果不符合用户机制。

正确方向：把匹配改成 **在预计算图章曲线上的构造式搜索**——

1. 用同色多章的 **并集外轮廓** 去拟合 **after_fit 贝塞尔环**；  
2. 用上层异色边界解释 **遮挡假边**；  
3. 用 **画布正方形裁切** 作为一等提案；  
4. 补缝只作约束内收尾。  

按 §8 的 P0→P3 改代码，即可在不推翻平面化/曲线库的前提下，把主路径扳回「拼合形状高度还原贝塞尔边界」。
