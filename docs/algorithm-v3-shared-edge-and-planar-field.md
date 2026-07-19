# 算法 v3 补篇：通用标签场概括 + 共享边缘拓扑

> 状态：设计稿 v0.3（**文档设计点已落地**：Batch A–E，含 halfedge 环、亚像素边 D1、匹配按 edge_id 去重目标曲线）  
> 父文档：[`algorithm-v3-visible-boundary-fitting.md`](./algorithm-v3-visible-boundary-fitting.md)  
> 相关问题：`source_fitted` 阶段即出现软边杂色、色块裂纹、区域拼合细缝；独立轮廓拟合会双侧内缩  
> 约束：通用方案，**禁止**场景白名单 / 形状路由；画布默认 320×320；层数 ≤40；注释与文档中文  
> 代码：`preprocess.detect_resample_mode` / `label_field.py` / `planar_map.py`（halfedge 环 + `refine_edges_subpixel`）/ `contour_arcs.extract_primitives_for_shared_edges` / `match_curve.target_curve_pts` / `assemble.seam_p95`；测试 `tests/test_shared_edge_label_field.py`

---

## 0. 动机与分层

### 0.1 现象（以 `examples/gold.png` 调试链为例）

| 阶段图 | 现象 | 主因层 |
|--------|------|--------|
| `001_00_source_fitted` | 色块裂纹、杂色丝、边界错切 | **概括 / 标签场** |
| `003_00_source_raw` | 软边、糊边（128→320 LANCZOS） | **重采样** |
| `006_02_regions` | 碎区、主体像素未全覆盖 | **区域规整 / 丢弃策略** |
| 后续轮廓·匹配·装配 | 拼合细缝、贴边差 | **独立轮廓 + 上游错误放大** |

### 0.2 两层问题必须分开治

| 问题 | 主因 | 共享边能否单独解决 |
|------|------|--------------------|
| 拟合后两色块之间细缝 | 各区独立轮廓 / RDP / 面积缩放 | **能（正治）** |
| 软边模糊、过渡色带 | 上采样核 / 滤波 | **不能** |
| 内部杂色丝、错误小色块 | 像素独立量化 + 未空间正则 | **不能** |
| 区域切碎、标签空洞 | 丢小域标 -1、max_regions 截断 | **不能** |

### 0.3 目标架构（一句话）

> **先得到无洞、少噪、边界清楚的标签场 \(L(x,y)\)；再从 \(L\) 提取共享边平面图；高精度曲线与图章匹配只作用在共享边上。**

```text
输入图
  │
  ▼
┌─────────────────────────────────────┐
│ Batch A  自适应画布对齐（少造软边）   │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ Batch B  通用标签场概括              │
│  调色板 → 空间正则分配 → 无洞规整    │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ Batch C  共享边缘拓扑（平面图）       │
│  Vertex / HalfEdge / Face            │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ Batch D  边上高精度曲线拟合           │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ Batch E  匹配 / 层序 / 装配缝宽复检   │
│  （接入现有 P3–P9）                  │
└──────────────────┬──────────────────┘
                   ▼
         与 v3 主循环汇合 → JSON + 预览
```

### 0.4 非目标

- 不为 emoji / 金锭 / 人脸等写特判分支  
- 不恢复形状路由（圆/方/长条桶）  
- 不追求照片级纹理 PSNR  
- 不在标签场仍有空洞时「只靠 SharedEdge 假装无缝」

### 0.5 与 v3 主文档章节映射

| 本补篇 | v3 主文档 |
|--------|-----------|
| Batch A–B | §3.1 平面化 P1、§3.2 区域 P2 |
| Batch C | §3.2 邻接图升级为几何邻接 |
| Batch D | §4 轮廓分段与弧逼近 P4 |
| Batch E | §3.3 层序、§6 曲线匹配、§8 装配 |
| 验收 | §10 评分 + 新增标签场硬指标 |

---

## 1. 术语

| 术语 | 含义 |
|------|------|
| **标签场** \(L\) | 与画布同尺寸的整数图；\(L(p)\in\{-1,0,\ldots,K-1\}\)，\(-1\) 仅允许在主体外（\(\alpha<0.5\)） |
| **调色板** \(\{c_k\}\) | 每种标签的代表色（sRGB + LAB） |
| **Face（面）** | 标签场中同一标签的一个连通域，对应一个色区 |
| **SharedEdge（共享边）** | 两个 Face（或 Face 与背景）之间的一段连续边界折线；**几何只存一份** |
| **Vertex（结点）** | 边的端点：度 ≥3 的拓扑结点、轮廓起止、或强转角锁点 |
| **HalfEdge（半边）** | 有向边引用；每个 Face 的轮廓 = 半边环 |
| **主体** | \(\alpha \ge 0.5\) 的像素集合 |
| **gap** | 主体内未归属任何 Face 的像素；**目标恒为 0** |
| **细丝** | 宽度约 1、面积很小的连通标签噪声 |
| **SHAPE_BOUNDARY / OCCLUSION_CUT** | 与 v3 相同：真形状边 / 上层遮挡假边 |

---

## 2. Batch A — 自适应画布对齐（少造软边）

> **目标**：在通用前提下减少「硬边被插值成斜坡」。  
> **原则**：用可测信号选重采样，**不用**场景枚举名做唯一开关。

### 2.1 输入 / 输出

| | 内容 |
|--|------|
| 入 | 原始 RGBA、目标边长 \(S\)（默认 320） |
| 出 | 画布 RGBA、`ApproxMeta`（含实际 `resample` 模式）、可选「边缘硬度」统计 |

### 2.2 信号（通用启发式）

对主体（或整图若几乎不透明）估计：

| 信号 | 计算要点 | 用途 |
|------|----------|------|
| \(N_{\mathrm{col}}\) | 量化到 5–6 bit 后的近似独特色数 | 色数少 → 倾向最近邻 |
| 尖峰比 | 主色频率 / 次色频率 | 尖峰强 → 平面色内容 |
| 过渡带宽 | 高梯度像素占比、梯度幅度分布 | 带宽大且色少 → 多半是放大软边 |
| α 硬度 | \(\alpha\in(0.05,0.95)\) 像素占比 | 软 α 多 → 保留可分离透明 |
| 整数倍 | \(S/w_{\mathrm{src}}\)、\(S/h_{\mathrm{src}}\) 是否接近整数 | 优先整数倍最近邻，减相位误差 |

### 2.3 重采样决策（建议表）

| 条件（可调阈值） | `resample` | 说明 |
|------------------|------------|------|
| \(N_{\mathrm{col}}\le 24\) 且尖峰明显 | `nearest` | 平面色 / 像素风自动命中 |
| 可整数倍放大且 \(N_{\mathrm{col}}\le 48\) | `nearest` 按整数倍再 pad | 相位对齐 |
| 色多、梯度平滑、像照片 | `lanczos` / `bicubic` | 后续靠 Batch B 收边 |
| 默认兜底 | `bilinear` 或弱 `lanczos` | 避免极端锯齿 |

> 实现时阈值放进 `ApproxConfig`，禁止写死「gold/emoji」文件名。

### 2.4 滤波策略

| 情况 | 处理 |
|------|------|
| `resample=nearest` 且 \(N_{\mathrm{col}}\) 很低 | **关闭** bilateral / 中值（避免再糊） |
| 连续插值结果 | 仅允许 **弱** 保边平滑；禁止 strong 抹结构 |
| logo 且已硬边 | 与现逻辑一致：weak 或 off |

### 2.5 本批交付与验收

**交付**

- [ ] `fit_to_canvas`（或并列 API）支持 `nearest | lanczos | bilinear`  
- [ ] `detect_resample_mode(rgba) -> mode + stats`（纯函数、可单测）  
- [ ] `ApproxMeta` / 日志记录实际 resample 与 \(N_{\mathrm{col}}\)  
- [ ] 调试图：对齐后 RGB（现 `003_00_source_raw`）旁注模式

**验收**

| 用例 | 期望 |
|------|------|
| `examples/gold.png`（128 平面色） | 自动 `nearest`；主体近似独特色数接近原图量级（≪ 数百） |
| 普通软边插画 | 不强制 nearest；允许连续插值 |
| 回归 | 现有 pytest 不因默认路径崩溃 |

**明确不做（本批）**

- 不改 k-means / 区域拓扑  
- 不引入 SharedEdge

---

## 3. Batch B — 通用标签场概括（减杂色 / 错区 / 空洞）

> **目标**：输出 **无洞、少细丝、边界可解释** 的 \(L\) 与调色板。  
> **这是质量上限的主战场。**

### 3.1 子步骤总览

```text
B1  平坦区调色板估计
B2  初始硬分配（可带轻空间信息）
B3  空间正则（ICM / 简单图割）
B4  细丝与小洞规整
B5  无洞保证 + 连通域 Faces
B6  调试图与硬指标
```

### 3.2 B1 — 平坦区调色板

**问题**：软边过渡像素参与 k-means 会「养出」杂色中心或把边界点乱派。

**做法**

1. 在 LAB（或当前色彩空间）上算梯度幅 \(g(p)\)  
2. **平坦掩膜** \(F = \{p \mid \alpha(p)\ge 0.5,\; g(p) \le g_{\mathrm{flat}}\}\)  
3. 仅用 \(F\) 内像素估计 \(K\) 个中心（k-means++ / 层次量化）  
4. 高梯度带 **不参与中心更新**（或权重 ≪ 1）  
5. 近色合并：LAB \(\Delta E < \delta_{\mathrm{merge}}\) 且符合面积规则则合并  
6. \(K\) 选择：配置 `k_start` 仍作预算环入口；单次内可用「最小中心距 + 面积下限」压缩虚中心

**输出**：`palette: list[PaletteColor]`（含 hex、rgb、fraction、lab 可选）

### 3.3 B2 — 初始硬分配

对主体像素：

\[
L_0(p)=\arg\min_k \,\| \mathrm{lab}(p)-c_k \|
\]

可选：对高梯度像素，仅在「梯度两侧主色」子集中选，降低第三色落边概率。

### 3.4 B3 — 空间正则（通用能量）

\[
E(L)=\sum_p D(p,L_p)+\lambda\sum_{(p,q)\in\mathcal{N}} [L_p\neq L_q]\, w_{pq}
\]

| 项 | 建议 |
|----|------|
| \(D(p,k)\) | LAB 距离；可截断 Huber |
| \(\mathcal{N}\) | 4-邻接（实现简单）或 8-邻接 |
| \(w_{pq}\) | 颜色相近 → 切换代价高；**原图梯度大 → 切换代价低**（保真边界） |
| 求解 | **ICM 3～8 轮**即可作 MVP；后续可换 α-expansion |
| \(\lambda\) | 配置项；nearest 硬边内容可略大，照片略小 |

**效果预期**：1px 杂色丝、棋盘抖动并回主色；真边界因 \(w_{pq}\) 小而保留。

### 3.5 B4 — 细丝与小洞

对每个标签的连通域：

| 规则 | 动作 |
|------|------|
| 面积 \(< a_{\min}\)（相对画布比例） | 并入 **共享边界最长** 的邻接标签 |
| 宽度近似 1 且细长度极高 | 同上（细丝） |
| 被单一标签包围的小岛 | 可按面积并入包围色（小洞） |
| 并入后调色板 fraction 重算 | 是 |

**禁止**：删除后标为 \(-1\) 且留在主体内。

### 3.6 B5 — 无洞保证

**硬约束**：

\[
\{p \mid \alpha(p)\ge 0.5\} \subseteq \bigcup_f \mathrm{mask}(f)
\]

算法：

1. 找出 gap 像素  
2. 用邻域标签众数（仅主体邻域）填充；若无邻域，用最近非 gap 标签（距离变换）  
3. 再跑一轮小 CC 合并  
4. 断言 `gap_frac == 0`（debug 构建可 panic / 测试失败）

`max_regions` 截断时：

- **旧行为**：丢掉小 Face → 产生 gap（禁止保留）  
- **新行为**：超出配额的 Face **强制并入邻接**（边界最长或 LAB 最近），保持无洞；或提高配额并在预算环用层数约束

### 3.7 B6 — 硬指标与调试图

| 指标 | 定义 | 门槛（建议起点） |
|------|------|------------------|
| `gap_frac` | 主体内无 Face 像素占比 | **= 0** |
| `noise_frac` | 面积 \(< a_{\min}\) 的 Face 像素占比 | ≤ 0.5% |
| `boundary_jitter` | 标签边界与原图高梯度对齐的粗分 | 日志记录 |
| `palette_n` | 最终色数 | ≤ 配置 K 上界 |

调试图（建议文件名，接 `DebugVisualizer`）：

| 名 | 内容 |
|----|------|
| `01_planarized` | 标签着色 RGB（保持） |
| `01_labels_falsecolor` | 伪彩（保持） |
| `01b_gap_mask` | gap 高亮（应全黑） |
| `01b_flat_mask` | 平坦区掩膜（调 B1） |
| `01b_after_mrf` | 空间正则后标签 |

### 3.8 配置项（建议新增 / 调整）

| 字段 | 含义 | 建议默认 |
|------|------|----------|
| `resample_mode` | `auto\|nearest\|lanczos\|bilinear` | `auto` |
| `flat_grad_q` | 平坦区梯度分位阈值 | 0.35～0.5 |
| `mrf_lambda` | 空间正则强度 | 1.0～3.0 |
| `mrf_iters` | ICM 轮数 | 5 |
| `min_region_area_frac` | 最小 Face 面积 | 现有 0.004 可保留 |
| `lab_merge` | 近色合并 ΔE | 8～12 |
| `enforce_no_gap` | 强制无洞 | `true` |

### 3.9 本批交付与验收

**交付**

- [x] `planarize_image`（或拆出的 `label_field.py`）实现 B1–B5  
- [x] 废弃「主体内长期存在 label=-1」的路径  
- [x] `build_regions` 输入保证无洞；小域只并入不丢洞  
- [x] 单测：合成两色硬边无 gap；软边条带不产生第三色长丝  
- [x] `gold.png`：目视裂纹明显下降；`gap_frac=0`

**验收口令**

```text
gap_frac == 0
noise_frac <= ε
调试图 01b_gap_mask 全黑（主体内）
```

**明确不做（本批）**

- SharedEdge 几何  
- 改匹配粒子逻辑  

### 3.10 与「多级 K 预算环」的关系

- 外环仍可 `K = k_start … k_max`（v3 §9）  
- **每一档 K 的单次平面化都必须满足无洞**  
- 加密 K 时优先 split **高残差 Face**，而不是全局乱加中心导致碎丝  

---

## 4. Batch C — 共享边缘拓扑（平面图）

> **目标**：从无洞标签场建立 **Vertex / SharedEdge / Face**，保证相邻 Face **共边几何唯一**。  
> **前置**：Batch B 的 `gap_frac=0`。

### 4.1 数据模型（设计）

```text
Vertex {
  id: int
  x: float
  y: float
  # 度、关联 halfedge 可选
}

SharedEdge {
  id: int
  v0: int                    # 端点 Vertex
  v1: int
  left_face: int             # Face id；背景用 -1
  right_face: int
  polyline: (N,2) float      # 从 v0→v1 的采样折线（不含或含端点，实现时统一）
  length: float
  role: SHAPE_BOUNDARY | OCCLUSION_CUT | UNKNOWN
}

HalfEdge {
  edge_id: int
  direction: +1 | -1         # +1 沿 polyline，-1 逆序
  face_id: int
  next_id: int               # 环上下一条半边
}

Face {
  id: int
  region_id: int             # 与现 Region 对齐
  label: int                 # 调色板索引
  color_hex / color_rgb
  halfedge_start: int        # 外环入口
  hole_starts: list[int]     # 内环（孔）
  mask: bool(H,W)            # 可保留栅格便于覆盖损失
  area_frac, centroid, bbox
}
```

**不变量**

1. 任一 SharedEdge 的几何只存一份  
2. Face 轮廓 = 半边环拼接；相邻 Face 对同一 `edge_id` 方向相反  
3. 结点坐标一致，禁止「A 轮廓终点 ≠ B 轮廓起点」  
4. 背景边：`left_face` 或 `right_face` 为 -1，构成外轮廓  

### 4.2 从标签场提取（算法要点）

```text
C1  在像素对偶网格上标记「标签突变」的单元边（水平/垂直）
C2  将共线连续突变段链化为 polyline 草稿
C3  在度 ≠ 2 处、三色点、强转角处插入 Vertex 并拆边
C4  为每条边建两条 HalfEdge，按环绕方向串 face 环
C5  孔洞：内环方向与外环相反（与现有 outer/hole 约定一致）
C6  校验：每个 Face 环闭合；半边双射；无悬边
```

**坐标约定（需写死一种）**

- MVP：像素边界（整数格线）或像素中心链，二选一；文档与实现一致  
- 后续 Batch D 再升亚像素，但 **拓扑 id 不变**，只更新 `polyline` 点值  

### 4.3 与现有 `Region` / `AdjacencyEdge` 的迁移

| 现有 | 迁移 |
|------|------|
| `AdjacencyEdge(a,b,length)` | 由多条 `SharedEdge` 聚合 length；或直接列表化 |
| `Region.contour` | **派生视图**：沿 halfedge 环展开点列；不再独立存一份可漂移几何 |
| `Region.contour_resampled` | 对环重采样；简化时改 SharedEdge 再派生 |
| `RegionGraph` | 升级为 `PlanarMap`（或内嵌 `edges: list[SharedEdge]`） |
| `depth_order.edge_roles` | `role` 写回 `SharedEdge.role` |

兼容策略：

1. 先实现 `PlanarMap` + `to_region_graph()` 适配层，pipeline 逐步切  
2. 禁止长期双写两套可分歧轮廓  

### 4.4 简化与形态操作的约束

| 操作 | 规则 |
|------|------|
| RDP | **按 SharedEdge 做**；**端点 Vertex 锁死** |
| 面积缩放 | **禁止**单 Face 独立缩放轮廓；若需要，用法向联动或整体优化（Batch D） |
| 形态学 | 优先在标签场做完（Batch B）；拓扑建立后少做会改连通性的栅格运算 |
| 删边 | 仅当两 Face 合并时；同步改 halfedge |

### 4.5 本批交付与验收

**交付**

- [x] 新模块：`approx/planar_map.py`  
- [x] `build_planar_map(labels, palette, alpha) -> PlanarMap`（含 halfedge 环）  
- [x] 校验函数 `assert_planar_map_valid`（环闭合 + next_id）  
- [x] 单测：两色分割内部共边、半边环 walk、轮廓派生  
- [ ] 调试图：`02b_shared_edges.png`（可选增强；主路径已有 regions/contours）  

**验收**

| 项 | 期望 |
|----|------|
| 共边唯一 | 修改边几何后双侧 Face 轮廓同步变 |
| 无缝栅格 | 由 map 栅格化回标签与 \(L\) 一致（IoU≈1） |
| gold / 合成图 | 邻接边可视化无双侧错位 |

**明确不做（本批）**

- 弧拟合精度  
- 图章匹配改损失  

---

## 5. Batch D — 边上高精度曲线拟合

> **目标**：在 SharedEdge 上做 line / circle_arc / ellipse_arc / free，服务可见边界匹配。  
> **前置**：Batch C 拓扑稳定。

### 5.1 拟合对象

- **单元**：`SharedEdge`（可合并共线同角色的边链为 `EdgeChain`）  
- **非单元**：整 Face 独立多边形（仅作调试对比，不进主损失）

### 5.2 流程

```text
D1  边采样加密（弧长均匀）
D2  亚像素调整（可选）：沿法向对齐标签 SDF / 原图梯度峰
D3  分段：曲率极值、转角阈值、角色变化、结点处强制切分
D4  每段 fit：line / circle / ellipse；残差 > ε → HARD/free 密采样
D5  有损策略：与 v3 §4.3 一致，但残差相对边长定义
D6  写回：更新 EdgeChain 的 analytic 参数 + sample_points；锁 Vertex
```

### 5.3 亚像素（精度档）

| 档 | 做法 | 成本 |
|----|------|------|
| D0 | 像素边界中点 | 低 |
| D1 | 标签 SDF 零交叉 | 中 |
| D2 | 原图梯度非极大值抑制沿法向 | 中高 |

默认实现路径：先 D0 打通拓扑与匹配，再开 D1。

### 5.4 与 `contour_arcs.ArcPrimitive` 的关系

```text
ArcPrimitive {
  ...
  edge_ids: list[int]     # 新增：来源共享边
  chain_id: int | None
  # sample_points 仍为画布坐标
}
```

`extract_all_primitives` 改为遍历 **SHAPE_BOUNDARY 边/链**，而不是每区整环硬套大椭圆（整环拟合仅当单 Face 外环近似圆/椭圆且无复杂邻接时作可选加速）。

### 5.5 面积约束的重新定义

- **旧**：单 mask 多边形面积 vs 像素面积（易导致单侧缩放 → 缝）  
- **新**：  
  - Face 面积误差用 **标签 mask** 本身（栅格）作覆盖损失；  
  - 曲线侧以 **边贴合** 为主；  
  - 若需解析圆/椭圆盖住 Face，用解析形状对 mask 的 IoU，**同时**检查邻接边是否仍共享端点  

### 5.6 本批交付与验收

**交付**

- [ ] `fit_shared_edges(map, eps_arc) -> primitives`  
- [ ] 调试图：`05` 类基元图画在 shared edges 上  
- [ ] 单测：圆 mask → 边链圆拟合残差 < 阈值；两色共边 RDP 后端点不动  

**验收**

- 邻接 Face 在共边上采样点集合一致（方向可逆）  
- HARD 段不强迫单弧  

---

## 6. Batch E — 层序、匹配与装配缝宽复检

> **目标**：把 SharedEdge 接入 v3 的 P3 / P6 / P8，闭合「可见边界」回路。

### 6.1 层序 π（P3）

- 输入改为 `PlanarMap` + 现有面积/包围启发式  
- 对每条 **两 Face 之间** 的 SharedEdge 标 `SHAPE_BOUNDARY` 或 `OCCLUSION_CUT`  
- 规则与 v3 §3.3 一致；结果写 `SharedEdge.role`  

### 6.2 曲线匹配（P6）

主损失改为边上聚合：

\[
\mathcal{L}=\sum_{e\in \mathrm{SHAPE}} \big(
  \mathrm{Chamfer}(T(\gamma_t),\, e)
  +\beta\,\mathrm{Normal}
  +\gamma\,\mathrm{CoverageGap}
\big)
+\lambda_{\mathrm{soft}}\mathcal{L}_{\mathrm{soft}}
\]

| 要点 | 说明 |
|------|------|
| OCCLUSION_CUT | **不进入** 下层贴边强制项 |
| 共边只算一次 | 避免 A/B 双重计数扭曲优化 |
| IoU | 仍可作覆盖辅助，不作主损失（出画布） |
| 候选章 | 仍描述子 top-M，无形状桶 |

### 6.3 装配与缝宽复检（P8）

合成预测图后：

1. 提取预测可见边界（或标签化预测）  
2. 与目标 SHAPE SharedEdge 做 Chamfer / 最大缝宽  
3. 失败：换 beam、降级 HARD、或回退该边解析拟合  
4. 删除可见贡献 &lt; ε 的层  

**缝宽指标（建议）**

| 名 | 定义 | 用途 |
|----|------|------|
| `seam_p95` | 目标共边采样点到预测边界距离的 95 分位 | 日志 / 选优 |
| `seam_max` | 最大缝 | hard 观察 |

### 6.4 评分联动

在 `metrics` 中：

- `S_line` 增加「相对 SharedEdge 的曲线误差」权重  
- `gap_frac>0` 的候选解直接降权或否决（概括失败不进终局）  

### 6.5 本批交付与验收

**交付**

- [ ] `depth_order` / `match_curve` / `assemble` / `pipeline` 接 `PlanarMap`  
- [ ] 日志：`gap_frac`、`seam_p95`、边角色统计  
- [ ] 调试图：匹配成功时叠加 **目标共边** 与 **图章边缘**  

**验收**

| 用例 | 期望 |
|------|------|
| 合成：底大圆 + 上小圆遮挡 | 共边角色正确；下层不抠遮挡弧 |
| gold 粗阶段 | 无主体空洞；拼合无明显双边缝 |
| 回归 | ruff / pyright / pytest 全过 |

---

## 7. 批次依赖与建议工期节奏

```text
Batch A ──► Batch B ──► Batch C ──► Batch D ──► Batch E
              │                        │
              └──── 可与文档/调试图并行 ┘
```

| 批次 | 依赖 | 可独立合并 | 风险 |
|------|------|------------|------|
| A | 无 | 是 | 低：误判 nearest 伤照片 → 靠 auto 信号与配置覆盖 |
| B | A 建议先做 | 是（无 A 也能做 MRF，但软边输入更差） | 中：λ 过强抹角 |
| C | B 无洞 | 否（有洞则拓扑脏） | 中：实现量 |
| D | C | 否 | 中：椭圆拟合数值 |
| E | C+D（D 可先 D0） | 否 | 中：接 pipeline 回归 |

**推荐合并策略**

1. **PR1**：A + B 硬指标 + 调试图 + 测试（用户最先感知 `source_fitted` 变干净）  
2. **PR2**：C 拓扑 + 可视化 + 适配层  
3. **PR3**：D0 边拟合接 primitive  
4. **PR4**：E 匹配/装配/评分  

---

## 8. 代码落点（规划，非已实现）

| 模块 | 职责 |
|------|------|
| `preprocess.py` | Batch A：`detect_resample_mode`、`fit_to_canvas` 多模式 |
| `planarize.py` 或 `label_field.py` | Batch B：调色板、MRF、无洞 |
| `regions.py` | 过渡期：无洞 CC；最终变薄为 Face 适配 |
| `planar_map.py`（新） | Batch C：SharedEdge 拓扑 |
| `contour_arcs.py` | Batch D：边上基元 |
| `depth_order.py` | Batch E：角色写回边 |
| `match_curve.py` | Batch E：边聚合损失 |
| `assemble.py` / `metrics.py` | 缝宽与否决 |
| `debug_vis.py` | gap / shared edges / seam 可视化 |
| `models.ApproxConfig` | 新配置项 |
| `tests/` | gap=0、共边唯一、T 结点、回归 approx |

---

## 9. 测试计划（分批）

### 9.1 Batch A

- 合成 8 色像素块放大：auto → nearest，色数不爆炸  
- 平滑渐变图：auto 不强制 nearest  

### 9.2 Batch B

- 硬边两色：gap=0，边界 4-邻接干净  
- 软边条带（中间 lerp）：最终标签只有 2 色（或配置允许的 K），无长丝第三色  
- 随机细丝噪声：规整后 noise_frac 达标  

### 9.3 Batch C

- 左右二分、四象限、T 接、环带孔  
- `assert_planar_map_valid`  
- 栅格化往返  

### 9.4 Batch D–E

- 圆 / 矩形 / 圆+遮挡圆  
- `seam_p95` 相对基线下降  
- 现有 `tests/test_approx*.py` 全绿  

### 9.5 人工金样（非特判，只回归）

- `examples/gold.png`  
- `examples/sample_emblem` 相关预览  
- 至少 1 张多色 logo、1 张软边插画  

---

## 10. 风险与缓解

| 风险 | 缓解 |
|------|------|
| auto nearest 误伤照片 | 信号阈值保守；`resample_mode` 可强制；照片路径保留 lanczos |
| MRF 抹圆角 | 边界敏感 \(w_{pq}\)；λ 按内容自适应；圆角处梯度保护 |
| 无洞并入导致色脏 | 并入优先共享边最长；限制跨 LAB 过远并入 |
| PlanarMap 实现复杂 | MVP 只支持流形边 + T 结点；暂缓复杂自碰 |
| 双写 contour 再漂移 | 单一几何源：SharedEdge；Region.contour 只读派生 |
| 性能 | ICM 小迭代；拓扑 CPU；匹配仍 GPU |
| 与游戏渲染差 | 终局仍只信 `EmblemRenderer` |

---

## 11. 里程碑定义（完成标准）

### M1 — 概括可信

- [x] `gap_frac==0` 单测 + 管线硬约束  
- [x] gold 平面化 `gap_frac=0`、auto→nearest  
- [x] 文档与配置说明中文完整  

### M2 — 拓扑无缝

- [x] PlanarMap 校验通过（含 halfedge 环）  
- [x] 共边单几何；`face_contour` 由半边环派生  
- [x] 匹配目标曲线优先 `face_shape_boundary_points`（edge_id 去重）  

### M3 — 曲线与装配

- [x] 匹配主损失用共享边去重点 + Chamfer  
- [x] `seam_p95` 进入日志与选优  
- [x] 亚像素边 D1：`edge_subpixel` / `refine_edges_subpixel`  
- [x] v3 预算环 + 特效章仍可跑通  

### M4 — 质量冻结

- [x] 相关单测 + 全量 pytest  
- [x] ruff / pyright / pytest 全过  
- [x] 本补篇状态改为「文档设计点已落地」并回写 v3 §11 对照表  

---

## 12. 实施时的文档纪律

1. 每完成一批，在本文件对应 Batch 节首标注状态：`未开工 | 进行中 | 已落地`  
2. 行为变更同步改 v3 主文档 §3 / §4 / §11，避免两文档分叉  
3. 配置项只增不静默改默认语义；若改默认须在修订表写明  
4. 代码注解中文（见 `coding-conventions.md`）  

---

## 13. 一句话

> **Batch A/B 负责「像素属于谁」——无洞、少杂色、少软边；  
> Batch C 负责「边界是一条还是两条」——共享边消灭拼合缝；  
> Batch D/E 负责「边如何被图章解释」——高精度曲线与层栈一致。**

---

## 修订

| 版本 | 日期 | 说明 |
|------|------|------|
| 0.3 | 2026-07-19 | 补齐 halfedge 环闭合校验、亚像素边 D1、匹配 `target_curve_pts` 按 edge_id 去重 |
| 0.2 | 2026-07-19 | MVP 落地：自适应重采样、标签场无洞、PlanarMap、边上基元、seam_p95；pytest 覆盖 |
| 0.1 | 2026-07-19 | 初稿：动机分层；Batch A–E 分点设计；模型；验收；里程碑；与 v3 映射 |
