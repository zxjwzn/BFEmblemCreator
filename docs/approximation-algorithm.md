# 图像近似算法设计（Stamp Matching Pursuit）

> 状态：设计稿（v0.1）  
> 前置依赖：已落地的离线渲染器（`EmblemRenderer` + `StampLayer` JSON）  
> 目标：输入任意图 → 输出 ≤40 层、可被游戏编辑器复现的 `EmblemDocument`

---

## 0. 一句话定义

在 **320×320 画布、≤40 层、256 种可染色 SVG 图章、中心原点 + 顺时针角 + 非均匀缩放 + flip + opacity** 的约束下，用**分层贪婪残差匹配（Matching Pursuit）+ 形状召回短名单 + 连续位姿精修**，把目标图像概括成可渲染的图章栈。

不追求照片级还原，追求：

1. 主体剪影可识别  
2. 主色块位置/面积大致正确  
3. 层数尽量少、结构尽量稳  
4. 输出 JSON 与离线预览、游戏内一致  

---

## 1. 问题形式化（与代码对齐）

### 1.1 决策变量（一层 = 一个 `StampLayer`）

| 字段 | 类型 | 搜索空间 |
|------|------|----------|
| `asset` | 离散 256 | 必须经召回缩到 top-k |
| `left`, `top` | 连续 | 画布中心坐标系，可越界（大章可伸出画布） |
| `width`, `height` | 连续 >0 | 允许非均匀缩放 |
| `angle` | 连续 | 度，顺时针 |
| `flipX`, `flipY` | 布尔 | 4 态 |
| `fill` | RGB | 连续或量化色板 |
| `opacity` | [0,1] | MVP 可固定 1.0 |
| 层序 | 排列 | 贪婪默认「后加在上」；后期可局部重排 |

### 1.2 前向模型（已有）

\[
I_{\mathrm{pred}} = \mathrm{Render}(L_1,\ldots,L_n),\quad n\le 40
\]

`Render` = 本文仓库的 `EmblemRenderer`（SSAA、预乘、白模染色）。  
**拟合与预览必须共用同一 Render**，避免 domain gap。

### 1.3 损失（建议默认）

在画布 α 有效区域内：

\[
\mathcal{L}
= \lambda_c\,\underbrace{\lVert w\odot(I_{\mathrm{pred}}-I_{\mathrm{tgt}})\rVert_1}_{\text{颜色}}
+ \lambda_e\,\underbrace{\lVert E(I_{\mathrm{pred}})-E(I_{\mathrm{tgt}})\rVert_1}_{\text{边缘}}
+ \lambda_a\,\underbrace{\lVert \alpha_{\mathrm{pred}}-\alpha_{\mathrm{tgt}}\rVert_1}_{\text{覆盖}}
\]

| 符号 | 含义 | 建议 |
|------|------|------|
| 颜色空间 | sRGB → 线性或 CIELAB | LAB 更稳；MVP 可用线性 RGB |
| \(w\) | 显著性/残差权重图 | 主体高、背景低 |
| \(E(\cdot)\) | Sobel 幅度或 Canny 软边缘 | 轮廓比纹理重要 |
| \(\alpha_{\mathrm{tgt}}\) | 目标不透明蒙版 | 无 alpha 时用「非背景」估计 |
| 可选 | 多尺度 pyramid loss | 粗层主导形状，细层补色 |

**不要**主用 MSE-on-pixels 死磕纹理；徽章问题是「色块 + 剪影」。

### 1.4 停止条件

满足任一即停：

- \(n = 40\)
- \(\mathcal{L} < \varepsilon\)
- 连续 \(k\) 层相对降损 \(< \delta\)（默认 k=3）
- 用户指定的时间/层数预算用尽

---

## 2. 总流程（推荐主算法）

```text
                    ┌─────────────────────────┐
  target image ──►  │ A. 概括预处理            │
                    │  resize/pad→320, 去背景  │
                    │  色量, 平滑, 显著性      │
                    └───────────┬─────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │ B. 构建/加载图章索引      │
                    │  mask, SDF, 描述子, 簇   │
                    └───────────┬─────────────┘
                                ▼
              layers=[], canvas=empty, residual=target
                                ▼
                    ┌─────────────────────────┐
              ┌───► │ C. 残差分析 → ROI 提案   │
              │     └───────────┬─────────────┘
              │                 ▼
              │     ┌─────────────────────────┐
              │     │ D. 图章召回 top-k         │
              │     └───────────┬─────────────┘
              │                 ▼
              │     ┌─────────────────────────┐
              │     │ E. 位姿+颜色离散搜索      │
              │     │    （短名单内）           │
              │     └───────────┬─────────────┘
              │                 ▼
              │     ┌─────────────────────────┐
              │     │ F. 连续精修（可选/每层）  │
              │     └───────────┬─────────────┘
              │                 ▼
              │     ┌─────────────────────────┐
              │     │ G. 接受/拒绝, 合成, 更新残差 │
              │     └───────────┬─────────────┘
              │                 ▼
              │          未满足停止？ ──yes──┘
              │                 │ no
              ▼                 ▼
                    ┌─────────────────────────┐
                    │ H. 全局精修 + 砍层/合并   │
                    └───────────┬─────────────┘
                                ▼
                         EmblemDocument JSON
                         + 预览 PNG
```

这是 **残差匹配追踪（Matching Pursuit）** 在「带 alpha 的可染色 sprite」上的版本，而不是端到端神经网络（可作为远期）。

---

## 3. 阶段详解

### A. 概括预处理（决定上限观感）

输入任意图 → 工作张量 `Target`（RGBA 或 RGB+mask），尺寸与渲染画布一致（默认 320）。

| 步骤 | 做法 | 目的 |
|------|------|------|
| 画布适配 | letterbox/cover 到 320，记录 transform | 与编辑器坐标一致 |
| 背景 | 洪水填充 / 角点色 / 用户蒙版 | 避免浪费层铺背景噪声 |
| 平滑 | bilateral 或轻微 Gaussian | 抑照片纹理 |
| 色量 | k-means / median cut，**K=4~8** | 逼近「可染色色块」域 |
| 显著性 | 中心先验 + 对比度 / 简单 U2Net（可选） | 权重图 \(w\) |
| 可选矢量 | 色区轮廓简化（RDP） | 给 ROI 提案用 |

输出：

- `I_tgt`：色量后的目标色  
- `α_tgt`：主体覆盖  
- `w`：损失权重  
- `palette`：主色列表（作 `fill` 候选）

**原则：先插画化，再拟合。** 照片直接拟合会把 40 层烧在噪点上。

---

### B. 图章索引（构建期，256 一次）

对每个 `asset` 预计算（缓存到磁盘）：

1. **多分辨率 alpha mask**（如 32/64/128）  
2. **SDF**（signed distance）——旋转/尺度对齐很稳  
3. **形状描述子**（任选组合）  
   - 7 Hu 矩 / 傅里叶描述子  
   - 径向直方图 + 孔洞数  
   - 主轴细长度、圆度、凸度、对称性  
4. **cluster_id**：近重复聚类（Square/盾牌变体/环等）  
5. **tags**：`basic | elongate | annular | symbol | complex`（可自动+手工）

**MVP 子集（强烈建议第一期只用这些跑通）：**

```text
Square, Circle, Triangle, RightTriangle, HalfCircle, OpenCircle,
Drop, Line, Stroke, StrokeBent, BentLine, Arrow, ArrowBent,
SquareCorner, CrescentMoon, Banner, Shield, Wingpart
```

约 15–25 个几何章，覆盖大部分色块徽章；256 全库放 Phase 2。

---

### C. 残差分析 → ROI 提案

当前合成 `I_pred`（初始全透明或可选底色一层）。

\[
R = I_{\mathrm{tgt}} - I_{\mathrm{pred}}
\quad\text{（在 α 与颜色通道上定义）}
\]

提案来源（可并行，取 top-m）：

| 提案器 | 方法 | 适合 |
|--------|------|------|
| 色块 | 残差图上 connected components（按 palette 色） | 大色区 |
| 峰值 | `|R|` 模糊后 local max | 局部错误 |
| 轮廓环 | 目标边缘 − 当前边缘 | 缺轮廓处 |
| 覆盖洞 | `α_tgt > 0.5` 且 `α_pred` 低 | 空洞 |
| 可选 | 超像素 / SLIC on residual | 碎片区 |

每个 ROI 带：bbox、mask、主色、面积、圆度/细长度等几何特征。

**层预算启发式：**

- 前 5 层：只接受「大面积 ROI」（>画布 3~5%）  
- 中段：中等块  
- 最后 10 层：小细节，且单层最小降损阈值提高  

---

### D. 图章召回（256 → k）

对每个 ROI 特征 \(f\)，从索引取：

\[
\mathrm{top\text{-}k}(f),\quad k \in [8, 24]
\]

规则路由（快且有效）：

| ROI 特征 | 优先章 |
|----------|--------|
| 圆度高 | Circle, OpenCircle, Drop, MedalRing* |
| 近似矩形 | Square, Banner, Shield* |
| 三角 | Triangle, RightTriangle |
| 细长 | Line, Stroke, StrokeBent, Arrow* |
| 弯月/缺口 | CrescentMoon, HalfCircle |
| 复杂/失败 | 回退 basic 集 + 代表元簇 |

复杂 Emblem/动物章：**仅当** ROI 与描述子异常接近时启用，避免早期乱入。

---

### E. 短名单内离散搜索（核心）

对候选 `asset` × flip × 粗角度 × 粗尺度，在 ROI 内求最佳放置。

#### E.1 几何目标（比直接 RGB 更稳）

用 **mask 域** 匹配为主：

\[
\mathrm{score}
= \mathrm{IoU}(\hat\alpha, \alpha_{\mathrm{ROI}})
+ \beta\cdot\mathrm{NCC}(\mathrm{SDF}_{stamp}, \mathrm{SDF}_{\mathrm{ROI}})
- \gamma\cdot\mathrm{colorPenalty}
\]

颜色：

\[
\mathrm{fill}^\star
= \mathrm{median}\{\,I_{\mathrm{tgt}}(p) \mid \hat\alpha(p)>\tau\,\}
\]

或投射到 `palette` 最近点。

#### E.2 位姿参数化

编辑器是 **中心 + width/height + angle**，搜索时建议：

1. 用 ROI 主轴得到初始 `angle`、中心 `(left,top)`  
2. `width/height` 由 ROI bbox 与图章 mask 的 AABB 比初始化  
3. 角度网格：粗 15°，细 3°（或根据对称性减半）  
4. 尺度：0.7/1.0/1.3 × 初始，再局部三分搜索  

**加速：**

- 在 64² / 128² 残差金字塔上搜，不在 320 SSAA 上搜  
- FFT / 滑动窗口仅对「平移」在固定角度尺度下使用  
- 同 cluster 只搜代表元，命中后再比簇内 2–3 个变体  

#### E.3 真正的降损验收

离散最优候选必须过 **渲染器前向**：

```text
trial = layers + [candidate]
L_new = loss(Render(trial), target)
accept if L_new < L_cur - min_gain
```

只用 IoU 容易「形状对、颜色/遮挡错」；**以 Render+Loss 为最终裁判**。

---

### F. 连续精修（每层或每 N 层）

固定 `asset`、flip；优化连续量：

\[
\theta = (\mathrm{left},\mathrm{top},\mathrm{width},\mathrm{height},\mathrm{angle},\mathrm{fill}_{rgb}[, \mathrm{opacity}])
\]

方法（按工程难度）：

| 级别 | 方法 | 说明 |
|------|------|------|
| F0 | 坐标扰动 / Nelder-Mead | 无梯度，易接现有 Render |
| F1 | 有限差分 + Adam | 慢但简单 |
| F2 | 可微近似渲染 | `grid_sample` 白模；与 PIL 路径略有差别，最后用真 Render 投影 |

约束：

- `width,height ≥ w_min`（游戏最小尺寸，待测，可先 4px）  
- 中心可在扩展 bbox 内  
- `fill` 可锁 palette 或中点松弛  

**注意：** 精修应用 **ss=1 或 2** 的快速 Render；最终预览再用 ss=4。

---

### G. 接受准则与残差更新

```text
if gain < min_gain: reject ROI, try next proposal
else:
  layers.append(layer)
  I_pred = Render(layers)
  residual = target - I_pred
```

可选 **backfitting**：每加 3~5 层，对最近几层再精修一轮（经典 MP 的 orthogonal 近似）。

---

### H. 全局后处理

| 操作 | 目的 |
|------|------|
| 遮挡剔除 | 若一层对最终 α 贡献 < ε，删除 |
| 同色合并 | 同 asset+近似位姿+同色 → 尝试合并失败则保留 |
| 层序局部搜索 | 对相邻层尝试交换，若 loss 降则接受 |
| 底色层 | 若背景大面积单色，强制 layer0 = 全画布 Square |
| 配额重分配 | 删层后若 budget 有余，再跑一轮小 ROI |
| 最终精修 | 全部连续参数联合微调（小学习率） |

输出：`EmblemDocument` + 预览 + 每层 gain 报告（可解释性）。

---

## 4. 损失与分辨率策略（对接抗锯齿结论）

| 阶段 | 分辨率 | SSAA | 损失 |
|------|--------|------|------|
| 召回/离散搜 | 64–128 | 1 | mask IoU + SDF |
| 降损验收 | 320 | 1–2 | 完整 \(\mathcal{L}\) |
| 连续精修 | 160–320 | 1–2 | \(\mathcal{L}\) |
| 导出预览 | 320 | 4 | 仅可视化 |

拟合时对 `I_tgt` 与 `I_pred` 做**相同**轻度模糊（σ≈0.5）可降锯齿噪声，但不改变 JSON。

---

## 5. 算法伪代码

```python
def approximate(image, stamps_index, max_layers=40) -> EmblemDocument:
    tgt, alpha, weight, palette = preprocess(image)      # A
    layers = []
    # 可选：大面积背景 Square
    pred = render(layers)
    loss = full_loss(pred, tgt, alpha, weight)

    while len(layers) < max_layers:
        rois = propose_rois(tgt, pred, alpha, weight)    # C
        best = None
        for roi in rois[:M]:
            candidates = recall(stamps_index, roi, k=16) # D
            for asset in candidates:
                layer0 = discrete_search(asset, roi, palette)  # E
                layer1 = continuous_refine(layer0, layers, tgt) # F
                trial = layers + [layer1]
                L = full_loss(render(trial), tgt, alpha, weight)
                if best is None or L < best.loss:
                    best = (L, layer1)
        if best is None or loss - best.loss < min_gain:
            break
        layers.append(best.layer)
        pred = render(layers)
        loss = best.loss
        if loss < eps:
            break

    layers = global_postprocess(layers, tgt, alpha, weight)  # H
    return EmblemDocument(layers)
```

---

## 6. 分期实现（与仓库模块映射）

### Phase 1 — 可演示 MVP（建议 1–2 周量级）

**范围：** basic 图章子集 + 色量目标 + 纯贪婪。

| 模块 | 路径建议 | 内容 |
|------|----------|------|
| 预处理 | `approx/preprocess.py` | 320、色量、背景 mask |
| 索引 | `approx/index.py` | 子集 mask+SDF 缓存 |
| 提案 | `approx/propose.py` | 残差色块 CC |
| 搜索 | `approx/search.py` | 角度/尺度网格 + median fill |
| 管线 | `approx/pipeline.py` | MP 主循环 |
| CLI | `bfemblem approx in.png -o out.json` | 一键 |

**成功标准：** 对 logo/简单头像，10–25 层内可辨认；JSON 可 `render` 回放。

### Phase 2 — 全库 256

- 描述子召回 + cluster  
- 金字塔搜索  
- backfitting  
- 层贡献剪枝  

### Phase 3 — 质量与速度

- 可微精修（PyTorch）  
- GPU 批量 score  
- 用户交互：锁定区域 / 禁止章 / 强制底色  
- 多假设 beam search（保留 top-B 层栈）  

---

## 7. 关键设计选择（决策记录）

### 7.1 为什么是贪婪 MP，而不是一次整数规划 / 纯网络？

- 40 层 × 256 章 × 连续位姿，联合离散优化不现实  
- 已有可微度有限的真实 Render（PIL/MuPDF），与游戏一致优先  
- MP 可解释、可中断、可加规则（徽章先验）  
- 网络方案缺大数据标签；可作为重排/召回补充，不作 MVP 主路径  

### 7.2 为什么 mask/SDF 先于 RGB？

- 图章是**硬形状 + 整章单色**，信息在边界不在纹理  
- 残差 RGB 受遮挡与层序影响大；局部 mask 更可分  

### 7.3 颜色怎么来？

整章单 `fill` ⇒ 最优色近似为覆盖区域内的目标色稳健统计量（median）。  
多色区域必须 **拆层** 或 **上盖**，不要幻想单章渐变。

### 7.4 opacity 用不用？

- MVP：固定 1.0（游戏里半透明易脏、难对齐）  
- 后期：仅用于软阴影/半透明装饰，且预算层紧时慎用  

### 7.5 复杂装饰章（Emblem1xx、动物）

作 **「高代价特殊词元」**：仅当 SDF 匹配分显著优于 basic 时选用。  
否则 40 层会被个别复杂章锁死局部。

---

## 8. 失败模式与对策

| 现象 | 原因 | 对策 |
|------|------|------|
| 层数用尽仍糊 | 未色量/抠图 | 加强 A；先剪影后填色 |
| 边缘双线/毛刺 | 重复描边层 | 边缘 gain 阈值；禁止同位置细条叠 |
| 颜色发灰 | 多章半透明或错误 median | opacity=1；覆盖区重估 fill |
| 大斜条对不齐 | 角度网格粗 | 主轴初始化 + 连续精修 |
| 细节乱飞 | 后期 ROI 过小 | 最小面积/最小 gain |
| 与游戏不一致 | 拟合用了不同 Render | **只准用 EmblemRenderer** |
| 搜索太慢 | 全库滑窗 | 召回 k≤16；64² 搜索；缓存 SDF |

---

## 9. 评估指标

| 指标 | 用途 |
|------|------|
| \(\mathcal{L}\) 全损失 | 主优化 |
| mask IoU（主体） | 剪影 |
| 色调直方图 L1 | 大色是否对 |
| 层数 / 有效层数 | 预算 |
| 人工 1–5 可识别分 | 产品真实标准 |
| 时延 | CLI 体验（MVP 目标：CPU < 60s/图） |

建立 **20 张黄金集**：简单几何 / 军章 / 二值 logo / 二次元头像剪影 / 用户真实图各若干。

---

## 10. 与现有代码的接口契约

```text
输入:  PIL.Image | Path
输出:  EmblemDocument  (已有 Pydantic)
校验:  EmblemRenderer.render(doc) → 预览
导出:  doc.save_json(...)
```

近似模块 **禁止** 另写第二套合成；最多为速度做「可微代理」，但接受层前必须真 Render 过线。

新增模型建议（仍 Pydantic）：

- `ApproxConfig`：max_layers, min_gain, palette_k, stamp_subset, pyramid_sizes…  
- `StampIndex` / `StampFeatures`：构建缓存元数据  
- `RoiProposal`：bbox, mask, score, suggest_tags  
- `ApproxResult`：document, final_loss, per_layer_gains, debug_images  

---

## 11. 建议的默认超参（MVP）

| 参数 | 默认 | 说明 |
|------|------|------|
| max_layers | 40 | 硬上限 |
| min_gain | 相对 loss 0.5% 或绝对阈值 | 防碎层 |
| palette_k | 6 | 色量 |
| recall_k | 12 | 短名单 |
| angle_coarse | 15° | 离散 |
| search_res | 128 | 几何匹配 |
| accept_ss | 1 | 验收渲染 |
| refine | 每层 Nelder-Mead 30–50 iter | F0 |
| min_roi_area | 0.5% 画布 | 后期可降 |

---

## 12. 总结：算法身份

| 名称 | 含义 |
|------|------|
| 主结构 | **Alpha-Sprite Matching Pursuit** |
| 形状 | SDF/IoU 召回 + 短名单位姿网格 |
| 颜色 | 覆盖区 median → `fill` |
| 精修 | 连续参数局部优化 |
| 裁判 | **同一套 EmblemRenderer + 复合损失** |
| 产品哲学 | **概括式插画拟合**，非照片重建 |

---

## 13. 下一步落地顺序（实现时）

1. `ApproxConfig` + `bfemblem approx` 空管线（读图 → 空 JSON）  
2. 预处理 + 可选「整图主色 Square 打底」  
3. basic 索引 + 单 ROI 拟合一层（人工可视化）  
4. MP 循环到 N 层  
5. 连续精修 + 砍层  
6. 扩展到 256 召回  

---

## 修订

| 版本 | 日期 | 说明 |
|------|------|------|
| 0.1 | 2026-07-19 | 初稿：MP 主路径、阶段、接口、分期、超参 |
