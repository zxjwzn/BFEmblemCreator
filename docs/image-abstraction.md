# 原始图像概括（Abstraction）设计

> 状态：设计稿 v0.1  
> 在近似管线中的位置：`原图 → 【概括】→ 形状分解/匹配 → ≤40 层图章`  
> 相关：[`approximation-algorithm.md`](./approximation-algorithm.md) 阶段 A

---

## 0. 为什么必须概括

战地图章的表达能力极度受限：

| 能力 | 图章世界 | 普通照片/原图 |
|------|----------|----------------|
| 颜色 | 每层 **一个** `fill` | 连续渐变、纹理、噪声 |
| 形状 | 256 个固定剪影 + 仿射 | 任意轮廓与细节 |
| 层数 | ≤ **40** | 等效「无限笔触」 |
| 混合 | 基本是不透明覆盖 | 半透明、阴影、高光 |

若把原图像素当拟合目标，优化会把层数浪费在：

- 皮肤/布料纹理  
- JPEG 噪点、抗锯齿杂边  
- 背景杂物、渐变天空  
- 头发丝、瞳孔高光等微细节  

**概括的目标**：把原图变成「用 40 个单色剪影拼得出来」的中间表示，而不是更清晰的照片。

成功标准（比 PSNR 重要）：

1. **几秒内能认出是谁/什么 logo**（剪影）  
2. **主色约 3–8 块**，位置稳  
3. **去掉**纹理、微渐变、背景噪声  
4. 输出可直接作为 Matching Pursuit 的 `I_tgt + α_tgt + w + palette`

---

## 1. 概括在整体中的位置

```text
用户原图 (任意尺寸/照片/插画/logo)
        │
        ▼
┌───────────────────────────────────────┐
│  Stage 0  画布对齐                     │  → 320×320 工作域
│  Stage 1  去背景 / 主体蒙版            │  → α_tgt
│  Stage 2  结构保持平滑                 │  → 去纹理，留边界
│  Stage 3  色彩量化 (K=4~8)             │  → 插画色块
│  Stage 4  色区规整 (形态学/小区合并)    │  → 可盖章的大块
│  Stage 5  显著性 / 损失权重            │  → w(x,y)
│  Stage 6  (可选) 分层与矢量轮廓        │  → ROI 种子
└───────────────────────────────────────┘
        │
        ▼
ApproxTarget {
  image_rgb,     # 概括后的目标色 (H,W,3)
  alpha,         # 主体 0~1
  weight,        # 损失权重
  palette,       # [(hex, fraction), ...]
  layers_hint,   # 可选：底色/主轮廓/细节优先级
  meta           # 原始尺寸、letterbox 变换等
}
```

下游 **只拟合 `ApproxTarget`**，不再看原图像素。

---

## 2. 按输入类型选择策略

同一套默认管线 + 模式开关：

| 模式 | 典型输入 | 概括重点 |
|------|----------|----------|
| `logo` | 扁平 logo、旗帜、徽章 | 几乎只需色量+锐边缘；少平滑 |
| `illustration` | 二次元/矢量风 | 中等平滑；保留清晰色界 |
| `photo_portrait` | 人脸照片 | 强平滑+少色+主体抠图；五官只留大块 |
| `photo_general` | 风景/物品照 | 显著性裁切+大幅色量；背景可丢 |
| `silhouette` | 已是剪影 | 单色/双色；专注轮廓 |

MVP 可自动粗分类：

- 颜色数已很少、大面积恒定 → `logo`  
- 人脸检测命中 → `photo_portrait`  
- 否则 → `illustration` / `photo_general`

---

## 3. 分阶段算法（推荐默认）

### Stage 0 — 画布对齐

**输入**：任意图像  
**输出**：`canvas_rgba` 320×320（与 `EmblemRenderer` 一致）

```text
1. 若有 alpha，保留；否则 alpha=1
2. 选择 fit 模式：
   - contain（letterbox）：完整装入，空余透明或纯色
   - cover：中心裁切填满，少黑边
3. 高质量缩放 (LANCZOS / area)
4. 记录 transform：scale, offset_x, offset_y
```

| 输入类型 | 推荐 fit |
|----------|----------|
| 完整 logo / 需保留全身 | `contain` |
| 头像 / 主体居中照片 | `cover`（默认对 photo） |

**不要**对非方图做非等比硬拉伸（除非用户明确要求）。

---

### 颜色空间约定

- 色量、色差：优先 **CIELAB**  
- 滤波：可在 sRGB 或线性 RGB  
- 输出给下游 / `fill`：稳定 **sRGB `#RRGGBB`**

---

### Stage 1 — 去背景 / 主体蒙版 \(\alpha\)

没有可靠 \(\alpha\)，层会浪费在背景上。

| 方法 | 适用 | 复杂度 |
|------|------|--------|
| **A. 角点/边缘洪水填充** | 纯色或简单背景 logo | 低，MVP 首选 |
| **B. 色键** | 绿幕/白底 | 低 |
| **C. 用户蒙版/矩形/背景色参数** | 交互 | 低，产品必备 |
| **D. 显著性** | 单主体照片 | 中 |
| **E. 抠图模型**（rembg / U2Net） | 人像/物品 | 中高，可选 |

**MVP 推荐：**

```text
α = flood_fill_from_corners(tolerance)
α = morph_close(α); keep_largest_cc(α)
if coverage(α) < 5% or > 98%:   # 自动失败
    α = saliency_threshold() or ones()
```

软边可保留 1–2px；进入色量前也可用 `α > 0.5` 硬化，便于连通域。

---

### Stage 2 — 结构保持平滑（去纹理，留边界）

目的：去掉毛孔、噪点、细纹理，**尽量保留色块边界**。

| 滤波 | 特点 | 建议 |
|------|------|------|
| **双边滤波 bilateral** | 保边去噪 | MVP 默认 |
| **引导滤波 guided** | 更快、边界稳 | 有 OpenCV 时优先 |
| **mean-shift** | 强力色块化 | 慢；logo 慎用 |
| **轻 Gaussian σ≈0.5** | 易糊界 | 仅辅助 |
| **形态学开闭** | 规整 α / 色区 | Stage 4 主力 |

强度：

- `logo`：弱或跳过  
- `illustration`：中等 bilateral 1 次  
- `photo_*`：较强 bilateral 1–2 次（如 d=7–9, σ_color=40–60, σ_space=7–9）

**禁止**锐化：会制造假色界，把层数骗进噪声边缘。

---

### Stage 3 — 色彩量化（概括的核心）

千万色 → **K 个代表色**。每层图章只有一个 `fill`，目标色数应接近「40 层里能养活的色块数」。

#### 3.1 K 的选择

| 场景 | K |
|------|-----|
| 极简 logo | 2–4 |
| 一般徽章/头像 | **5–8（默认 6）** |
| 复杂插画 | 8–12（仍可能不够；宁小勿大） |

经验法则：

```text
K_eff ≈ min(K, max_layers / 3)
```

同色还要多层拼形状，K 太大只会得到「看起来高级、盖不出来」的目标。

#### 3.2 算法

| 算法 | 说明 |
|------|------|
| **K-means in LAB**（仅 α>0.5 像素） | MVP 默认 |
| Median cut | 经典调色板备选 |
| K-means++ 多次初始化取最佳 | 减抖动 |
| SLIC 超像素均色 → 再聚类 | 空间更稳，Phase 2 |

```text
pixels = { LAB(p) | α(p) > 0.5 }
palette = kmeans(pixels, K)
label(p) = nearest_palette(LAB(p))   # α 外可标 background
image_q(p) = sRGB(palette[label(p)])
```

#### 3.3 调色板后处理

1. **合并过近色**：LAB 距离 < τ 则并簇  
2. **剔极小色**：面积比 < 0.5% 并入最近色（中心/高权重区可豁免）  
3. **按面积排序**：`palette[0]` → 底色候选  
4. 导出 `[(#RRGGBB, fraction), ...]` 供搜索时的 `fill` 候选

#### 3.4 全局量化的局限与改进

纯全局 k-means 可能把「不相邻同色」或「受光/阴影」绑成脏色。改进路径：

- 超像素均色后再 k-means  
- 仅在平滑后的图上量化（Stage 2 要够）  
- 用户锁定主色（品牌色）

---

### Stage 4 — 色区规整（让块「像章能盖住」）

量化后常见椒盐碎点：

```text
for each color label k:
    mask = (label == k) & (α > 0.5)
    mask = open(mask, 1px)     # 去毛刺
    mask = close(mask, 2px)    # 补小洞
    drop CC with area < A_min  # 并入邻域众数色
rewrite image_q
```

`A_min` 建议：**画布面积的 0.15%–0.4%**（320×320 约 150–400 px）。  
更小的块用基础图章也盖不准，不如并掉，省层数。

可选（logo）：

- 轮廓 `approxPolyDP` 后再填充 → 边界更贴纸化、更易被 Square/Triangle 命中  

---

### Stage 5 — 显著性与损失权重 \(w\)

\[
w = \mathrm{normalize}(
  w_\alpha \alpha + w_s S + w_e E + w_c C
)
\]

| 项 | 含义 | 备注 |
|----|------|------|
| \(\alpha\) | 主体内 | 基础门控 |
| \(S\) | 显著性 | 无模型：对比度+中心；有模型更好 |
| \(E\) | 概括图边缘强度 | **轮廓优先**，权重要给够 |
| \(C\) | 中心高斯 | 头像开、满版 logo 关 |

归一化使 `mean(w)≈1`，便于 `min_gain` 等阈值跨图稳定。

---

### Stage 6 —（可选）结构分层提示

不是最终图章层，而是给 MP 的「施工顺序」：

| 提示 | 来源 | 下游 |
|------|------|------|
| 底色 | 面积最大 palette 色 | 强制先铺大 Square |
| 主剪影 | α 最大连通域 | 优先大 Circle/Shield/Square |
| 主色块列表 | 各色大 CC 按面积排序 | ROI 提案顺序 |
| 细节区 | 小 CC + 高边缘带 | 仅最后约 10 层预算 |

几何特征（圆度、细长度、矩形度、凸度）从 CC 或简化轮廓计算，供图章召回路由。

---

## 4. 端到端伪代码

```python
def abstract_image(image, mode="auto", k=6, size=320) -> ApproxTarget:
    mode = detect_mode(image) if mode == "auto" else mode
    fit = "cover" if mode.startswith("photo") else "contain"

    rgba, meta = fit_to_canvas(image, size=size, how=fit)
    alpha = estimate_alpha(rgba, mode=mode)
    rgb = rgba[:, :, :3]

    if mode != "logo":
        rgb = bilateral(rgb, strength=mode)

    labels, palette = quantize_lab(rgb, alpha, k=k)
    labels, palette = merge_tiny_and_close(labels, palette, alpha)
    labels = regularize_regions(labels, alpha, a_min=0.002 * size * size)
    image_q = palette_lookup(labels, palette)

    weight = build_weight(alpha, image_q, mode=mode)
    hints = build_layer_hints(labels, palette, alpha)  # optional

    return ApproxTarget(
        image_rgb=image_q,
        alpha=alpha,
        weight=weight,
        palette=palette,
        layers_hint=hints,
        meta=meta,
    )
```

---

## 5. 参数默认表（MVP）

| 参数 | 默认 | 说明 |
|------|------|------|
| canvas | 320 | 与渲染器一致 |
| fit | photo→cover, else contain | |
| K | 6 | 色量 |
| bilateral | photo 开 / logo 关 | |
| A_min | 0.2% 画布 | 最小色区 |
| 近色合并 τ | LAB ≈ 8–12 | |
| 极小色 | <0.5% 面积 | 并入近色 |
| 边缘权重 w_e | ≥ 颜色项 | 剪影优先 |

---

## 6. 可视化调试（强烈建议做）

概括阶段应导出拼图，便于调参：

```text
[原图缩放] [α蒙版] [平滑后] [量化后] [色区边界] [权重 w]
```

没有这张调试图，后续拟合出问题会分不清是「概括过死」还是「匹配太差」。

---

## 7. 好坏对照（设计直觉）

| 原图现象 | 错误概括 | 正确概括 |
|----------|----------|----------|
| 人脸照片 | 保留皮肤渐变 20 色 | 肤色 1–2 块 + 头发块 + 五官大色 |
| 有杂乱背景 | 背景一起量化 | α 裁掉背景 |
| 渐变天空 | 一层层条纹色 | 并成 1 色或当背景丢弃 |
| 细线文字 | 碎成散点 | 并入底色或整块色带（文字本就难） |
| 扁平 logo | 过度模糊 | 少平滑、保锐界、K=原色数 |

**文字、精细纹章、真实发丝**：概括阶段就应承认「还原不了」，改为色块/剪影近似，或留给用户手修。

---

## 8. 与拟合算法的接口契约

```text
ApproxTarget
├── image_rgb: uint8 (H,W,3)     # 只含概括色，无纹理
├── alpha:     float (H,W)       # 主体
├── weight:    float (H,W)       # loss 权重
├── palette:   list[PaletteColor]
├── layers_hint: optional
└── meta:      canvas size, fit, mode, transforms
```

约定：

1. 拟合损失只对 `ApproxTarget` 计算  
2. `fill` 搜索优先从 `palette` 取，再允许小范围连续微调  
3. `α≈0` 区域不提案 ROI（除非用户要背景层）  
4. 调试时允许把 `image_rgb` 当「假原图」单独 `render` 对比  

Pydantic 模型建议：`ApproxConfig`, `PaletteColor`, `ApproxTarget`, `ApproxMeta`。

---

## 9. 分期

### P0 — 最小可用

- fit 320 + 角点洪水 α  
- LAB k-means（K=6）  
- 小区合并  
- 导出调试拼图 + `ApproxTarget`  

### P1 — 稳

- bilateral  
- 模式 auto（logo/photo）  
- 权重图 w  
- 用户背景色 / mask 输入  

### P2 — 强

- rembg 可选  
- SLIC 空间量化  
- 轮廓简化 + layers_hint  
- 品牌色锁定  

---

## 10. 一句话策略

> **先变成「印刷贴纸风」的少色块画，再交给图章去盖。**  
> 概括越克制（色越少、块越大、背景越干净），40 层越够用；  
> 概括越想留住照片真实感，拟合越必然失败。

---

## 修订

| 版本 | 日期 | 说明 |
|------|------|------|
| 0.1 | 2026-07-19 | 初稿：六阶段管线、模式、参数、接口、分期 |
