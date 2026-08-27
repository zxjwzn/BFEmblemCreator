# 徽章离线渲染器

> 战地1 图章编辑器的**离线数字孪生**：输入编辑器导出的徽章 JSON，输出 PNG。
> 它既是最终交付物（JSON→图），也是学习模型的**评测/导出**后端。

---

## 1. 徽章 JSON 格式（与编辑器导出一致）

徽章是一个**有序图章层栈**，下标 0 为最底层，按下标从底到顶依次叠加（画家算法）。

`StampLayer` 字段（`src/bf_emblem_creator/models.py`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `asset` | str | 图章 id，对应 `assets/stamps/{asset}.svg`（校验时剥掉 `.svg` 后缀） |
| `left` / `top` | float | 图章**中心点**坐标（画布像素，可在画布外） |
| `width` / `height` | float | **旋转前**包围盒宽高（可远大于画布，大章只露局部） |
| `angle` | float | 旋转角（**顺时针**，度） |
| `flipX` / `flipY` | bool | 水平 / 垂直镜像 |
| `fill` | `#RRGGBB` | 染色颜色（规范为大写十六进制） |
| `opacity` | float | 不透明度 0–1（默认 1） |
| `selectable` | bool | 编辑器 UI 字段；渲染忽略，导出时保留 |

`EmblemDocument = RootModel[list[StampLayer]]`，提供 `load_json` / `save_json` / `from_layers`。

## 2. 渲染管线（`render.py` 的 `EmblemRenderer`）

逐层「光栅化 → 染色 → 几何变换 → 中心原点 alpha 叠加」，最后整体降采样：

1. **光栅化**：`rasterize_svg` 把 SVG 白模按 `width/height × 超采样倍率` 渲成 RGBA（见 §3）。
2. **染色**：把图章当形状蒙版，**覆盖率只信 alpha**，凡 alpha>0 处 RGB 一律写 `fill`
   （`fill` 为单色的前提；直通 alpha，避免把残留白/彩带进描边）。
3. **几何变换在预乘 alpha 空间进行**：先镜像（flipX/flipY）、双线性缩放到 `width×height`，
   再旋转。**角度取负**——编辑器角为顺时针，PIL `rotate` 为逆时针；`expand=True` 保持视觉中心。
4. **中心原点叠加**：`left/top` 是未旋转包围盒中心，`alpha_composite` 到（可能放大的）画布。
5. **降采样（SSAA）**：在 **float 预乘 alpha** 空间做盒式/面积平均，避免 uint8 预乘在低 alpha
   处反预乘爆炸（白/彩边）；全透明像素 RGB 清零。

> 关键不变量：几何变换一律在**超采样高分辨率**下完成，再降到目标画布，
> 避免「先降到 320 再旋转」的台阶锯齿。默认 `supersample=4`（320 画布内部按 1280 渲染）。

## 3. SVG 光栅化（`raster.py`）

- 基于 **PyMuPDF**（`fitz`）：SVG 作为单页文档打开，按 `out_width×out_height` 非等比缩放光栅化
  （以匹配编辑器对原生宽高比不同的图章做 `width/height` 覆盖）。
- 输出 `shape=(H,W,4)` 的 RGBA uint8；带 `lru_cache`（按路径+尺寸）。
- 缺 alpha 通道时补 255；偶发尺寸舍入偏差用最近邻拉回目标尺寸。

## 4. 图章库（`stamps.py` 的 `StampLibrary`）

- 基于文件系统：`{id}.svg` 目录；`list_ids()` 列出全部 id，`resolve(asset)` 解析为 `StampAssetInfo`。
- 大小写不敏感回退（Windows/Linux 差异）；读取 SVG 原生宽高（优先 `viewBox`，其次 `width/height`）。
- 图章为**白模**：形状信息全在 alpha；透明背景，无不透明底矩形。

## 5. 配置

`CanvasConfig`：`width/height`（默认 320×320）、`background`（null = 透明）。
`RenderConfig`：`canvas`、`stamps_dir`、`supersample`（1–8，默认 4）、`stamp_raster_scale`（1–4，默认 1.5）。
坐标约定仅支持 `origin="center"`（与编辑器导出一致）。

## 6. 命令行

```bash
uv run bfemblem render examples/sample_emblem.json -o out/sample.png   # JSON → PNG
uv run bfemblem validate examples/sample_emblem.json                  # 校验 JSON 与 asset 存在性
uv run bfemblem list-stamps                                           # 列出图章 id
uv run bfemblem export-json in.json -o out.json                       # 校验并重写
```

## 7. 与编辑器对齐的要点（数字孪生验收）

- **坐标系**：画布 320×320，原点左上；`left/top` 为中心点。
- **角度方向**：导出 JSON 顺时针；渲染时取负传给 PIL。
- **染色模型**：自由 RGB（`fill`），白模 × 颜色；不支持逐层渐变/多色。
- **验收**：给定同一层列表，离线合成与游戏内预览在几何与颜色上足够一致。

## 8. 与学习模型的关系

本渲染器是**最终高保真导出与评测**的后端。学习模型的训练内环另需一个 **GPU 可微批量渲染器**
（`grid_sample` 硬 alpha 即可提供足够梯度）作为「环境」，规格见 `docs/research/emblem-rl-model.md` §2。
两者职责分离：可微版服务训练，本渲染器锚定交付与评测口径。
