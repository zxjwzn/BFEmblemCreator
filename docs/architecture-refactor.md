# 近似管线架构（已落地，无兼容层）

> 唯一配置入口：`ModeRecipe`  
> 唯一执行入口：`ApproxEngine` / `approximate_image(recipe=...)`

---

## 调用链

```text
approximate_image(image, recipe: ModeRecipe | None)
  └─ ApproxEngine(recipe or illustration 默认)
       ├─ StampLoader → StampCatalog
       ├─ StampRenderer
       ├─ ImageProcessor → ProcessedImage
       ├─ RegionPartitioner → RegionPartition
       │     └─ curve_fit 时 simplify_planar_map_curves（chain 贝塞尔）
       │         → SharedEdge.polyline 写回后：
       │           Region.contour / face_shape_boundary_points / 匹配目标 同源
       └─ StampMatchAssembler → EmblemDocument
            ├─ Phase1 Cover-Union：每 Face 同色多章，目标=after_fit Γ_F
            ├─ Phase2 Occlusion-Carve（可选）
            └─ Phase3 ConstrainedGapFill：补缝回绑 Face/Γ_F
```

> 设计与参数：[`stamp-constructive-matching.md`](./stamp-constructive-matching.md)  
> 实现：`approx/union_cover.py`、`processors/match_assembler.py`；测试 `tests/test_constructive_cover.py`

| 组件 | 路径 |
|------|------|
| 配方/子配置 | `approx/recipe.py` |
| Engine | `approx/engine.py` |
| 处理器 | `approx/processors/` |
| 入口 | `approx/pipeline.py` |
| 数据模型 | `approx/models.py`（无扁平 ApproxConfig） |

---

## ModeRecipe

```python
from bf_emblem_creator import approximate_image, AbstractionMode, default_recipe_for_mode

recipe = default_recipe_for_mode(AbstractionMode.illustration).override(
    stamps_dir="assets/stamps",
    num_colors=6,
    max_layers=40,
    debug_dir="out/debug",
)
result = approximate_image("in.png", recipe)
```

| Mode | boundary | assets | angles |
|------|----------|--------|--------|
| logo / illustration / photo_* / silhouette | curve_fit | 全库/allowlist | free |
| pixel | dense | Square | 0° / 90° |

---

## 子配置（Pydantic）

- `StampLoaderConfig` / `ImageProcessorConfig` / `RegionPartitionerConfig`
- `StampMatchAssemblerConfig`（`AngleConstraint`、`UnionCoverConfig`）
- `StampRendererConfig` / `CurveFitConfig`
- `ModeRecipe.override(...)` 覆盖常用字段

**已删除**：扁平 `ApproxConfig`、`recipe_from_approx_config`、`budget_loop` K 外环、pipeline 内 `_run_once` 巨石。

---

## CLI

```bash
uv run bfemblem approx img.png -o out.json --mode illustration --num-colors 6
uv run bfemblem approx img.png -o out.json --mode pixel -k 8
```

内部构造 `default_recipe_for_mode(mode).override(...)`。

---

## 测试

- `tests/test_mode_recipe_engine.py`
- `tests/test_edge_curve_fit.py`
- 其余测试均改为 `ModeRecipe` / `ImageProcessorConfig`

交付门禁：`ruff` + `pyright` + `pytest`。
