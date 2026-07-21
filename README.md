# BF Emblem Creator

战地图章徽章工具：离线渲染编辑器 JSON，并按 **可见边界曲线拟合 + 共享边拓扑 + 线条质量 + GPU 粒子** 做自动近似。

## 规范

- **注释 / 文档字符串 / CLI 说明 / 用户文案：一律中文**（见 [docs/coding-conventions.md](docs/coding-conventions.md)）
- 数据结构使用 Pydantic；依赖使用 **uv** 管理
- 近似搜索优先 **CUDA**（`torch` cu128）

## 环境

```bash
uv sync --all-groups
# 确认 GPU
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## 徽章 JSON 格式

与编辑器导出一致（图层列表，**底层 → 顶层**）：

- `asset` → `assets/stamps/{asset}.svg`
- `left` / `top` → 图章**中心点**（可在画布外）
- `width` / `height` → 可**远大于**画布（大章只露局部）
- `angle` → 顺时针（度）
- `fill` → `#RRGGBB`

## 命令行

```bash
# JSON → PNG
uv run bfemblem render examples/sample_emblem.json -o out/sample.png

# 图像 → 图章 JSON
uv run bfemblem approx examples/😄.png -o out/smile.json -p out/smile.png

# 严格色量（可用颜色数）
uv run bfemblem approx examples/gold.png -o out/gold.json --num-colors 6 -d out/debug

# 评分（sim / line / simple / overall）
uv run bfemblem score examples/😄.png out/smile.png --layers 2

uv run bfemblem validate examples/sample_emblem.json
uv run bfemblem list-stamps
```

## 近似算法（摘要）

详见 [docs/visible-boundary-fitting.md](docs/visible-boundary-fitting.md) 与 [docs/shared-edge-and-planar-field.md](docs/shared-edge-and-planar-field.md)。

1. 严格 LAB k-means 色量（`num_colors`）+ 共享边平面图  
2. 图章边缘曲线库：并行 SVG 栅格 + 批量 GPU SDF + 外环/内孔多环；缓存缺则补算，全量重算用 `prefit-stamps`  
3. **大尺度 / 出画布** GPU 粒子匹配，层间重叠造型  
4. 评分：`overall ≈ sim × line × simple`，**线条硬门槛**  

## Python API

```python
from bf_emblem_creator import approximate_image, ApproxConfig

result = approximate_image("examples/😄.png", ApproxConfig(num_colors=6), n_particles=256)
print(result.score.summary())
result.document.save_json("out/smile.json")
```

## 交付前质量门禁

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
uv run pytest
```

## 文档

- [可见边界曲线拟合](docs/visible-boundary-fitting.md)
- [通用标签场 + 共享边缘拓扑](docs/shared-edge-and-planar-field.md)
- [技术方案](docs/technical-design.md)
- [图像概括](docs/image-abstraction.md)
- [编码规范](docs/coding-conventions.md)
