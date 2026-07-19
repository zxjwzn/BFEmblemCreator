# BF Emblem Creator

战地图章徽章工具：将编辑器导出的 JSON 离线渲染为图片，并为后续自动摆放算法打基础。

## 规范

- **注释 / 文档字符串 / CLI 说明 / 用户文案：一律中文**（见 [docs/coding-conventions.md](docs/coding-conventions.md)）
- 数据结构使用 Pydantic；依赖使用 uv 管理

## 环境

```bash
uv sync --all-groups
```

## 徽章 JSON 格式

与编辑器导出一致（图层列表，**底层 → 顶层**）：

```json
[
  {
    "asset": "Square",
    "opacity": 1,
    "angle": 0,
    "flipX": false,
    "flipY": false,
    "top": 160,
    "left": 160,
    "height": 100,
    "width": 100,
    "fill": "#3A9FFA"
  }
]
```

- `asset` → `assets/stamps/{asset}.svg`
- `left` / `top` → 图章中心点（默认画布 320×320）
- `angle` → 角度（度，**顺时针**）
- `fill` → 染色颜色（`#RRGGBB`）

## 命令行

```bash
# JSON → PNG
uv run bfemblem render examples/sample_emblem.json -o out/sample.png

# 不透明背景
uv run bfemblem render examples/sample_emblem.json -o out/sample.png -b "#000000"

# 校验 + 规范化 JSON
uv run bfemblem validate examples/sample_emblem.json
uv run bfemblem export-json examples/sample_emblem.json -o out/normalized.json

# 列出图章 id
uv run bfemblem list-stamps
```

## Python API

```python
from bf_emblem_creator import EmblemDocument, EmblemRenderer, RenderConfig

doc = EmblemDocument.load_json("examples/sample_emblem.json")
renderer = EmblemRenderer(RenderConfig(stamps_dir="assets/stamps"))
renderer.render_to_path(doc, "out/sample.png")
doc.save_json("out/copy.json")
```

## 交付前质量门禁

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
uv run pytest
```

或先格式化再检查：

```bash
uv run ruff format src tests
uv run ruff check src tests
uv run pyright
uv run pytest
```

## 文档

- [技术方案](docs/technical-design.md)
- [编码规范](docs/coding-conventions.md)
