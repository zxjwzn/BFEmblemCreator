# BF Emblem Creator

战地1 图章徽章工具：**编辑器导出 JSON 的离线渲染器**

## 规范

- **注释 / 文档字符串 / CLI 说明 / 用户文案：一律中文**
- 标识符与编辑器 JSON 字段名保持英文（如 `flipX`、`asset`）
- 数据结构使用 **Pydantic v2**；依赖使用 **uv** 管理

## 环境

```bash
uv sync --all-groups
```

## 徽章 JSON 格式

与编辑器导出一致（图层列表，**底层 → 顶层**）：

- `asset` → `assets/stamps/{asset}.svg`
- `left` / `top` → 图章**中心点**（可在画布外）
- `width` / `height` → 可**远大于**画布（大章只露局部）
- `angle` → 顺时针（度）
- `flipX` / `flipY` → 水平 / 垂直镜像
- `opacity` → 不透明度，0–1
- `fill` → `#RRGGBB`
- `selectable` → 编辑器 UI 字段；渲染忽略，导出时保留

## 命令行

```bash
# JSON → PNG
uv run bfemblem render examples/sample_emblem.json -o out/sample.png

# 校验 JSON 与图章存在性 / 列出图章 / 重写 JSON
uv run bfemblem validate examples/sample_emblem.json
uv run bfemblem list-stamps
uv run bfemblem export-json examples/sample_emblem.json -o out/copy.json
```

## Python API

```python
from bf_emblem_creator import EmblemDocument, EmblemRenderer, RenderConfig

doc = EmblemDocument.load_json("examples/sample_emblem.json")
EmblemRenderer(RenderConfig()).render_to_path(doc, "out/sample.png")
```

## 交付前质量门禁

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
uv run pytest
```
