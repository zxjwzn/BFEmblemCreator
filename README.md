# BF Emblem Creator

战地1 图章徽章工具：**编辑器导出 JSON 的离线渲染器**，以及**学习式徽章生成模型**的预研
（见 `docs/research/`）。

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
- `fill` → `#RRGGBB`

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

## 文档

分类索引见 [docs/README.md](docs/README.md)。

- [徽章离线渲染器](docs/renderer/emblem-renderer.md) — 渲染器技术规格（JSON / 管线 / CLI）
- [通用图章生成模型调研](docs/research/generative-model-survey.md) — 概念综述
- [徽章生成模型设计（RL 闭环）](docs/research/emblem-rl-model.md) — **主攻方向**：自监督预训练 + 蒸馏 + GRPO
