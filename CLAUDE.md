# BFEmblemCreator — Claude 项目说明

## 项目

战地图章徽章工具：离线渲染编辑器导出 JSON，后续扩展自动摆放。

## 必须遵守的规范

### 注解与文档使用中文

**所有注释、docstring、Field/CLI 说明、用户可见文案、项目文档使用中文。**

详见 [docs/coding-conventions.md](docs/coding-conventions.md)。

- 标识符与编辑器 JSON 字段名保持英文（如 `flipX`、`asset`）
- 不要把已有中文注解改回英文

### 工程约定

- 包管理：**uv**（`uv sync` / `uv run`）
- 数据模型：**Pydantic v2**（`src/bf_emblem_creator/models.py`）
- 图章资源：`assets/stamps/{Asset}.svg`（256 个）
- 导出 JSON：图层数组，底层在前；`left`/`top` 为中心点；`angle` 顺时针（度）
- 画布默认 320×320
- **无版本分叉 / 无兼容旧路径**：见 [docs/coding-conventions.md](docs/coding-conventions.md)「实现与文档的时效性」
- **图章曲线缓存**：缺文件才补算；全量重算仅 `prefit-stamps`（或 `force_refit=True`）

### 交付前检查

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
uv run pytest
```

### 关键路径

| 路径 | 说明 |
|------|------|
| `docs/technical-design.md` | 技术方案 |
| `docs/architecture-refactor.md` | 近似管线架构重构（模式配方 / 处理器类 / pixel） |
| `docs/stamp-constructive-matching.md` | 图章构造式匹配（同色并集 / 异色遮挡 / 画布裁切） |
| `docs/coding-conventions.md` | 编码规范（含中文注解、无版本分叉） |
| `src/bf_emblem_creator/` | 主代码 |
| `examples/sample_emblem.json` | 示例导出 |
