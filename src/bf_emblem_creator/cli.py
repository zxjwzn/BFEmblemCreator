"""命令行：徽章 JSON 渲染为图片，以及校验/重写 JSON。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from bf_emblem_creator.models import CanvasConfig, EmblemDocument, HexColor, RenderConfig
from bf_emblem_creator.render import EmblemRenderer
from bf_emblem_creator.stamps import StampLibrary

app = typer.Typer(
    name="bfemblem",
    help="战地图章徽章离线渲染器",
    add_completion=False,
    no_args_is_help=True,
)

InputJsonArg = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, readable=True, show_default=False),
]
StampsDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--stamps-dir",
        "-s",
        help="图章 SVG 目录（默认: assets/stamps）",
    ),
]


def _default_stamps_dir() -> Path:
    """优先使用当前工作目录下的 assets/stamps，否则回退到仓库相对路径。"""
    cwd_candidate = Path.cwd() / "assets" / "stamps"
    if cwd_candidate.is_dir():
        return cwd_candidate
    repo_candidate = Path(__file__).resolve().parents[2] / "assets" / "stamps"
    if repo_candidate.is_dir():
        return repo_candidate
    return cwd_candidate


def _parse_background(value: str | None) -> HexColor | None:
    """解析并校验背景色参数。"""
    if value is None:
        return None
    # 复用 CanvasConfig 的颜色校验逻辑。
    return CanvasConfig(background=value).background


@app.command("render")
def render_cmd(
    input_json: InputJsonArg,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="输出图片路径（.png / .webp / .jpg）",
        ),
    ],
    stamps_dir: StampsDirOpt = None,
    width: Annotated[int, typer.Option("--width", "-W", min=1, help="画布宽度")] = 320,
    height: Annotated[int, typer.Option("--height", "-H", min=1, help="画布高度")] = 320,
    background: Annotated[
        str | None,
        typer.Option(
            "--background",
            "-b",
            help="背景十六进制色；省略则为透明",
        ),
    ] = None,
    supersample: Annotated[
        float,
        typer.Option(
            "--supersample",
            min=1.0,
            max=8.0,
            help="全画布超采样倍率（默认 4；越大越抗锯齿，越慢）",
        ),
    ] = 4.0,
    stamp_raster_scale: Annotated[
        float,
        typer.Option(
            "--stamp-raster-scale",
            min=1.0,
            max=4.0,
            help="图章 SVG 额外光栅倍率（默认 1.5）",
        ),
    ] = 1.5,
) -> None:
    """将编辑器导出的徽章 JSON 渲染为图片。"""
    doc = EmblemDocument.load_json(input_json)
    try:
        bg = _parse_background(background)
    except ValidationError as exc:
        typer.secho(f"无效的 --background: {background!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    cfg = RenderConfig(
        canvas=CanvasConfig(width=width, height=height, background=bg),
        stamps_dir=stamps_dir or _default_stamps_dir(),
        supersample=supersample,
        stamp_raster_scale=stamp_raster_scale,
    )
    path = EmblemRenderer(cfg).render_to_path(doc, output)
    typer.echo(f"已写入 {path}（{len(doc)} 层）")


@app.command("export-json")
def export_json_cmd(
    input_json: InputJsonArg,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="规范化后的徽章 JSON 路径",
        ),
    ],
    indent: Annotated[int, typer.Option("--indent", min=0, max=8, help="JSON 缩进")] = 2,
) -> None:
    """加载、经 Pydantic 校验后重写徽章 JSON。"""
    doc = EmblemDocument.load_json(input_json)
    doc.save_json(output, indent=indent)
    typer.echo(f"已写入 {output}（{len(doc)} 层）")


@app.command("list-stamps")
def list_stamps_cmd(stamps_dir: StampsDirOpt = None) -> None:
    """列出可用图章 asset id。"""
    lib = StampLibrary(stamps_dir or _default_stamps_dir())
    ids = lib.list_ids()
    for stamp_id in ids:
        typer.echo(stamp_id)
    typer.echo(f"# 共 {len(ids)} 个图章", err=True)


@app.command("validate")
def validate_cmd(
    input_json: InputJsonArg,
    stamps_dir: StampsDirOpt = None,
) -> None:
    """校验 JSON 结构，并确认每个 asset 在磁盘上存在。"""
    doc = EmblemDocument.load_json(input_json)
    lib = StampLibrary(stamps_dir or _default_stamps_dir())
    missing: list[str] = []
    for layer in doc:
        try:
            lib.resolve(layer.asset)
        except FileNotFoundError:
            missing.append(layer.asset)
    if missing:
        typer.secho(
            "缺失图章: " + ", ".join(sorted(set(missing))),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"通过: {len(doc)} 层，全部 asset 已解析")


def main() -> None:
    """CLI 入口。"""
    app()


if __name__ == "__main__":
    main()
