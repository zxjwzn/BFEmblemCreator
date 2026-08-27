"""命令行：徽章 JSON 渲染、重写校验、图章列表。"""

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
    help="战地图章徽章：编辑器导出 JSON 的离线渲染",
    add_completion=False,
    no_args_is_help=True,
)

InputJsonArg = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, readable=True, show_default=False),
]
StampsDirOpt = Annotated[
    Path | None,
    typer.Option("--stamps-dir", "-s", help="图章 SVG 目录（默认: assets/stamps）"),
]


def _default_stamps_dir() -> Path:
    """优先使用当前工作目录下的 assets/stamps。"""
    cwd_candidate = Path.cwd() / "assets" / "stamps"
    if cwd_candidate.is_dir():
        return cwd_candidate
    repo_candidate = Path(__file__).resolve().parents[2] / "assets" / "stamps"
    if repo_candidate.is_dir():
        return repo_candidate
    return cwd_candidate


def _parse_background(value: str | None) -> HexColor | None:
    if value is None:
        return None
    return CanvasConfig(background=value).background


@app.command("render")
def render_cmd(
    input_json: InputJsonArg,
    output: Annotated[Path, typer.Option("--output", "-o", help="输出图片路径")],
    stamps_dir: StampsDirOpt = None,
    width: Annotated[int, typer.Option("--width", "-W", min=1)] = 320,
    height: Annotated[int, typer.Option("--height", "-H", min=1)] = 320,
    background: Annotated[str | None, typer.Option("--background", "-b")] = None,
    supersample: Annotated[float, typer.Option("--supersample", min=1.0, max=8.0)] = 4.0,
    stamp_raster_scale: Annotated[float, typer.Option("--stamp-raster-scale", min=1.0, max=4.0)] = 1.5,
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
    output: Annotated[Path, typer.Option("--output", "-o")],
    indent: Annotated[int, typer.Option("--indent", min=0, max=8)] = 2,
) -> None:
    """校验并重写徽章 JSON。"""
    doc = EmblemDocument.load_json(input_json)
    doc.save_json(output, indent=indent)
    typer.echo(f"已写入 {output}（{len(doc)} 层）")


@app.command("list-stamps")
def list_stamps_cmd(stamps_dir: StampsDirOpt = None) -> None:
    """列出图章 id。"""
    lib = StampLibrary(stamps_dir or _default_stamps_dir())
    ids = lib.list_ids()
    for stamp_id in ids:
        typer.echo(stamp_id)
    typer.echo(f"# 共 {len(ids)} 个图章", err=True)


@app.command("validate")
def validate_cmd(input_json: InputJsonArg, stamps_dir: StampsDirOpt = None) -> None:
    """校验 JSON 与 asset 存在性。"""
    doc = EmblemDocument.load_json(input_json)
    lib = StampLibrary(stamps_dir or _default_stamps_dir())
    missing: list[str] = []
    for layer in doc:
        try:
            lib.resolve(layer.asset)
        except FileNotFoundError:
            missing.append(layer.asset)
    if missing:
        typer.secho("缺失图章: " + ", ".join(sorted(set(missing))), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"通过: {len(doc)} 层")


def main() -> None:
    """CLI 入口。"""
    app()


if __name__ == "__main__":
    main()
