"""命令行：徽章 JSON 渲染、校验，以及图像近似拟合。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from bf_emblem_creator.approx.metrics import SimilarityReport, score_fit
from bf_emblem_creator.approx.models import AbstractionMode, ApproxConfig
from bf_emblem_creator.approx.pipeline import approximate_image
from bf_emblem_creator.approx.preprocess import abstract_image, save_debug_montage
from bf_emblem_creator.models import CanvasConfig, EmblemDocument, HexColor, RenderConfig
from bf_emblem_creator.render import EmblemRenderer
from bf_emblem_creator.stamps import StampLibrary

app = typer.Typer(
    name="bfemblem",
    help="战地图章徽章：离线渲染与自动近似",
    add_completion=False,
    no_args_is_help=True,
)

InputJsonArg = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, readable=True, show_default=False),
]
InputImageArg = Annotated[
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


@app.command("approx")
def approx_cmd(
    input_image: InputImageArg,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="输出徽章 JSON 路径"),
    ],
    preview: Annotated[
        Path | None,
        typer.Option("--preview", "-p", help="可选：渲染预览 PNG"),
    ] = None,
    debug_montage: Annotated[
        Path | None,
        typer.Option("--debug-montage", help="可选：概括调试拼图 PNG"),
    ] = None,
    stamps_dir: StampsDirOpt = None,
    max_layers: Annotated[int, typer.Option("--max-layers", min=1, max=40, help="最大层数")] = 24,
    palette_k: Annotated[int, typer.Option("--palette-k", min=2, max=16, help="色量 K")] = 6,
    pass_score: Annotated[
        float,
        typer.Option("--pass-score", min=0.0, max=1.0, help="综合相似度达标线"),
    ] = 0.48,
    mode: Annotated[
        AbstractionMode,
        typer.Option("--mode", help="概括模式"),
    ] = AbstractionMode.auto,
    refine: Annotated[bool, typer.Option("--refine/--no-refine", help="是否局部精修")] = True,
) -> None:
    """将输入图像概括并拟合为 ≤40 层图章 JSON。"""
    from PIL import Image

    cfg = ApproxConfig(
        stamps_dir=stamps_dir or _default_stamps_dir(),
        max_layers=max_layers,
        palette_k=palette_k,
        pass_score=pass_score,
        mode=mode,
        refine=refine,
    )
    result = approximate_image(input_image, cfg)
    result.document.save_json(output)
    if preview is not None and result.preview_rgb is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.preview_rgb, mode="RGBA").save(preview)
    if debug_montage is not None:
        save_debug_montage(result.target, debug_montage)

    sim = result.similarity
    assert isinstance(sim, SimilarityReport)
    typer.echo(f"已写入 {output}（{len(result.document)} 层，loss={result.final_loss:.4f}）")
    typer.echo(sim.summary())
    if not sim.passed:
        typer.secho(
            f"相似度未达阈值 {pass_score:.2f}，请检查预览或调参。",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2)


@app.command("score")
def score_cmd(
    input_image: InputImageArg,
    pred_image: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, show_default=False),
    ],
    pass_score: Annotated[float, typer.Option("--pass-score", min=0.0, max=1.0)] = 0.52,
    palette_k: Annotated[int, typer.Option("--palette-k", min=2, max=16)] = 6,
) -> None:
    """将预测图与输入图的概括目标比较，输出相似度。"""
    from PIL import Image

    cfg = ApproxConfig(palette_k=palette_k, pass_score=pass_score)
    target = abstract_image(input_image, cfg)
    pred = Image.open(pred_image).convert("RGBA")
    size = target.meta.canvas_size
    if pred.size != (size, size):
        pred = pred.resize((size, size), Image.Resampling.LANCZOS)
    report = score_fit(pred, target, pass_score=pass_score)
    typer.echo(report.summary())
    if not report.passed:
        raise typer.Exit(code=2)


def main() -> None:
    """CLI 入口。"""
    app()


if __name__ == "__main__":
    main()
