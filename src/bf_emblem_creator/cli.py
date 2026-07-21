"""命令行：徽章 JSON 渲染、校验、图像近似、图章曲线预拟合。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from bf_emblem_creator.approx.blocks import abstract_to_blocks
from bf_emblem_creator.approx.metrics import score_prediction
from bf_emblem_creator.approx.models import AbstractionMode
from bf_emblem_creator.approx.pipeline import approximate_image
from bf_emblem_creator.approx.recipe import default_recipe_for_mode
from bf_emblem_creator.approx.stamp_curves import StampCurveLibrary
from bf_emblem_creator.models import CanvasConfig, EmblemDocument, HexColor, RenderConfig
from bf_emblem_creator.render import EmblemRenderer
from bf_emblem_creator.stamps import StampLibrary

app = typer.Typer(
    name="bfemblem",
    help="战地图章徽章：离线渲染与自动近似（ModeRecipe + 五大处理器）",
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


@app.command("prefit-stamps")
def prefit_stamps_cmd(
    stamps_dir: StampsDirOpt = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", "-c", help="曲线缓存目录（默认 assets/.cache/stamp_curves）"),
    ] = None,
    tex_size: Annotated[
        int,
        typer.Option("--tex-size", min=64, max=512, help="预拟合栅格边长（默认 256，内部闭合边需高分辨率）"),
    ] = 256,
) -> None:
    """
    强制重新拟合目录下全部图章 SVG 的边缘曲线并持久化。

    Moore 跟踪外轮廓+内部孔；高分辨率栅格；写入 assets/.cache/stamp_curves。
    """
    sdir = stamps_dir or _default_stamps_dir()
    cdir = cache_dir or (Path(sdir).parent / ".cache" / "stamp_curves")
    typer.echo(f"预拟合图章: {sdir} → {cdir}（tex_size={tex_size}，强制重算，多环高精度）")
    lib = StampCurveLibrary.prefit_directory(sdir, cache_dir=cdir, tex_size=tex_size)
    tag_counts: dict[str, int] = {}
    hole_stamps = 0
    for e in lib.entries:
        for t in e.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        if e.n_holes > 0:
            hole_stamps += 1
    typer.echo(f"已写入 {len(lib.entries)} 个图章曲线缓存（含孔洞图章 {hole_stamps} 个）")
    if tag_counts:
        typer.echo("效果标签: " + ", ".join(f"{k}={v}" for k, v in sorted(tag_counts.items())))


@app.command("approx")
def approx_cmd(
    input_image: InputImageArg,
    output: Annotated[Path, typer.Option("--output", "-o", help="输出徽章 JSON")],
    preview: Annotated[Path | None, typer.Option("--preview", "-p", help="预览 PNG")] = None,
    stamps_dir: StampsDirOpt = None,
    max_layers: Annotated[
        int,
        typer.Option("--max-layers", min=1, max=40, help="图层上限（默认 40）"),
    ] = 40,
    num_colors: Annotated[
        int,
        typer.Option("--num-colors", "-k", min=2, max=64, help="严格 LAB 色量 K"),
    ] = 6,
    pass_score: Annotated[float, typer.Option("--pass-score", min=0.0, max=1.0)] = 0.48,
    particles: Annotated[int, typer.Option("--particles", min=32, max=4096, help="每区域粒子数")] = 384,
    mode: Annotated[
        AbstractionMode,
        typer.Option(
            "--mode",
            help="概括模式：logo/illustration/photo_portrait/photo_general/silhouette/pixel（默认 illustration）",
        ),
    ] = AbstractionMode.illustration,
    cpu: Annotated[bool, typer.Option("--cpu", help="强制 CPU")] = False,
    no_fx: Annotated[bool, typer.Option("--no-fx", help="禁用特殊图章渐变通道")] = False,
    debug_dir: Annotated[
        Path | None,
        typer.Option(
            "--debug-dir",
            "-d",
            help="逐步调试图目录；不写粒子搜索过程",
        ),
    ] = None,
) -> None:
    """ModeRecipe 近似：平面化 + 共享边贝塞尔 + 图章匹配（输出 JSON 与预览）。"""
    from PIL import Image

    recipe = default_recipe_for_mode(mode).override(
        stamps_dir=stamps_dir or _default_stamps_dir(),
        max_layers=max_layers,
        num_colors=num_colors,
        pass_score=pass_score,
        n_particles=particles,
        use_cuda=not cpu,
        enable_special_fx=not no_fx,
        debug_dir=debug_dir,
    )
    result = approximate_image(input_image, recipe, n_particles=particles)
    result.document.save_json(output)
    if preview is not None and result.preview_rgb is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.preview_rgb, mode="RGBA").save(preview)
    typer.echo(
        f"已写入 {output}（{len(result.document)}/{max_layers} 层，"
        f"区域={result.blocks_found}，num_colors={result.k_used}，mode={result.mode}，"
        f"device={result.device}，{result.elapsed_sec:.2f}s）"
    )
    typer.echo(f"停止原因: {result.stop_reason}")
    typer.echo(f"边界一致性: {result.boundary_score:.3f}")
    if result.special_fx_assets:
        typer.echo("特效章: " + ", ".join(result.special_fx_assets))
    for line in result.log_lines:
        typer.echo(f"  {line}")
    if result.debug_images:
        typer.echo(f"调试图: {len(result.debug_images)} 张 → {debug_dir}")
    typer.echo(result.score.summary())
    if not result.score.passed:
        typer.secho(
            "评分未达标（JSON/预览仍已写出；可用 --pass-score 放宽，或检查 line/sim/curve 分项）。",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2)


@app.command("score")
def score_cmd(
    input_image: InputImageArg,
    pred_image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    pass_score: Annotated[float, typer.Option("--pass-score")] = 0.48,
    n_layers: Annotated[int, typer.Option("--layers", min=0, max=40)] = 1,
) -> None:
    """对预测图打分（曲线边界 + 色彩 + line / simple / overall）。"""
    from PIL import Image

    target = abstract_to_blocks(input_image)
    pred = Image.open(pred_image).convert("RGBA")
    report = score_prediction(pred, target, n_layers=n_layers, pass_overall=pass_score)
    typer.echo(report.summary())
    typer.echo(
        f"  detail: iou={report.sim.alpha_iou:.3f} color={report.sim.color_score:.3f} "
        f"curve={report.sim.edge_score:.3f} chamfer={report.curve_chamfer:.2f} "
        f"jagged={report.line.jaggedness:.3f} corners={report.line.corner_density:.3f} "
        f"frag={report.line.fragment_ratio:.3f}"
    )
    if not report.passed:
        raise typer.Exit(code=2)


def main() -> None:
    """CLI 入口。"""
    app()


if __name__ == "__main__":
    main()
