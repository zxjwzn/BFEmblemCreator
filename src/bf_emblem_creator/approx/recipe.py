"""模式配方与各处理器子配置（Pydantic；唯一配置体）。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bf_emblem_creator.approx.models import AbstractionMode, ResampleMode


class BoundaryPolicy(str, Enum):
    """共享边几何策略。"""

    dense = "dense"
    curve_fit = "curve_fit"


class AngleMode(str, Enum):
    """图章角度搜索策略。"""

    free = "free"
    discrete = "discrete"


class FitPolicy(str, Enum):
    """画布 fit 策略。"""

    contain = "contain"
    cover = "cover"


class BilateralStrength(str, Enum):
    """保边平滑强度。"""

    off = "off"
    weak = "weak"
    medium = "medium"
    strong = "strong"


class AngleConstraint(BaseModel):
    """角度约束。"""

    model_config = ConfigDict(extra="forbid")

    mode: AngleMode = Field(default=AngleMode.free, description="free 连续搜索；discrete 仅允许 angles_deg")
    angles_deg: list[float] = Field(
        default_factory=lambda: [0.0, 90.0],
        description="discrete 时允许的角度（度）",
    )
    angle_step_deg: float = Field(default=30.0, gt=0.0, description="free 模式粗搜步长（度）")


class CurveFitConfig(BaseModel):
    """共享边宏观贝塞尔拟合参数。"""

    model_config = ConfigDict(extra="forbid")

    max_vertices: int = Field(default=8, ge=3, le=512, description="链/边顶点数超过则拟合")
    min_arc_length_px: float = Field(default=6.0, ge=0.0, description="链/边弧长超过则拟合")
    line_flat_eps_px: float = Field(default=2.0, ge=0.1, le=16.0, description="近直判定像素阈")
    corner_deg: float = Field(default=50.0, ge=10.0, le=120.0, description="宏观尖角钉扎角度")
    smooth_radius_px: float = Field(default=8.0, ge=1.5, le=40.0, description="宏观切向半窗")
    min_anchor_spacing_px: float = Field(default=10.0, ge=1.0, le=80.0, description="锚点最小弧长间距")
    samples_per_seg: int = Field(default=6, ge=3, le=32, description="每段贝塞尔采样点数")


class StampLoaderConfig(BaseModel):
    """图章加载器配置。"""

    model_config = ConfigDict(extra="forbid")

    stamps_dir: Path = Field(default=Path("assets/stamps"), description="图章 SVG 目录")
    cache_dir: Path | None = Field(default=None, description="曲线缓存目录；None 用 stamps 旁 .cache")
    tex_size: int = Field(default=256, ge=32, le=512, description="图章栅格边长")
    resample_n: int = Field(default=256, ge=32, le=512, description="轮廓重采样点数")
    asset_allowlist: list[str] | None = Field(default=None, description="允许的 asset；None 为全库")
    asset_blocklist: list[str] = Field(default_factory=list, description="排除的 asset")
    force_rebuild: bool = Field(default=False, description="忽略缓存强制重建")
    max_workers: int = Field(default=8, ge=1, le=32, description="并行栅格线程数")

    @field_validator("stamps_dir", "cache_dir", mode="before")
    @classmethod
    def _path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value)


class ImageProcessorConfig(BaseModel):
    """图像处理器配置。"""

    model_config = ConfigDict(extra="forbid")

    canvas_size: int = Field(default=320, gt=0, description="工作画布边长")
    resample_mode: ResampleMode = Field(default=ResampleMode.auto, description="重采样模式")
    fit_policy: FitPolicy = Field(default=FitPolicy.contain, description="画布 fit：contain/cover")
    num_colors: int = Field(default=6, ge=2, le=64, description="严格 LAB 色量 K")
    bilateral: bool = Field(default=True, description="是否保边平滑")
    bilateral_strength: BilateralStrength = Field(
        default=BilateralStrength.medium,
        description="平滑强度；off 等价关闭",
    )
    mrf_lambda: float = Field(default=2.5, ge=0.0, le=20.0, description="MRF 强度")
    mrf_iters: int = Field(default=5, ge=0, le=30, description="ICM 轮数")
    min_region_area_frac: float = Field(default=0.004, ge=0.0, le=0.1, description="小连通域面积比")
    enforce_no_gap: bool = Field(default=True, description="主体无标签洞")
    seed: int = Field(default=0, description="k-means 种子")
    use_cuda: bool = Field(default=True, description="是否优先 CUDA（平面化侧）")


class RegionPartitionerConfig(BaseModel):
    """区域划分器配置。"""

    model_config = ConfigDict(extra="forbid")

    max_faces: int = Field(default=24, ge=2, le=64, description="平面图最多 Face 数")
    min_area_frac: float = Field(default=0.004, ge=0.0, le=0.1, description="Face 最小面积比")
    edge_subpixel: bool = Field(default=False, description="共享边亚像素")
    boundary_policy: BoundaryPolicy = Field(
        default=BoundaryPolicy.curve_fit,
        description="dense=保持 dual；curve_fit=宏观贝塞尔",
    )
    curve_fit: CurveFitConfig = Field(default_factory=CurveFitConfig, description="贝塞尔拟合参数")
    eps_arc: float = Field(default=0.05, ge=0.01, le=0.2, description="弧基元相关阈值")


class UnionCoverConfig(BaseModel):
    """同色多图章并集覆盖。"""

    model_config = ConfigDict(extra="forbid")

    max_stamps_per_region: int = Field(default=4, ge=1, le=12, description="每区最多同色章")
    min_cover: float = Field(default=0.82, ge=0.3, le=1.0, description="覆盖目标")
    min_cover_gain: float = Field(default=0.045, ge=0.0, le=0.5, description="追加最小增益")
    max_leak: float = Field(default=0.42, ge=0.0, le=1.0, description="最大泄漏比")


class StampMatchAssemblerConfig(BaseModel):
    """图章匹配装配器配置。"""

    model_config = ConfigDict(extra="forbid")

    max_layers: int = Field(default=40, ge=1, le=40, description="层硬上限")
    n_particles: int = Field(default=384, ge=32, le=4096, description="每区域粒子数")
    recall_k: int = Field(default=48, ge=1, le=128, description="描述子召回数")
    refine: bool = Field(default=True, description="是否局部精修")
    refine_iters: int = Field(default=3, ge=0, description="精修迭代")
    angle: AngleConstraint = Field(default_factory=AngleConstraint, description="角度约束")
    force_uniform_scale: bool = Field(default=False, description="是否强制 width==height")
    allowed_assets: list[str] | None = Field(default=None, description="装配侧过滤；None 用 catalog")
    union_cover: UnionCoverConfig = Field(default_factory=UnionCoverConfig, description="并集覆盖")
    enable_special_fx: bool = Field(default=True, description="特效章通道")
    prefer_primitive_seed: bool = Field(default=False, description="圆/椭圆种子")
    stall_patience: int = Field(default=2, ge=1, description="残差无增益耐心")
    seed: int = Field(default=0, description="匹配随机种子")
    pass_score: float = Field(default=0.48, ge=0.0, le=1.0, description="综合分门槛")
    pass_sim: float = Field(default=0.50, ge=0.0, le=1.0, description="相似度门槛")
    pass_line: float = Field(default=0.55, ge=0.0, le=1.0, description="线条门槛")


class StampRendererConfig(BaseModel):
    """图章渲染器配置。"""

    model_config = ConfigDict(extra="forbid")

    canvas_size: int = Field(default=320, gt=0, description="画布边长")
    stamp_tex_size: int = Field(default=256, ge=32, le=512, description="图章纹理边长")
    use_cuda: bool = Field(default=True, description="优先 CUDA")
    supersample: float = Field(default=1.0, ge=1.0, le=4.0, description="超采样")


class ModeRecipe(BaseModel):
    """某一 AbstractionMode 的完整处理配方（唯一配置入口）。"""

    model_config = ConfigDict(extra="forbid")

    mode: AbstractionMode = Field(..., description="绑定的概括模式")
    description: str = Field(default="", description="配方说明")
    stamp_loader: StampLoaderConfig = Field(default_factory=StampLoaderConfig)
    image: ImageProcessorConfig = Field(default_factory=ImageProcessorConfig)
    region: RegionPartitionerConfig = Field(default_factory=RegionPartitionerConfig)
    match: StampMatchAssemblerConfig = Field(default_factory=StampMatchAssemblerConfig)
    renderer: StampRendererConfig = Field(default_factory=StampRendererConfig)
    debug_dir: Path | None = Field(default=None, description="调试图目录")

    def override(
        self,
        *,
        stamps_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
        asset_allowlist: list[str] | None = None,
        num_colors: int | None = None,
        canvas_size: int | None = None,
        max_layers: int | None = None,
        max_faces: int | None = None,
        n_particles: int | None = None,
        use_cuda: bool | None = None,
        enable_special_fx: bool | None = None,
        debug_dir: Path | str | None = None,
        seed: int | None = None,
        pass_score: float | None = None,
        refine: bool | None = None,
        **nested: Any,
    ) -> Self:
        """返回覆盖常用字段后的新配方（不修改 self）。"""
        sl = self.stamp_loader
        im = self.image
        rg = self.region
        mt = self.match
        rd = self.renderer
        if stamps_dir is not None:
            sl = sl.model_copy(update={"stamps_dir": Path(stamps_dir)})
        if cache_dir is not None:
            sl = sl.model_copy(update={"cache_dir": Path(cache_dir)})
        if asset_allowlist is not None:
            sl = sl.model_copy(update={"asset_allowlist": list(asset_allowlist)})
            mt = mt.model_copy(update={"allowed_assets": list(asset_allowlist)})
        if num_colors is not None:
            im = im.model_copy(update={"num_colors": int(num_colors)})
        if canvas_size is not None:
            im = im.model_copy(update={"canvas_size": int(canvas_size)})
            rd = rd.model_copy(update={"canvas_size": int(canvas_size)})
        if max_layers is not None:
            mt = mt.model_copy(update={"max_layers": int(max_layers)})
        if max_faces is not None:
            rg = rg.model_copy(update={"max_faces": int(max_faces)})
        if n_particles is not None:
            mt = mt.model_copy(update={"n_particles": int(n_particles)})
        if use_cuda is not None:
            im = im.model_copy(update={"use_cuda": bool(use_cuda)})
            rd = rd.model_copy(update={"use_cuda": bool(use_cuda)})
        if enable_special_fx is not None:
            mt = mt.model_copy(update={"enable_special_fx": bool(enable_special_fx)})
        if seed is not None:
            im = im.model_copy(update={"seed": int(seed)})
            mt = mt.model_copy(update={"seed": int(seed)})
        if pass_score is not None:
            mt = mt.model_copy(update={"pass_score": float(pass_score)})
        if refine is not None:
            mt = mt.model_copy(update={"refine": bool(refine)})
        dbg = Path(debug_dir) if debug_dir is not None else self.debug_dir
        _ = nested
        return self.model_copy(
            update={
                "stamp_loader": sl,
                "image": im,
                "region": rg,
                "match": mt,
                "renderer": rd,
                "debug_dir": dbg,
            }
        )


def _base_curve_recipe(mode: AbstractionMode, *, fit: FitPolicy, bilateral: BilateralStrength) -> ModeRecipe:
    bil_on = bilateral != BilateralStrength.off
    return ModeRecipe(
        mode=mode,
        description=f"默认配方：{mode.value}",
        image=ImageProcessorConfig(
            fit_policy=fit,
            bilateral=bil_on,
            bilateral_strength=bilateral,
        ),
        region=RegionPartitionerConfig(boundary_policy=BoundaryPolicy.curve_fit),
        match=StampMatchAssemblerConfig(
            angle=AngleConstraint(mode=AngleMode.free, angle_step_deg=30.0),
        ),
        renderer=StampRendererConfig(),
    )


def default_recipe_for_mode(mode: AbstractionMode) -> ModeRecipe:
    """模式 → 默认 ModeRecipe。"""
    if mode == AbstractionMode.logo:
        return _base_curve_recipe(mode, fit=FitPolicy.contain, bilateral=BilateralStrength.weak)
    if mode == AbstractionMode.illustration:
        return _base_curve_recipe(mode, fit=FitPolicy.contain, bilateral=BilateralStrength.medium)
    if mode in {AbstractionMode.photo_portrait, AbstractionMode.photo_general}:
        return _base_curve_recipe(mode, fit=FitPolicy.cover, bilateral=BilateralStrength.strong)
    if mode == AbstractionMode.silhouette:
        return _base_curve_recipe(mode, fit=FitPolicy.contain, bilateral=BilateralStrength.medium)
    if mode == AbstractionMode.pixel:
        return ModeRecipe(
            mode=mode,
            description="像素风：仅 Square，角度 0/90，dense 边界",
            stamp_loader=StampLoaderConfig(asset_allowlist=["Square"]),
            image=ImageProcessorConfig(
                fit_policy=FitPolicy.contain,
                bilateral=False,
                bilateral_strength=BilateralStrength.off,
                resample_mode=ResampleMode.nearest,
            ),
            region=RegionPartitionerConfig(boundary_policy=BoundaryPolicy.dense),
            match=StampMatchAssemblerConfig(
                allowed_assets=["Square"],
                angle=AngleConstraint(mode=AngleMode.discrete, angles_deg=[0.0, 90.0]),
                force_uniform_scale=False,
                enable_special_fx=False,
            ),
            renderer=StampRendererConfig(),
        )
    return _base_curve_recipe(AbstractionMode.illustration, fit=FitPolicy.contain, bilateral=BilateralStrength.medium)
