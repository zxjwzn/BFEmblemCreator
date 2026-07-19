"""近似管线的 Pydantic 配置与结果模型。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bf_emblem_creator.models import EmblemDocument, HexColor

U8Arr = NDArray[np.uint8]
FloatArr = NDArray[np.floating]


class AbstractionMode(str, Enum):
    """图像概括模式。"""

    auto = "auto"
    logo = "logo"
    illustration = "illustration"
    photo_portrait = "photo_portrait"
    photo_general = "photo_general"
    silhouette = "silhouette"


class ResampleMode(str, Enum):
    """画布对齐重采样模式（Batch A）。"""

    auto = "auto"
    nearest = "nearest"
    lanczos = "lanczos"
    bilinear = "bilinear"


class PaletteColor(BaseModel):
    """调色板中的一种代表色。"""

    model_config = ConfigDict(extra="forbid")

    hex: HexColor = Field(..., description="sRGB 十六进制色")
    fraction: float = Field(..., ge=0.0, le=1.0, description="在主体内的面积占比")
    rgb: tuple[int, int, int] = Field(..., description="RGB 元组")


class ApproxMeta(BaseModel):
    """概括阶段元数据。"""

    model_config = ConfigDict(extra="forbid")

    source_width: int = Field(..., description="原图宽")
    source_height: int = Field(..., description="原图高")
    canvas_size: int = Field(default=320, description="工作画布边长")
    fit: str = Field(..., description="contain 或 cover")
    mode: AbstractionMode = Field(..., description="实际使用的概括模式")
    scale: float = Field(default=1.0, description="缩放比例")
    offset_x: float = Field(default=0.0, description="水平偏移（画布坐标）")
    offset_y: float = Field(default=0.0, description="垂直偏移（画布坐标）")
    resample: str = Field(default="lanczos", description="实际重采样：nearest/lanczos/bilinear")
    approx_color_count: int = Field(default=0, ge=0, description="主体近似独特色数（检测用）")
    gap_frac: float = Field(default=0.0, ge=0.0, description="主体内标签空洞占比（目标 0）")
    noise_frac: float = Field(default=0.0, ge=0.0, description="过小 Face 像素占比")
    seam_p95: float = Field(default=0.0, ge=0.0, description="共享可见边缝宽 95 分位（像素）")


class LayerHint(BaseModel):
    """给匹配追踪的施工顺序提示。"""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., description="bottom_color / silhouette / region / detail")
    color: HexColor | None = Field(default=None, description="相关颜色")
    area_fraction: float = Field(default=0.0, ge=0.0, description="面积占比")
    bbox: tuple[int, int, int, int] | None = Field(
        default=None,
        description="(x0,y0,x1,y1) 像素包围盒",
    )


class ApproxTarget(BaseModel):
    """
    概括后的拟合目标。

    数组字段允许任意类型以便承载 numpy；序列化时一般不直接 dump 大数组。
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    image_rgb: Any = Field(..., description="概括后 RGB uint8 (H,W,3)")
    alpha: Any = Field(..., description="主体蒙版 float (H,W) 0~1")
    weight: Any = Field(..., description="损失权重 float (H,W)")
    labels: Any = Field(..., description="色区标签 int (H,W)，-1 为背景")
    palette: list[PaletteColor] = Field(default_factory=list, description="调色板")
    layers_hint: list[LayerHint] = Field(default_factory=list, description="分层提示")
    meta: ApproxMeta = Field(..., description="元数据")

    def numpy_rgb(self) -> U8Arr:
        """返回 RGB 数组。"""
        return np.asarray(self.image_rgb, dtype=np.uint8)

    def numpy_alpha(self) -> FloatArr:
        """返回 alpha 数组。"""
        return np.asarray(self.alpha, dtype=np.float64)

    def numpy_weight(self) -> FloatArr:
        """返回权重数组。"""
        return np.asarray(self.weight, dtype=np.float64)

    def numpy_labels(self) -> NDArray[np.int32]:
        """返回标签数组。"""
        return np.asarray(self.labels, dtype=np.int32)


class ApproxConfig(BaseModel):
    """近似算法总配置（v3：可见边界曲线拟合）。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # 画布 / 概括
    canvas_size: int = Field(default=320, gt=0, description="工作画布边长")
    mode: AbstractionMode = Field(default=AbstractionMode.auto, description="概括模式")
    resample_mode: ResampleMode = Field(
        default=ResampleMode.auto,
        description="画布重采样：auto 按色数/尖峰启发式选择 nearest 或 lanczos",
    )
    palette_k: int = Field(default=4, ge=2, le=16, description="默认色量 K（由简入繁：先少色）")
    min_region_area_frac: float = Field(
        default=0.004,
        ge=0.0,
        le=0.1,
        description="最小保留色区面积占比",
    )
    bilateral: bool = Field(default=True, description="是否做保边平滑")
    # Batch B：标签场
    flat_grad_q: float = Field(
        default=0.45,
        ge=0.05,
        le=0.95,
        description="平坦区梯度分位阈值（仅平坦像素参与调色板）",
    )
    mrf_lambda: float = Field(default=2.0, ge=0.0, le=20.0, description="空间正则 Potts 强度")
    mrf_iters: int = Field(default=5, ge=0, le=30, description="ICM 轮数；0 关闭")
    lab_merge: float = Field(default=10.0, ge=0.0, le=40.0, description="近色合并 LAB ΔE 阈值")
    enforce_no_gap: bool = Field(default=True, description="主体内强制无标签空洞")
    # Batch C/D/E
    use_planar_map: bool = Field(default=True, description="使用共享边平面图作为轮廓几何源")
    edge_subpixel: bool = Field(
        default=False,
        description="共享边亚像素（标签 SDF）；默认像素边界 MVP",
    )

    # 多级 K 预算环（P9）：由简入繁 — 先粗后细
    k_start: int = Field(default=4, ge=2, le=16, description="起始色量 K（粗阶段，少色块）")
    delta_k: int = Field(default=2, ge=1, le=4, description="加密时 K 增量")
    k_max: int = Field(default=12, ge=2, le=16, description="色量 K 上限")
    k_max_iters: int = Field(default=4, ge=1, le=6, description="K 外环最大迭代")
    n_margin: int = Field(default=6, ge=0, le=20, description="留给渐变/细节的空余层提示")
    eps_arc: float = Field(default=0.05, ge=0.01, le=0.2, description="弧拟合有损阈值")
    max_contour_area_rel_err: float = Field(
        default=0.03,
        ge=0.0,
        le=0.2,
        description="闭合色块轮廓面积相对 mask 面积的最大相对误差（默认 3%）",
    )
    coarse_max_regions: int = Field(default=8, ge=2, le=40, description="粗阶段最多放置区域数")
    prefer_primitive_seed: bool = Field(default=True, description="优先用圆/椭圆弧基元初始化简洁图章")

    # 匹配
    max_layers: int = Field(
        default=40,
        ge=1,
        le=40,
        description="图层数上限（游戏硬限制 40）",
    )
    min_gain: float = Field(default=0.004, ge=0.0, description="接受新层的最小相对降损")
    stall_patience: int = Field(default=2, ge=1, description="连续无增益则停止的次数")
    loss_epsilon: float = Field(default=0.03, ge=0.0, description="损失足够低则停止")
    recall_k: int = Field(default=48, ge=1, le=128, description="描述子 top-M 召回数（无形状桶）")
    max_rois_per_iter: int = Field(default=3, ge=1, description="每轮最多尝试 ROI 数")
    search_size: int = Field(default=96, ge=32, le=256, description="几何搜索分辨率")
    angle_step_deg: float = Field(default=30.0, gt=0.0, description="粗角度步长（度）")
    refine: bool = Field(default=True, description="是否对接受层做局部连续精修")
    refine_iters: int = Field(default=3, ge=0, description="精修迭代次数")
    add_bottom_fill: bool = Field(default=True, description="是否尝试底色打底层")
    enable_special_fx: bool = Field(default=True, description="是否启用特殊图章渐变通道")
    stamp_subset: list[str] | None = Field(
        default=None,
        description="限定图章 id 列表；None 表示全库可检索",
    )
    stamps_dir: Path = Field(default=Path("assets/stamps"), description="图章目录")
    stamp_curve_cache: Path | None = Field(
        default=None,
        description="图章曲线缓存目录；None 则 assets/.cache/stamp_curves",
    )
    debug_dir: Path | None = Field(
        default=None,
        description="逐步调试图输出目录；None 表示不写。匹配阶段仅输出通过并入层的结果",
    )
    render_supersample: float = Field(
        default=1.0,
        ge=1.0,
        le=4.0,
        description="拟合过程渲染超采样（1 最快）",
    )
    seed: int = Field(default=0, description="随机种子（色量初始化等）")

    # 相似度评判：曲线比对 + 色彩
    pass_score: float = Field(
        default=0.48,
        ge=0.0,
        le=1.0,
        description="综合 overall 达标线",
    )
    pass_sim: float = Field(default=0.50, ge=0.0, le=1.0, description="S_sim 门槛")
    pass_line: float = Field(default=0.55, ge=0.0, le=1.0, description="S_line 硬门槛")
    n_particles: int = Field(default=384, ge=32, le=4096, description="每区域粒子数")
    use_cuda: bool = Field(default=True, description="优先使用 CUDA")

    @field_validator("stamps_dir", "stamp_curve_cache", "debug_dir", mode="before")
    @classmethod
    def _path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value)


class ApproxResult(BaseModel):
    """一次近似的完整结果。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    document: EmblemDocument = Field(..., description="输出图章文档")
    target: ApproxTarget = Field(..., description="概括目标")
    final_loss: float = Field(..., description="最终复合损失")
    per_layer_gains: list[float] = Field(default_factory=list, description="每层降损")
    similarity: Any = Field(default=None, description="SimilarityReport，可选")
    preview_rgb: Any = Field(default=None, description="渲染预览 RGB uint8，可选")
