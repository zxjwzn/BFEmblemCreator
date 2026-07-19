"""P9：层预算与多级 K 自适应环。"""

from __future__ import annotations

from dataclasses import dataclass

from bf_emblem_creator.approx.models import ApproxConfig


@dataclass
class BudgetState:
    """预算环状态。"""

    k: int
    iteration: int
    n_layers: int
    score: float
    stop_reason: str = ""


def initial_k(cfg: ApproxConfig) -> int:
    """起始 K：由简入繁，取 k_start 与 palette_k 的较小者（先少色大块）。"""
    return int(min(cfg.k_start, cfg.palette_k))


def is_coarse_phase(k: int, cfg: ApproxConfig) -> bool:
    """是否仍处粗概括阶段（色量接近起点）。"""
    return k <= cfg.k_start + cfg.delta_k


def next_k(state: BudgetState, cfg: ApproxConfig) -> int | None:
    """
    由简入繁：未达标且有空余层预算时提高 K 并重算。

    空余层判断：n_layers + n_margin < max_layers 时才加密（避免层已满仍涨 K）。
    """
    if state.iteration >= cfg.k_max_iters:
        return None
    if state.score >= cfg.pass_score:
        return None
    if state.n_layers + cfg.n_margin >= cfg.max_layers:
        return None
    nk = state.k + cfg.delta_k
    if nk <= cfg.k_max:
        return nk
    return None


def should_merge_for_overflow(n_layers: int, max_layers: int) -> bool:
    """层将爆时合并弱弧/区域。"""
    return n_layers >= max_layers
