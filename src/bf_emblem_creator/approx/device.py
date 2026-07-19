"""设备选择：优先 CUDA，否则 CPU（并给出一次警告）。"""

from __future__ import annotations

import warnings

import torch

_WARNED = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """返回计算设备。"""
    global _WARNED
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_cuda and not _WARNED:
        warnings.warn(
            "未检测到 CUDA，近似搜索将在 CPU 上运行（仍用批量张量，但会更慢）。",
            stacklevel=2,
        )
        _WARNED = True
    return torch.device("cpu")
