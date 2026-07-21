"""处理器包。"""

from bf_emblem_creator.approx.processors.image_processor import ImageProcessor, ProcessedImage
from bf_emblem_creator.approx.processors.match_assembler import AssemblyResult, StampMatchAssembler
from bf_emblem_creator.approx.processors.region_partitioner import RegionPartition, RegionPartitioner
from bf_emblem_creator.approx.processors.stamp_loader import StampCatalog, StampLoader
from bf_emblem_creator.approx.processors.stamp_renderer import StampRenderer

__all__ = [
    "AssemblyResult",
    "ImageProcessor",
    "ProcessedImage",
    "RegionPartition",
    "RegionPartitioner",
    "StampCatalog",
    "StampLoader",
    "StampMatchAssembler",
    "StampRenderer",
]
