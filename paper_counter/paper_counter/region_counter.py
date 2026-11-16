"""Compatibility imports for the extracted region stages.

Material dispatch is confined to the legacy adapter; policies live in pipelines.
"""
from .shared.region_stages import (
    vertical_iou,region_nms,select_nonconflicting,extract_bands,complete_regions,infer_structure,
)


def process_regions(image,cfg):
    from .registry import get_registry,legacy_pipeline_id
    pipeline=get_registry().create(legacy_pipeline_id(cfg))
    if not hasattr(pipeline,'run_prepared'):
        raise ValueError('process_regions requires a regions pipeline')
    return pipeline.run_prepared(image,cfg)
