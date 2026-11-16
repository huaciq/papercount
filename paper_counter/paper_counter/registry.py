"""Explicit pipeline registry. Registering a material does not edit the engine."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable,Protocol


class DetectionPipeline(Protocol):
    pipeline_id: str

    def run(self,frame,recipe):
        """Return a PipelineExecution with explicit prepared-image geometry."""
        ...


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    factory: Callable[[],DetectionPipeline]
    parameters_type: type
    material_families: tuple[str,...]
    count_units: tuple[str,...]
    method_family: str


class PipelineRegistry:
    def __init__(self):self._specs={}

    def register(self,spec):
        if spec.pipeline_id in self._specs:
            raise ValueError(f'Pipeline already registered: {spec.pipeline_id}')
        self._specs[spec.pipeline_id]=spec

    def get(self,pipeline_id):
        if not isinstance(pipeline_id,str):raise ValueError('pipeline must be a string identifier')
        try:return self._specs[pipeline_id]
        except KeyError:raise ValueError(f'Unknown pipeline: {pipeline_id!r}; choose {sorted(self._specs)}') from None

    def create(self,pipeline_id):
        pipeline=self.get(pipeline_id).factory()
        if pipeline.pipeline_id!=pipeline_id:
            raise ValueError('Registered factory returned a different pipeline')
        return pipeline

    def identifiers(self):return tuple(sorted(self._specs))


@lru_cache(maxsize=1)
def get_registry():
    from .pipelines.regions_generic.pipeline import GenericRegionsPipeline
    from .pipelines.regions_generic.parameters import RegionsParameters
    from .pipelines.corrugated_body.pipeline import CorrugatedBodyPipeline
    from .pipelines.corrugated_body.parameters import CorrugatedParameters
    from .pipelines.tracks_existing.pipeline import ExistingTracksPipeline
    from .pipelines.tracks_existing.parameters import TracksParameters
    registry=PipelineRegistry()
    for spec in (
        PipelineSpec('regions_generic_v1',GenericRegionsPipeline,RegionsParameters,('generic_layers',),('optical_unit',),'regions'),
        PipelineSpec('corrugated_body_v1',CorrugatedBodyPipeline,CorrugatedParameters,('corrugated',),('board',),'regions'),
        PipelineSpec('tracks_existing_v1',ExistingTracksPipeline,TracksParameters,('generic_layers',),('paper_edge',),'tracks'),
    ):registry.register(spec)
    return registry


def legacy_pipeline_id(config):
    """One compatibility boundary for old algorithm/material selectors."""
    if config.algorithm not in {'tracks','regions'}:
        raise ValueError('Only tracks and regions are supported')
    return {('tracks','generic'):'tracks_existing_v1',
            ('regions','generic'):'regions_generic_v1',
            ('regions','corrugated'):'corrugated_body_v1'}.get(
                (config.algorithm,config.region_material)) or _invalid_legacy_material()


def _invalid_legacy_material():
    raise ValueError('Unsupported legacy algorithm/material combination')
