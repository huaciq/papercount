"""Strict configuration helpers; no material decisions live here."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import math


def mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a mapping')
    if any(not isinstance(k, str) for k in value):
        raise ValueError(f'{name} keys must be strings')
    return deepcopy(value)


def strict_fields(cls, value, name):
    data=mapping(value,name)
    unknown=set(data)-{f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f'Unknown {name} keys: {sorted(unknown)}')
    return data


def positive(value, name, *, integer=False):
    valid=type(value) is int if integer else type(value) in (int,float)
    if not valid or not math.isfinite(value) or value<=0:
        raise ValueError(f'{name} must be a finite positive {"integer" if integer else "number"}')


@dataclass(frozen=True)
class RuntimeParameters:
    display_max_height: int = 4000
    display_max_width: int = 1600
    debug_chunk_rows: int = 1024
    export: str = 'compact'

    def __post_init__(self):
        for name in ('display_max_height','display_max_width','debug_chunk_rows'):
            positive(getattr(self,name),name,integer=True)
        if not isinstance(self.export,str) or self.export not in {'compact','full_debug'}:
            raise ValueError('runtime.export must be compact or full_debug')

    @classmethod
    def from_dict(cls,data):
        return cls(**strict_fields(cls,data,'runtime'))

    @classmethod
    def from_legacy(cls,cfg):
        return cls(cfg.display_max_height,cfg.display_max_width,cfg.debug_chunk_rows)

    def legacy_fields(self):
        return {k:v for k,v in asdict(self).items() if k!='export'}


@dataclass(frozen=True)
class RegionPreparationParameters:
    automatic_geometry: bool = True
    roi_policy: str = 'bright_core_with_dim_edges'
    target_spacing_px: float = 14.0
    sampling_width_px: int = 225

    def __post_init__(self):
        if type(self.automatic_geometry) is not bool:
            raise ValueError('automatic_geometry must be boolean')
        positive(self.target_spacing_px,'target_spacing_px')
        positive(self.sampling_width_px,'sampling_width_px',integer=True)
        if not 8<=self.target_spacing_px<=32 or not 16<=self.sampling_width_px<=1024:
            raise ValueError('region sampling width/spacing is outside the supported range')
        if self.roi_policy!='bright_core_with_dim_edges':
            raise ValueError('Unsupported regions ROI policy')

    @classmethod
    def from_dict(cls,data):
        return cls(**strict_fields(cls,data,'preparation'))

    @classmethod
    def from_legacy(cls,cfg):
        return cls(cfg.auto_geometry_enabled,'bright_core_with_dim_edges',cfg.geometry_target_spacing,cfg.narrow_width)

    def legacy_fields(self):
        return dict(auto_geometry_enabled=self.automatic_geometry,geometry_target_spacing=self.target_spacing_px,
                    narrow_width=self.sampling_width_px)


@dataclass(frozen=True)
class RegionObservationParameters:
    illumination_sigma: float = 15.0
    epsilon: float = 1e-6
    max_period_px: int = 50

    def __post_init__(self):
        positive(self.illumination_sigma,'illumination_sigma')
        positive(self.epsilon,'epsilon')
        positive(self.max_period_px,'max_period_px',integer=True)
        if self.max_period_px<5:
            raise ValueError('max_period_px must be at least 5')

    @classmethod
    def from_dict(cls,data):
        return cls(**strict_fields(cls,data,'observation'))

    @classmethod
    def from_legacy(cls,cfg):
        return cls(cfg.illumination_sigma,cfg.epsilon,cfg.max_period)

    def legacy_fields(self):
        return dict(illumination_sigma=self.illumination_sigma,epsilon=self.epsilon,max_period=self.max_period_px)
