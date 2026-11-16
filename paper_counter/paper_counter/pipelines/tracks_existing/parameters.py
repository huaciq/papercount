from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict,dataclass,field,fields
from pathlib import Path
from ...config import CounterConfig
from ..parameters import strict_fields,mapping,positive


@dataclass(frozen=True)
class TracksPreparationParameters:
    automatic_geometry: bool = False
    target_spacing_px: float = 14.
    preserve_extent: bool = True
    roi_policy: str = 'foreground_envelope'

    def __post_init__(self):
        if type(self.automatic_geometry) is not bool or type(self.preserve_extent) is not bool:
            raise ValueError('tracks geometry flags must be boolean')
        positive(self.target_spacing_px,'target_spacing_px')
        if not 8<=self.target_spacing_px<=32 or self.roi_policy!='foreground_envelope':
            raise ValueError('Unsupported tracks geometry settings')

    @classmethod
    def from_dict(cls,data):return cls(**strict_fields(cls,data,'tracks preparation'))


EXCLUDED={f.name for f in fields(CounterConfig) if f.name.startswith('region_')}|{
    'algorithm','narrow_width','auto_geometry_enabled','geometry_target_spacing','tracks_geometry_preserve_extent',
    'display_max_height','display_max_width','debug_chunk_rows'}


@dataclass(frozen=True)
class TracksParameters:
    profile: str = 'e3_3'
    preparation: TracksPreparationParameters = field(default_factory=TracksPreparationParameters)
    overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.profile,str) or self.profile not in {'base','e3_3','paper_layers_v4'}:
            raise ValueError('Unknown tracks profile')
        values=mapping(self.overrides,'tracks overrides')
        unknown=set(values)-({f.name for f in fields(CounterConfig)}-EXCLUDED)
        if unknown:raise ValueError(f'Unknown tracks overrides: {sorted(unknown)}')
        object.__setattr__(self,'overrides',values)

    @classmethod
    def from_dict(cls,data):
        values=strict_fields(cls,data,'tracks parameters')
        if 'preparation' in values:values['preparation']=TracksPreparationParameters.from_dict(values['preparation'])
        return cls(**values)

    @classmethod
    def from_legacy(cls,cfg):
        defaults=CounterConfig().to_dict()
        overrides={k:deepcopy(v) for k,v in cfg.to_dict().items() if k not in EXCLUDED and v!=defaults[k]}
        return cls('base',TracksPreparationParameters(cfg.auto_geometry_enabled,cfg.geometry_target_spacing,
                                                     cfg.tracks_geometry_preserve_extent),overrides)

    def to_config(self,runtime):
        if self.profile=='base':data=CounterConfig().to_dict()
        else:
            name='default.yaml' if self.profile=='e3_3' else 'paper_layers_v4.yaml'
            data=CounterConfig.from_yaml(Path(__file__).resolve().parents[2]/'presets'/name).to_dict()
        data.update(deepcopy(self.overrides))
        data.update(auto_geometry_enabled=self.preparation.automatic_geometry,
                    geometry_target_spacing=self.preparation.target_spacing_px,
                    tracks_geometry_preserve_extent=self.preparation.preserve_extent,
                    **runtime.legacy_fields())
        return CounterConfig(**data)

    def to_dict(self):return asdict(self)
