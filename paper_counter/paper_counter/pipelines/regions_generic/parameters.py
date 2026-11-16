from __future__ import annotations
from dataclasses import asdict,dataclass,field
from ...config import CounterConfig
from ..parameters import RegionPreparationParameters,RegionObservationParameters,strict_fields


@dataclass(frozen=True)
class RegionsParameters:
    preparation: RegionPreparationParameters = field(default_factory=RegionPreparationParameters)
    observation: RegionObservationParameters = field(default_factory=RegionObservationParameters)
    structure: str = 'auto'
    signal_mode: str = 'three_strip'

    def __post_init__(self):
        if not isinstance(self.structure,str) or self.structure not in {'auto','single_bright','double_bright'}:
            raise ValueError('structure must be auto, single_bright or double_bright')
        if not isinstance(self.signal_mode,str) or self.signal_mode not in {'global','three_strip'}:
            raise ValueError('signal_mode must be global or three_strip')

    @classmethod
    def from_dict(cls,data):
        values=strict_fields(cls,data,'regions parameters')
        for name,kind in [('preparation',RegionPreparationParameters),('observation',RegionObservationParameters)]:
            if name in values:values[name]=kind.from_dict(values[name])
        return cls(**values)

    @classmethod
    def from_legacy(cls,cfg):
        return cls(RegionPreparationParameters.from_legacy(cfg),RegionObservationParameters.from_legacy(cfg),
                   cfg.region_structure,cfg.region_signal_mode)

    def to_config(self,runtime):
        return CounterConfig(algorithm='regions',region_material='generic',region_structure=self.structure,
            region_signal_mode=self.signal_mode,**self.preparation.legacy_fields(),
            **self.observation.legacy_fields(),**runtime.legacy_fields())

    def to_dict(self):return asdict(self)
