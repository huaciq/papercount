from __future__ import annotations
from dataclasses import asdict,dataclass,field
from ...config import CounterConfig
from ..parameters import RegionPreparationParameters,RegionObservationParameters,strict_fields


@dataclass(frozen=True)
class CorrugatedParameters:
    preparation: RegionPreparationParameters = field(default_factory=RegionPreparationParameters)
    observation: RegionObservationParameters = field(default_factory=RegionObservationParameters)

    @classmethod
    def from_dict(cls,data):
        values=strict_fields(cls,data,'corrugated parameters')
        for name,kind in [('preparation',RegionPreparationParameters),('observation',RegionObservationParameters)]:
            if name in values:values[name]=kind.from_dict(values[name])
        return cls(**values)

    @classmethod
    def from_legacy(cls,cfg):
        return cls(RegionPreparationParameters.from_legacy(cfg),RegionObservationParameters.from_legacy(cfg))

    def to_config(self,runtime):
        return CounterConfig(algorithm='regions',region_material='corrugated',region_structure='auto',
            region_signal_mode='three_strip',**self.preparation.legacy_fields(),
            **self.observation.legacy_fields(),**runtime.legacy_fields())

    def to_dict(self):return asdict(self)
