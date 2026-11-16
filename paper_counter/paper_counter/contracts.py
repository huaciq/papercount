"""Pipeline data contracts. Working and original coordinates remain explicit."""
from __future__ import annotations
from dataclasses import asdict,dataclass,field
from typing import Any
import numpy as np
from .models import PaperCountResult


@dataclass(frozen=True)
class InputFrame:
    image: np.ndarray = field(repr=False)
    image_id: str | None = None


@dataclass
class PreparedFrame:
    image: np.ndarray = field(repr=False)
    geometry: dict | None = None
    nonlinear_maps: dict[str,np.ndarray] = field(default_factory=dict,repr=False)

    def to_input(self,x,y,*,space='prepared'):
        """Map a measured prepared/aligned point back to the engine input."""
        if space!='prepared':
            if space not in self.nonlinear_maps:raise ValueError(f'Unknown coordinate space: {space}')
            mapping=self.nonlinear_maps[space]
            xx=float(np.clip(x,0,mapping.shape[1]-1));yy=float(np.clip(y,0,mapping.shape[0]-1))
            x0,y0=int(np.floor(xx)),int(np.floor(yy))
            x1,y1=min(x0+1,mapping.shape[1]-1),min(y0+1,mapping.shape[0]-1)
            fx,fy=xx-x0,yy-y0
            y=float((1-fy)*((1-fx)*mapping[y0,x0]+fx*mapping[y0,x1])+
                    fy*((1-fx)*mapping[y1,x0]+fx*mapping[y1,x1]))
        matrix=np.asarray((self.geometry or {}).get('detection_to_input',np.eye(3)),dtype=float)
        point=matrix@np.array([x,y,1.])
        return float(point[0]/point[2]),float(point[1]/point[2])


@dataclass
class PipelineExecution:
    result: PaperCountResult
    prepared: PreparedFrame


@dataclass(frozen=True)
class Evidence:
    kind: str
    source_id: str
    observed: bool
    coordinate_system: str = 'original_input'


@dataclass(frozen=True)
class PaperInstance:
    instance_id: str
    count_unit: str
    geometry_kind: str
    geometry: dict[str,Any]
    evidence: tuple[Evidence,...]
    state: str = 'unverified'
    counted: bool = True
    coordinate_system: str = 'original_input'

    def to_dict(self):return asdict(self)
