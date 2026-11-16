"""Versioned recipes and capture metadata, independent of material algorithms."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict,dataclass,field
from pathlib import Path
import re
import json
import yaml
from .config import CounterConfig
from .pipelines.parameters import RuntimeParameters,mapping,strict_fields,positive
from .registry import get_registry,legacy_pipeline_id

RESOURCE_ROOT=Path(__file__).parent/'recipes'


def identifier(value,name):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',value):
        raise ValueError(f'{name} must be a nonempty identifier')
    return value


@dataclass(frozen=True)
class CaptureProfile:
    capture_id: str = 'unspecified_v1'
    camera: str | None = None
    lighting: str | None = None
    pixel_size_mm: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.capture_id,'capture_id')
        for name in ('camera','lighting'):
            if getattr(self,name) is not None and not isinstance(getattr(self,name),str):
                raise ValueError(f'capture.{name} must be a string or null')
        if self.pixel_size_mm is not None:positive(self.pixel_size_mm,'pixel_size_mm')
        object.__setattr__(self,'metadata',mapping(self.metadata,'capture metadata'))
        try:json.dumps(self.metadata,allow_nan=False)
        except (ValueError,TypeError) as error:raise ValueError('capture metadata must be finite JSON data') from error

    @classmethod
    def from_dict(cls,data):return cls(**strict_fields(cls,data,'capture profile'))

    @classmethod
    def load(cls,value):
        if isinstance(value,dict):return cls.from_dict(value)
        identifier(value,'capture_profile')
        path=RESOURCE_ROOT/'capture'/f'{value}.yaml'
        if not path.is_file():raise ValueError(f'Unknown capture profile: {value}')
        with path.open(encoding='utf-8') as stream:return cls.from_dict(yaml.safe_load(stream))


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    material_family: str
    count_unit: str
    pipeline: str
    parameters: object
    capture_profile: CaptureProfile = field(default_factory=CaptureProfile)
    runtime: RuntimeParameters = field(default_factory=RuntimeParameters)
    schema_version: int = 1
    compatibility_config: CounterConfig | None = field(default=None,repr=False,compare=False)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version!=1:
            raise ValueError('Unsupported recipe schema_version; expected 1')
        identifier(self.recipe_id,'recipe_id')
        spec=get_registry().get(self.pipeline)
        if self.material_family not in spec.material_families:
            raise ValueError(f'{self.pipeline} does not support material_family={self.material_family!r}')
        if self.count_unit not in spec.count_units:
            raise ValueError(f'{self.pipeline} does not support count_unit={self.count_unit!r}')
        if not isinstance(self.parameters,spec.parameters_type):
            raise ValueError(f'Wrong parameters type for {self.pipeline}')
        # Trigger the existing low-level numeric validation at recipe load.
        self.to_counter_config()

    def to_counter_config(self):
        if self.compatibility_config is not None:return deepcopy(self.compatibility_config)
        return self.parameters.to_config(self.runtime)

    def to_dict(self):
        result=dict(schema_version=self.schema_version,recipe_id=self.recipe_id,pipeline=self.pipeline,
            material_family=self.material_family,count_unit=self.count_unit,
            capture_profile=asdict(self.capture_profile),parameters=self.parameters.to_dict(),runtime=asdict(self.runtime))
        if self.compatibility_config is not None:
            result['compatibility']=dict(format='counter_config_v1',config=self.compatibility_config.to_dict())
        return result

    @classmethod
    def from_dict(cls,value):
        data=mapping(value,'recipe')
        allowed={'schema_version','recipe_id','material_family','count_unit','pipeline','capture_profile','parameters','runtime','compatibility'}
        unknown=set(data)-allowed
        if unknown:raise ValueError(f'Unknown recipe keys: {sorted(unknown)}')
        required={'schema_version','recipe_id','material_family','count_unit','pipeline'}
        if required-set(data):raise ValueError(f'Missing recipe keys: {sorted(required-set(data))}')
        spec=get_registry().get(data['pipeline'])
        params=spec.parameters_type.from_dict(data.get('parameters',{}))
        runtime=RuntimeParameters.from_dict(data.get('runtime',{}))
        capture=CaptureProfile.load(data.get('capture_profile','unspecified_v1'))
        compatibility=data.get('compatibility')
        config=None
        if compatibility is not None:
            item=mapping(compatibility,'compatibility')
            if set(item)!={'format','config'} or item['format']!='counter_config_v1':
                raise ValueError('Unsupported compatibility configuration')
            try:config=CounterConfig(**mapping(item['config'],'compatibility.config'))
            except TypeError as error:raise ValueError(str(error)) from error
            if legacy_pipeline_id(config)!=data['pipeline']:
                raise ValueError('Compatibility configuration selects a different pipeline')
            if spec.parameters_type.from_legacy(config).to_dict()!=params.to_dict():
                raise ValueError('Compatibility configuration conflicts with recipe parameters')
            original_runtime=RuntimeParameters.from_legacy(config).legacy_fields()
            if original_runtime!=runtime.legacy_fields():
                raise ValueError('Compatibility configuration conflicts with runtime')
        return cls(data['recipe_id'],data['material_family'],data['count_unit'],data['pipeline'],params,
                   capture,runtime,data['schema_version'],config)

    @classmethod
    def from_legacy(cls,config,*,recipe_id='legacy_config_v1'):
        config=deepcopy(config)
        pipeline=legacy_pipeline_id(config)
        spec=get_registry().get(pipeline)
        return cls(recipe_id,spec.material_families[0],spec.count_units[0],pipeline,
                   spec.parameters_type.from_legacy(config),runtime=RuntimeParameters.from_legacy(config),
                   compatibility_config=config)

    @classmethod
    def load(cls,value):
        if isinstance(value,cls):return deepcopy(value)
        if not isinstance(value,(str,Path)):raise TypeError('recipe must be a name, YAML path or Recipe')
        path=Path(value)
        if not path.is_file():
            if isinstance(value,Path) or path.suffix:
                raise FileNotFoundError(path)
            identifier(value,'recipe name')
            path=RESOURCE_ROOT/'materials'/f'{value}.yaml'
            if not path.is_file():raise ValueError(f'Unknown recipe: {value!r}')
        with path.open(encoding='utf-8') as stream:return cls.from_dict(yaml.safe_load(stream))


def load_config_or_recipe(path):
    path=Path(path)
    with path.open(encoding='utf-8') as stream:data=yaml.safe_load(stream) or {}
    data=mapping(data,'configuration')
    if {'schema_version','pipeline','recipe_id'} & set(data):return Recipe.from_dict(data)
    return Recipe.from_legacy(CounterConfig.from_yaml(path))
