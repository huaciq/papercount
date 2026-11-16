"""Common execution and provenance; material decisions belong to pipelines."""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np
from .contracts import InputFrame,PipelineExecution
from .recipe import Recipe
from .registry import get_registry


class DetectionEngine:
    def __init__(self,recipe:Recipe):self.recipe=deepcopy(recipe)

    def execute(self,image,*,image_id=None):
        if not isinstance(image,np.ndarray):raise TypeError('image must be a numpy array')
        if image.ndim not in (2,3) or image.size==0 or image.dtype not in (np.uint8,np.uint16):
            raise ValueError('image must be nonempty uint8/uint16 gray or color data')
        pipeline=get_registry().create(self.recipe.pipeline)
        execution=pipeline.run(InputFrame(image.copy(),image_id),self.recipe)
        if not isinstance(execution,PipelineExecution):
            raise TypeError('Pipeline must return PipelineExecution')
        result=execution.result
        if type(result.count) is not int or result.count<0:
            raise ValueError('Pipeline produced an invalid count')
        if not math.isfinite(float(result.confidence)):
            raise ValueError('Pipeline produced a non-finite consistency score')
        # The diagnostic additions do not reinterpret any legacy coordinate or
        # score field. Full recipes make a named or compatibility run replayable.
        result.debug_info['pipeline']=dict(id=self.recipe.pipeline,recipe_id=self.recipe.recipe_id,
            material_family=self.recipe.material_family,count_unit=self.recipe.count_unit)
        result.debug_info['resolved_recipe']=self.recipe.to_dict()
        return execution

    def process(self,image,*,image_id=None):return self.execute(image,image_id=image_id).result
