"""Recipe contracts, pipeline extension and compatibility at public boundaries."""
from copy import deepcopy
from dataclasses import dataclass,replace
from pathlib import Path
import json,os,subprocess,sys
import cv2
import numpy as np
import pytest

from paper_counter import (PaperDetector,CounterConfig,Recipe,CaptureProfile,DetectionEngine,
    PipelineSpec,PipelineRegistry,get_registry,PreparedFrame,PipelineExecution,PaperCountResult)
from paper_counter.pipelines.regions_generic.parameters import RegionsParameters
from test_counter import synthetic_papers

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('recipe,preset',[
    ('regions_generic_v1','regions'),('corrugated_side_v1','corrugated'),('tracks_existing_v1','tracks')])
def test_named_recipes_preserve_existing_effective_settings(recipe,preset):
    assert PaperDetector(recipe=recipe).config==PaperDetector(preset=preset).config


@pytest.mark.parametrize('preset',['tracks','regions','corrugated','tracks_preprocessed','paper_layers_v4'])
def test_full_compatibility_recipe_is_replayable(preset):
    detector=PaperDetector(preset=preset)
    payload=detector.recipe.to_dict()
    replay=Recipe.from_dict(payload)
    assert replay.to_counter_config()==detector.config
    assert replay.to_dict()==payload
    json.dumps(payload,allow_nan=False)


def test_compatibility_retains_unused_legacy_fields_and_rejects_shadow_settings():
    cfg=CounterConfig(algorithm='regions',num_strips=5,narrow_width=180)
    recipe=Recipe.from_legacy(cfg)
    assert Recipe.from_dict(recipe.to_dict()).to_counter_config()==cfg
    payload=recipe.to_dict()
    payload['parameters']['preparation']['sampling_width_px']=225
    with pytest.raises(ValueError,match='conflicts'):
        Recipe.from_dict(payload)


@pytest.mark.parametrize('path,value',[
    (('schema_version',),True),
    (('pipeline',),'unknown_v1'),
    (('count_unit',),'bright_band'),
    (('material_family',),'generic_layers'),
    (('parameters','tracks_preprocessing_enabled'),True),
    (('parameters','preparation','sampling_width_px'),0),
    (('parameters','preparation','roi_policy'),'unknown'),
    (('runtime','unexpected'),1),
    (('unexpected',),1),
])
def test_recipe_errors_fail_before_inference(path,value):
    payload=Recipe.load('corrugated_side_v1').to_dict()
    target=payload
    for key in path[:-1]:target=target[key]
    target[path[-1]]=value
    with pytest.raises(ValueError):Recipe.from_dict(payload)


def test_capture_profile_is_metadata_not_an_implicit_image_transform():
    with pytest.raises(ValueError):CaptureProfile(pixel_size_mm=0)
    with pytest.raises(ValueError):CaptureProfile(metadata={'x':float('nan')})
    base=Recipe.load('regions_generic_v1')
    changed=replace(base,capture_profile=CaptureProfile('camera_a_v1','camera_a','side',.01))
    assert changed.to_counter_config()==base.to_counter_config()
    assert Recipe.from_dict(changed.to_dict()).capture_profile==changed.capture_profile


def test_original_instance_geometry_and_legacy_fields_agree():
    image,_=synthetic_papers(count=8)
    original=np.rot90(image,-1).copy()
    before=original.copy()
    result=PaperDetector(recipe='regions_generic_v1').detect(original,rotate_ccw=True)
    assert result.count>0
    assert len(result.instances)==result.count==len(result.boxes)
    for instance,box in zip(result.instances,result.boxes):
        assert instance.geometry['polygon']==box['input_polygon']
        assert instance.coordinate_system=='original_input'
        assert instance.count_unit=='optical_unit'
    assert np.array_equal(before,original)
    json.dumps(result.to_dict(),allow_nan=False)


def test_affine_and_tangent_coordinates_are_composed_without_changing_spaces():
    y,x=np.mgrid[:20,:30]
    frame=PreparedFrame(np.zeros((20,30),np.uint8),
        {'detection_to_input':[[2,0,5],[0,3,7],[0,0,1]]},{'aligned':y.astype(np.float32)+.1*x})
    assert frame.to_input(10,5)==(25.,22.)
    assert frame.to_input(10,5,space='aligned')==pytest.approx((25.,25.))
    with pytest.raises(ValueError):frame.to_input(10,5,space='unknown')


def test_a_new_registered_pipeline_requires_no_engine_or_counter_branch(monkeypatch):
    registry=get_registry()
    monkeypatch.setattr(registry,'_specs',dict(registry._specs))

    class ProbePipeline:
        pipeline_id='probe_v1'
        def run(self,frame,recipe):
            frame.image[:]=99
            result=PaperCountResult(0,[],[],[],[],None,'region',0.,
                dict(status='review_required',image_shape=list(frame.image.shape),roi=dict(x1=0,y1=0,x2=20,y2=20)),{})
            return PipelineExecution(result,PreparedFrame(frame.image))

    spec=PipelineSpec('probe_v1',ProbePipeline,RegionsParameters,('test_material',),('test_unit',),'regions')
    registry.register(spec)
    recipe=Recipe('probe_recipe_v1','test_material','test_unit','probe_v1',RegionsParameters())
    image=np.zeros((20,20),np.uint8)
    result=DetectionEngine(recipe).process(image)
    assert not image.any()
    assert result.debug_info['pipeline']['id']=='probe_v1'
    assert result.debug_info['resolved_recipe']['count_unit']=='test_unit'
    with pytest.raises(ValueError,match='already registered'):registry.register(spec)


def test_new_and_legacy_configuration_inputs_are_unambiguous(tmp_path):
    recipe=Recipe.load('corrugated_side_v1')
    with pytest.raises(ValueError):PaperDetector(recipe=recipe,preset='tracks')
    with pytest.raises(ValueError):PaperDetector(recipe=recipe,config=CounterConfig())
    import yaml
    path=tmp_path/'recipe.yaml'
    path.write_text(yaml.safe_dump(recipe.to_dict()),encoding='utf-8')
    assert PaperDetector(config=path).recipe.to_dict()==recipe.to_dict()
    assert PaperDetector(recipe=path).config==PaperDetector(preset='corrugated').config


def test_frozen_batch_replays_a_named_recipe_and_exports_provenance(tmp_path):
    folder=tmp_path/'input';folder.mkdir()
    image,_=synthetic_papers(count=8)
    cv2.imencode('.png',image)[1].tofile(folder/'paper.png')
    output=tmp_path/'run'
    completed=subprocess.run([sys.executable,str(ROOT/'scripts/batch_detect.py'),'--input',str(folder),
        '--recipe','regions_generic_v1','--workers','1','--output',str(output)],
        capture_output=True,text=True,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    assert completed.returncode==0,completed.stdout+completed.stderr
    verification=json.loads((output/'audit/verification.json').read_text(encoding='utf-8'))
    assert verification['source_matches_current'] and verification['inputs_unchanged']
    assert verification['pipeline']=='regions_generic_v1'
    assert verification['recipe_id']=='regions_generic_v1'
    saved=Recipe.load(output/'audit/recipe.yaml')
    assert saved.to_dict()==Recipe.load('regions_generic_v1').to_dict()
    detail=output/'audit/details/paper.png'
    assert Recipe.load(detail/'resolved_recipe.yaml').pipeline=='regions_generic_v1'
    instances=json.loads((detail/'instances.json').read_text(encoding='utf-8'))
    assert instances['coordinate_system']=='original_input'
    assert instances['count_unit']=='optical_unit'
    assert instances['count']==len(instances['instances'])
    debug=json.loads((detail/'debug_info.json').read_text(encoding='utf-8'))
    assert debug['pipeline']['id']=='regions_generic_v1'
