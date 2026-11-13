from dataclasses import replace
import cv2
import numpy as np
import pytest
from paper_counter.paper_geometry import estimate_angle, measure_spacing, prepare_geometry
from paper_counter.coordinates import detection_to_original,original_to_detection
from paper_counter.config import CounterConfig


def test_angle_and_spacing_are_measured_without_count_target():
    y,x = np.mgrid[:280,:600]
    slope = np.tan(np.deg2rad(3.))
    image = np.clip(120+100*np.cos(2*np.pi*(y-slope*x)/8),0,255).astype(np.uint8)
    angle,info = estimate_angle(image)
    assert info['reliable'] and abs(angle-3)<.3
    flat = np.clip(120+100*np.cos(2*np.pi*y/8),0,255).astype(np.uint8)
    period,info = measure_spacing(flat)
    assert info['reliable'] and period==8


def test_prepared_points_roundtrip_crop_rotation_resize_and_external_rotation():
    y,x = np.mgrid[:240,:420]
    stack = (120+100*np.cos(2*np.pi*(y-.04*x)/5)).astype(np.uint8)
    image = np.zeros((500,700),np.uint8)
    image[70:310,100:520] = stack
    original = image.copy()
    prepared,info = prepare_geometry(image)
    assert np.array_equal(image,original)
    assert info['applied'] and info['scale_x']>1
    forward = np.array(info['input_to_detection'])
    for point in [(200,140),(400,220)]:
        dx,dy,_ = forward @ [*point,1]
        assert np.allclose(detection_to_original(dx,dy,0,700,500,info),point)
        assert np.allclose(detection_to_original(dx,dy,90,500,700,info),
                           [499-point[1],point[0]])
        assert np.allclose(original_to_detection(499-point[1],point[0],90,500,700,info),[dx,dy])


def test_blank_input_does_not_invent_angle_or_scale():
    image = np.zeros((100,200),np.uint8)
    prepared,info = prepare_geometry(image)
    assert prepared is image and not info['applied']
    assert detection_to_original(30,40,0,200,100,info)==(30,40)


def test_guarded_center_is_sampled_once_with_exact_affine_mapping():
    y,x=np.mgrid[:800,:900]
    image=np.clip(125+100*np.cos(2*np.pi*(y-.025*x)/8),0,255).astype(np.uint8)
    prepared,info=prepare_geometry(image,center_width=225,bounded_angle_analysis=True)
    assert prepared.shape[1]==225
    assert info['detector_resampling_passes']==1
    assert info['angle_estimation']['selected_source']=='center_guard'
    matrix=np.asarray(info['input_to_detection'])
    for p in [(450,120),(470,600)]:
        q=matrix@[*p,1.]
        assert np.allclose(detection_to_original(q[0],q[1],0,900,800,info),p)
    direct=cv2.warpAffine(image,matrix[:2],(prepared.shape[1],prepared.shape[0]),flags=cv2.INTER_CUBIC)
    assert np.array_equal(prepared,direct)


def test_config_rejects_roi_that_would_have_wrong_coordinate_meaning():
    with pytest.raises(ValueError,match='automatic ROI'):
        CounterConfig(auto_geometry_enabled=True,roi=dict(x1=0,y1=0,x2=20,y2=20))
    assert not CounterConfig().auto_geometry_enabled


def test_api_and_export_both_map_prepared_observations_back_to_input(tmp_path):
    import json
    from paper_counter import PaperDetector
    from test_counter import synthetic_papers
    image,_ = synthetic_papers(count=8)
    cfg = CounterConfig(auto_geometry_enabled=True)
    result = PaperDetector(config=cfg).detect(image,output_dir=tmp_path)
    info = result.raw_result.debug_info['geometry_preprocessing']
    assert result.detection_shape == tuple(result.raw_result.debug_info['image_shape'])
    assert result.count>0
    exported = json.loads((tmp_path/'detection_coordinates.json').read_text(encoding='utf-8'))
    for api_track,export_track in zip(result.tracks,exported['tracks']):
        for api_point,export_point in zip(api_track['observations'],export_track['observations']):
            expected = detection_to_original(api_point['detection_x'],api_point['detection_y'],
                                             0,image.shape[1],image.shape[0],info)
            assert np.allclose([api_point['input_x'],api_point['input_y']],expected)
            assert np.allclose([export_point['input_x'],export_point['input_y']],expected)
