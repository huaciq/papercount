from dataclasses import replace
import json
import numpy as np
import pytest
from paper_counter import CounterConfig,PaperCounter,PaperDetector
from paper_counter.region_counter import region_nms,select_nonconflicting


def stack(count=12,width=300,noise=0,weak=False):
    y,x=np.mgrid[:(count+3)*30,:width]
    curve=.02*(x-width/2)+1.2*np.sin(x/45)
    raw=np.full(y.shape,15.,dtype=float)
    for i in range(count):
        center=45+30*i
        for offset,amplitude in [(-5,130),(5,70 if weak else 130)]:
            # Some horizontal texture, without requiring a uniformly bright body.
            level=amplitude*(.8+.2*np.sin(x/15+i)**2)
            raw+=level*np.exp(-.5*((y-curve-center-offset)/1.4)**2)
    raw+=np.random.default_rng(5).normal(0,noise,raw.shape)
    return np.clip(raw,0,255).astype(np.uint8)


@pytest.mark.parametrize('weak',[False,True])
def test_complete_double_structure_is_one_box_not_locally_one_or_two(weak):
    result=PaperCounter(CounterConfig(algorithm='regions',region_structure='double_bright')).process(stack(noise=1,weak=weak))
    assert result.count==len(result.region_boxes)==12
    assert result.period==pytest.approx(30,abs=1)
    assert result.paper_edge_tracks==[]
    used=[]
    for b in result.region_boxes:
        assert b['structure']=='double_bright'
        assert b['evidence']['bright_lobes']==2
        assert b['evidence']['dark_depth']>0
        used+=b['band_ids']
    assert len(used)==len(set(used))


def test_single_band_mode_is_an_explicit_different_unit():
    image=stack()
    a=PaperCounter(CounterConfig(algorithm='regions',region_structure='double_bright')).process(image)
    b=PaperCounter(CounterConfig(algorithm='regions',region_structure='single_bright')).process(image)
    assert a.count==12 and b.count==24
    assert all(r['structure']=='single_bright' for r in b.region_boxes)


@pytest.mark.parametrize('value',[0,30,255])
def test_blank_regions_have_no_boxes(value):
    result=PaperCounter(CounterConfig(algorithm='regions')).process(np.full((300,225),value,np.uint8))
    assert result.count==0


def test_large_black_gap_is_not_a_complete_region():
    image=stack(count=12)
    image[170:290]=15
    result=PaperCounter(CounterConfig(algorithm='regions',region_structure='double_bright')).process(image)
    assert result.count==8
    assert not any(b['aligned_xyxy'][1]<190 and b['aligned_xyxy'][3]>270 for b in result.region_boxes)


def test_two_isolated_bands_cannot_box_the_large_black_background():
    y=np.arange(400)[:,None]
    image=np.broadcast_to(15+130*np.exp(-.5*((y-60)/1.5)**2)+130*np.exp(-.5*((y-330)/1.5)**2),(400,225))
    result=PaperCounter(CounterConfig(algorithm='regions',region_structure='double_bright')).process(image.astype(np.uint8))
    assert result.count==0


def test_missing_lobe_remains_unresolved_instead_of_changing_counting_unit():
    image=stack(count=12)
    image[196:205]=15 # Remove the first lobe of the sixth complete structure.
    result=PaperCounter(CounterConfig(algorithm='regions',region_structure='double_bright')).process(image)
    assert result.count==11
    assert result.debug_info['region_detection']['unresolved_bands']
    assert all(b['evidence']['bright_lobes']==2 for b in result.region_boxes)


def test_nms_and_ownership_are_different_operations():
    a=dict(id='a',top=0.,bottom=12.,score=3.,band_ids=[0,1])
    duplicate=dict(id='duplicate',top=1.,bottom=13.,score=2.,band_ids=[0,1])
    shared=dict(id='shared',top=10.,bottom=22.,score=2.,band_ids=[1,2])
    next_unit=dict(id='next',top=20.,bottom=32.,score=3.,band_ids=[2,3])
    kept,log=region_nms([a,duplicate,shared,next_unit])
    assert {b['id'] for b in kept}=={'a','shared','next'}
    assert log[0]['candidate']=='duplicate'
    assert [b['id'] for b in select_nonconflicting(kept)]==['a','next']


def test_api_compact_and_full_boxes_use_explicit_coordinates(tmp_path):
    from paper_counter.region_output import save_regions
    image=stack(count=6)
    cfg=CounterConfig(algorithm='regions',region_structure='double_bright')
    api=PaperDetector(config=cfg).detect(image,output_dir=tmp_path)
    assert api.count==6 and len(api.boxes)==6 and not api.tracks
    assert api.to_dict()['boxes']==api.boxes
    exported=json.loads((tmp_path/'detection_boxes.json').read_text(encoding='utf-8'))
    assert exported['count']==6 and not (tmp_path/'detection_points.csv').exists()
    for box in exported['boxes']:
        x1,y1,x2,y2=box['detection_xyxy']
        assert np.allclose(box['input_xyxy'],[x1+(image.shape[1]-225)//2,y1,x2+(image.shape[1]-225)//2,y2])
        assert box['overlay_xyxy'][1]==pytest.approx(y1+116)
    save_regions(api.raw_result,tmp_path/'full',cfg,full=True)
    assert (tmp_path/'full/region_arrays.npz').is_file()


def test_invalid_structure_is_rejected():
    with pytest.raises(ValueError,match='region_structure'):
        replace(CounterConfig(),region_structure='auto_switch_per_peak')


def test_auto_structure_is_chosen_once_and_detects_alternating_two_lobe_motif():
    cfg=CounterConfig(algorithm='regions')
    result=PaperCounter(cfg).process(stack(count=14,weak=True))
    assert result.debug_info['region_detection']['identity']['optical_structure_before_physical_sequence']=='double_bright'
    assert result.count==14
    assert {b['structure'] for b in result.region_boxes}=={'paper_body'}


def test_auto_does_not_blindly_pair_equally_spaced_single_bands():
    y=np.arange(450)[:,None];image=np.full((450,225),15.)
    for center in range(35,420,18):image+=130*np.exp(-.5*((y-center)/1.5)**2)
    result=PaperCounter(CounterConfig(algorithm='regions')).process(image.astype(np.uint8))
    assert result.count==len(range(35,420,18))
    rd=result.debug_info['region_detection']
    assert rd['identity']['optical_structure_before_physical_sequence']=='single_bright'
    assert rd['identity']['structure_selection']['physical_unit_ambiguous']


def test_auto_recognizes_equal_gap_but_alternating_width_lobes():
    y=np.arange(650)[:,None];image=np.full((650,225),15.)
    centers=list(range(40,610,20))
    for i,center in enumerate(centers):
        sigma=1.1 if i%2 else 2.8
        image+=130*np.exp(-.5*((y-center)/sigma)**2)
    result=PaperCounter(CounterConfig(algorithm='regions')).process(image.astype(np.uint8))
    rd=result.debug_info['region_detection']
    assert rd['identity']['optical_structure_before_physical_sequence']=='double_bright'
    assert rd['identity']['structure_selection']['alternating_shape_fractions']['width']>.55
    # There are 14 complete motifs plus one unmatched lobe. Explicit optical
    # mode still counts only complete motifs; physical mode may infer a partial
    # section, but must keep the full 40px cycle and expose that uncertainty.
    assert rd['thickness_postprocess']['normal_height']==pytest.approx(40,abs=1)
    assert len(centers)//2<=result.count<=len(centers)//2+1
    if result.count>len(centers)//2:
        assert rd['thickness_postprocess']['inferred_regions']>0
    explicit=PaperCounter(CounterConfig(algorithm='regions',region_structure='double_bright')).process(image.astype(np.uint8))
    assert explicit.count==len(centers)//2


def test_box_input_polygon_handles_requested_rotation_and_geometry(tmp_path):
    from paper_counter.coordinates import detection_to_original
    image=stack(count=6)
    vertical=np.rot90(image,-1).copy()
    cfg=CounterConfig(algorithm='regions',auto_geometry_enabled=True,region_structure='double_bright')
    a=PaperDetector(config=cfg).detect(image)
    b=PaperDetector(config=cfg).detect(vertical,rotate_ccw=True,output_dir=tmp_path)
    assert a.count==b.count==6
    for box in b.boxes:
        x1,y1,x2,y2=box['detection_xyxy']
        for (x,y),mapped in zip([(x1,y1),(x2,y1),(x2,y2),(x1,y2)],box['input_polygon']):
            px,py=detection_to_original(x-.5,y-.5,90,vertical.shape[1],vertical.shape[0],b.raw_result.debug_info['geometry_preprocessing'])
            assert np.allclose(mapped,[px+.5,py+.5])
