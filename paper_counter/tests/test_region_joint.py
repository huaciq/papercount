import json
import numpy as np
import pytest
from paper_counter import CounterConfig,PaperCounter,PaperDetector
from paper_counter.region_joint import joint_boundaries


def luminous_stack(count=14,width=225,only_strip=None,weak=None):
    y,x=np.mgrid[:(count+4)*20,:width]
    image=np.full(y.shape,15.)
    centers=(np.arange(count)+2)*20
    for i,c in enumerate(centers):
        feature=(9. if weak==i else 140.)*np.exp(-.5*((y-c)/1.7)**2)
        if only_strip is not None:
            edges=np.linspace(0,width,4,dtype=int)
            feature*=((x>=edges[only_strip])&(x<edges[only_strip+1]))
        image+=feature
    return image.astype(np.uint8),centers


def assert_disjoint(result):
    boxes=result.region_boxes
    assert len(boxes)==result.count
    assert all(a['detection_xyxy'][3]<=b['detection_xyxy'][1] for a,b in zip(boxes,boxes[1:]))
    assert all(b['detection_xyxy'][3]>b['detection_xyxy'][1] for b in boxes)


def test_three_independent_signals_are_kept_and_final_boxes_do_not_overlap():
    image,centers=luminous_stack()
    result=PaperCounter(CounterConfig(algorithm='regions')).process(image)
    assert result.count==len(centers)
    rd=result.debug_info['region_detection'];obs=rd['three_strip_observation']
    assert obs['bounds']==[[0,75],[75,150],[150,225]]
    assert obs['fusion_order']=='independent_native_candidates_then_joint_evidence'
    assert obs['fused_signal_role']=='auxiliary_only'
    data=result.artifacts['three_strip']
    assert data['signals'].shape==data['raw_signals'].shape==data['weights'].shape==(3,len(image))
    assert np.all(data['weights'].sum(axis=0)<=1.00001)
    assert rd['joint_body_boundaries']['peak_width_is_body_height'] is False
    assert_disjoint(result)


@pytest.mark.parametrize('strip',[0,1,2])
def test_one_clear_strip_does_not_need_two_positive_votes(strip):
    image,centers=luminous_stack(only_strip=strip)
    result=PaperCounter(CounterConfig(algorithm='regions',region_structure='single_bright')).process(image)
    assert result.count==len(centers)
    for box in result.region_boxes:
        evidence=box['evidence']['strip_support']
        assert evidence[0]['source_strips']==[strip]
    assert_disjoint(result)


def test_real_weak_signal_is_retained_as_candidate_before_fusion():
    image,centers=luminous_stack(weak=6)
    result=PaperCounter(CounterConfig(algorithm='regions',region_structure='single_bright')).process(image)
    assert result.count==len(centers)
    assert any(abs((b['detection_xyxy'][1]+b['detection_xyxy'][3])/2-centers[6])<5 for b in result.region_boxes)
    assert_disjoint(result)


def test_one_clear_strip_can_also_support_a_complete_double_structure():
    y,x=np.mgrid[:480,:225];image=np.full(y.shape,15.)
    centers=list(range(45,430,30))
    for c in centers:
        image+=(x>=150)*(120*np.exp(-.5*((y-c+5)/1.4)**2)+100*np.exp(-.5*((y-c-5)/1.4)**2))
    result=PaperCounter(CounterConfig(algorithm='regions',region_structure='double_bright')).process(image.astype(np.uint8))
    assert result.count==len(centers)
    assert_disjoint(result)


def test_dark_gap_is_preserved_without_period_tiling():
    image,_=luminous_stack(count=18)
    image[145:220]=15
    result=PaperCounter(CounterConfig(algorithm='regions',region_structure='single_bright')).process(image)
    assert not any(b['detection_xyxy'][1]<160 and b['detection_xyxy'][3]>205 for b in result.region_boxes)
    assert_disjoint(result)


def boundary_fixture(thick=False):
    h=350;y=np.arange(h);profile=np.full(h,10.)
    spans=[(20+25*i,37+25*i) for i in range(10)]
    if thick:
        spans=[(a+(17 if i>4 else 0),b+(17 if i>=4 else 0)) for i,(a,b) in enumerate(spans)]
    bands=[];units=[]
    for i,(a,b) in enumerate(spans):
        c=(a+b-1)/2
        profile[a:b]=90
        profile+=25*np.exp(-.5*((y-c)/(1.+i%3))**2)
        bands.append(dict(center=c))
        units.append(dict(id=str(i),top=c-2,bottom=c+2,band_ids=[i],score=3.))
    observations=dict(signals=np.tile(profile,(3,1)),raw_signals=np.tile(profile,(3,1)),
        weights=np.full((3,h),1/3))
    return units,bands,observations


def test_body_height_comes_from_interfaces_not_variable_bright_peak_widths():
    units,bands,observation=boundary_fixture()
    bodies,plan=joint_boundaries(units,bands,observation)
    heights=np.array([b['bottom']-b['top'] for b in bodies])
    assert np.std(heights[1:-1])<1.5
    assert np.median(heights)==pytest.approx(17,abs=2)
    assert plan['boundary_source']=='three_strip_gray_interfaces'


def test_clear_genuinely_thick_body_is_allowed_by_soft_prior():
    units,bands,observation=boundary_fixture(thick=True)
    bodies,plan=joint_boundaries(units,bands,observation)
    heights=np.array([b['bottom']-b['top'] for b in bodies])
    assert len(bodies)==10
    assert heights[4]>1.7*np.median(heights)
    assert plan['thickness'][4]['soft_cost']>0


def test_exported_horizontal_sections_are_disjoint_and_curves_are_separate(tmp_path):
    image,_=luminous_stack(count=7)
    y,x=np.mgrid[:image.shape[0],:image.shape[1]]
    # Distort image in space, rather than giving each strip an artificial
    # periodic shift; native direction alignment must handle the curvature.
    import cv2
    distorted=cv2.remap(image,x.astype(np.float32),(y+.035*(x-112)).astype(np.float32),cv2.INTER_LINEAR)
    result=PaperDetector(config=CounterConfig(algorithm='regions')).detect(distorted,output_dir=tmp_path)
    assert result.count==7
    assert_disjoint(result.raw_result)
    exported=json.loads((tmp_path/'detection_boxes.json').read_text(encoding='utf-8'))
    assert all(len(b['boundary_samples']['top'])==3 for b in exported['boxes'])
    assert all(a['overlay_xyxy'][3]<=b['overlay_xyxy'][1] for a,b in zip(exported['boxes'],exported['boxes'][1:]))
    preview=cv2.imdecode(np.fromfile(tmp_path/'result_overlay.png',dtype=np.uint8),cv2.IMREAD_COLOR)
    roi=result.raw_result.artifacts['roi'];height,width=roi.shape[:2]
    assert preview.shape[1]==2*width
    unmarked=preview[116:116+height,width:2*width]
    expected=cv2.cvtColor(roi,cv2.COLOR_GRAY2BGR) if roi.ndim==2 else roi
    assert np.array_equal(unmarked,expected[:,::-1])
    assert exported['render_metadata']['comparison_panel']['horizontal_mirror'] is True
    assert exported['render_metadata']['comparison_panel']['gap_px']==0
