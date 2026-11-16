"""Physical seams and flute webs have different counting semantics."""
import cv2
import numpy as np
import pytest

from paper_counter import CounterConfig, PaperDetector
from paper_counter.corrugated_regions import segment_corrugated_bodies, corroborate_seam_scale
from paper_counter.paper_geometry import stack_bounds
from paper_counter.region_postvalidation import identify_suspicious_boxes


def observations(image, centers):
    bands = [dict(center=float(c), score=3.) for c in centers]
    units = [dict(id=f'u{i}', top=float(c-3), bottom=float(c+3), band_ids=[i])
             for i, c in enumerate(centers)]
    return bands, units, np.ones_like(image, dtype=bool)


def test_flute_webs_prevent_cavity_subdivision():
    raw = np.full((850,225), 8., np.float32)
    centers = []
    for top in range(40, 820, 130):
        bottom = top+120
        raw[top:top+5] = 230
        raw[bottom-5:bottom] = 230
        for x in (25,125):
            polygon=np.array([[x,top],[x+14,top],[x+65,bottom-1],[x+51,bottom-1]],np.int32)
            cv2.fillConvexPoly(raw,polygon,230)
        centers.extend([top+2,bottom-3])
    bands, units, valid = observations(raw,centers)
    result, audit = segment_corrugated_bodies(raw,valid,bands,units,40)
    assert len(result) == 6
    assert audit['observed_seams']['cavity_rows_rejected'] > 300
    assert all(100 < u['bottom']-u['top'] < 150 for u in result)
    assert not any(u['thickness_inferred'] for u in result)


def test_compressed_cavity_shoulders_are_not_weak_contact_seams():
    raw=np.zeros((460,225),np.float32)
    centers=[]
    for top in range(35,430,68):
        raw[top:top+60]=220
        raw[top+20:top+35,:160]=8
        centers.extend([top+7,top+27,top+50])
    bands,units,valid=observations(raw,centers)
    result,audit=segment_corrugated_bodies(raw,valid,bands,units,30)
    assert not audit['observed_seams']['open_flute_structure']
    assert len(result)==6
    assert all(u['bottom']-u['top']>=55 for u in result)


def test_false_short_seam_spacing_needs_independent_raw_recurrence():
    y=np.arange(850)
    signal=np.full(850,15.)
    for center in range(50,800,36):
        signal+=180*np.exp(-.5*((y-center)/5)**2)+45*np.exp(-.5*((y-center-17)/3)**2)
    raw=np.tile(signal[:,None],(1,225)).astype(np.float32)
    centers=list(range(50,800,18))
    _,units,_=observations(raw,centers)
    reference,audit=corroborate_seam_scale(15.,raw,(35,810),units,18)
    assert 34<=reference<=38
    assert audit['override']


def test_no_visible_seam_cannot_be_created_from_thickness():
    raw=np.zeros((420,225),np.float32)
    raw[40:380]=220
    bands,units,valid=observations(raw,range(60,380,40))
    result,audit=segment_corrugated_bodies(raw,valid,bands,units,40)
    assert len(result)==1
    assert audit['selected_seams']==[]
    assert result[0]['top']==pytest.approx(40,abs=1)
    assert result[0]['bottom']==pytest.approx(380,abs=1)


def test_gray_contact_seams_are_detected_without_a_black_air_gap():
    raw=np.full((300,225),15.,np.float32)
    raw[40:250]=210
    for cut in range(70,250,30):
        raw[cut-1:cut+1]=165
    bands,units,valid=observations(raw,range(55,250,30))
    result,audit=segment_corrugated_bodies(raw,valid,bands,units,30)
    assert len(result)==7
    assert len(audit['selected_seams'])==6


def test_strict_roi_ignores_fixture_and_keeps_dim_outer_paper():
    raw=np.full((1800,300),35,np.uint8)
    raw[80:84]=250
    for top in range(600,1040,40):
        raw[top:top+28]=240
    raw[1040:1068]=90
    bounds,located=stack_bounds(raw,strict=True)
    assert located
    assert 500 < bounds[1] <= 600
    assert 1060 < bounds[3] < 1200


def test_free_black_interval_is_not_a_counted_body():
    raw=np.zeros((400,225),np.float32)
    centers=[]
    for top in (30,90,150,270,330):
        raw[top:top+40]=220
        centers.append(top+20)
    bands,units,valid=observations(raw,centers)
    result,_=segment_corrugated_bodies(raw,valid,bands,units,60)
    assert len(result)==5
    assert not any(u['top']<220<u['bottom'] for u in result)


def test_tiny_outer_liner_is_excluded_but_real_neighbor_is_preserved():
    raw=np.zeros((360,225),np.float32)
    raw[10:15]=220
    centers=[12]
    for top in range(35,330,48):
        raw[top:top+40]=220
        centers.append(top+20)
    bands,units,valid=observations(raw,centers)
    result,audit=segment_corrugated_bodies(raw,valid,bands,units,48)
    assert audit['excluded_partial_edges']
    assert len(result)==7
    assert result[0]['top']>=30


def test_live_gray_arrays_reach_postvalidation():
    raw=np.full((3,140),10,np.float32)
    raw[:,40:48]=180
    boxes=[dict(number=i+1,detection_xyxy=[0,float(i*20),225,float(i*20+(8 if i==2 else 20))],
                band_ids=[i],thickness_inferred=False) for i in range(6)]
    suspicious=identify_suspicious_boxes(boxes,{'region_detection':{'three_strip_observation':{}}},
                                         {'raw_signals':raw})
    issues=next(s['issues'] for s in suspicious if s['box_number']==3)
    assert any(i['type']=='small_thickness_outlier' for i in issues)
    assert not any(i['type']=='weak_top_boundary' for i in issues)


def test_corrugated_is_a_regions_preset_and_validates_material():
    detector=PaperDetector(preset='corrugated')
    assert detector.config.algorithm=='regions'
    assert detector.config.region_material=='corrugated'
    with pytest.raises(ValueError,match='corrugated material requires'):
        CounterConfig(region_material='corrugated')
