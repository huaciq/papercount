from types import SimpleNamespace as NS
import numpy as np
from paper_counter import CounterConfig
from paper_counter.background_region_guard import apply_background_region_guard
from paper_counter.stripe_evidence import independent_identity_duplicate, independent_gap_observations
from paper_counter.track_identity import extract_track_aligned_profiles,build_edge_template
from paper_counter.models import StripPeakCandidate
from paper_counter.tracking import make_track


def test_unobservable_profile_cannot_be_a_template_reference():
    cfg=CounterConfig(num_strips=6)
    t=make_track([StripPeakCandidate(s,1.,10,10,2,1,'strong') for s in range(6)],20,6,cfg)
    t.track_id='edge'
    raw=np.ones((6,100))*100
    p=extract_track_aligned_profiles(t,20,raw,raw,raw,raw,0,cfg)
    assert len(p.raw_by_strip)==0
    template=build_edge_template({'edge':p},['edge'],cfg)
    assert template.status=='insufficient' and not template.reference_track_ids


def fixture():
    cfg=CounterConfig(evidence_model=True)
    def track(name,y,support=12):
        return NS(track_id=name,center_y=y,source='detected',support_ratio=support/12,
            sequence_decision='selected',strip_positions={s:y for s in range(support)},reason=None)
    anchors=[track(str(i),30*i) for i in range(6)]
    features={t.track_id:NS(complete_identity_strips=12,above_raw_level=150.,below_raw_level=20.,
        raw_signed_contrast=130.,local_spacing=30.) for t in anchors}
    return cfg,track,anchors,features


def test_dim_reflection_outside_broad_paper_region_is_rejected():
    cfg,track,anchors,features=fixture();reflection=track('reflection',400,5)
    features['reflection']=NS(complete_identity_strips=5,above_raw_level=14.,below_raw_level=10.,
        raw_signed_contrast=4.,local_spacing=30.)
    kept,rejected,diag=apply_background_region_guard(anchors+[reflection],features,cfg,{})
    assert rejected==[reflection] and len(kept)==6 and diag['paper_reference']==150


def test_separated_dark_full_width_paper_remains_counted():
    cfg,track,anchors,features=fixture();dark=track('dark',400)
    features['dark']=NS(complete_identity_strips=12,above_raw_level=14.,below_raw_level=0.,
        raw_signed_contrast=14.,local_spacing=30.)
    kept,rejected,_=apply_background_region_guard(anchors+[dark],features,cfg,{})
    assert dark in kept and not rejected


def test_interior_dark_layer_is_not_removed_as_exterior_reflection():
    cfg,track,anchors,features=fixture();dark=track('dark',75,5)
    features['dark']=NS(complete_identity_strips=5,above_raw_level=14.,below_raw_level=10.,
        raw_signed_contrast=4.,local_spacing=30.)
    kept,rejected,_=apply_background_region_guard(anchors+[dark],features,cfg,{})
    assert dark in kept and not rejected


def test_native_tracks_that_touch_and_diverge_are_not_same_identity():
    a=NS(track_id='seed_a',strip_positions={0:10.,1:20.})
    b=NS(track_id='seed_b',strip_positions={0:10.,1:35.})
    assert not independent_identity_duplicate(a,b,'positive')
    b.strip_positions={0:10.,1:20.}
    assert independent_identity_duplicate(a,b,'positive')


def test_single_bright_patch_does_not_establish_an_independent_gap_layer():
    model=NS(canonical_track=lambda t:t,local=NS(at=lambda y:20.))
    left=NS(center_y=0.,strip_positions={0:0.,1:0.})
    right=NS(center_y=40.,strip_positions={0:40.,1:40.})
    candidate=NS(center_y=20.,strip_positions={0:20.})
    assert not independent_gap_observations(candidate,left,right,model)['verified']
    candidate.strip_positions[1]=20.
    assert independent_gap_observations(candidate,left,right,model)['verified']


def test_single_strip_complete_three_body_identity_can_recover_a_layer():
    left=NS(track_id='left',center_y=0.,strip_positions={0:0.})
    candidate=NS(track_id='middle',center_y=20.,strip_positions={0:20.})
    right=NS(track_id='right',center_y=40.,strip_positions={0:40.})
    ids={'left':0,'middle':1,'right':2}
    model=NS(canonical_track=lambda t:t,local=NS(at=lambda y:20.),
        bindings=lambda t:{0:ids[t.track_id]},
        bands=[[{'upper':0.,'lower':12.},{'upper':20.,'lower':32.},{'upper':40.,'lower':52.}]],
        strong_bands=[[True,True,True]],
        pair_evidence=lambda a,b:{'samples':[{'strip':0,'separate_bodies':True}]})
    proof=independent_gap_observations(candidate,left,right,model)
    assert proof['verified'] and proof['single_strip_complete_identity']==[0]
    model.bands[0][1]['lower']=23.
    assert not independent_gap_observations(candidate,left,right,model)['verified']
    model.bands[0][1]['lower']=32.
    model.pair_evidence=lambda a,b:{'samples':[{'strip':0,'separate_bodies':a.track_id!='middle'}]}
    assert not independent_gap_observations(candidate,left,right,model)['verified']


def test_outer_component_does_not_bypass_same_body_deduplication():
    cfg=CounterConfig(num_strips=6)
    def track(name,positions):
        t=make_track([StripPeakCandidate(s,y,10,10,2,1,'strong') for s,y in positions.items()],20,6,cfg)
        t.track_id=name;t.boundary_polarity='negative'
        return t
    a=track('seed_00001',{0:10.,1:20.})
    b=track('raw_outer_component_1',{0:10.,1:21.})
    assert independent_identity_duplicate(a,b,'negative')


def test_partial_aligned_window_cannot_override_actual_bright_paper_samples():
    cfg,track,anchors,features=fixture();paper=track('partial',400,8)
    paper.local_evidence={'paper_body':{'samples':[{'state':'confirmed','body_level':160.,'contrast':100.} for _ in range(8)]}}
    features['partial']=NS(complete_identity_strips=2,above_raw_level=0.,below_raw_level=0.,
        raw_signed_contrast=0.,local_spacing=30.)
    kept,rejected,_=apply_background_region_guard(anchors+[paper],features,cfg,{})
    assert paper in kept and not rejected


def test_unobservable_outer_reflection_uses_available_raw_samples_not_zero_placeholder():
    cfg,track,anchors,features=fixture();reflection=track('partial',400,8)
    features['partial']=NS(complete_identity_strips=0,above_raw_level=0.,below_raw_level=0.,
        raw_signed_contrast=0.,local_spacing=30.,
        partial_endpoint={'samples':[{'paper_median':5.,'gap_median':5.,'contrast':0.} for _ in range(5)]})
    _,rejected,diag=apply_background_region_guard(anchors+[reflection],features,cfg,{})
    assert rejected==[reflection]
    assert diag['decisions'][0]['sample_source']=='available_partial_RAW_samples'
