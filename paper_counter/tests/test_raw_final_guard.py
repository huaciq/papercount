from dataclasses import replace
from types import SimpleNamespace
import pytest
from paper_counter import CounterConfig
from paper_counter.raw_final_guard import apply_raw_observation_guard

def case(support=.33,body=False,position=98,neighbor=70):
    a=SimpleNamespace(track_id='partial',center_y=position,strip_positions={0:position,1:position+1},
                      support_ratio=support,local_evidence={'raw_body_confirmation':{'supported':body}})
    b=SimpleNamespace(track_id='strong',center_y=neighbor,strip_positions={0:neighbor},support_ratio=1.,local_evidence={})
    features={t.track_id:SimpleNamespace(local_spacing=10.) for t in (a,b)}
    debug={'count_after':2,'proposed_reject_track_ids':[],'applied_reject_track_ids':[],
           'track_decisions':[{'track_id':t.track_id,'applied':False} for t in (a,b)]}
    return [a,b],features,{'y1':0,'y2':100},debug

def test_isolated_partial_track_with_no_raw_window_is_rejected_and_logged():
    ts,fs,bounds,d=case();cfg=replace(CounterConfig(),raw_final_observation_guard_enabled=True,final_track_verifier_enabled=True,final_track_verifier_shadow_mode=False)
    kept,rejected,info=apply_raw_observation_guard(ts,fs,bounds,cfg,d)
    assert [t.track_id for t in kept]==['strong']
    assert [t.track_id for t in rejected]==['partial']
    assert info['decisions'][0]['usable_raw_identity_strips']==0
    assert d['applied_reject_track_ids']==['partial'] and d['count_after']==1
    assert d['track_decisions'][0]['terminal_decision']=='REJECT_UNOBSERVABLE_ISOLATED_TRACK'

@pytest.mark.parametrize('changes',[{'support':.5},{'body':True},{'neighbor':90},{'position':90}])
def test_observed_endpoint_raw_body_neighbor_or_usable_window_prevents_rejection(changes):
    ts,fs,bounds,d=case(**changes);cfg=replace(CounterConfig(),raw_final_observation_guard_enabled=True,final_track_verifier_enabled=True,final_track_verifier_shadow_mode=False)
    kept,rejected,_=apply_raw_observation_guard(ts,fs,bounds,cfg,d)
    assert kept==ts and not rejected

def test_default_flag_preserves_existing_behavior():
    ts,fs,bounds,d=case();kept,rejected,info=apply_raw_observation_guard(ts,fs,bounds,CounterConfig(),d)
    assert kept is ts and not rejected and not info['enabled']

def test_guard_requires_active_verifier():
    with pytest.raises(ValueError):replace(CounterConfig(),raw_final_observation_guard_enabled=True,final_track_verifier_enabled=False)
    with pytest.raises(ValueError):replace(CounterConfig(),raw_final_observation_guard_enabled=True,final_track_verifier_enabled=True,final_track_verifier_shadow_mode=True)

@pytest.mark.parametrize('body,pair,contrast,reject',[(False,0,1,True),(True,0,1,False),(False,1,1,False),(False,0,8,False)])
def test_dark_reflection_guard_requires_absence_of_independent_body_evidence(body,pair,contrast,reject):
    ts,fs,bounds,d=case(position=65,neighbor=20,body=body)
    ts[0].strip_positions={s:65 for s in range(12)}
    for f in fs.values():
        f.paper_support_above=0.01
        f.positive_pair_support=pair
        f.raw_signed_contrast=contrast
    cfg=replace(CounterConfig(),raw_final_observation_guard_enabled=True,
                final_track_verifier_enabled=True,final_track_verifier_shadow_mode=False,
                isolated_dark_track_guard_enabled=True)
    kept,rejected,info=apply_raw_observation_guard(ts,fs,bounds,cfg,d)
    assert bool(rejected)==reject
    if reject:
        assert d['track_decisions'][0]['terminal_decision']=='REJECT_ISOLATED_DARK_REFLECTION'


@pytest.mark.parametrize('complete,pair,support,reject',[(12,0,0,True),(0,0,0,False),(12,1,0,False),(12,0,.5,False)])
def test_recovery_guard_distinguishes_measured_absence_from_missing_data(complete,pair,support,reject):
    ts,fs,bounds,d=case(position=65,neighbor=20)
    ts[0].strip_positions={s:65 for s in range(12)}
    for t in ts:t.source='detected'
    ts[0].source='recovered'
    for f in fs.values():
        f.paper_support_above=.01;f.positive_pair_support=pair
        f.partial_endpoint={'complete_identity_strips':complete,'body_support_ratio':support}
    cfg=replace(CounterConfig(),raw_final_observation_guard_enabled=True,
                final_track_verifier_enabled=True,final_track_verifier_shadow_mode=False,
                recovered_raw_body_guard_enabled=True)
    kept,rejected,info=apply_raw_observation_guard(ts,fs,bounds,cfg,d)
    assert bool(rejected)==reject
    if reject:assert d['track_decisions'][0]['terminal_decision']=='REJECT_RECOVERY_WITHOUT_RAW_BODY'
