from types import SimpleNamespace
import numpy as np
from paper_counter import CounterConfig
from paper_counter.partial_endpoint import partial_endpoint_evidence


def test_observed_clipped_positive_body_is_supported():
    cfg=CounterConfig();raw=np.zeros((12,100));raw[:,5:20]=90
    t=SimpleNamespace(strip_positions={s:4.5 for s in range(12)},continuity_ratio=1.,smoothness_score=1.)
    result=partial_endpoint_evidence(t,30,raw,0,'positive',cfg)
    assert result['confirmed'] and result['complete_identity_strips']==0
    assert result['body_support_ratio']==1


def test_absent_samples_and_black_background_cannot_confirm_paper():
    cfg=CounterConfig();raw=np.zeros((12,100))
    t=SimpleNamespace(strip_positions={s:4.5 for s in range(12)},continuity_ratio=1.,smoothness_score=1.)
    assert not partial_endpoint_evidence(t,30,raw,0,'positive',cfg)['confirmed']
    raw[:,:]=90
    assert not partial_endpoint_evidence(t,30,raw,0,'positive',cfg)['confirmed']
    t.strip_positions={s:0. for s in range(12)}
    assert not partial_endpoint_evidence(t,30,raw,0,'negative',cfg)['confirmed']


def test_interior_and_wrong_polarity_never_get_partial_override():
    cfg=CounterConfig();raw=np.zeros((12,150));raw[:,50:70]=90
    t=SimpleNamespace(strip_positions={s:49.5 for s in range(12)},continuity_ratio=1.,smoothness_score=1.)
    assert not partial_endpoint_evidence(t,30,raw,0,'positive',cfg)['confirmed']
    raw[:,:]=0;raw[:,5:20]=90;t.strip_positions={s:4.5 for s in range(12)}
    assert not partial_endpoint_evidence(t,30,raw,0,'negative',cfg)['confirmed']


def test_disconnected_clipped_visibility_cannot_override_identity():
    cfg=CounterConfig();raw=np.zeros((12,100));raw[:,5:20]=90
    positions={s:4.5 for s in range(12)};positions[5]=positions[6]=0
    t=SimpleNamespace(strip_positions=positions,continuity_ratio=1.,smoothness_score=1.)
    proof=partial_endpoint_evidence(t,30,raw,0,'positive',cfg)
    assert proof['body_support_ratio']>.6
    assert not proof['continuous_visibility'] and not proof['confirmed']


def test_one_unobservable_strip_does_not_split_observed_sheet():
    cfg=CounterConfig();raw=np.zeros((12,100));raw[:,5:20]=90
    positions={s:4.5 for s in range(12)};positions[5]=0
    t=SimpleNamespace(strip_positions=positions,continuity_ratio=1.,smoothness_score=1.)
    proof=partial_endpoint_evidence(t,30,raw,0,'positive',cfg)
    assert proof['longest_unobservable_run']==1 and proof['confirmed']
