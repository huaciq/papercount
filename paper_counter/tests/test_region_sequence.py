import numpy as np
from paper_counter.region_sequence import paper_support,repair_sequence
from test_region_thickness import fixture as thickness_fixture


def fixture(heights):
    units,bands,obs=thickness_fixture(heights)
    for band in bands:band['score']=3.
    return units,bands,obs


def observations(units,background=12.):
    h=int(units[-1]['bottom']+60);y=np.arange(h);raw=np.full(h,background)
    for u in units:
        a=int(u['top']);b=int(u['bottom']);raw[a:b]=110
        raw+=25*np.exp(-.5*((y-(a+b)/2)/2)**2)
    return dict(signals=np.tile(raw,(3,1)),raw_signals=np.tile(raw,(3,1)),weights=np.full((3,h),1/3))


def test_sensor_glow_is_not_paper_and_is_not_subdivided():
    units,bands,_=fixture([25]*20)
    obs=observations(units)
    obs['raw_signals'][:,400:]=15;obs['signals'][:,400:]=15
    # Strong paper at 110 and a broad background region at 15 coexist.
    out,audit=repair_sequence(units,bands,obs,units,25)
    assert audit['paper_support']['intervals'][-1][1]<410
    assert all(u['bottom']<410 for u in out)


def test_complete_raw_cycle_can_override_half_height_boxes():
    units,bands,_=fixture([18]*40)
    h=800;y=np.arange(h);raw=np.full(h,10.)
    for c in range(40,740,36):
        raw+=130*np.exp(-.5*((y-c)/4)**2)+65*np.exp(-.5*((y-c-16)/3)**2)
    obs=dict(signals=np.tile(raw,(3,1)),raw_signals=np.tile(raw,(3,1)),weights=np.full((3,h),1/3))
    out,audit=repair_sequence(units,bands,obs,units,36)
    assert audit['normal_height']>30
    assert len(out)<26
    assert audit['scale_model']['source'].startswith('raw_')


def test_joint_boundaries_can_cross_old_boxes_and_leave_real_gap_free():
    units,bands,_=fixture([40]*16)
    obs=observations(units);obs['raw_signals'][:,260:380]=12;obs['signals'][:,260:380]=12
    units[3]['bottom']-=10;units[4]['top']-=10
    out,audit=repair_sequence(units,bands,obs,units,40)
    assert all(a['bottom']<=b['top'] for a,b in zip(out,out[1:]))
    assert not any(u['top']<280 and u['bottom']>350 for u in out)
    assert audit['algorithm']=='supported_joint_resegmentation'


def test_one_clear_stripe_can_define_credible_paper_support():
    units,_,_=fixture([30]*12);obs=observations(units)
    raw=obs['raw_signals'].copy();raw[:2]=12
    mask,core,audit=paper_support(raw,30)
    assert np.any(mask[40:300])
    assert audit['stripes'][2]['credible']
