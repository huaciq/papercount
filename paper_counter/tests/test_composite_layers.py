from types import SimpleNamespace as NS
import numpy as np
from paper_counter import CounterConfig
from paper_counter.composite_layers import CompositeLayerModel,distinct_composite_observations,same_composite_slot

def model():
    m=object.__new__(CompositeLayerModel);m.cfg=CounterConfig(evidence_model=True,composite_layers_enabled=True)
    m.offset=0;m.period=20;m.local=NS(at=lambda y:20.);m.relative=np.linspace(-.75,.75,31)
    y=np.arange(300);wave=25*np.cos(2*np.pi*y/20)+18*np.cos(4*np.pi*y/20)
    # 不同横向窗口具有不同内部明暗结构，包括亮度方向相反的窗口。
    m.views=np.array([[120+wave,150-wave,90+wave+8*np.sin(2*np.pi*y/20)]])
    m.references=[NS(track_id=str(i),strip_positions={0:float(i)},center_y=float(i)) for i in (60,80,100,120,160,180,200)]
    return m

def test_complete_composite_signature_tolerates_horizontal_texture_differences():
    m=model();t=m.template(0,140.)
    assert t is not None
    score,valid,_=m.score(0,[140.],t)
    assert valid[0] and score[0]>.95
    # 半周期纹理不能自动代表另一张纸。
    _,valid,_=m.score(0,[150.],t)
    assert not valid[0]


def test_repeatable_sub_gray_level_structure_is_not_lost_to_bright_band_cutoff():
    m=model();m.views=150+(m.views-np.mean(m.views,axis=2,keepdims=True))*.01
    t=m.template(0,140.)
    assert t is not None
    _,valid,_=m.score(0,[140.],t)
    assert valid[0]

def test_black_region_and_unobservable_windows_do_not_establish_structure():
    m=model();t=m.template(0,140.)
    m.views[:,:,125:156]=0
    _,valid,_=m.score(0,[140.,0.],t)
    assert not np.any(valid)


def test_out_of_frame_direction_samples_are_unknown_even_when_stored_values_are_nonzero():
    m=model();t=m.template(0,140.)
    masks=np.ones((3,300),bool);masks[:2,125:156]=False
    m.alignment=[{'view_available_rows':masks}]
    _,valid,_=m.score(0,[140.],t)
    assert not valid[0]


def test_ambiguous_half_cycle_does_not_create_extra_layers():
    m=model();wave=20*np.cos(4*np.pi*np.arange(300)/20)
    m.views=np.array([[120+wave,130+wave,140+wave]])
    assert m.template(0,140.) is None

def test_distinct_layer_identity_requires_multiple_real_strip_observations():
    p={'verified':True,'samples':[{'period':20.}]}
    a=NS(local_evidence={'composite_structure':p},strip_positions={0:20.,1:20.})
    b=NS(local_evidence={},strip_positions={0:40.,1:40.})
    assert distinct_composite_observations(a,b)
    b.strip_positions={0:20.,1:40.}
    assert not distinct_composite_observations(a,b)


def test_disjoint_fragments_share_one_layer_only_with_same_two_anchor_phase():
    left=NS(track_id='left',strip_positions={0:10.,1:12.,2:20.,3:22.})
    right=NS(track_id='right',strip_positions={0:50.,1:52.,2:60.,3:62.})
    a=NS(track_id='a',strip_positions={0:30.,1:32.})
    b=NS(track_id='b',strip_positions={2:40.,3:42.})
    assert same_composite_slot(a,b,left,right,20.)
    b.strip_positions={2:55.,3:57.}
    assert not same_composite_slot(a,b,left,right,20.)
