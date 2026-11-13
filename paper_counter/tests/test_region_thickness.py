import numpy as np
from paper_counter.region_thickness import repair_thickness


def fixture(heights):
    h=int(sum(heights)+60);y=np.arange(h);signal=np.full(h,15.)
    units=[];bands=[];top=20.
    for i,height in enumerate(heights):
        bottom=top+height;center=(top+bottom)/2
        signal[int(top):int(bottom)]=100
        signal+=2*np.exp(-.5*((y-center)/1.5)**2)
        units.append(dict(id=str(i),top=top,bottom=bottom,band_ids=[i],score=3.,
                          structure='single_bright',evidence={}))
        bands.append(dict(center=center))
        top=bottom
    observation=dict(signals=np.tile(signal,(3,1)),raw_signals=np.tile(signal,(3,1)),
                     weights=np.full((3,h),1/3))
    return units,bands,observation


def test_four_fragments_can_be_merged_as_a_group():
    units,bands,obs=fixture([40]*7+[10]*4+[40]*7)
    result,audit=repair_thickness(units,bands,obs,units,40)
    assert len(result)==15
    action=next(a for a in audit['actions'] if len(a['source_ids'])==4)
    assert action['source_ids']==['7','8','9','10']
    assert audit['normal_height']==40
    assert result[7]['bottom']-result[7]['top']==40


def test_small_neighbor_is_preferred_over_a_large_normal_neighbor():
    units,bands,obs=fixture([40]*7+[16,8,40]+[40]*7)
    result,audit=repair_thickness(units,bands,obs,units,40)
    assert any(a['source_ids']==['7','8'] for a in audit['actions'])
    assert all(a['source_ids']!=['8','9'] for a in audit['actions'])


def test_background_gap_protects_real_thin_papers_from_merge():
    units,bands,obs=fixture([40]*7+[10]*4+[40]*7)
    for unit in units[6:11]:
        unit['bottom']-=2
        obs['raw_signals'][:,int(unit['bottom']):int(unit['bottom'])+2]=15
    result,audit=repair_thickness(units,bands,obs,units,40)
    assert len(result)==len(units)
    assert not audit['actions']


def test_large_body_can_be_inferred_without_native_complete_units():
    units,bands,obs=fixture([40]*7+[80]+[40]*7)
    result,audit=repair_thickness(units,bands,obs,units,40)
    assert len(result)==len(units)+1
    assert audit['inferred_regions']==2
    assert all(u['evidence']['thickness_repair']['boundary_origin']=='thickness_prediction_adjusted_by_auxiliary_gray'
               for u in result if u.get('thickness_inferred'))


def test_large_body_split_requires_measured_cut_and_normal_children():
    units,bands,obs=fixture([40]*7+[80]+[40]*7)
    big=units[7];mid=(big['top']+big['bottom'])/2
    bands.extend([dict(center=mid-20),dict(center=mid+20)])
    left=dict(big,id='left',band_ids=[len(bands)-2])
    right=dict(big,id='right',band_ids=[len(bands)-1])
    obs['signals'][:,int(mid)-1:int(mid)+2]=25
    obs['raw_signals'][:,int(mid)-1:int(mid)+2]=25
    result,audit=repair_thickness(units,bands,obs,units+[left,right],40)
    assert len(result)==len(units)+1
    assert any(a['action']=='split' for a in audit['actions'])
    assert all(a['bottom']<=b['top'] for a,b in zip(result,result[1:]))


def test_regular_or_insufficient_sequences_are_unchanged():
    for heights in ([40]*15,[10,10,40]):
        units,bands,obs=fixture(heights)
        result,audit=repair_thickness(units,bands,obs,units,40)
        assert result==units and not audit['actions']


def test_two_moderately_thin_boxes_are_reviewed_without_half_height_gate():
    units,bands,obs=fixture([34]*7+[19,21]+[34]*7)
    result,audit=repair_thickness(units,bands,obs,units,34)
    assert any(a['source_ids']==['7','8'] for a in audit['actions'])
    assert audit['fixed_anomaly_ratio_gates'] is False


def test_thickness_can_infer_three_papers_inside_one_box():
    units,bands,obs=fixture([40]*7+[120]+[40]*7)
    result,audit=repair_thickness(units,bands,obs,units,40)
    assert len(result)==len(units)+2
    assert audit['inferred_regions']==3


def test_empty_large_box_does_not_acquire_inferred_sheets():
    units,bands,obs=fixture([40]*7+[80]+[40]*7)
    a=int(units[7]['top']);b=int(units[7]['bottom'])
    obs['raw_signals'][:,a:b]=15;obs['signals'][:,a:b]=15
    result,audit=repair_thickness(units,bands,obs,units,40)
    assert len(result)==len(units)
    assert audit['inferred_regions']==0


def test_adaptive_spread_is_wider_for_naturally_variable_stack():
    sigmas=[]
    for heights in ([40]*24,[30,34,38,42,46,50]*4):
        units,bands,obs=fixture(heights)
        _,audit=repair_thickness(units,bands,obs,units,40)
        sigmas.append(audit['log_thickness_sigma'])
    assert sigmas[1]>sigmas[0]


def test_double_structure_identity_is_a_soft_cost_not_a_split_veto():
    units,_,obs=fixture([40]*7+[120]+[40]*7)
    bands=[]
    for u in units:
        center=(u['top']+u['bottom'])/2
        u['structure']='double_bright';u['band_ids']=[len(bands),len(bands)+1]
        bands.extend([dict(center=center-4),dict(center=center+4)])
    result,audit=repair_thickness(units,bands,obs,units,40)
    assert len(result)>len(units)
    assert audit['inferred_regions']>0
