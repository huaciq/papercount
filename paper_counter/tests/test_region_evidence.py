import numpy as np
import pytest
from paper_counter import CounterConfig,PaperCounter
from paper_counter.region_evidence import review_thickness,limit_to_observed_neighbors


def test_paper_internal_dark_region_is_not_shorter_external_gap():
    y,x=np.mgrid[:500,:250]
    image=np.zeros(y.shape,dtype=float)
    centers=[40,90,114,138,162,186,210,234,258,282,306,330]
    for center in centers:
        inside=np.abs(y-center)<7
        image+=inside*(45+10*np.sin(x/20)**2)
        image+=125*np.exp(-.5*((y-center+8)/1.6)**2)
        image+=100*np.exp(-.5*((y-center-8)/1.2)**2)
    result=PaperCounter(CounterConfig(algorithm='regions',region_structure='double_bright')).process(image.astype(np.uint8))
    assert result.count==len(centers)
    assert all(b['band_ids']==[2*i,2*i+1] for i,b in enumerate(result.region_boxes))
    model=result.debug_info['region_detection']['identity']['pair_phase_model']
    assert model['reliable'] and len(model['reference_ids'])>=2


def make_bands(centers,wide=None):
    return [dict(id=f'band_{i}',index=i,center=float(c),top=float(c-4),bottom=float(c+4),
        score=2.,prominence=80.,coverage=1.) if wide!=c else
        dict(id=f'band_{i}',index=i,center=float(c),top=float(c-5),bottom=float(c+25),
        score=2.,prominence=80.,coverage=1.) for i,c in enumerate(centers)]


def test_large_region_with_real_internal_boundary_is_split():
    y=np.arange(220);centers=[20,45,70,95,115,145,170,195]
    profile=sum(100*np.exp(-.5*((y-c)/2.5)**2) for c in centers).astype(np.float32)
    bands=make_bands([20,45,70,95,145,170,195],wide=95)
    raw=np.tile(profile[:,None],(1,80));valid=np.ones(raw.shape,bool)
    result,review=review_thickness(bands,profile,raw,valid)
    assert len(result)==8
    split=[r for r in review if r['decision']=='split_at_observed_internal_boundaries']
    assert len(split)==1 and split[0]['children']==2
    assert all(p['column_support']>.9 for p in split[0]['boundaries'])


def test_truly_thick_smooth_paper_is_preserved_without_a_real_divider():
    y=np.arange(220);centers=[20,45,70,95,145,170,195]
    profile=sum(100*np.exp(-.5*((y-c)/(7.5 if c==95 else 2.5))**2) for c in centers).astype(np.float32)
    bands=make_bands(centers,wide=95)
    raw=np.tile(profile[:,None],(1,80));valid=np.ones(raw.shape,bool)
    result,review=review_thickness(bands,profile,raw,valid)
    assert len(result)==len(bands)
    assert not any(r['decision']=='split_at_observed_internal_boundaries' for r in review)


def test_huge_contour_over_existing_peaks_only_rebounds_no_duplicate_count():
    y=np.arange(220);centers=[20,45,70,95,120,145,170,195]
    profile=sum(100*np.exp(-.5*((y-c)/2.5)**2) for c in centers).astype(np.float32)
    bands=make_bands(centers);bands[2].update(top=5.,bottom=205.)
    clipped,events=limit_to_observed_neighbors(bands,profile)
    assert len(clipped)==len(bands)
    assert clipped[2]['top']>45 and clipped[2]['bottom']<95
    assert events[0]['count_changed'] is False
    raw=np.tile(profile[:,None],(1,80))
    after,_=review_thickness(clipped,profile,raw,np.ones(raw.shape,bool))
    assert len(after)==len(bands)
