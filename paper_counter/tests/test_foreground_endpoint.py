from types import SimpleNamespace
import pytest
from paper_counter import CounterConfig
from paper_counter.foreground_endpoint import verified_foreground_endpoint


@pytest.mark.parametrize('outer,support,complete,pair,distance,expected',[
    (True,.92,12,.83,14,True),(False,.92,12,.83,14,False),
    (True,.5,12,.83,14,False),(True,.92,0,.83,14,False),
    (True,.92,12,0,14,False),(True,.92,12,.83,50,False)])
def test_foreground_endpoint_requires_independent_observations(outer,support,complete,pair,distance,expected):
    track=SimpleNamespace(strip_positions={s:100 for s in range(12)},support_ratio=1,continuity_ratio=1,smoothness_score=1)
    f=SimpleNamespace(partial_endpoint={'body_support_ratio':support,'complete_identity_strips':complete},
        false_family='BACKGROUND_FALSE_EDGE',evidence_families=[],distance_to_stack_boundary=distance,
        local_spacing=33,background_below_score=.99,positive_pair_support=pair)
    assert verified_foreground_endpoint(track,f,outer,CounterConfig())==expected
