from dataclasses import replace
import numpy as np
import pytest
from paper_counter import CounterConfig
from paper_counter.models import StripPeakCandidate
from paper_counter.tracking import make_track
from paper_counter.endpoint_association import repair_endpoint_association


@pytest.mark.parametrize('raw_body,complete_candidates,expected',[(True,True,True),(False,True,False),(True,False,False)])
def test_fragment_repair_needs_existing_candidates_and_real_raw_body(raw_body,complete_candidates,expected):
    cfg=replace(CounterConfig(),endpoint_reassociation_enabled=True)
    def peak(s,y):return StripPeakCandidate(s,y,30,30,5,2,'strong')
    seed=make_track([peak(s,4.5 if s<6 else 9.5) for s in range(1,7)],20,12,cfg)
    seed.track_id='seed'
    neighbor=make_track([peak(s,24.5) for s in range(12)],20,12,cfg);neighbor.track_id='neighbor'
    raw=np.zeros((12,100),dtype=np.float32)
    if raw_body:raw[:,5:15]=90;raw[:,25:35]=90
    candidates=[[peak(s,4.5),peak(s,24.5)] if complete_candidates or 1<=s<=6 else [peak(s,24.5)] for s in range(12)]
    tracks,replaced,debug=repair_endpoint_association([seed,neighbor],candidates,raw,0,'positive',20,cfg)
    assert bool(replaced)==expected
    assert len(tracks)==2
    if expected:
        repaired=tracks[0]
        assert repaired.support_count==12
        assert all(y==4.5 for y in repaired.strip_positions.values())
        assert replaced==[seed]
