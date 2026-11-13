from dataclasses import replace
import numpy as np
import pytest
from paper_counter import CounterConfig
from paper_counter.raw_period import raw_interface_period
from paper_counter.strip_sequence import opposite_duplicates
from paper_counter.strip_sequence import choose_body_polarity, associate_oriented_candidates, suppress_shared_raw_bodies
from paper_counter.models import StripPeakCandidate
from paper_counter.tracking import make_track


def profiles(period=24):
    y = np.arange(1600)
    return np.array([140+65*np.cos(2*np.pi*(y-s*.3)/period) for s in range(12)])


def test_raw_interfaces_correct_doubled_prior_without_generating_locations():
    period, debug = raw_interface_period(profiles(), 48., CounterConfig())
    assert period == 24 and debug['changed']
    assert debug['agreeing_strips'] == 12
    assert debug['period_generated_paper_count'] == 0


@pytest.mark.parametrize('initial', [24., 30., 12.])
def test_correct_or_nonharmonic_prior_is_preserved(initial):
    period, debug = raw_interface_period(profiles(), initial, CounterConfig())
    assert period == initial and not debug['changed']


def test_local_reflection_and_flat_raw_cannot_change_prior():
    raw = np.full((12,1600), 140.)
    raw[:3] = profiles()[:3]
    period, debug = raw_interface_period(raw,48.,CounterConfig())
    assert period == 48 and not debug['changed']
    assert raw_interface_period(np.full((12,1600),140.),48.,CounterConfig())[0] == 48


def track(y):
    cfg = CounterConfig()
    peaks = [StripPeakCandidate(s,float(y),20.,10.,1.,2.,'strong') for s in range(5)]
    return make_track(peaks,30.,12,cfg)


def test_adaptive_pair_matching_tolerates_localization_but_preserves_adjacent_body():
    a,b = track(50),track(75)
    a.local_evidence={'spacing_from_local_candidates':30.,'opposite_positions':{str(s):70. for s in range(5)}}
    cfg=replace(CounterConfig(),strip_sequence_adaptive_body_dedup_enabled=True)
    assert not opposite_duplicates(a,b)
    assert opposite_duplicates(a,b,cfg)
    assert not opposite_duplicates(a,track(100),cfg)
    assert not opposite_duplicates(a,track(75),CounterConfig())


def test_polarity_selection_requires_clear_body_and_support_advantage():
    def branch(support,pair,n=8):
        tracks=[track(50+30*i) for i in range(n)]
        for t in tracks:t.local_evidence={'weighted_support':support,'paired_score':pair}
        return tracks
    chosen,debug=choose_body_polarity({'positive':branch(.65,.6),'negative':branch(.95,.8)},'positive')
    assert chosen=='negative' and debug['changed']
    assert choose_body_polarity({'positive':branch(.95,.8),'negative':branch(.96,.8)},'positive')[0]=='positive'
    assert choose_body_polarity({'positive':branch(.95,.8),'negative':branch(.7,.95)},'positive')[0]=='positive'


def test_ordered_raw_orientation_recovers_tilt_and_uses_each_peak_once():
    cfg=CounterConfig(num_strips=5)
    strips=[(s*40,(s+1)*40) for s in range(5)]
    # Drift is larger than the ordinary 0.5T gate, but RAW slope explains it.
    candidates=[[StripPeakCandidate(s,float(80+30*i+16*s),20.,10.,1.,2.,'strong')
                 for i in range(6)] for s in range(5)]
    slopes=[np.full(400,.4) for _ in range(5)]
    quality=np.ones((5,400))
    diagnostic={'polarities':{'positive':{'strips':[{'observations':[]} for _ in range(5)]}}}
    tracks,debug=associate_oriented_candidates(candidates,diagnostic,'positive',strips,slopes,quality,0,30,cfg)
    assert len(tracks)==6 and all(t.support_count==5 for t in tracks)
    assert sum(t.support_count for t in tracks)==30
    assert all(len(set(t.strip_positions))==t.support_count for t in tracks)
    assert debug['generated_observations']==0


def test_shared_pair_dedup_keeps_raw_interface_missing_from_normalized_candidates():
    cfg=CounterConfig(num_strips=5)
    def diagnostic(ys,pair):
        return {'polarities':{'positive':{'strips':[{'observations':[
            {'y':float(y),'opposite_pair_y':float(pair),'pair_score':.8,'raw_significance':1.}
            for y in ys]} for _ in range(5)]}}}
    raw={'negative':np.zeros((5,200))};raw['negative'][:,80]=50.
    kept,rejected,debug=suppress_shared_raw_bodies([track(40),track(46)],diagnostic([40,46],80),raw,0,'positive',cfg)
    assert len(kept)==1 and len(rejected)==1 and len(debug['decisions'])==1
    # An actual intervening RAW paper interface proves that these are distinct
    # even though weak normalized evidence paired both to the later boundary.
    raw['negative'][:,65]=40.
    kept,rejected,_=suppress_shared_raw_bodies([track(40),track(72)],diagnostic([40,72],80),raw,0,'positive',cfg)
    assert len(kept)==2 and not rejected
