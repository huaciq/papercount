"""Crowding, missed observed bands and real dark intervals have distinct rules."""
from dataclasses import replace
import unittest
import numpy as np

from paper_counter import CounterConfig
from paper_counter.models import StripPeakCandidate
from paper_counter.tracking import make_track
from paper_counter.body_evidence import attach_body_evidence
from paper_counter.stripe_evidence import stripe_proposals,deduplicate_body_identities
from paper_counter.signal_gaps import SignalGapModel
from paper_counter.vertical_sequence import optimize_vertical_sequence,transition_score,unary_score
from paper_counter.track_identity import apply_final_track_verifier
import test_track_identity as identity_test_support


class SignalGapTests(unittest.TestCase):
    def setUp(self):
        self.cfg=CounterConfig(evidence_model=True,num_strips=6,min_period=5,max_period=40,
            sequence_period_multiple_mode=False,sequence_short_spacing_weight=3.)
        self.strips=[(10*s,10*(s+1)) for s in range(6)]

    def model(self,raw,period=20,offset=0):
        _,diag=stripe_proposals(raw,self.strips,[np.zeros(raw.shape[1]) for _ in range(6)],
            offset,period,'positive',self.cfg)
        return SignalGapModel(raw,diag,offset,period,'positive',self.cfg)

    def track(self,y,name,raw,period=20,offset=0,strips=None,polarity='positive',score=.8):
        strips=range(6) if strips is None else strips
        t=make_track([StripPeakCandidate(s,float(y),10,10,3,1,'strong') for s in strips],period,6,self.cfg)
        t.track_id=name;t.boundary_polarity=polarity;t.total_score=score;t.candidate_class='strong'
        attach_body_evidence([t],raw,offset,lambda y:period,polarity,self.cfg)
        return t

    def test_cluster_inside_one_measured_body_cannot_count_as_two(self):
        raw=np.zeros((6,200));raw[:,40:53]=80;raw[:,60:73]=80
        model=self.model(raw)
        a=self.track(40,'a',raw);texture=self.track(44,'texture',raw);b=self.track(60,'b',raw)
        self.assertTrue(model.same_identity(a,texture))
        self.assertIsNone(transition_score(a,texture,model.local,self.cfg))
        result=optimize_vertical_sequence([a,texture,b],model.local,self.cfg)
        self.assertEqual(len(result.selected),2)

    def test_black_gap_is_not_missing_paper(self):
        raw=np.zeros((6,240));raw[:,20:30]=80;raw[:,180:190]=25
        model=self.model(raw)
        a=self.track(20,'a',raw);b=self.track(180,'b',raw)
        proof=model.pair_evidence(a,b)
        self.assertEqual(proof['state'],'measured_background_break')
        edge=transition_score(a,b,model.local,self.cfg)
        self.assertEqual(edge['long_gap_penalty'],0.)
        self.assertEqual(edge['spacing_prior_penalty'],0.)
        self.assertEqual(len(optimize_vertical_sequence([a,b],model.local,self.cfg).selected),2)

    def test_gap_with_observed_band_prefers_real_middle_candidate(self):
        raw=np.zeros((6,240))
        for y in (20,40,60,80,100,120,140,160,180):raw[:,y:y+9]=60
        model=self.model(raw)
        a=self.track(60,'a',raw);missing=self.track(80,'middle',raw,score=.35);b=self.track(100,'b',raw)
        self.assertEqual(model.pair_evidence(a,b)['state'],'observed_bands_inside_gap')
        result=optimize_vertical_sequence([a,missing,b],model.local,self.cfg)
        self.assertEqual({t.track_id for t in result.selected},{'a','middle','b'})
        self.assertEqual(len(optimize_vertical_sequence([a,b],model.local,self.cfg).selected),2)

    def test_single_clear_missing_segment_survives_without_snr_bonus(self):
        raw=np.zeros((6,240));raw[:,60:69]=80;raw[:,100:109]=80;raw[2,80:89]=80
        model=self.model(raw)
        a=self.track(60,'a',raw);middle=self.track(80,'middle',raw,strips=[2],score=.2);b=self.track(100,'b',raw)
        result=optimize_vertical_sequence([a,middle,b],model.local,self.cfg)
        self.assertEqual({t.track_id for t in result.selected},{'a','middle','b'})
        self.assertLessEqual(unary_score(middle,self.cfg),self.cfg.min_track_score)

    def test_distinct_tightly_spaced_bodies_are_not_deleted_by_period_alone(self):
        raw=np.zeros((6,240));raw[:,60:64]=80;raw[:,67:71]=80
        model=self.model(raw)
        a=self.track(60,'a',raw);b=self.track(67,'b',raw)
        proof=model.pair_evidence(a,b)
        self.assertEqual(proof['state'],'close_distinct_bodies')
        self.assertEqual(transition_score(a,b,model.local,self.cfg)['short_gap_penalty'],0.)

    def test_signal_scale_is_independent_of_dense_candidate_list(self):
        raw=np.zeros((6,400))
        for y in range(20,380,20):raw[:,y:y+9]=80
        model=self.model(raw)
        self.assertAlmostEqual(model.local.at(200),20,delta=1)
        dense=[self.track(y,f't{y}',raw) for y in range(100,151,2)]
        optimize_vertical_sequence(dense,model.local,self.cfg)
        self.assertAlmostEqual(model.local.at(200),20,delta=1)

    def test_opposite_edges_and_roi_offset_share_one_identity(self):
        raw=np.zeros((6,200));raw[:,40:53]=80
        model=self.model(raw,offset=100)
        a=self.track(140,'upper',raw,offset=100)
        b=self.track(152,'lower',raw,offset=100,polarity='negative')
        before=dict(b.strip_positions)
        kept,rejected,_=deduplicate_body_identities([a,b],'positive',model)
        self.assertEqual(len(kept),1)
        self.assertEqual(b.strip_positions,before)
        self.assertTrue(all(p['upper']>=100 for row in model.bands for p in row))

    def test_unknown_does_not_reduce_original_gradient_score(self):
        raw=np.zeros((6,200));t=self.track(80,'unknown',raw,score=.9)
        self.assertAlmostEqual(unary_score(t,self.cfg),unary_score(t,replace(self.cfg,evidence_model=False)))

    def test_local_texture_repeat_cannot_replace_established_scale(self):
        raw=np.zeros((6,400))
        for y in range(12,380,12):raw[:,y:y+5]=80
        model=self.model(raw,period=20)
        self.assertTrue(np.all(model.local.values>=16))
        self.assertTrue(np.all(model.local.values<=24))

    def test_local_body_confirmation_is_not_terminal_texture_exemption(self):
        raw=np.zeros((6,200));raw[:,40:53]=80
        t=self.track(40,'texture',raw)
        self.assertTrue(t.local_evidence['paper_body']['verified'])
        feature=identity_test_support.VerifierDecisionTests.Feature('REJECT_SECONDARY_TEXTURE')
        cfg=replace(self.cfg,strip_sequence_enabled=True,strip_sequence_raw_body_safeguard_enabled=True,
            final_track_verifier_enabled=True,final_track_verifier_shadow_mode=False)
        kept,rejected,_=apply_final_track_verifier([t],{'texture':feature},cfg)
        self.assertEqual(kept,[]);self.assertEqual(rejected,[t])

    def test_ordinary_native_sequence_is_preserved_by_window_reconciliation(self):
        from paper_counter.stripe_evidence import reconcile_signal_regions
        raw=np.zeros((6,240))
        for y in range(20,220,20):raw[:,y:y+9]=80
        model=self.model(raw)
        base=[self.track(y,f't{y}',raw) for y in range(40,180,20)]
        before={t.track_id:dict(t.strip_positions) for t in base}
        kept,_,_=reconcile_signal_regions(base,[],model,self.cfg)
        self.assertEqual({t.track_id:dict(t.strip_positions) for t in kept},before)

    def test_local_gap_window_recovers_observed_band_and_preserves_anchors(self):
        from paper_counter.stripe_evidence import reconcile_signal_regions
        raw=np.zeros((6,240))
        for y in range(20,220,20):raw[:,y:y+9]=80
        model=self.model(raw)
        base=[self.track(y,f't{y}',raw) for y in (40,60,100,120)]
        middle=self.track(80,'body_middle',raw,score=.35)
        kept,_,_=reconcile_signal_regions(base,[middle],model,self.cfg)
        self.assertEqual({t.track_id for t in kept},{'t40','t60','body_middle','t100','t120'})

    def test_normal_spacing_prevents_false_merge_from_overlapping_bands(self):
        raw=np.zeros((6,240));raw[:,40:62]=80
        model=self.model(raw)
        a=self.track(40,'a',raw);b=self.track(60,'b',raw)
        self.assertFalse(model.same_identity(a,b))
        self.assertNotEqual(model.pair_evidence(a,b)['state'],'same_measured_body')

    def test_monotonic_raw_transition_cannot_override_measured_absence(self):
        raw=np.zeros((6,180));raw[:,:30]=180;raw[:,30:70]=np.linspace(25,1,40)
        model=self.model(raw,period=35)
        t=self.track(69,'tail',raw,period=35,polarity='negative')
        self.assertEqual(t.local_evidence['paper_body']['state'],'absent')
        self.assertTrue(model.transition_evidence(t)['verified'])
        model.protect_observed_transitions([t])
        self.assertEqual(t.local_evidence['paper_body']['state'],'absent')

    def test_broad_bounded_body_conflict_is_unknown_not_absence(self):
        narrow=np.zeros((6,180));narrow[:,:30]=180;narrow[:,30:70]=np.linspace(25,1,40)
        model=self.model(narrow,period=35)
        primary=np.zeros((6,180));primary[:,56:70]=80;model.primary_raw=primary
        t=self.track(69,'actual_native_body',narrow,period=35,polarity='negative')
        self.assertEqual(t.local_evidence['paper_body']['state'],'absent')
        model.protect_observed_transitions([t])
        self.assertEqual(t.local_evidence['paper_body']['state'],'unknown')
        self.assertFalse(t.local_evidence['paper_body']['verified'])

    def test_disjoint_fragments_need_matching_two_measured_boundaries(self):
        raw=np.zeros((6,200));raw[:,40:53]=80;raw[:,60:73]=80
        _,diag=stripe_proposals(raw,self.strips,[np.zeros(200) for _ in range(6)],0,20,'positive',self.cfg)
        model=SignalGapModel(raw,diag,0,20,'positive',self.cfg,self.strips,[np.zeros(200) for _ in range(6)])
        left=self.track(40,'left',raw,strips=[0,1,2])
        right=self.track(40,'right',raw,strips=[3,4,5])
        other=self.track(60,'other',raw,strips=[3,4,5])
        self.assertTrue(model.same_identity(left,right))
        self.assertFalse(model.same_identity(left,other))


if __name__=='__main__':unittest.main()
