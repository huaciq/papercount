"""One-sided endpoints, background negatives and ordinary terminal parity."""
from types import SimpleNamespace
from dataclasses import replace
import unittest
import numpy as np
from paper_counter import CounterConfig
from paper_counter.models import StripPeakCandidate
from paper_counter.tracking import make_track
from paper_counter.raw_mask import RawStackMask, StackState
from paper_counter.track_identity import endpoint_identity_v2, apply_final_track_verifier


def fixture(y=1, bright=True, side='top'):
    cfg = CounterConfig(num_strips=6, endpoint_identity_v2_enabled=True,
                        final_track_verifier_shadow_mode=False)
    raw = np.zeros((6, 200), np.float32)
    raw[:, 0:80] = 100 if bright else 0
    raw[:, 100:160] = 100
    if side == 'bottom':
        raw = raw[:, ::-1].copy()
    states = np.where(np.median(raw,axis=0)>0, StackState.CORE, StackState.INVALID).astype(np.uint8)
    mask = RawStackMask(states,states!=StackState.INVALID,{'row_median':np.median(raw,axis=0)}, {})
    peaks = [StripPeakCandidate(s,float(y),10,10,1,2,'strong',local_prominence_reference=5) for s in range(6)]
    track = make_track(peaks,20,6,cfg)
    track.track_id = 'observed'
    track.candidate_class = 'strong'
    feature = SimpleNamespace(local_spacing=20., evidence_families=['background','paper_support'],
        false_family='BACKGROUND_FALSE_EDGE',proposed_decision='REJECT_BACKGROUND',
        identity_score=.2,identity_z_score=-5,raw_template_similarity=.1,
        normalized_template_similarity=.1,positive_profile_corr=.1,negative_template_similarity=.1,
        template_outlier_z=6.)
    feature.to_dict = lambda: {k:v for k,v in vars(feature).items() if k!='to_dict'}
    return cfg,raw,mask,track,feature


class EndpointIdentityTests(unittest.TestCase):
    def test_top_truncated_body_survives_old_background_veto(self):
        cfg,raw,mask,track,feature = fixture()
        feature.endpoint_identity = endpoint_identity_v2(track,feature,raw,0,mask,cfg)
        kept,rejected,debug = apply_final_track_verifier([track],{track.track_id:feature},cfg)
        self.assertEqual((len(kept),len(rejected)),(1,0))
        self.assertEqual(debug['legacy_terminal']['count'],0)
        self.assertFalse(feature.endpoint_identity['outside_context_required'])

    def test_reflected_bottom_uses_inward_upper_body(self):
        cfg,raw,mask,track,feature = fixture(y=198,side='bottom')
        result = endpoint_identity_v2(track,feature,raw,0,mask,cfg)
        self.assertEqual((result['side'],result['inward_direction'],result['decision']),
                         ('bottom',-1,'KEEP_ENDPOINT_V2'))

    def test_coherent_black_background_does_not_pass(self):
        cfg,raw,mask,track,feature = fixture(bright=False)
        result = endpoint_identity_v2(track,feature,raw,0,mask,cfg)
        self.assertEqual(result['decision'],'REJECT_ENDPOINT_V2')
        self.assertFalse(result['gates']['inward_raw_body'])
        self.assertFalse(result['gates']['foreground_support'])

    def test_no_opposite_peak_or_template_needed(self):
        cfg,raw,mask,track,feature = fixture()
        result = endpoint_identity_v2(track,feature,raw,0,mask,cfg)
        self.assertFalse(result['opposite_edge_required'])
        self.assertFalse(result['template_required'])
        self.assertEqual(result['decision'],'KEEP_ENDPOINT_V2')

    def test_roi_offset_invariance(self):
        cfg,raw,mask,track,feature = fixture()
        original = endpoint_identity_v2(track,feature,raw,0,mask,cfg)
        track.center_y += 1000
        track.strip_positions = {s:y+1000 for s,y in track.strip_positions.items()}
        mask.offset = 1000
        shifted = endpoint_identity_v2(track,feature,raw,1000,mask,cfg)
        self.assertEqual(original['decision'],shifted['decision'])
        self.assertEqual(original['inward_raw_body_support'],shifted['inward_raw_body_support'])

    def test_single_strip_cannot_claim_endpoint(self):
        cfg,raw,mask,track,feature = fixture()
        track.strip_positions = {0:1.}
        result = endpoint_identity_v2(track,feature,raw,0,mask,cfg)
        self.assertFalse(result['gates']['multiple_observed_strips'])
        self.assertEqual(result['decision'],'REJECT_ENDPOINT_V2')

    def test_discontinuous_geometry_cannot_pass_bright_body(self):
        cfg,raw,mask,track,feature = fixture()
        track.smoothness_score = .1
        result = endpoint_identity_v2(track,feature,raw,0,mask,cfg)
        self.assertEqual(result['decision'],'REJECT_ENDPOINT_V2')

    def test_internal_terminal_rejection_identical(self):
        cfg,raw,mask,track,feature = fixture(y=110)
        feature.endpoint_identity = endpoint_identity_v2(track,feature,raw,0,mask,cfg)
        self.assertFalse(feature.endpoint_identity['is_endpoint'])
        new = apply_final_track_verifier([track],{track.track_id:feature},cfg)
        old = apply_final_track_verifier([track],{track.track_id:feature},replace(cfg,endpoint_identity_v2_enabled=False))
        self.assertEqual([t.track_id for t in new[1]],[t.track_id for t in old[1]])
        self.assertTrue(new[2]['endpoint_identity_v2']['internal_decisions_unchanged'])

    def test_shadow_preserves_original_observations(self):
        cfg,raw,mask,track,feature = fixture(bright=False)
        feature.endpoint_identity = endpoint_identity_v2(track,feature,raw,0,mask,cfg)
        kept,rejected,_ = apply_final_track_verifier([track],{track.track_id:feature},replace(cfg,final_track_verifier_shadow_mode=True))
        self.assertEqual((len(kept),len(rejected)),(1,0))

    def test_roi_band_is_spacing_based(self):
        cfg,raw,mask,track,feature = fixture(y=21)
        result = endpoint_identity_v2(track,feature,raw,0,mask,cfg)
        self.assertFalse(result['is_endpoint'])
        self.assertTrue(endpoint_identity_v2(track,feature,raw,0,mask,replace(cfg,endpoint_roi_band_ratio=1.1))['is_endpoint'])

    def test_background_is_a_penalty_not_an_unconditional_veto(self):
        cfg,raw,mask,track,feature = fixture()
        mild = endpoint_identity_v2(track,feature,raw,0,mask,cfg)
        severe = endpoint_identity_v2(track,feature,raw,0,mask,replace(cfg,endpoint_background_penalty=1.0))
        self.assertEqual(mild['decision'],'KEEP_ENDPOINT_V2')
        self.assertEqual(severe['decision'],'REJECT_ENDPOINT_V2')
        self.assertFalse(mild['background_is_absolute_veto'])
