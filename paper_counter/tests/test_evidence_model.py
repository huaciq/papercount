"""Behavioral coverage for measured bodies, direction, local anchors and period."""
import itertools
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np

from paper_counter import CounterConfig
from paper_counter.body_evidence import measure_body,attach_body_evidence
from paper_counter.models import StripPeakCandidate
from paper_counter.tracking import make_track,classify_tracks,associate_strip_peaks
from paper_counter.stripe_evidence import stripe_proposals,fuse_body_proposals,deduplicate_body_identities,select_body_sequence,stripe_responses
from paper_counter.boundary_direction import trace_boundary_motion,predicted_shift
from paper_counter.isolated_component import recover_isolated_outer_components
from paper_counter.track_identity import conservative_identity_decision
from paper_counter.vertical_sequence import optimize_vertical_sequence,transition_score,unary_score
from paper_counter.regularity import evaluate_middle_track_competition
from paper_counter.confidence import compute_geometry_confidence
from test_vertical_sequence import track,prior


class EvidenceModelTests(unittest.TestCase):
    def setUp(self):
        self.cfg=CounterConfig(evidence_model=True,num_strips=6,min_period=5,max_period=30)

    def test_dark_real_body_width_independent_of_large_gap(self):
        line=np.zeros(180);line[80:93]=40
        proof=measure_body(line,80,'positive',38)
        self.assertEqual(proof['state'],'confirmed')
        self.assertLess(proof['width'],15)
        a,b=proof['sample_bounds']
        self.assertGreater(a,79);self.assertLess(b,94)

    def test_monotonic_background_tail_has_no_paper_band(self):
        line=np.zeros(180);line[:30]=180
        line[30:70]=np.linspace(25,1,40)
        proof=measure_body(line,69,'negative',35)
        self.assertEqual(proof['state'],'absent')

    def test_empty_background_cannot_generate_stripe_tracks(self):
        raw=np.zeros((6,180))
        proposals,_=stripe_proposals(raw,[(s*10,(s+1)*10) for s in range(6)],
            [np.zeros(180) for _ in range(6)],0,15,'positive',self.cfg)
        self.assertEqual(proposals,[])

    def test_unmeasurable_scale_and_flat_v4_image_return_zero(self):
        from paper_counter import PaperCounter
        cfg=CounterConfig.from_yaml(Path(__file__).resolve().parents[1]/'configs/paper_layers_v4.yaml')
        result=PaperCounter(cfg).process(np.zeros((180,120),np.uint8))
        self.assertEqual(result.count,0)
        self.assertEqual(stripe_proposals(np.zeros((6,180)),[],[],0,None,'positive',self.cfg)[0],[])

    def test_morphological_contrast_keeps_raw_image_and_body_decision(self):
        line=np.arange(180)*.1+30;line[80:93]+=3
        before=line.copy();smooth,bright,dark=stripe_responses(line,25)
        np.testing.assert_array_equal(line,before)
        self.assertGreater(bright[85],2)
        self.assertLess(abs(bright[30]),.2)
        self.assertTrue(np.all(dark>=0))

    def test_partial_first_body_uses_visible_opposite_edge(self):
        raw=np.zeros((6,180));raw[:,:12]=70;raw[:,60:72]=90;raw[:,120:132]=90
        proposals,_=stripe_proposals(raw,[(s*10,(s+1)*10) for s in range(6)],
            [np.zeros(180) for _ in range(6)],0,20,'positive',self.cfg)
        first=[t for t in proposals if t.center_y<20]
        self.assertTrue(first)
        self.assertTrue(any(t.boundary_polarity=='negative' for t in first))

    def test_direction_follows_tilt_instead_of_nearest_other_layer(self):
        rows=[]
        for s in range(6):
            rows.append([StripPeakCandidate(s,float(y+12*s),5,5,2,1,'strong') for y in (100,120)])
        plain=associate_strip_peaks(rows,20,self.cfg)
        oriented=associate_strip_peaks(rows,20,self.cfg,strip_centers=np.arange(6)*100.,
            orientation_fields=[np.full(512,.12) for _ in range(6)])
        complete=[t for t in oriented if t.support_count==6]
        self.assertEqual(len(complete),2)
        self.assertEqual([complete[0].strip_positions[s] for s in range(6)],[100+12*s for s in range(6)])
        self.assertFalse(any(t.support_count==6 and t.strip_positions[0]==100 and t.strip_positions[5]==160 for t in plain))

    def test_one_clear_body_anchor_is_not_vetoed_by_missing_strips(self):
        p=StripPeakCandidate(2,80.,20,20,2,2,'strong',local_prominence_reference=10)
        t=make_track([p],15,6,self.cfg)
        raw=np.zeros((6,180));raw[2,80:93]=80
        attach_body_evidence([t],raw,0,lambda y:15,'positive',self.cfg)
        strong,marginal,garbage=classify_tracks([t],self.cfg)
        self.assertEqual(strong,[t]);self.assertFalse(garbage)
        self.assertEqual(t.support_count,1)

    def test_clear_single_segment_survives_final_joint_sequence(self):
        raw=np.zeros((6,180));raw[:,20:30]=80;raw[:,60:70]=80;raw[2,40:50]=80
        ts=[]
        for name,y,strips in [('upper',20,range(6)),('local',40,[2]),('lower',60,range(6))]:
            t=make_track([StripPeakCandidate(s,float(y),20,20,2,2,'strong',local_prominence_reference=10) for s in strips],20,6,self.cfg)
            t.track_id=name;t.boundary_polarity='positive';ts.append(t)
        attach_body_evidence(ts,raw,0,lambda y:20,'positive',self.cfg)
        strong,_,_=classify_tracks(ts,self.cfg)
        class R:
            def at(self,y):return .9
        kept,rejected,_=select_body_sequence(strong,prior(20),'positive',self.cfg,R())
        self.assertEqual({t.track_id for t in kept},{'upper','local','lower'})
        self.assertFalse(rejected)

    def test_stable_profile_matches_flat_historical_parameters(self):
        root=Path(__file__).resolve().parents[1]
        default=CounterConfig.from_yaml(root/'configs/default.yaml')
        old=CounterConfig.from_yaml(root/'paper_counter/presets/e3_3.yaml')
        self.assertFalse(default.evidence_model)
        self.assertEqual(default.to_dict(),old.to_dict())

    def test_full_coverage_body_does_not_receive_local_visibility_bonus(self):
        t=track(40);t.support_ratio=1.;t.total_score=.5;t.candidate_class='strong'
        t.local_evidence['paper_body']={'verified':True,'local_anchor':True,
            'samples':[{'state':'confirmed','snr':100.}]}
        ordinary=replace(self.cfg,evidence_model=False)
        self.assertAlmostEqual(unary_score(t,self.cfg),unary_score(t,ordinary))

    def test_period_accumulates_measured_weak_stripes(self):
        raw=np.full((6,240),20.)
        for y in range(12,225,12):raw[:,y:y+5]+=1.8
        proposals,debug=stripe_proposals(raw,[(s*10,(s+1)*10) for s in range(6)],
            [np.zeros(240) for _ in range(6)],0,12,'positive',self.cfg)
        weak=[t for t in proposals if t.local_evidence.get('period_assisted')]
        self.assertTrue(weak)
        self.assertTrue(any(row['periodic_weak_search'] for row in debug['strips']))
        self.assertTrue(all(t.peak_candidates for t in weak))

    def test_existing_outer_representative_receives_component_proof(self):
        gray=np.zeros((180,120),np.uint8);gray[25:38]=80;gray[100:113]=90
        strips=[(s*20,(s+1)*20) for s in range(6)]
        peaks=[StripPeakCandidate(s,25.,10,10,2,1,'strong') for s in range(6)]
        t=make_track(peaks,20,6,self.cfg);t.track_id='existing'
        tracks,replaced,debug=recover_isolated_outer_components(gray,strips,[t],20,0,'positive',self.cfg)
        self.assertTrue(t.local_evidence['isolated_raw_component']['verified'])
        self.assertIn(t,tracks);self.assertNotIn(t,replaced)

    def test_correlated_background_score_is_not_two_independent_votes(self):
        family,decision=conservative_identity_decision(['background','paper_support'],True,-5,self.cfg)
        self.assertEqual(decision,'AMBIGUOUS_KEEP')

    def test_unknown_windows_do_not_outvote_measured_background(self):
        raw=np.zeros((6,180));raw[:3,:30]=180;raw[:3,30:70]=np.linspace(25,1,40)
        peaks=[StripPeakCandidate(s,69.,10,10,2,1,'strong') for s in range(6)]
        t=make_track(peaks,35,6,self.cfg)
        attach_body_evidence([t],raw,0,lambda y:35,'negative',self.cfg)
        self.assertEqual(t.local_evidence['paper_body']['state'],'absent')

    def test_raw_body_coordinates_include_roi_offset(self):
        raw=np.zeros((6,180));raw[:,80:93]=80
        t=make_track([StripPeakCandidate(s,180.,10,10,2,1,'strong') for s in range(6)],20,6,self.cfg)
        attach_body_evidence([t],raw,100,lambda y:20,'positive',self.cfg)
        proof=t.local_evidence['paper_body']['samples'][0]
        self.assertGreater(proof['pair_y'],185)
        self.assertTrue(all(179<=v<=194 for v in proof['sample_bounds']))

    def test_same_paper_opposite_edges_count_once_but_adjacent_body_survives(self):
        raw=np.zeros((6,180));raw[:,40:53]=80;raw[:,65:78]=90
        def item(y,polarity,id):
            t=make_track([StripPeakCandidate(s,y,10,10,2,1,'strong') for s in range(6)],25,6,self.cfg)
            t.track_id=id;t.boundary_polarity=polarity
            attach_body_evidence([t],raw,0,lambda y:25,polarity,self.cfg)
            return t
        upper=item(40.,'positive','first_upper');lower=item(52.,'negative','first_lower')
        next_sheet=item(65.,'positive','second')
        kept,rejected,diagnostic=deduplicate_body_identities([upper,lower,next_sheet],'positive')
        self.assertEqual(len(kept),2);self.assertEqual(len(rejected),1)
        self.assertIn(next_sheet,kept)

    def test_body_proposal_competes_in_same_sequence_without_faking_polarity(self):
        raw=np.zeros((6,180));raw[:,40:53]=80;raw[:,65:78]=90
        proposals,_=stripe_proposals(raw,[(s*10,(s+1)*10) for s in range(6)],
            [np.zeros(180) for _ in range(6)],0,25,'positive',self.cfg)
        merged,_,_=deduplicate_body_identities(proposals,'positive')
        before={t.track_id:(t.boundary_polarity,dict(t.strip_positions)) for t in merged}
        kept,_,_=select_body_sequence(merged,prior(25),'positive',self.cfg)
        self.assertEqual(len(kept),2)
        for t in kept:self.assertEqual(before[t.track_id],(t.boundary_polarity,t.strip_positions))

    def test_raw_direction_trace_follows_local_wave_with_roi_offset(self):
        width=600;height=220;xx=np.arange(width)
        wave=np.rint(14*np.sin(xx*2*np.pi/300)).astype(int)
        gray=np.zeros((height,width),np.uint8)
        for x in range(width):
            for y in (45,80,115,150):gray[y+wave[x]:y+wave[x]+20,x]=180
        strips=[(s*100,(s+1)*100) for s in range(6)]
        fields=trace_boundary_motion(gray,strips,[np.zeros(height) for _ in range(6)],self.cfg)
        centers=np.array([(a+b-1)/2 for a,b in strips])
        y=80+wave[49]+100
        shift=predicted_shift(fields,centers,0,1,y,100)
        expected=float(wave[149]-wave[49])
        self.assertAlmostEqual(shift,expected,delta=2)

    def test_retired_options_load_for_compatibility_but_are_not_effective_parameters(self):
        cfg=replace(self.cfg,pass2_exception_only=False,identity_low_template_similarity=.9,
                    raw_mask_brightness_k=1.,raw_mask_close_ratio=10.,raw_mask_min_active_ratio=.1,
                    mixed_spacing_prior_weight=2.,regular_gap_multiple_tolerance=.1)
        self.assertNotIn('pass2_exception_only',cfg.to_dict())
        self.assertNotIn('identity_low_template_similarity',cfg.to_dict())
        self.assertEqual(cfg.to_dict(),self.cfg.to_dict())

    def test_config_profile_is_small_but_serializes_effective_settings(self):
        cfg=CounterConfig.from_yaml(Path(__file__).resolve().parents[1]/'configs/paper_layers_v4.yaml')
        self.assertTrue(cfg.evidence_model)
        clone=CounterConfig(**cfg.to_dict())
        self.assertEqual(cfg.to_dict(),clone.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.yaml';path.write_text('profile: paper_layers_v4\nsignal:\n  nonsense: 1\n',encoding='utf-8')
            with self.assertRaises(ValueError):CounterConfig.from_yaml(path)

    def test_three_track_dp_matches_exhaustive_objective(self):
        class R:
            def at(self,y):return .9
        cfg=CounterConfig.from_yaml(Path(__file__).resolve().parents[1]/'configs/paper_layers_v4.yaml')
        positions=[100,119,138,162,194,206,225]
        scores=[.650279053,.776156484,.638710762,.883104727,.628660725,.690194535,.713593603]
        tracks=[track(y,score) for y,score in zip(positions,scores)]
        for i,t in enumerate(tracks):t.track_id=f't{i}';t.candidate_class='strong'
        local=prior(25);r=R()
        def objective(indices):
            edges=[transition_score(tracks[a],tracks[b],local,cfg,r) for a,b in zip(indices,indices[1:])]
            if any(e is None for e in edges):return -np.inf
            return (sum(unary_score(tracks[i],cfg) for i in indices)+sum(e['pair_score'] for e in edges)
                    -sum(max(0,evaluate_middle_track_competition(tracks[a],tracks[b],tracks[c],25,cfg,.9)['competition_score'])
                         for a,b,c in zip(indices,indices[1:],indices[2:])))
        paths=[p for n in range(1,8) for p in itertools.combinations(range(7),n)]
        best=max(paths,key=objective)
        result=optimize_vertical_sequence(tracks,local,cfg,r)
        actual=tuple(int(t.track_id[1:]) for t in result.selected)
        self.assertAlmostEqual(objective(actual),objective(best))

    def test_rescue_ratio_uses_unique_surviving_identity(self):
        from paper_counter.vertical_sequence import SequenceResult
        ts=[track(100+25*i) for i in range(4)]
        for i,t in enumerate(ts):t.track_id=str(i)
        sequence=SequenceResult(selected=ts,rescued=[ts[0],ts[0],track(800)])
        _,parts=compute_geometry_confidence(ts,sequence,self.cfg)
        self.assertEqual(parts['rescued_ratio'],.25)
