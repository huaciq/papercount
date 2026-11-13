"""压实层纹可积累；随机纹理、黑区和周期预测不能凭空计数。"""
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from paper_counter import CounterConfig,PaperCounter
from paper_counter.compressed_layers import accumulated_profiles,measure_weak_layers,compressed_proposals,compressed_identity_context,prepare_compressed_candidates,recheck_compressed_identity,fragment_relation
from paper_counter.boundary_direction import DirectionFields
from paper_counter.models import StripPeakCandidate
from paper_counter.tracking import make_track


class CompressedLayerTests(unittest.TestCase):
    def setUp(self):
        self.cfg=CounterConfig(evidence_model=True,num_strips=6,min_period=5,max_period=40)

    def test_collective_weak_repeat_without_bounded_bright_body(self):
        y=np.arange(360)
        groups=np.tile(140+1.1*np.sin(2*np.pi*y/20),(3,1))
        accepted,_=measure_weak_layers(groups,20,'positive',self.cfg)
        self.assertGreater(len(accepted),10)
        self.assertTrue(all(p['horizontal_votes']==3 for p in accepted))
        import json
        json.dumps(accepted, allow_nan=False)
        # 候选只能落在实测上升斜率的峰附近，而非任意补入周期位置。
        self.assertTrue(all(min(p['y']%20,20-p['y']%20)<=1 for p in accepted))

    def test_period_alone_cannot_create_layers_on_constant_image(self):
        accepted,_=measure_weak_layers(np.full((3,300),140.),20,'positive',self.cfg)
        self.assertEqual(accepted,[])

    def test_one_patterned_window_cannot_outvote_two_unobserved_windows(self):
        y=np.arange(360);groups=np.full((3,360),140.)
        groups[0]+=2*np.sin(2*np.pi*y/20)
        accepted,_=measure_weak_layers(groups,20,'positive',self.cfg)
        self.assertEqual(accepted,[])

    def test_dark_break_is_not_compressed_paper(self):
        y=np.arange(360);groups=np.tile(1+1*np.sin(2*np.pi*y/20),(3,1))
        accepted,_=measure_weak_layers(groups,20,'positive',self.cfg)
        self.assertEqual(accepted,[])

    def test_faint_one_pixel_seams_are_accumulated_without_black_gaps(self):
        line=np.full(360,140.);line[np.arange(360)%20==0]-=3.
        accepted,_=measure_weak_layers(np.tile(line,(3,1)),20,'positive',self.cfg)
        self.assertGreater(len(accepted),10)

    def test_normal_dark_seams_do_not_activate_weak_texture_identity_model(self):
        y=np.arange(360);line=140+1.1*np.sin(2*np.pi*(y-5)/20)
        line[y%20==0]=20.
        accepted,_=measure_weak_layers(np.tile(line,(3,1)),20,'positive',self.cfg)
        self.assertEqual(accepted,[])

    def test_one_faint_interface_among_normal_seams_is_not_lost_by_region_gate(self):
        line=np.full(360,140.);line[np.arange(360)%20==0]-=120.;line[160]=137.
        accepted,_=measure_weak_layers(np.tile(line,(3,1)),20,'positive',self.cfg)
        self.assertTrue(any(abs(p['y']-161)<=1 for p in accepted))

    def test_partial_fragment_cannot_be_counted_again_at_other_columns(self):
        def track(name,positions):
            t=make_track([StripPeakCandidate(s,y,2,2,1,1,'strong') for s,y in positions.items()],20,6,self.cfg)
            t.track_id=name;t.boundary_polarity='positive';return t
        old=track('old',{0:40.,1:42.});fragment=track('new',{2:44.,3:46.})
        motion={(a,b):np.full(100,2.*(b-a)) for a in range(6) for b in range(6) if a!=b}
        fields=DirectionFields([np.zeros(100) for _ in range(6)],motion,{k:np.ones(100,bool) for k in motion})
        self.assertEqual(fragment_relation(fragment,old,[(10*s,10*(s+1)) for s in range(6)],fields,0,self.cfg),'possible_existing_fragment')
        independent=track('other_paper',{2:64.,3:66.})
        self.assertIsNone(fragment_relation(independent,old,[(10*s,10*(s+1)) for s in range(6)],fields,0,self.cfg))

    def test_mixed_layer_fragment_is_identity_conflict_not_another_sheet(self):
        def track(name,positions):
            t=make_track([StripPeakCandidate(s,y,2,2,1,1,'strong') for s,y in positions.items()],20,6,self.cfg)
            t.track_id=name;t.boundary_polarity='positive';return t
        old=track('old',{0:40.,1:40.});mixed=track('mixed',{0:40.,1:60.})
        self.assertEqual(fragment_relation(mixed,old),'existing_identity_conflict')

    def test_opposite_boundary_of_measured_same_body_is_not_another_sheet(self):
        def track(name,y,polarity):
            t=make_track([StripPeakCandidate(s,y,2,2,1,1,'strong') for s in range(6)],20,6,self.cfg)
            t.track_id=name;t.boundary_polarity=polarity;return t
        upper=track('upper',60.,'positive');lower=track('lower',70.,'negative')
        lower.local_evidence['paper_body']={'samples':[{'strip':s,'state':'confirmed','pair_y':60.,'width':10.} for s in range(6)]}
        self.assertEqual(fragment_relation(upper,lower,cfg=self.cfg),'already_observed_opposite_boundary')

    def test_random_unrelated_windows_do_not_create_repeat(self):
        rng=np.random.default_rng(43)
        groups=140+rng.normal(0,2,(3,600))
        accepted,_=measure_weak_layers(groups,20,'positive',self.cfg)
        self.assertEqual(accepted,[])

    def test_half_period_alias_is_reported_not_counted_as_trusted_layers(self):
        y=np.arange(360);groups=np.tile(140+1.1*np.sin(2*np.pi*y/10),(3,1))
        accepted,debug=measure_weak_layers(groups,20,'positive',self.cfg)
        self.assertEqual(accepted,[])
        self.assertTrue(any(p['reason']=='ambiguous_half_period' for p in debug['rejected']))

    def test_accumulation_retains_repeat_under_independent_pixel_noise(self):
        rng=np.random.default_rng(2);h,w=360,240;y=np.arange(h)[:,None]
        image=(140+1.8*np.sin(2*np.pi*y/20)+rng.normal(0,1,(h,w))).astype(np.float32)
        profiles,_=accumulated_profiles(image,[(0,w)],20,self.cfg)
        accepted,_=measure_weak_layers(profiles[0],20,'positive',self.cfg)
        self.assertGreater(len(accepted),10)

    def test_faint_seams_survive_pixel_noise_with_known_neighbor_direction(self):
        rng=np.random.default_rng(17);h,w=360,240
        line=np.full(h,140.);line[np.arange(h)%20==0]-=2.
        image=(line[:,None]+rng.normal(0,4,(h,w))).astype(np.float32)
        # 这里隔离检验累积能力：方向由邻近清晰边界提供，已知为水平。
        fields=DirectionFields([np.zeros(h),np.zeros(h)],{(0,1):np.zeros(h)},
            {(0,1):np.ones(h,bool)})
        profiles,_=accumulated_profiles(image,[(0,w//2),(w//2,w)],20,self.cfg,fields)
        accepted,_=measure_weak_layers(profiles[0],20,'positive',self.cfg)
        self.assertGreaterEqual(len(accepted),2)
        self.assertTrue(all(min(abs(p['y']%20-1),abs(p['y']%20-21))<=3 for p in accepted))

    def test_direction_accumulation_follows_measured_slant(self):
        h,w=360,180;y=np.arange(h)[:,None];x=np.arange(w)[None,:]
        image=(140+3*np.sin(2*np.pi*(y-.15*(x-89.5))/20)).astype(np.float32)
        profiles,_=accumulated_profiles(image,[(0,w)],20,self.cfg)
        self.assertLess(np.max(np.std(profiles[0,:,50:-50],axis=0)),.8)
        self.assertGreater(np.std(profiles[0].mean(axis=0)[50:-50]),1.5)

    def test_proposals_keep_actual_coordinates_and_do_not_claim_body_identity(self):
        h,w=360,180;y=np.arange(h)[:,None]
        image=np.broadcast_to(140+1.1*np.sin(2*np.pi*y/20),(h,w)).astype(np.float32)
        strips=[(30*i,30*(i+1)) for i in range(6)]
        tracks,debug=compressed_proposals(image,strips,[np.zeros(h) for _ in strips],100,20,'positive',self.cfg)
        self.assertGreater(len(tracks),10)
        self.assertFalse(debug['period_creates_positions'])
        self.assertTrue(all(t.center_y>=100 and len(t.peak_candidates)==6 for t in tracks))
        self.assertTrue(all('paper_body' not in t.local_evidence for t in tracks))
        self.assertTrue(all(t.candidate_class=='marginal' for t in tracks))

    def test_opt_in_configuration_runs_through_mature_hypothesis(self):
        cfg=CounterConfig.from_yaml(Path(__file__).resolve().parents[1]/'configs/tracks_variants/compressed_layers.yaml')
        y=np.arange(300)[:,None]
        image=np.broadcast_to(140+3*np.sin(2*np.pi*y/20),(300,240)).astype(np.uint8)
        result=PaperCounter(cfg).process(image)
        self.assertTrue(result.debug_info['compressed_layers']['enabled'])
        self.assertEqual(result.count,len(result.paper_edge_tracks))

    def test_shared_observations_compare_tilted_partial_layer_without_center_bias(self):
        def track(name,positions):
            t=make_track([StripPeakCandidate(s,y,2,2,1,1,'strong') for s,y in positions.items()],20,6,self.cfg)
            t.track_id=name;t.boundary_polarity='positive';return t
        a=track('a',{0:20.,1:30.,2:40.,3:50.,4:60.,5:70.})
        b=track('b',{0:60.,1:70.,2:80.,3:90.,4:100.,5:110.})
        partial=track('weak',{0:40.,1:50.})
        self.assertEqual(a.center_y,partial.center_y)  # 中位坐标相等不表示同一张纸。
        partial.local_evidence={'allocated_signal_gap':{'left':'a','right':'b'},
            'compressed_layers':{'verified':True,'samples':[{'strip':s,'verified':True,
                'period':20,'local_template_correlation':.95} for s in (0,1)]}}
        proof=compressed_identity_context(partial,[a,partial,b],self.cfg)
        self.assertTrue(proof['verified'])
        partial.strip_positions[1]=31.
        self.assertFalse(compressed_identity_context(partial,[a,partial,b],self.cfg)['verified'])

    def test_weak_proof_is_propagated_to_existing_candidate_without_double_count(self):
        def track(name,y):
            t=make_track([StripPeakCandidate(s,y,2,2,1,1,'strong') for s in range(6)],20,6,self.cfg)
            t.track_id=name;t.boundary_polarity='positive';return t
        a,b,c=track('a',40.),track('b',60.),track('c',80.)
        donor=track('compressed',60.)
        donor.local_evidence['compressed_layers']={'verified':True,'samples':[
            {'strip':s,'y':60.,'verified':True,'period':20.,'local_template_correlation':.95} for s in range(6)]}
        kept,_=prepare_compressed_candidates([a,b,c],[donor],self.cfg)
        self.assertEqual(kept,[])
        feature=SimpleNamespace(evidence_families=['template','negative_shape'],
            proposed_decision='REJECT_SECONDARY_TEXTURE',false_family='SECONDARY_TEXTURE',
            needs_final_verification=True,identity_z_score=-1.)
        record=recheck_compressed_identity(b,feature,[a,b,c],self.cfg)
        self.assertTrue(record['used_local_identity'])
        self.assertNotEqual(feature.proposed_decision,'REJECT_SECONDARY_TEXTURE')
        feature.evidence_families=['background','template'];feature.proposed_decision='REJECT_BACKGROUND'
        recheck_compressed_identity(b,feature,[a,b,c],self.cfg)
        self.assertEqual(feature.proposed_decision,'REJECT_BACKGROUND')
        feature.evidence_families=['template'];feature.proposed_decision='REJECT_MEASURED_NON_PAPER'
        recheck_compressed_identity(b,feature,[a,b,c],self.cfg)
        self.assertEqual(feature.proposed_decision,'REJECT_MEASURED_NON_PAPER')


if __name__=='__main__':unittest.main()
