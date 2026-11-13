import copy
import unittest
from paper_counter.stage_evaluation import validate_ground_truth,evaluate_debug,match_tracks

def fixture():
    entry = {'image_sha256':'a'*64,'rotation_ccw_degrees':90,
        'count':{'value':1,'status':'verified'},'track_annotation_status':'verified',
        'track_completeness':'partial','tracks':[{'id':'paper1','status':'verified','endpoint':'top','source':'synthetic geometry',
            'tolerance_px':2,'boundaries':[{'polarity':'positive','points':[[0,2],[30,2]]}]}],
        'background_regions':[{'id':'black','status':'verified','source':'synthetic background','bbox':[0,80,30,100]}]}
    good = {'track_id':'a','strip_positions':{'0':2.,'1':2.,'2':2.}}
    false = {'track_id':'b','strip_positions':{'0':90.,'1':90.,'2':90.}}
    debug = {'gradient_polarity':'positive','rotation_ccw_degrees':90,
        'roi':{'x1':0},'strip_bounds_roi_x':[[0,10],[10,20],[20,30]],'count':1,
        'evaluation_stages':{'pre_verifier':[good,false],'final':[good]}}
    return entry,debug

class StageEvaluationTests(unittest.TestCase):
    def test_verified_endpoint_and_background_attribution(self):
        entry,debug=fixture()
        result=evaluate_debug(debug,entry,'a'*64)
        self.assertEqual(result['stages']['final']['endpoint_recall'],1.)
        self.assertEqual(result['stages']['pre_verifier']['background_fp'],1)
        self.assertEqual(result['stages']['final']['background_fp'],0)
        self.assertEqual(result['count_error'],0)
        self.assertIsNone(result['stages']['final']['precision'])

    def test_pending_gt_does_not_create_fake_zero_fp_or_recall(self):
        entry,debug=fixture()
        entry['tracks'][0]['status']='pending'
        entry['background_regions'][0]['status']='pending'
        result=evaluate_debug(debug,entry,'a'*64)['stages']['final']
        self.assertIsNone(result['endpoint_recall'])
        self.assertIsNone(result['background_fp'])

    def test_hash_and_rotation_mismatch_fail_closed(self):
        entry,debug=fixture()
        with self.assertRaises(ValueError):evaluate_debug(debug,entry,'other')
        debug['rotation_ccw_degrees']=0
        with self.assertRaises(ValueError):evaluate_debug(debug,entry,'a'*64)

    def test_one_prediction_cannot_match_two_sheets(self):
        entry,debug=fixture()
        duplicate=copy.deepcopy(entry['tracks'][0]);duplicate['id']='paper2'
        truth=entry['tracks']+[duplicate]
        self.assertEqual(len(match_tracks(truth,debug['evaluation_stages']['final'],debug)),1)

    def test_crossing_curve_not_accepted_by_center_only(self):
        entry,debug=fixture()
        debug['evaluation_stages']['final'][0]['strip_positions']={'0':0,'1':2,'2':7}
        result=evaluate_debug(debug,entry,'a'*64)
        self.assertEqual(result['stages']['final']['endpoint_recall'],0.)

    def test_same_paper_alternative_polarity_is_one_truth(self):
        entry,debug=fixture()
        entry['tracks'][0]['boundaries'].append({'polarity':'negative','points':[[0,5],[30,5]]})
        debug['evaluation_stages']['final'].append({'track_id':'lower','boundary_polarity':'negative',
            'strip_positions':{'0':5,'1':5,'2':5}})
        result=evaluate_debug(debug,entry,'a'*64)['stages']['final']
        self.assertEqual(result['matched'],1)

    def test_no_extrapolation_to_unannotated_strip(self):
        entry,debug=fixture()
        entry['tracks'][0]['boundaries'][0]['points']=[[10,2],[15,2]]
        result=evaluate_debug(debug,entry,'a'*64)['stages']['final']
        self.assertEqual(result['matched'],0)

    def test_gt_format_rejects_overlay_coordinates(self):
        entry,_=fixture()
        with self.assertRaises(ValueError):
            validate_ground_truth({'schema_version':1,'coordinate_space':'result_overlay','images':{'x':entry}})

    def test_complete_gt_cannot_contain_pending_track(self):
        entry,_=fixture();entry['track_completeness']='complete';entry['tracks'][0]['status']='pending'
        with self.assertRaises(ValueError):
            validate_ground_truth({'schema_version':1,'coordinate_space':'detection_image','images':{'x':entry}})

    def test_explicit_ignore_region_excludes_only_track_metrics(self):
        entry,debug=fixture()
        entry['ignore_regions']=[{'id':'ambiguous','status':'verified','source':'explicit annotation','bbox':[0,80,30,100]}]
        result=evaluate_debug(debug,entry,'a'*64)
        self.assertEqual(result['stages']['pre_verifier']['background_fp'],0)
        self.assertEqual(result['stages']['pre_verifier']['ignored_track_ids'],['b'])
        self.assertEqual(result['count_error'],0)
