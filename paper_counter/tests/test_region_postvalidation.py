import numpy as np
from paper_counter import CounterConfig, PaperCounter

from paper_counter.region_postvalidation import (
    check_physical_constraints,
    validate_box_sequence,
)


def _box(number, top, bottom, inferred=False):
    return {
        'id': f'r{number}', 'number': number,
        'detection_xyxy': [0.0, float(top), 225.0, float(bottom)],
        'aligned_xyxy': [0.0, float(top), 225.0, float(bottom)],
        'score': 1.0, 'band_ids': [],
        'thickness_inferred': inferred,
    }


def test_thickness_outlier_is_retained_for_review_instead_of_deleted():
    boxes = [_box(i, i * 20, i * 20 + (8 if i == 2 else 20), inferred=i == 2)
             for i in range(6)]
    observation = {'raw_signals': np.ones((3, 130), dtype=np.float32) * 100.0}
    corrected, debug = validate_box_sequence(boxes, [], observation, {
        'region_detection': {'three_strip_observation': observation}
    })
    assert len(corrected) == len(boxes)
    assert debug['removed_boxes'] == []
    assert 2 in debug['review_boxes']


def test_overlap_is_clipped_and_reported_as_a_constraint_repair():
    boxes = [_box(1, 0, 25), _box(2, 20, 45)]
    corrected, debug = validate_box_sequence(boxes, [], {}, {})
    assert corrected[0]['detection_xyxy'][3] == corrected[1]['detection_xyxy'][1]
    assert debug['corrections_applied'][0]['type'] == 'clip_overlap'
    assert check_physical_constraints(corrected)['passed']


def test_regions_expose_stage_counts_and_real_postvalidation():
    image = np.full((420, 225), 15, dtype=np.uint8)
    for center in range(45, 390, 30):
        image[max(0, center - 4):center + 5] = 180
    result = PaperCounter(CounterConfig(algorithm='regions')).process(image)
    stages = result.debug_info['stage_counts']
    assert stages['final_count'] == result.count
    assert result.debug_info['post_validation']['enabled'] is True
    assert result.debug_info['post_validation'].get('error') is None
    assert result.debug_info['final_filter']['enabled'] is False


def test_edge_background_candidate_is_removed_only_with_missing_body_support():
    edge = _box(1, 0, 9, inferred=True)
    edge['band_ids'] = [1]
    edge['score'] = 4.0
    edge['evidence'] = {'joint_resegmentation': {
        'source_unit_ids': [], 'top_gray_support': 0.0,
        'bottom_gray_support': 0.3,
    }}
    normal = _box(2, 20, 40)
    normal2 = _box(3, 50, 70)
    boxes, debug = validate_box_sequence(
        [edge, normal, normal2], [], {'raw_signals': np.ones((3, 80), dtype=np.float32)},
        {'region_detection': {'three_strip_observation': {}}})
    assert len(boxes) == 2
    assert debug['removed_boxes'] == [1]
