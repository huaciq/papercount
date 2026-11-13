"""Count GT safeguards and evidence boundaries; no detector runs."""
from pathlib import Path
import importlib.util
import pytest

spec = importlib.util.spec_from_file_location('e15', Path(__file__).resolve().parents[1] / 'scripts/evaluate_e1_5.py')
e15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e15)


@pytest.mark.parametrize('text,status,count', [('40张', 'verified', 40), ('170', 'verified', 170),
    ('成像条件太差，170左右', 'unscorable', None), ('成像不好', 'unscorable', None),
    ('', 'pending', None), ('待确认40', 'pending', None), ('40或41', 'pending', None)])
def test_status_full_line_only(text, status, count):
    assert e15.classify_gt(text)[:2] == (status, count)


def test_mapping_timestamp_and_duplicates_not_serial_number():
    a = Path('Pic_2026_09_21_104944_2练功卷1 (2).bmp')
    b = Path('Pic_2026_09_21_141037_1-练功卷1-1.bmp')
    c = Path('Pic_2026_09_21_104944_2练功卷1 (1).bmp')
    records = e15.pair_table([a, b, c], ['91', '91', '90'])
    assert [r[0] for r in records] == [c, a, b]
    assert [r[1] for r in records] == ['91', '91', '90']


def test_broadcast_only_qualitative_and_cardinality_errors():
    images = [Path(f'Pic_2026_09_21_14043{i}_{i}.bmp') for i in (1, 2)]
    assert len(e15.pair_table(images, ['成像不好'])) == 2
    assert all(r[1] == '' for r in e15.pair_table(images, []))
    with pytest.raises(ValueError):
        e15.pair_table(images, ['170'])


def test_empty_middle_row_is_not_shifted():
    images = [Path(f'Pic_2026_09_21_14043{i}_{i}.bmp') for i in (1, 2, 3)]
    rows = e15.pair_table(images, ['40', '', '41'])
    assert rows[1][1:3] == ('', 2)
    assert rows[2][1:3] == ('41', 3)


def test_metrics_sign_distribution_and_no_gt_empty():
    m = e15.metric([-3, 0, 1, 2])
    assert m['n'] == 4 and m['MAE'] == 1.5 and m['signed_error'] == 0
    assert m['overcount_images'] == 2 and m['undercount_images'] == 1
    assert e15.metric([])['MAE'] is None
    assert e15.change(39, 41, 40) == 'unchanged'


def test_stack_templates_do_not_merge_different_capture_series():
    folder = '测试集/260921练功卷/'
    a = e15.metadata(folder + 'Pic_2026_09_21_104944_2练功卷1 (1).bmp')
    b = e15.metadata(folder + 'Pic_2026_09_21_141037_1-练功卷1-1.bmp')
    assert a['pile_group'] != b['pile_group']
    coated = e15.metadata('测试集/260921-覆膜铜板纸3/Pic_2026_09_21_115745_3-覆膜铜板纸3-1-1.bmp')
    assert coated['pile_group'].endswith('stack_1')


def test_historical_stages_not_fabricated():
    track = {'track_id': 'a'}
    stages, origin = e15.stages_from_saved({'pre_verifier_sequence': {'track_ids': ['a']}, 'paper_edge_tracks': [track]})
    assert list(stages) == ['pre_verifier', 'final']
    with pytest.raises(ValueError):
        e15.stages_from_saved({'pre_verifier_sequence': {'track_ids': ['b']}, 'paper_edge_tracks': [track]})


def simple_prediction(count, ids, endpoints=()):
    return {'count': count, 'pre_geometry_sha256': 'same', 'final_ids': ids, 'legacy_kept_ids': ['a'],
            'endpoints': list(endpoints), 'path': 'example.json'}


def endpoint(tid):
    return {'track_id': tid, 'center_y': 8.5, 'side': 'top', 'observed_strips': 12,
            'inward_raw_body_support': 1, 'foreground_support': 1, 'evidence_score': .8,
            'background_negative_evidence': 1, 'legacy_false_family': 'BACKGROUND_FALSE_EDGE',
            'decision': 'KEEP_ENDPOINT_V2', 'gates': {'body': True}, 'failed_gates': []}


def test_endpoint_count_worsening_is_not_background_GT():
    old = simple_prediction(1, ['a'])
    new = simple_prediction(2, ['a', 'b'], [endpoint('b')])
    result = e15.audit_endpoint_pair(old, new, {'status': 'verified', 'count': 1})[0]
    assert result['individual_count_effect'] == 'worsened'
    assert result['track_identity_status'].startswith('unknown')
    assert result['background_FP_status'].startswith('unknown')
    result = e15.audit_endpoint_pair(old, new, {'status': 'pending', 'count': None})[0]
    assert result['individual_count_effect'] == 'not_evaluable'


def test_endpoint_rejects_internal_change():
    with pytest.raises(ValueError):
        e15.audit_endpoint_pair(simple_prediction(1, ['a']), simple_prediction(2, ['a', 'internal']),
                               {'status': 'verified', 'count': 2})


def test_same_verified_cohort_and_nonverified_exclusion():
    entries = {key: {'status': status, 'count': count, 'raw_annotation': '', 'source': None, 'source_line': None,
                     **e15.metadata(key)} for key, status, count in
               [('验证集/a.bmp', 'verified', 1), ('验证集/b.bmp', 'unscorable', None),
                ('验证集/c.bmp', 'verified', 1), ('验证集/d.bmp', 'pending', None)]}
    predictions = {model: {key: simple_prediction(1, ['a']) for key in entries} for model in e15.MODELS}
    del predictions['Dual E1']['验证集/c.bmp']
    tables = e15.build_tables(entries, predictions)
    assert tables['common'] == ['验证集/a.bmp']
    assert all(m['n'] == 1 for m in tables['metrics'])
    assert len(tables['excluded']) == 3


def test_count_only_diagnosis_does_not_confirm_spatial_loss():
    p = simple_prediction(2, ['a', 'b'])
    p.update({'stage_counts': {'classified': 3, 'sequence_pass1': 2, 'pre_verifier': 2, 'final': 2},
              'stage_provenance': 'native', 'period': 40, 'median_local_period': 20,
              'period_to_local_ratio': 2, 'raw_foreground_outside_roi': False, 'roi': {},
              'active_roi_reason': 'test', 'crossing_hard_reject_count': 0,
              'sequence_suppressed_count': 1, 'sequence_suppressed_ids': ['c'], 'applied_reject_ids': []})
    rows = e15.diagnosis_rows('test', 'Baseline E1', 3, p)
    by_category = {r['category']: r for r in rows}
    assert by_category['少计']['status'] == 'confirmed'
    assert by_category['序列']['status'] == 'suspected'
    assert by_category['倍周期']['status'] == 'suspected'
    assert by_category['空间根因']['status'] == 'unknown'


def test_recovery_interval_confirms_only_count_change():
    p = simple_prediction(4, ['a', 'b', 'c', 'd'])
    p.update({'stage_counts': {'classified': 3, 'sequence_pass1': 3, 'after_recovery': 4,
                              'pre_verifier': 4, 'final': 4},
              'stage_provenance': 'native', 'period': 20, 'median_local_period': 20,
              'period_to_local_ratio': 1, 'raw_foreground_outside_roi': False, 'roi': {},
              'active_roi_reason': 'test', 'crossing_hard_reject_count': 0,
              'sequence_suppressed_count': 0, 'sequence_suppressed_ids': [], 'applied_reject_ids': []})
    rows = e15.diagnosis_rows('test', 'Dual E1', 3, p)
    row = next(r for r in rows if r['category'] == '恢复/融合区间')
    assert row['status'] == 'confirmed'
    assert row['evidence_scope'].startswith('aggregate_')
    assert row['spatial_root_cause_status'].startswith('unknown')
