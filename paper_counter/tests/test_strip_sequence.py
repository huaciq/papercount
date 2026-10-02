from dataclasses import replace
import numpy as np
import pytest
from paper_counter import CounterConfig, PaperCounter
from paper_counter.profiles import estimate_strip_orientation, make_strip_profiles_diagnostic
from paper_counter.preprocessing import signed_gradient
from paper_counter.strip_sequence import ordered_match, quality_profiles, choose_strip_spacing, attach_raw_body_confirmation


def settings(**changes):
    return replace(CounterConfig(), stack_roi_mode='foreground_envelope',
                   strip_sequence_mode='replace',
                   strip_sequence_enabled=True, strip_sequence_dual_enabled=True,
                   final_track_verifier_enabled=False, **changes)


def stack(n=18, tilt=0.0, gaps=None, saturated=False):
    image = np.zeros((800, 240), np.uint8)
    for i in range(n):
        top = (gaps[i] if gaps is not None else 55+30*i)
        for x in range(240):
            y = int(top+tilt*x)
            image[max(0,y):min(800,y+21), x] = 170
    if saturated:
        image[:, 40:200] = 255
    return image


def test_ordered_matching_preserves_missing_edge_without_crossing():
    pairs, extra = ordered_match(np.array([10, 30, 50, 70]), np.array([12, 32, 72, 110]), np.full(4, 7))
    assert pairs == [(0, 0), (1, 1), (3, 2)]
    assert extra == [3]


def test_dual_edges_count_each_bright_sheet_once():
    result = PaperCounter(settings()).process(stack())
    assert result.count == 18
    assert result.debug_info['strip_consensus']['opposite_same_sheet_suppressed'] == 18
    assert all(t.local_evidence['raw_observation_count'] > 0 for t in result.paper_edge_tracks)


def test_periodic_empty_gap_never_creates_paper():
    result = PaperCounter(settings()).process(stack(n=6, gaps=[55, 85, 115, 355, 385, 415]))
    assert result.count == 6
    assert result.debug_info['strip_consensus']['period_missing_observations'] == 0


def test_flat_raw_image_never_yields_enhancement_only_papers():
    cfg = settings(multiscale_gradient_enabled=True, gradient_scales=[0.0, 0.7, 1.4])
    assert PaperCounter(cfg).process(np.full((800, 240), 180, np.uint8)).count == 0


def test_local_quality_is_not_recomputed_from_detected_peak_count():
    image = stack(saturated=True)
    cfg = settings()
    q = quality_profiles(image, [(0, 40), (40, 80), (200, 240)], cfg)
    assert np.mean(q[0, 100:500]) > 3*np.mean(q[1, 100:500])
    assert np.mean(q[0, 100:500]) == pytest.approx(np.mean(q[2, 100:500]))


def test_clear_strips_survive_unobservable_saturated_middle():
    result = PaperCounter(settings()).process(stack(saturated=True))
    assert result.count == 18
    assert any(t.support_count < 8 for t in result.paper_edge_tracks)


def test_orientation_alignment_preserves_tilted_raw_boundaries():
    image = stack(n=10, tilt=0.28)
    cfg = settings(strip_aggregation_mode='orientation_aligned')
    fields = estimate_strip_orientation(image, cfg)
    assert np.median(fields[4][100:300]) == pytest.approx(0.28, abs=0.12)
    result = PaperCounter(cfg).process(image)
    assert result.count == 10
    assert max(t.y_range for t in result.paper_edge_tracks) > 35


def test_orientation_requires_raw_direction_not_period_lag_guess():
    with pytest.raises(ValueError, match='RAW-derived'):
        make_strip_profiles_diagnostic(signed_gradient(stack()), 'positive', settings(strip_aggregation_mode='orientation_aligned'))


def test_single_and_dual_primary_positions_agree_on_clean_paper_stack():
    image = stack()
    single = PaperCounter(replace(settings(), strip_sequence_dual_enabled=False)).process(image)
    dual = PaperCounter(settings()).process(image)
    assert single.count == dual.count == 18
    assert single.peak_positions == dual.peak_positions


def test_weak_horizontal_texture_inside_paper_is_not_an_additional_sheet():
    image = stack()
    for i in range(18):
        image[65+30*i:68+30*i] = 155
    result = PaperCounter(settings()).process(image)
    assert result.count == 18


def test_augmentation_does_not_double_existing_clean_sheet_boundaries():
    cfg = replace(settings(), strip_sequence_mode='augment', track_validate_before_claim=True)
    result = PaperCounter(cfg).process(stack())
    assert result.count == 18
    assert result.debug_info['strip_consensus']['augmentation']['added'] == 0


def test_shadow_keeps_existing_count_and_records_independent_evidence():
    image = stack()
    baseline = PaperCounter(replace(settings(), strip_sequence_enabled=False, strip_sequence_dual_enabled=False)).process(image)
    shadow = PaperCounter(replace(settings(), strip_sequence_mode='shadow')).process(image)
    assert shadow.count == baseline.count
    assert shadow.peak_positions == baseline.peak_positions
    assert shadow.debug_info['strip_consensus']['augmentation']['mode'] == 'shadow'


def test_new_pipeline_handles_uint16_without_hardcoded_8bit_thresholds():
    image = stack().astype(np.uint16)*257
    assert PaperCounter(settings()).process(image).count == 18


def test_validate_before_claim_keeps_unique_ids_through_terminal_verifier():
    cfg = replace(settings(), strip_sequence_mode='augment', track_validate_before_claim=True,
                  final_track_verifier_enabled=True, final_track_verifier_shadow_mode=False)
    result = PaperCounter(cfg).process(stack())
    ids = [t.track_id for t in result.paper_edge_tracks]
    assert result.count == 18
    assert len(ids) == len(set(ids))
    assert len(result.debug_info['track_identity']['features']) == result.count


def test_raw_body_and_native_strip_acf_can_correct_doubled_global_scale():
    cfg = settings()
    stripes = [{'strip': s, 'observations': [
        {'y': float(y), 'raw_significance': 1., 'pair_score': .8} for y in range(10, 230, 20)]}
        for s in range(cfg.num_strips)]
    diagnostic = {'polarities': {'positive': {'strips': stripes, 'strip_period_hypotheses': [
        [{'period': 40, 'acf_score': .9}, {'period': 20, 'acf_score': .6}] for _ in stripes]}}}
    scale, evidence = choose_strip_spacing(diagnostic, 'positive', 40, cfg)
    assert scale == 20 and evidence['changed']
    assert evidence['period_generated_paper_count'] == 0


def test_acf_without_raw_body_candidates_cannot_override_spacing():
    cfg = settings()
    stripes = [{'strip': s, 'observations': []} for s in range(cfg.num_strips)]
    diagnostic = {'polarities': {'positive': {'strips': stripes, 'strip_period_hypotheses': [
        [{'period': 20, 'acf_score': .95}] for _ in stripes]}}}
    scale, evidence = choose_strip_spacing(diagnostic, 'positive', 40, cfg)
    assert scale == 40 and not evidence['changed']


def test_raw_body_confirmation_requires_multiple_spatial_regions():
    image = stack()
    cfg = settings()
    result = PaperCounter(cfg).process(image)
    tracks = result.paper_edge_tracks
    d = result.debug_info['strip_consensus']
    attach_raw_body_confirmation(tracks, d, 'positive', cfg)
    assert all(t.local_evidence['raw_body_confirmation']['supported'] for t in tracks)
    for polarity in ('positive', 'negative'):
        for row in d['polarities'][polarity]['strips'][4:]:
            row['observations'] = []
    attach_raw_body_confirmation(tracks, d, 'positive', cfg)
    assert not any(t.local_evidence['raw_body_confirmation']['supported'] for t in tracks)


def test_terminal_raw_conflict_is_reviewable_and_does_not_bypass_background_rejection():
    from paper_counter.track_identity import TrackIdentityFeatures, apply_final_track_verifier
    cfg = replace(settings(), final_track_verifier_enabled=True, final_track_verifier_shadow_mode=False)
    result = PaperCounter(cfg).process(stack())
    tracks = result.paper_edge_tracks
    attach_raw_body_confirmation(tracks, result.debug_info['strip_consensus'], 'negative', cfg)
    features = {r['track_id']: TrackIdentityFeatures(**r) for r in result.debug_info['track_identity']['features']}
    first = features[tracks[0].track_id]
    first.false_family, first.proposed_decision = 'SECONDARY_TEXTURE', 'REJECT_SECONDARY_TEXTURE'
    first.evidence_families = ['template']
    kept, rejected, _ = apply_final_track_verifier(tracks, features, cfg)
    assert tracks[0] in rejected
    guarded = replace(cfg, strip_sequence_raw_body_safeguard_enabled=True)
    kept, rejected, debug = apply_final_track_verifier(tracks, features, guarded)
    assert tracks[0] in kept and not rejected
    assert debug['raw_body_conflicts_retained'] == [tracks[0].track_id]
    first.false_family, first.proposed_decision = 'BACKGROUND_FALSE_EDGE', 'REJECT_BACKGROUND'
    kept, rejected, _ = apply_final_track_verifier(tracks, features, guarded)
    assert tracks[0] in rejected
