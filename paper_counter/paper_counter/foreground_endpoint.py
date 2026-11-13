"""Verify an existing exposed outer edge using measured foreground evidence."""


def verified_foreground_endpoint(track,feature,is_outer,cfg):
    proof=feature.partial_endpoint or {}
    return bool(is_outer and feature.false_family=='BACKGROUND_FALSE_EDGE'
        and 'competition' not in (feature.evidence_families or [])
        and feature.distance_to_stack_boundary<=.5*feature.local_spacing
        and feature.background_below_score>=.9
        and feature.positive_pair_support>=.6
        and proof.get('complete_identity_strips',0)>=.75*len(track.strip_positions)
        and proof.get('body_support_ratio',0)>=.8
        and track.support_ratio>=cfg.min_track_support_ratio
        and track.continuity_ratio>=cfg.min_continuity_ratio
        and track.smoothness_score>=cfg.min_track_smoothness)
