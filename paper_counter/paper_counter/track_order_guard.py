"""Reject auxiliary tracks that cross an independently observed primary layer."""


def apply_track_order_guard(tracks, preferred, cfg):
    debug={'enabled':cfg.auxiliary_order_guard_enabled or cfg.auxiliary_verified_body_dedup_enabled,'decisions':[]}
    if not debug['enabled']:return tracks,[],debug
    primary=[t for t in tracks if (t.boundary_polarity or preferred)==preferred
             and t.support_ratio>=.75 and t.smoothness_score>=cfg.min_track_smoothness]
    rejected=[]
    for track in tracks:
        if (track.boundary_polarity or preferred)==preferred:continue
        for anchor in primary:
            common=sorted(set(track.strip_positions)&set(anchor.strip_positions))
            signed=[float(track.strip_positions[s]-anchor.strip_positions[s]) for s in common]
            # Two actual observations on each side prevent one noisy position
            # from establishing a crossing. No interpolated positions are used.
            if cfg.auxiliary_order_guard_enabled and sum(d>3 for d in signed)>=2 and sum(d< -3 for d in signed)>=2:
                track.reason='auxiliary_crosses_observed_primary_layer'
                track.sequence_decision='suppressed'
                rejected.append(track)
                debug['decisions'].append(dict(track_id=track.track_id,anchor_id=anchor.track_id,
                    common_strips=common,signed_differences=signed))
                break
        if any(t is track for t in rejected) or not cfg.auxiliary_verified_body_dedup_enabled:continue
        body=track.local_evidence.get('raw_body_confirmation',{})
        observations=body.get('observations',[]) if body.get('supported') else []
        for anchor in primary:
            matches=[o for o in observations if o.get('pair_y') is not None
                     and o['strip'] in anchor.strip_positions
                     and abs(o['pair_y']-anchor.strip_positions[o['strip']])<=3]
            if len(matches)>=max(3,cfg.min_absolute_track_support) and len(matches)>=.8*len(observations):
                track.reason='auxiliary_verified_pair_same_physical_body'
                track.sequence_decision='suppressed';rejected.append(track)
                debug['decisions'].append(dict(track_id=track.track_id,anchor_id=anchor.track_id,
                    reason=track.reason,verified_pair_observations=matches))
                break
    ids={t.track_id for t in rejected}
    return [t for t in tracks if t.track_id not in ids],rejected,debug
