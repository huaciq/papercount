"""Terminal observation guard for isolated, partial, unmeasurable tracks.

No intensity value of zero is treated as measured background: availability is
computed from actual observation coordinates and the RAW sampling window.
"""
from __future__ import annotations

def apply_raw_observation_guard(tracks, features, bounds, cfg, verifier_debug):
    diagnostic={'enabled':cfg.raw_final_observation_guard_enabled,'decisions':[],
                'count_before':len(tracks),'count_after':len(tracks)}
    if not cfg.raw_final_observation_guard_enabled:
        return tracks, [], diagnostic
    rejected=[]
    for track in tracks:
        feature=features.get(track.track_id)
        if feature is None:
            continue
        radius=cfg.identity_profile_extent_ratio*max(feature.local_spacing,cfg.epsilon)
        usable=sum(bounds['y1'] <= float(y)-radius and float(y)+radius <= bounds['y2']-1
                   for y in track.strip_positions.values())
        gap=min((abs(track.center_y-other.center_y) for other in tracks
                 if other.track_id!=track.track_id),default=0.)/max(feature.local_spacing,cfg.epsilon)
        body=track.local_evidence.get('raw_body_confirmation',{}).get('supported',False)
        if cfg.evidence_model:
            body=(track.local_evidence.get('paper_body',{}).get('verified',False)
                  or track.local_evidence.get('isolated_raw_component',{}).get('verified',False))
            if cfg.composite_layers_enabled:
                body=body or track.local_evidence.get('composite_structure',{}).get('verified',False)
        reject=(usable==0 and track.support_ratio<min(.5,cfg.min_track_support_ratio)
                and gap>cfg.sequence_large_gap_anomaly_ratio and not body)
        dark_isolated=(cfg.isolated_dark_track_guard_enabled and usable>=cfg.min_absolute_track_support
                and gap>cfg.sequence_large_gap_anomaly_ratio and not body
                and feature.paper_support_above<cfg.identity_low_paper_support
                and feature.positive_pair_support==0
                and feature.raw_signed_contrast<cfg.strip_sequence_raw_min_contrast)
        proof=getattr(feature,'partial_endpoint',None) or {}
        measured_gap_transition=(cfg.evidence_model and track.local_evidence.get('allocated_signal_gap')
            and track.local_evidence.get('raw_transition',{}).get('verified'))
        unsupported_recovery=(cfg.recovered_raw_body_guard_enabled and not measured_gap_transition and
                track.source=='recovered' and not body and
                proof.get('complete_identity_strips',0)>=.75*len(track.strip_positions) and
                proof.get('body_support_ratio',1)==0 and
                feature.paper_support_above<cfg.identity_low_paper_support and
                feature.positive_pair_support==0)
        reject=reject or dark_isolated or unsupported_recovery
        if reject:
            rejected.append(track)
            diagnostic['decisions'].append({'track_id':track.track_id,
                'reason':('recovered_track_without_measured_raw_body' if unsupported_recovery else
                          'isolated_dark_reflection_without_paper_body' if dark_isolated else
                          'isolated_partial_track_without_raw_identity_window'),
                'usable_raw_identity_strips':usable,'observed_support_ratio':track.support_ratio,
                'nearest_gap_ratio':gap,'raw_body_supported':body})
    ids={t.track_id for t in rejected}
    kept=[t for t in tracks if t.track_id not in ids]
    diagnostic['count_after']=len(kept)
    if ids:
        verifier_debug['count_after']=len(kept)
        for key in ('proposed_reject_track_ids','applied_reject_track_ids'):
            verifier_debug[key].extend(t.track_id for t in rejected if t.track_id not in verifier_debug[key])
        verifier_debug['proposed_reject_count']=len(verifier_debug['proposed_reject_track_ids'])
        verifier_debug['applied_reject_count']=len(verifier_debug['applied_reject_track_ids'])
        for row in verifier_debug['track_decisions']:
            if row['track_id'] in ids:
                reason=next(d['reason'] for d in diagnostic['decisions'] if d['track_id']==row['track_id'])
                row['terminal_decision']=('REJECT_RECOVERY_WITHOUT_RAW_BODY' if reason=='recovered_track_without_measured_raw_body' else
                                          'REJECT_ISOLATED_DARK_REFLECTION' if reason=='isolated_dark_reflection_without_paper_body'
                                          else 'REJECT_UNOBSERVABLE_ISOLATED_TRACK')
                row['applied']=True
    verifier_debug['raw_observation_guard']=diagnostic
    return kept,rejected,diagnostic
