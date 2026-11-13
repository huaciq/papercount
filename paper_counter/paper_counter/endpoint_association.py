"""Repair a fragmented outer track using existing, RAW-supported observations."""
import numpy as np
from .partial_endpoint import partial_endpoint_evidence
from .tracking import make_track


def repair_endpoint_association(tracks,candidates,raw,offset,polarity,period,cfg):
    debug={'enabled':cfg.endpoint_reassociation_enabled,'decisions':[]}
    if not cfg.endpoint_reassociation_enabled or len(tracks)<2 or not period:return tracks,[],debug
    ordered=sorted(tracks,key=lambda t:t.center_y)
    seed=ordered[0] if polarity=='positive' else ordered[-1]
    distance=(seed.center_y-offset if polarity=='positive' else offset+raw.shape[1]-1-seed.center_y)
    if distance>2*period or seed.support_ratio>=.75 or len(seed.strip_positions)<3:return tracks,[],debug
    points=sorted(seed.strip_positions.items())
    slopes=[(y2-y1)/(s2-s1) for i,(s1,y1) in enumerate(points) for s2,y2 in points[i+1:]]
    slope=float(np.median(slopes));intercept=float(np.median([y-slope*s for s,y in points]))
    tolerance=max(2.,.15*period)
    observed=[]
    for s,strip in enumerate(candidates):
        options=[]
        for peak in strip:
            residual=abs(peak.y-(intercept+slope*s))
            if residual>tolerance:continue
            if any(t is not seed and s in t.strip_positions and abs(t.strip_positions[s]-peak.y)<=3 for t in tracks):continue
            single=make_track([peak],period,cfg.num_strips,cfg,source='recovered')
            proof=partial_endpoint_evidence(single,period,raw,offset,polarity,cfg)
            if not any(v['supported'] for v in proof['samples']):continue
            options.append((residual,peak))
        if options:observed.append(min(options,key=lambda v:v[0])[1])
    if len(observed)<max(np.ceil(.75*cfg.num_strips),len(seed.strip_positions)+2):return tracks,[],debug
    repaired=make_track(observed,period,cfg.num_strips,cfg,source='recovered')
    if repaired.continuity_ratio<cfg.min_continuity_ratio or repaired.smoothness_score<cfg.min_track_smoothness:return tracks,[],debug
    repaired.track_id='endpoint_reassociated_'+seed.track_id
    repaired.boundary_polarity=polarity;repaired.sequence_decision='rescued'
    repaired.local_evidence['endpoint_reassociation']={'seed_track_id':seed.track_id,
        'original_positions':seed.strip_positions,'observed_positions':repaired.strip_positions,
        'policy':'robust_fit_selects_existing_RAW_supported_candidates_only'}
    seed.reason='replaced_by_measured_endpoint_reassociation'
    debug['decisions'].append(repaired.local_evidence['endpoint_reassociation'])
    return sorted([t for t in tracks if t is not seed]+[repaired],key=lambda t:t.center_y),[seed],debug
