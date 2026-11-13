"""Observed outer paper components, including slanted sheets across wide dark gaps."""
import cv2
import numpy as np
from .models import StripPeakCandidate
from .tracking import make_track


def recover_isolated_outer_components(gray, strips, tracks, period, offset, polarity, cfg):
    """One physical connected RAW component -> at most one boundary track.

    Only detached outer components are considered. Connectivity, width, paper
    thickness, signed RAW transitions and low exterior are all measured.
    The period bounds thickness only; it never supplies a missing position.
    """
    raw=gray.astype(np.float32)*(255/np.iinfo(gray.dtype).max)
    background,bright=np.percentile(raw[::3,::3],[2,95])
    threshold=float(background+max(4.,.1*(bright-background)))
    binary=(raw>threshold).astype(np.uint8)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(binary,8)
    wide=[i for i in range(1,n) if stats[i,cv2.CC_STAT_WIDTH]>=.65*gray.shape[1]]
    diagnostic=dict(threshold=threshold,decisions=[],added=0,replaced_ids=[])
    if len(wide)<2:return tracks,[],diagnostic
    ordered=sorted(wide,key=lambda i:centers[i,1])
    selected=list(tracks);removed=[]
    for component,neighbor in [(ordered[0],ordered[1]),(ordered[-1],ordered[-2])]:
        observations=[];thickness=[];separations=[];support=[]
        for s,(a,b) in enumerate(strips):
            x=int((a+b)/2)
            ys=np.flatnonzero(labels[:,x]==component)
            other=np.flatnonzero(labels[:,x]==neighbor)
            if not len(ys) or not len(other):continue
            top,bottom=int(ys[0]),int(ys[-1])
            if len(ys)!=bottom-top+1:continue
            gap=max(top-other[-1],other[0]-bottom)
            if gap<.75*period:continue
            width=bottom-top+1
            if not 2<=width<=1.4*period:continue
            # Narrow horizontal support follows the actual measured component.
            line=np.median(raw[:,max(0,x-2):x+3],axis=1)
            edge=top if polarity=='positive' else bottom
            derivative=np.gradient(line)*(1 if polarity=='positive' else -1)
            lo=max(1,edge-2);hi=min(len(line)-1,edge+3)
            if hi<=lo:continue
            y=lo+int(np.argmax(derivative[lo:hi]))
            body=float(np.median(line[top:bottom+1]))
            exterior=np.r_[line[max(0,top-4):top],line[bottom+1:min(len(line),bottom+5)]]
            if len(exterior)<4:continue
            contrast=body-float(np.median(exterior))
            if contrast<cfg.strip_sequence_raw_min_contrast or derivative[y]<=0:continue
            support.append(dict(strip=s,upper=top+offset,lower=bottom+offset,
                                boundary=y+offset,body=body,contrast=contrast))
            thickness.append(width);separations.append(float(gap))
            observations.append(StripPeakCandidate(s,float(y+offset),float(derivative[y]),
                float(derivative[y]),float(width),cfg.strip_sequence_raw_min_contrast,'strong',
                local_prominence_reference=max(float(derivative[y]),cfg.strip_sequence_raw_min_contrast),
                observation_source='detached_RAW_component_boundary'))
        if len(observations)<np.ceil(.75*len(strips)):continue
        if np.median(np.abs(np.asarray(thickness)-np.median(thickness)))>.25*period:continue
        candidate=make_track(observations,period,len(strips),cfg,source='recovered')
        if candidate.continuity_ratio<cfg.min_continuity_ratio or candidate.smoothness_score<cfg.min_track_smoothness:continue
        candidate.track_id=f'raw_outer_component_{component}'
        candidate.boundary_polarity=polarity
        candidate.sequence_decision='rescued';candidate.candidate_class='strong'
        candidate.local_evidence['isolated_raw_component']=dict(verified=True,samples=support,
            median_thickness=float(np.median(thickness)),median_observed_gap=float(np.median(separations)))
        by_strip={v['strip']:v for v in support}
        matches=[]
        for old in selected:
            common=set(old.strip_positions)&set(by_strip)
            inside=[by_strip[s]['upper']-3<=old.strip_positions[s]<=by_strip[s]['lower']+3 for s in common]
            if len(inside)>=3 and np.mean(inside)>=.75:matches.append(old)
        if len(matches)>1:
            diagnostic['decisions'].append(dict(component=component,decision='ambiguous_multiple_existing_tracks'))
            continue
        # Preserve a good existing representative; repair fragmented geometry only.
        if matches and matches[0].support_ratio>=.75:
            if cfg.evidence_model:
                # Preserve the identity as well as its already-good geometry.
                matches[0].local_evidence['isolated_raw_component']=candidate.local_evidence['isolated_raw_component']
            continue
        for old in matches:
            selected.remove(old);removed.append(old);diagnostic['replaced_ids'].append(old.track_id)
        selected.append(candidate)
        diagnostic['added']+=1
        diagnostic['decisions'].append(dict(component=component,track_id=candidate.track_id,
            decision='measured_outer_paper',proof=candidate.local_evidence['isolated_raw_component']))
    return sorted(selected,key=lambda t:t.center_y),removed,diagnostic
