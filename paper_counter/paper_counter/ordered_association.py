"""One-to-one ordered association using direction residuals, not absolute jumps."""
from __future__ import annotations
import numpy as np
from .strip_sequence import ordered_match
from .tracking import make_track
from .boundary_direction import predicted_shift


def associate_directional(candidates, strips, slopes, offset, period, cfg, valid_count=None, quality=None):
    centers=np.array([sum(v)/2 for v in strips])
    groups=[]; decisions=[]
    # Solve adjacent observation sets jointly in vertical order. A clear peak
    # never claims an unrelated neighbor merely because its absolute y is near.
    anchor=int(np.argmax(np.mean(quality,axis=1))) if quality is not None else len(strips)//2
    order=sorted(range(len(strips)),key=lambda s:abs(s-anchor))
    for s in order:
        row=candidates[s]
        usable=sorted((p for p in row if p.reason is None),key=lambda p:p.y)
        active=[g for g in groups if min(abs(s-p.strip_index) for p in g)<=cfg.max_skip_strips+1]
        predictions=[]; gates=[]
        for group in active:
            nearest=sorted(group,key=lambda p:abs(p.strip_index-s))
            last=nearest[0]; previous=last.strip_index
            y=int(np.clip(round(last.y)-offset,0,len(slopes[previous])-1))
            measured=float((slopes[previous][y]+slopes[s][y])*.5)
            if len(group)>=2:
                old=nearest[1]
                path_slope=(last.y-old.y)/(centers[previous]-centers[old.strip_index])
                # A local RAW direction follows turns; past trajectory is a
                # fallback only when the tensor is uninformative.
                if abs(measured)<cfg.epsilon:
                    measured=float(np.clip(path_slope,-cfg.orientation_max_slope,cfg.orientation_max_slope))
            predictions.append(last.y+measured*(centers[s]-centers[previous]))
            if getattr(slopes,'motion',None):
                predictions[-1]=last.y+predicted_shift(slopes,centers,previous,s,last.y,offset)
            gates.append(max(2.,cfg.max_neighbor_shift_ratio*period)*np.sqrt(abs(s-previous)))
        permutation=np.argsort(predictions)
        active=[active[i] for i in permutation]
        predicted=np.array(predictions)[permutation]
        gate=np.array(gates)[permutation]
        pairs,new=ordered_match(predicted,np.array([p.y for p in usable]),gate)
        for i,j in pairs:
            active[i].append(usable[j])
            decisions.append({'strip':s,'y':usable[j].y,'predicted_y':float(predicted[i]),
                              'residual':float(usable[j].y-predicted[i]),'gate':float(gate[i])})
        groups.extend([[usable[j]] for j in new])
    tracks=[make_track(group,period,valid_count or len(strips),cfg) for group in groups]
    return sorted(tracks,key=lambda t:t.center_y), {'enabled':True,'method':'direction_residual_monotone_assignment',
        'generated_observations':0,'matches':decisions}
