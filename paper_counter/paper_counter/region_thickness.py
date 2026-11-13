"""Thickness-led sequence repair; optical boundaries are auxiliary evidence."""
from copy import deepcopy
import numpy as np


def weighted_median(values, weights):
    order=np.argsort(values);v=np.asarray(values)[order];w=np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w),w.sum()/2)])


def repair_thickness(units,bands,observation,proposals,period_hint=None):
    """Compare all keep/merge/split plans without fixed anomaly ratio gates.

    Inferred subdivisions are allowed, explicitly labelled, and do not count
    as independently observed paper. Real empty intervals remain empty.
    """
    audit=dict(before_count=len(units),after_count=len(units),actions=[],unresolved=[],applied=False,
               algorithm='adaptive_thickness_sequence',fixed_anomaly_ratio_gates=False)
    if len(units)<6:
        audit['reason']='insufficient_thickness_anchors';return units,audit
    heights=np.array([u['bottom']-u['top'] for u in units]);logs=np.log(np.maximum(heights,1.))
    # Four fragments must not outvote one full body merely by their count.
    mass=np.minimum(heights/np.median(heights),2.)
    bandwidth=float(np.clip(.6*np.subtract(*np.percentile(logs,[75,25])),.12,.3))
    density=np.exp(-.5*((logs[:,None]-logs[None,:])/bandwidth)**2)@mass
    mode=float(heights[np.argmax(density)])
    if period_hint and abs(np.log(period_hint/mode))<2*bandwidth:
        near=np.abs(np.log(heights/period_hint))<2*bandwidth
        if np.count_nonzero(near)>=max(5,.15*len(units)):mode=float(np.median(heights[near]))
    weights_anchor=np.exp(-.5*(np.log(heights/mode)/bandwidth)**2)*mass
    normal=weighted_median(heights,weights_anchor)
    residual=np.abs(np.log(heights/normal))
    # Natural log-thickness spread plus one-pixel localization uncertainty.
    sigma=float(np.clip(np.hypot(1.4826*weighted_median(residual,weights_anchor),1/normal),.09,.4))
    anchors=residual<=max(2.5*sigma,bandwidth)
    confidence=float(np.clip(np.sum(mass[anchors])/np.sum(mass),.15,1.))
    audit.update(normal_height=normal,log_thickness_sigma=sigma,model_confidence=confidence,
        anchor_indices=np.flatnonzero(anchors).tolist(),
        scale_source='height_weighted_observed_mode_with_optional_acf_corroboration',
        original_units=[dict(id=u['id'],top=u['top'],bottom=u['bottom'],band_ids=u['band_ids']) for u in units])
    if np.count_nonzero(anchors)<5:
        audit['reason']='no_reliable_thickness_mode';return units,audit
    references=[];spreads=[]
    for i in range(len(units)):
        lo=max(0,i-8);hi=min(len(units),i+9);local=heights[lo:hi][anchors[lo:hi]]
        ref=float(np.median(local)) if len(local)>=3 else normal
        references.append(ref)
        mad=float(np.median(abs(np.log(local/ref)))) if len(local)>=5 else 0.
        spreads.append(max(sigma,float(min(.4,1.4826*mad))))
    # Rectangle height also contains separator-placement error. Independent
    # native centers reveal that uncertainty: a short rectangle whose centers
    # still follow the normal pitch should not automatically erase a paper.
    centers=np.array([np.mean([bands[k]['center'] for k in u['band_ids']]) for u in units])
    localization=[]
    for i in range(len(units)):
        distances=np.diff(centers[max(0,i-1):min(len(units),i+2)])
        span=float(np.mean(distances)) if len(distances) else heights[i]
        error=abs(np.log(heights[i]/max(span,1.)))
        if heights[i]<references[i] or i in (0,len(units)-1):
            spreads[i]=max(spreads[i],min(.4,error))
        localization.append(dict(native_center_span=span,separator_log_uncertainty=float(error)))
    audit['separator_localization_uncertainty']=localization
    def loss(height,ref,spread):
        if height<=0:return 1e9
        z=abs(np.log(height/ref))/spread
        return float(confidence*2*(np.sqrt(1+z*z)-1))
    signals=observation['signals'];raw=observation['raw_signals'];strip_weights=observation['weights']
    bg=np.percentile(raw,2,axis=1);floor=bg+np.maximum(1.5,.02*(np.percentile(raw,95,axis=1)-bg))
    def cut_evidence(y,ref):
        pos=int(np.clip(round(y),1,signals.shape[1]-2));radius=max(2,int(.4*ref))
        lo=max(0,pos-radius);hi=min(signals.shape[1],pos+radius+1);depths=[]
        for k in range(3):
            if np.mean(strip_weights[k,lo:hi])<.03:continue
            s=signals[k];left=float(np.max(s[lo:pos]));right=float(np.max(s[pos+1:hi]))
            valley=float(np.min(s[max(lo,pos-1):min(hi,pos+2)]))
            depths.append(float(np.clip((min(left,right)-valley)/max(np.ptp(s[lo:hi]),3.),0,1)))
        return dict(y=float(y),depth=float(np.median(depths)) if depths else 0.,supporting_strips=len(depths))
    cuts=[cut_evidence((a['bottom']+b['top'])/2,normal) for a,b in zip(units,units[1:])]
    audit['interfaces']=cuts
    def foreground(a,b):
        lo=max(0,int(np.ceil(a)));hi=min(raw.shape[1],int(np.ceil(b)))
        if hi<=lo:return 0.
        # Blank stripes do not veto a clear one; empty spans cannot be tiled.
        return float(np.max(np.mean(raw[:,lo:hi]>floor[:,None],axis=1)))
    def split_plan(unit,ref,spread):
        a=unit['top'];b=unit['bottom'];height=b-a;plans=[]
        # Stack endpoints have only one neighboring sheet and may include an
        # exposed face or a cropped body. Missing outside context increases
        # the cost of inferring another sheet, rather than forbidding it.
        outside_context_cost=2. if unit is units[0] or unit is units[-1] else 1.
        # This integer-count search is performed for every box, rather than
        # only after an arbitrary small/large threshold has fired.
        maximum=min(64,max(2,int(np.ceil(height/ref))+1))
        for count in range(2,maximum+1):
            targets=np.linspace(a,b,count+1);states=[(0.,a,[],[])]
            for j in range(1,count):
                target=targets[j];radius=max(1,int(round(spread*ref)))
                locations=np.unique(np.r_[target,np.arange(max(a+1,target-radius),min(b-1,target+radius)+1,1.)])
                next_states=[]
                for y in locations:
                    if y<=a or y>=b:continue
                    cut=cut_evidence(y,ref)
                    options=[(v+loss(y-prev,ref,spread)+.35*(1-cut['depth']),ys+[float(y)],cs+[cut])
                        for v,prev,ys,cs in states if y>prev and foreground(prev,y)>=.5]
                    if options:
                        best=min(options,key=lambda q:q[0]);next_states.append((best[0],float(y),best[1],best[2]))
                states=next_states
                if not states:break
            endings=[(v+loss(b-prev,ref,spread)+1.2*outside_context_cost*(count-1)/confidence,ys,cs)
                for v,prev,ys,cs in states if len(ys)==count-1 and b>prev and foreground(prev,b)>=.5]
            for value,ys,cs in endings:
                # A clearly observed two-lobe identity is additional evidence,
                # not an absolute veto. Splitting it into children with no
                # complete motif must beat a stronger identity-preservation cost.
                if unit.get('structure')=='double_bright':
                    bounds=[a]+ys+[b]
                    missing=sum(not any(p['band_ids'] and all(top<=bands[k]['center']<bottom for k in p['band_ids'])
                        for p in proposals) for top,bottom in zip(bounds,bounds[1:]))
                    value+=float(unit['score'])*missing
                plans.append((value,ys,cs))
        if not plans:return None
        score,ys,cs=min(plans,key=lambda q:q[0]);bounds=[a]+ys+[b];children=[]
        for j,(top,bottom) in enumerate(zip(bounds,bounds[1:])):
            native=[p for p in proposals if p['band_ids'] and all(top<=bands[k]['center']<bottom for k in p['band_ids'])]
            observed=max(native,key=lambda p:p['score']) if native else None
            child=deepcopy(observed if observed is not None else unit)
            ids=observed['band_ids'] if observed is not None else [k for k in unit['band_ids'] if top<=bands[k]['center']<bottom]
            child.update(top=float(top),bottom=float(bottom),band_ids=list(ids),id=f"split_{unit['id']}_{j}",thickness_inferred=True)
            child['evidence']['thickness_repair']=dict(action='thickness_led_split',source_unit_id=unit['id'],
                native_complete_candidate=observed is not None,reference=ref,
                boundary_origin='thickness_prediction_adjusted_by_auxiliary_gray',physical_identity='inferred_requires_review')
            child['score']=float(child['score']*.6)
            children.append(child)
        return score,children,cs
    costs=[0.];paths=[[]];comparisons=[]
    for end in range(1,len(units)+1):
        i=end-1;ref=references[i];spread=spreads[i];unit=units[i]
        keep=loss(heights[i],ref,spread);best=costs[i]+keep
        path=paths[i]+[(i,end,'keep',None)];alternatives=[dict(action='keep',cost=keep)]
        # Moderate thin neighbors (e.g. 19px+21px) compete as a union too.
        for start in range(i-1,-1,-1):
            group=units[start:end]
            if any(b['top']-a['bottom']>1. for a,b in zip(group,group[1:])):break
            ref_group=float(np.median(references[start:end]));s=float(np.median(spreads[start:end]))
            height=group[-1]['bottom']-group[0]['top'];internal=cuts[start:end-1]
            penalty=sum(.65+.4*c['depth'] for c in internal)/confidence
            penalty+=.5*sum(np.exp(-.5*(np.log(heights[k]/references[k])/spreads[k])**2) for k in range(start,end))
            score=loss(height,ref_group,s)+penalty
            alternatives.append(dict(action='merge',source_start=start,cost=score))
            if costs[start]+score<best:
                best=costs[start]+score;path=paths[start]+[(start,end,'merge',dict(reference=ref_group,removed_interfaces=internal))]
        split=split_plan(unit,ref,spread)
        if split:
            score,children,interfaces=split;alternatives.append(dict(action='split',cost=score,children=len(children)))
            if costs[i]+score<best:
                best=costs[i]+score;path=paths[i]+[(i,end,'split',dict(children=children,interfaces=interfaces,reference=ref))]
        comparisons.append(dict(unit_id=unit['id'],reference=ref,spread=spread,keep_cost=keep,
            alternatives=alternatives,best_prefix_cost=float(best)))
        costs.append(best);paths.append(path)
    result=[]
    for start,end,action,detail in paths[-1]:
        source=units[start:end]
        if action=='keep':
            result.append(source[0])
            if loss(heights[start],references[start],spreads[start])>1.:
                audit['unresolved'].append(dict(unit_id=source[0]['id'],ratio=float(heights[start]/references[start]),
                    reason='keep_preferred_by_sequence_cost',thickness_loss=loss(heights[start],references[start],spreads[start])))
            continue
        event=dict(action=action,source_ids=[u['id'] for u in source],**detail)
        if action=='merge':
            merged=deepcopy(source[0]);merged.update(top=source[0]['top'],bottom=source[-1]['bottom'],
                id='merged_'+source[0]['id'],band_ids=sorted({k for u in source for k in u['band_ids']}),
                score=float(np.mean([u['score'] for u in source])))
            merged['evidence']['thickness_repair']=dict(action='merge_fragments',source_units=deepcopy(source),**detail)
            output=[merged]
        else:
            output=detail['children'];event['inference']='thickness_led_not_independently_observed_sheets'
        result.extend(output);event.pop('children',None);audit['actions'].append(event)
    assert all(a['bottom']<=b['top']+1e-8 for a,b in zip(result,result[1:]))
    ids=[k for u in result for k in u['band_ids']];assert len(ids)==len(set(ids))
    audit.update(applied=bool(audit['actions']),after_count=len(result),local_references=references,
        local_spreads=spreads,sequence_cost=float(costs[-1]),comparisons=comparisons,
        inferred_regions=sum(u.get('thickness_inferred',False) for u in result))
    return result,audit
