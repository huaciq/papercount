# -*- coding: utf-8 -*-
"""Three independent grayscale observations, joint ordered body boundaries.

The fused profile is auxiliary. Native strip candidates are kept before any
signal fusion; an uninformative strip is not a negative vote. Peak contours
describe candidate lobes only, never the final paper thickness.
"""
from copy import deepcopy
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter,maximum_filter1d,minimum_filter1d
from scipy.signal import find_peaks
from scipy.stats import trim_mean


def weighted_median(values,weights):
    order=np.argsort(values);v=np.asarray(values)[order];w=np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w),.5*np.sum(w))])


def observe_strips(aligned,raw,valid,max_period,extract):
    """Extract per-strip candidates first, then form ordered common observations."""
    h,w=aligned.shape;edges=np.linspace(0,w,4,dtype=int)
    signals=[];raw_signals=[];weights=[];native=[];bounds=[];assessments=[]
    for strip,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        bands,signal,pitch,strength,_=extract(aligned[:,a:b],raw[:,a:b],valid[:,a:b],max_period)
        # Gray DC is preserved. Reliability normalizes the contribution, not
        # the physical gray levels needed to distinguish paper from background.
        raw_signal=trim_mean(raw[:,a:b],.1,axis=1).astype(np.float32)
        local_range=maximum_filter1d(signal,61)-minimum_filter1d(signal,61)
        noise=1.4826*median_filter(np.abs(signal-gaussian_filter1d(signal,.65)),size=61)
        raw_range=maximum_filter1d(raw_signal,31)-minimum_filter1d(raw_signal,31)
        reliability=local_range/np.maximum(local_range+6*noise,1.)
        reliability*=raw_range/np.maximum(raw_range+2.,1.)
        reliability*=gaussian_filter1d(valid[:,a:b].mean(axis=1),2.)
        reliability[raw_range<1.]=0.
        signals.append(signal);raw_signals.append(raw_signal);weights.append(reliability)
        bounds.append([int(a),int(b)])
        assessments.append(dict(strip=strip,period=pitch,acf_score=strength,candidate_count=len(bands)))
        for band in bands:
            item=deepcopy(band);item.update(strip=strip,native_center=band['center'],
                reliability=float(reliability[int(band['center'])]),native_kind='primary')
            native.append(item)
        # Weak observations remain candidates before fusion. Search only
        # actual gray maxima in an observed missing interval or at an end;
        # this is not periodic interpolation and does not add a grid location.
        positions=np.array(sorted(q['center'] for q in bands))
        spacing=float(np.median(np.diff(positions))) if len(positions)>2 else 14.
        weak_peaks,weak_props=find_peaks(gaussian_filter1d(signal,.5),
            prominence=np.maximum(.7,np.maximum(.8*noise,.035*local_range)),distance=2)
        bg=float(np.percentile(raw[:,a:b],2));floor=bg+max(2.,.03*float(np.percentile(raw[:,a:b],95)-bg))
        for p,prom in zip(weak_peaks,weak_props['prominences']):
            if not len(positions) or np.min(np.abs(positions-p))<=max(2.,.32*spacing):continue
            before=positions[positions<p];after=positions[positions>p]
            missing=(len(before) and len(after) and after[0]-before[-1]>1.5*spacing)
            endpoint=(p<positions[0] and positions[0]-p<1.5*spacing) or (p>positions[-1] and p-positions[-1]<1.5*spacing)
            if not (missing or endpoint):continue
            lo=max(0,int(p-.55*spacing));hi=min(h-1,int(p+.55*spacing))
            left=lo+int(np.argmin(signal[lo:p+1]));right=p+int(np.argmin(signal[p:hi+1]))
            if left>=p or right<=p:continue
            peak_raw=raw[max(0,p-1):min(h,p+2),a:b].max(axis=0)
            contrast=peak_raw-np.maximum(raw[left,a:b],raw[right,a:b])
            observed=(contrast>.7)&(peak_raw>floor)&valid[p,a:b]
            coverage=float(np.mean(observed))
            if coverage<.55:continue
            native.append(dict(id=f'weak_{strip}_{p}',top=float(left),bottom=float(right),center=float(p),
                prominence=float(prom),coverage=coverage,score=float(coverage*np.log1p(prom/max(1.,noise[p]))),
                scale=.5,strip=strip,native_center=float(p),reliability=float(reliability[p]),native_kind='weak'))
    signals=np.asarray(signals);raw_signals=np.asarray(raw_signals);weights=np.asarray(weights)
    total=weights.sum(axis=0)
    normalized_weights=weights/np.maximum(total,1e-9)
    fused=np.sum(signals*normalized_weights,axis=0)
    fused[total<=1e-9]=np.mean(signals[:,total<=1e-9],axis=0)
    # Each strip is already mapped by the same tangent flow into center-column
    # coordinates. No periodic cross-correlation shift is permitted here.
    spacing=[]
    for strip in range(3):
        positions=sorted(n['center'] for n in native if n['strip']==strip)
        spacing.extend(np.diff(positions))
    typical=float(np.median(spacing)) if spacing else 14.
    # A broad/flat luminous rim has an uncertain maximum even after tangent
    # alignment. Match within that sub-half-cycle uncertainty; never shift a
    # whole strip by a periodic lag just to make its maxima coincide.
    gate=max(1.5,min(5.5,.32*typical))
    groups=[]
    for n in sorted(native,key=lambda q:(-q['score']*max(q['reliability'],.1),q['center'])):
        nearby=[g for g in groups if abs(g['center']-n['center'])<=gate and all(o['strip']!=n['strip'] for o in g['observations'])]
        if nearby:
            group=min(nearby,key=lambda g:abs(g['center']-n['center']))
            group['observations'].append(n)
            group['center']=weighted_median([o['center'] for o in group['observations']],
                [max(.1,o['reliability'])*o['score'] for o in group['observations']])
        else:groups.append(dict(center=n['center'],observations=[n]))
    # A second native observation in a strip may be the same blurred maximum.
    # Merge only when *all* native center positions stay within the small gate.
    groups.sort(key=lambda g:g['center']);merged=[]
    for g in groups:
        if merged and g['center']-merged[-1]['center']<max(1.5,.55*gate):
            prior=merged[-1];by_strip={o['strip']:o for o in prior['observations']}
            for o in g['observations']:
                if o['strip'] not in by_strip or o['score']>by_strip[o['strip']]['score']:by_strip[o['strip']]=o
            prior['observations']=list(by_strip.values())
            prior['center']=weighted_median([o['center'] for o in prior['observations']],
                [max(.1,o['reliability'])*o['score'] for o in prior['observations']])
        else:merged.append(g)
    bands=[]
    for group in merged:
        observations=group['observations']
        strongest=max(observations,key=lambda o:o['score']*max(o['reliability'],.1))
        # One clear strip is sufficient; three adjacent stripes are correlated
        # observations, so evidence is not multiplied into three independent votes.
        if len(observations)==1 and (strongest['coverage']<.5 or strongest['score']<1.5):continue
        local_weights=[max(.1,o['reliability'])*o['score'] for o in observations]
        band=dict(strongest,center=weighted_median([o['center'] for o in observations],local_weights),
            top=weighted_median([o['top'] for o in observations],local_weights),
            bottom=weighted_median([o['bottom'] for o in observations],local_weights),
            score=float(max(o['score'] for o in observations)+.15*np.mean([o['score'] for o in observations])),
            coverage=float(max(o['coverage'] for o in observations)),
            strip_observations=observations,source_strips=sorted(o['strip'] for o in observations))
        bands.append(band)
    bands.sort(key=lambda b:b['center'])
    for i,b in enumerate(bands):b.update(id=f'band_{i:05d}',index=i)
    return bands,fused.astype(np.float32),dict(signals=signals,raw_signals=raw_signals,
        weights=normalized_weights,bounds=bounds,assessments=assessments,native_candidates=native,
        matching_gate=gate,typical_lobe_spacing=typical,
        fusion_order='independent_native_candidates_then_joint_evidence',
        fused_signal_role='auxiliary_only',minimum_supporting_strips=1)


def select_units(candidates,bands):
    """Ordered body selection with a soft *full-unit spacing* prior."""
    if not candidates:return [],dict(typical_spacing=None)
    ordered=sorted(candidates,key=lambda c:((bands[c['band_ids'][0]]['center']+bands[c['band_ids'][-1]]['center'])/2))
    centers=np.array([(bands[c['band_ids'][0]]['center']+bands[c['band_ids'][-1]]['center'])/2 for c in ordered])
    # Adjacent double-lobe alternatives interleave. Estimate the unit spacing
    # from noncompeting pairs rather than mistaking their half-phase for a sheet.
    gaps=[]
    for i,c in enumerate(ordered[:-1]):
        nxt=next((j for j in range(i+1,len(ordered)) if min(ordered[j]['band_ids'])>max(c['band_ids'])),None)
        if nxt is not None:gaps.append(centers[nxt]-centers[i])
    positive=np.asarray([v for v in gaps if v>2.])
    typical=float(np.median(positive)) if len(positive) else None
    scale=max(.5,float(np.median([c['score'] for c in ordered])))
    values=[];paths=[]
    for i,c in enumerate(ordered):
        best=c['score'];path=[i]
        for j in range(i):
            if max(ordered[j]['band_ids'])>=min(c['band_ids']):continue
            ratio=(centers[i]-centers[j])/typical if typical else 1.
            # Dense competing fragments are penalized, not physically valid
            # long gaps. No count is produced to fill an empty interval.
            penalty=.8*scale*max(0.,(.75-ratio)/.25)**2
            score=values[j]+c['score']-penalty
            if score>best:best=score;path=paths[j]+[i]
        values.append(best);paths.append(path)
    selected=[ordered[i] for i in paths[int(np.argmax(values))]]
    return selected,dict(typical_spacing=typical,spacing_source='ordered_noncompeting_complete_units',
        thickness_prior='soft_not_a_period_grid',excluded=[c['id'] for c in ordered if c not in selected])


def _interface(left,right,bands,observation):
    signals=observation['signals'];raws=observation['raw_signals'];weights=observation['weights']
    a=int(round(bands[max(left['band_ids'])]['center']));b=int(round(bands[min(right['band_ids'])]['center']))
    if b<=a+1:return [dict(bottom=float(a+.5),top=float(a+.5),score=0.,evidence=[])]
    proposals=[]
    for strip in range(3):
        signal=gaussian_filter1d(signals[strip],.5);raw=raws[strip]
        minima,_=find_peaks(-signal[a+1:b])
        minima=minima+a+1
        if not len(minima):minima=np.array([a+1+np.argmin(signal[a+1:b])])
        for y in minima:
            depth=min(signal[a],signal[b])-signal[y]
            local_range=max(float(np.ptp(signal[a:b+1])),1.)
            quality=float(np.mean(weights[strip,a:b+1]))
            if depth<=.5 or quality<=.01:continue
            proposals.append(dict(y=int(y),strip=strip,score=float(quality*np.clip(depth/local_range,0,1))))
    if not proposals:
        # Existing bodies still need a geometric separator, but lack of image
        # evidence is explicit and carries no claim of a measured boundary.
        middle=(a+b)/2
        return [dict(bottom=middle,top=middle,score=0.,evidence=[],status='boundary_unresolved')]
    groups=[]
    for p in sorted(proposals,key=lambda p:-p['score']):
        g=next((g for g in groups if abs(g['y']-p['y'])<=2.),None)
        if g is None:groups.append(dict(y=float(p['y']),observations=[p]))
        else:
            g['observations'].append(p)
            g['y']=weighted_median([q['y'] for q in g['observations']],[q['score'] for q in g['observations']])
    choices=[]
    for group in groups:
        y=group['y'];evidence=group['observations'];bottom=y;top=y
        # A truly dark contiguous background run is left empty. It is not
        # divided into paper rectangles merely to make their heights uniform.
        gaps=[]
        for strip in range(3):
            raw=raws[strip];background=float(np.percentile(raw,2))
            cutoff=background+max(1.5,.02*(float(np.percentile(raw,95))-background))
            quality=float(np.mean(weights[strip,a:b+1]))
            if quality<=.03 or min(float(raw[a]),float(raw[b]))<=cutoff:continue
            pos=int(round(y))
            if raw[pos]>cutoff:continue
            lo=pos;hi=pos
            while lo>a+1 and raw[lo-1]<=cutoff:lo-=1
            while hi<b-1 and raw[hi+1]<=cutoff:hi+=1
            if hi-lo>=2:gaps.append((lo,hi+1,quality))
        if len(gaps)>=2 or (gaps and max(g[2] for g in gaps)>=.6):
            bottom=weighted_median([g[0] for g in gaps],[g[2] for g in gaps])
            top=weighted_median([g[1] for g in gaps],[g[2] for g in gaps])
        by_strip={p['strip']:p for p in evidence}
        score=max(p['score'] for p in by_strip.values())+.15*sum(p['score'] for p in by_strip.values())
        choices.append(dict(bottom=float(bottom),top=float(top),score=float(score),evidence=evidence,status='observed'))
    return sorted(choices,key=lambda c:-c['score'])[:8]


def joint_boundaries(units,bands,observation):
    """Joint shared separators; paper height is NOT a relative peak contour."""
    if not units:return [],dict(interfaces=[],thickness=[])
    signals=observation['raw_signals'];h=signals.shape[1]
    unit_centers=np.array([(bands[min(u['band_ids'])]['center']+bands[max(u['band_ids'])]['center'])/2 for u in units])
    gaps=np.diff(unit_centers);nominal=float(np.median(gaps)) if len(gaps) else max(8.,units[0]['bottom']-units[0]['top'])
    def outer(unit,upper):
        center=int(round(bands[min(unit['band_ids']) if upper else max(unit['band_ids'])]['center']))
        a=max(0,int(center-1.2*nominal)) if upper else center
        b=center if upper else min(h-1,int(center+1.2*nominal))
        estimates=[]
        for strip,raw in enumerate(signals):
            background=float(np.percentile(raw,2));peak=float(raw[center])
            threshold=background+max(1.5,.05*(peak-background))
            if peak<=threshold:continue
            current=center;direction=-1 if upper else 1
            while a<=current+direction<=b and raw[current+direction]>threshold:current+=direction
            estimates.append(float(current if upper else current+1))
        return float(np.median(estimates)) if estimates else float(unit['top'] if upper else unit['bottom'])
    first=outer(units[0],True);last=outer(units[-1],False)
    choices=[_interface(left,right,bands,observation) for left,right in zip(units,units[1:])]
    initial=[c[0] for c in choices]
    initial_tops=[first]+[c['top'] for c in initial]
    initial_bottoms=[c['bottom'] for c in initial]+[last]
    observed_heights=np.maximum(1.,np.asarray(initial_bottoms)-np.asarray(initial_tops))
    # Learn from whole-body interfaces, not from bright-lobe widths. Nearby
    # measured bodies define a robust reference, with genuine outliers allowed.
    references=[]
    for i in range(len(units)):
        neighbors=np.r_[observed_heights[max(0,i-6):i],observed_heights[i+1:i+7]]
        references.append(float(np.median(neighbors)) if len(neighbors) else float(observed_heights[i]))
    def height_cost(height,index):
        if height<=0:return 1e6
        z=abs(np.log(height/max(1.,references[index])))/.25
        return .12*min(4.,z*z/2 if z<1 else z-.5)
    values=[0.];states=[dict(top=first)];paths=[[]]
    for i,options in enumerate(choices):
        next_values=[];next_paths=[]
        for c in options:
            scores=[values[j]+c['score']-height_cost(c['bottom']-state['top'],i) for j,state in enumerate(states)]
            best=int(np.argmax(scores));next_values.append(scores[best]);next_paths.append(paths[best]+[c])
        values=next_values;paths=next_paths;states=options
    if choices:
        end=int(np.argmax([values[j]-height_cost(last-state['top'],len(units)-1) for j,state in enumerate(states)]))
        chosen=paths[end]
    else:chosen=[]
    tops=[first]+[c['top'] for c in chosen];bottoms=[c['bottom'] for c in chosen]+[last]
    result=[];thickness=[]
    for i,(unit,top,bottom) in enumerate(zip(units,tops,bottoms)):
        # Disjoint separator corridors already enforce order; these assertions
        # make an accidental overlap a failure instead of a hidden bad export.
        assert bottom>top
        if i:assert top>=bottoms[i-1]-1e-8
        result.append(dict(unit,top=float(top),bottom=float(bottom)))
        thickness.append(dict(unit_id=unit['id'],height=float(bottom-top),local_reference=references[i],
            ratio=float((bottom-top)/max(references[i],1.)),soft_cost=height_cost(bottom-top,i)))
    return result,dict(interfaces=chosen,thickness=thickness,initial_heights=observed_heights.tolist(),
        hard_constraints=['positive_height','ordered','nonoverlapping'],
        boundary_source='three_strip_gray_interfaces',peak_width_is_body_height=False)
