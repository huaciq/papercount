# -*- coding: utf-8 -*-
"""Paper-interior phase evidence and evidence-gated thickness review."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def limit_to_observed_neighbors(bands,profile):
    """A peak's global prominence contour may enclose many already found peaks.

    Correct that measurement artifact at actual valleys before looking for
    missing paper. Otherwise splitting the giant contour double-counts the
    neighboring papers which already have their own observations.
    """
    output=[];events=[]
    for i,band in enumerate(bands):
        item=dict(band);center=int(round(band['center']))
        if i and item['top']<=bands[i-1]['center']:
            left=int(round(bands[i-1]['center']))
            item['top']=float(left+np.argmin(profile[left:center+1]))
        if i+1<len(bands) and item['bottom']>=bands[i+1]['center']:
            right=int(round(bands[i+1]['center']))
            item['bottom']=float(center+np.argmin(profile[center:right+1]))
        if item['top']!=band['top'] or item['bottom']!=band['bottom']:
            events.append(dict(band_id=band['id'],decision='rebound_at_observed_neighbor_valleys',
                original_bounds=[band['top'],band['bottom']],bounds=[item['top'],item['bottom']],
                count_changed=False))
        output.append(item)
    return output,events


def gap_observations(bands,raw):
    background=float(np.percentile(raw,2))
    cutoff=background+max(1.5,.015*float(np.percentile(raw,95)-background))
    output=[]
    for left,right in zip(bands,bands[1:]):
        a=max(0,int(np.ceil(left['bottom'])));b=min(len(raw),int(np.floor(right['top'])))
        if b<=a:a=int(left['center'])+1;b=int(right['center'])
        patch=raw[a:max(a+1,b)]
        output.append(dict(top=a,bottom=b,span=max(0,b-a),median_gray=float(np.median(patch)),
            black_fraction=float(np.mean(patch<=cutoff)),background_cutoff=cutoff))
    return output


def score_pair_phase(candidates,bands,profile,raw):
    """Learn ordered whole-unit shape from units next to observed empty gaps.

    Internal darkness is not the same as background. A long, mostly empty gap
    is an anchor separating units. Learn left/right lobe order and the full
    waveform from nearby complete candidates, then score all alternatives.
    No fixed 0/1 band parity is imposed across a missing observation.
    """
    if not candidates:return dict(enabled=True,reference_ids=[],gaps=[])
    gaps=gap_observations(bands,raw)
    median_width=float(np.median([b['bottom']-b['top'] for b in bands]))
    barriers={i for i,g in enumerate(gaps) if g['black_fraction']>.6 and g['span']>max(2.,.8*median_width)}
    references=[]
    for c in candidates:
        i,j=c['band_ids']
        if i in barriers:continue
        beside_background=(i>0 and i-1 in barriers) or j in barriers
        if beside_background and gaps[i]['black_fraction']<.5:
            references.append(c)
    # Insufficient observed anchors is genuine ambiguity, not permission to
    # force an arbitrary global pairing or choose a phase by total count.
    if len(references)<2:
        return dict(enabled=True,reference_ids=[c['id'] for c in references],
            reliable=False,gaps=gaps,background_barrier_indices=sorted(barriers),reason='insufficient_background_anchored_units')
    def descriptor(c):
        a,b=[bands[i] for i in c['band_ids']]
        widths=[a['bottom']-a['top'],b['bottom']-b['top']]
        return np.array([np.log(widths[0]/widths[1]),
            np.log(max(a['prominence'],1e-6)/max(b['prominence'],1e-6)),
            np.log(c['bottom']-c['top'])])
    def waveform(c):
        rows=np.linspace(c['top'],c['bottom'],64)
        z=np.interp(rows,np.arange(len(profile)),profile);z-=np.mean(z)
        return z/max(np.linalg.norm(z),1e-6)
    descriptors=np.array([descriptor(c) for c in references])
    center=np.median(descriptors,axis=0)
    spread=np.maximum(1.4826*np.median(np.abs(descriptors-center),axis=0),[.18,.3,.2])
    template=np.median([waveform(c) for c in references],axis=0)
    template-=template.mean();template/=max(np.linalg.norm(template),1e-6)
    reference_gray=float(np.median([gaps[c['band_ids'][0]]['median_gray'] for c in references]))
    # No penalty is based on absolute darkness alone: the paper in another
    # material may genuinely have a nearly black internal texture.
    for c in candidates:
        i,j=c['band_ids'];z=(descriptor(c)-center)/spread
        shape=float(np.exp(-.5*np.mean(np.minimum(z*z,9))))
        correlation=float(np.dot(waveform(c),template))
        wave=float(np.clip((correlation+.2)/1.2,0,1))
        g=gaps[i]
        interior=float(np.clip((g['median_gray']-g['background_cutoff'])/max(reference_gray-g['background_cutoff'],3.),0,1))
        penalty=.04 if i in barriers else 1.
        c['score_before_phase']=c['score']
        c['score']*=float((.15+.85*shape)*(.2+.8*wave)*(.15+.85*interior)*penalty)
        c['phase_rejected']=bool(i in barriers and (shape<.5 or correlation<.3))
        c['evidence']['pair_phase']=dict(shape_similarity=shape,waveform_correlation=correlation,
            interior_gray=g['median_gray'],background_fraction=g['black_fraction'],
            crosses_observed_background=i in barriers,reference_ids=[r['id'] for r in references])
    return dict(enabled=True,reliable=True,reference_ids=[c['id'] for c in references],
        gaps=gaps,background_barrier_indices=sorted(barriers),
        ordered_descriptor_median=center.tolist(),ordered_descriptor_scale=spread.tolist())


def review_thickness(bands,profile,raw,valid):
    """Large height is a search trigger; only measured internal valleys split it."""
    if len(bands)<5:return bands,[]
    reviewed=[];output=[];h,w=raw.shape
    smoothed=gaussian_filter1d(profile,.45)
    for index,band in enumerate(bands):
        neighbors=bands[max(0,index-6):index]+bands[index+1:index+7]
        heights=np.array([b['bottom']-b['top'] for b in neighbors])
        typical=float(np.median(heights));mad=float(1.4826*np.median(np.abs(heights-typical)))
        height=band['bottom']-band['top']
        if height<=max(1.65*typical,typical+3*mad):output.append(band);continue
        a=max(0,int(np.floor(band['top'])));b=min(h-1,int(np.ceil(band['bottom'])))
        local=smoothed[a:b+1]
        # Lower prominence is restricted to a suspicious *observed region*.
        # Search maxima/valleys on the actual image, never on a period grid.
        peaks,props=find_peaks(local,prominence=max(.8,.025*float(np.ptp(local))),
            distance=max(2,int(.45*typical)))
        peaks=peaks+a
        if len(peaks)<2:
            output.append(band);reviewed.append(dict(band_id=band['id'],height=height,
                local_typical_height=typical,decision='keep_thick_no_internal_boundary'))
            continue
        cuts=[];proof=[]
        for p,q in zip(peaks,peaks[1:]):
            if q-p<max(2.,.5*typical):continue
            cut=int(p+np.argmin(smoothed[p:q+1]))
            if cut<=p or cut>=q:continue
            left=raw[max(a,p-1):min(b,p+1)+1].max(axis=0)
            right=raw[max(a,q-1):min(b,q+1)+1].max(axis=0)
            dip=raw[max(a,cut-1):min(b,cut+1)+1].min(axis=0)
            depth=np.minimum(left,right)-dip
            needed=max(.7,.025*float(np.median(np.maximum(left,right))-np.median(dip)))
            supported=(depth>needed)&valid[p]&valid[q]&valid[cut]
            coverage=float(np.mean(supported))
            # A meaningful signal valley and broad real-column evidence are
            # both necessary. A thick smooth sheet with noise stays one box.
            normalized_depth=min(float(smoothed[p]),float(smoothed[q]))-float(smoothed[cut])
            if coverage>=.55 and normalized_depth>=max(.8,.04*float(np.ptp(local))):
                cuts.append(cut);proof.append(dict(y=cut,left_peak=int(p),right_peak=int(q),
                    column_support=coverage,raw_depth=float(np.median(depth)),normalized_depth=normalized_depth))
        if not cuts:
            output.append(band);reviewed.append(dict(band_id=band['id'],height=height,
                local_typical_height=typical,decision='keep_thick_internal_evidence_insufficient'))
            continue
        boundaries=[band['top'],*sorted(set(cuts)),band['bottom']]
        children=[]
        for left,right in zip(boundaries,boundaries[1:]):
            lo=max(a,int(np.ceil(left)));hi=min(b,int(np.floor(right)))
            if hi<=lo:continue
            peak=int(lo+np.argmax(smoothed[lo:hi+1]))
            # Use each child's measured intensity contour, not the entire
            # valley-to-valley interval (which includes inter-paper darkness).
            def contour(bound,direction):
                baseline=float(smoothed[bound]);level=baseline+.35*(float(smoothed[peak])-baseline)
                previous=peak
                for row in range(peak+direction,bound+direction,direction):
                    if smoothed[row]<=level:
                        delta=float(smoothed[previous]-smoothed[row])
                        fraction=float((smoothed[previous]-level)/max(delta,1e-9))
                        return float(previous+direction*fraction)
                    previous=row
                return float(bound)
            core_left=contour(lo,-1);core_right=contour(hi,1)
            child=dict(band,top=core_left,bottom=core_right,center=float(peak),
                parent_band_id=band['id'],thickness_split_evidence=proof)
            children.append(child)
        if len(children)<2:
            output.append(band);continue
        # The split pieces should also resemble ordinary observed paper
        # bodies. This prevents a faint texture dip in a wide bright body
        # from becoming two incompatible half-body boxes.
        def wave(left,right):
            z=np.interp(np.linspace(left,right,32),np.arange(h),smoothed)
            z-=z.mean();return z/max(np.linalg.norm(z),1e-6)
        reference_bands=[q for q in neighbors if .65*typical<q['bottom']-q['top']<1.5*typical]
        template=np.median([wave(q['top'],q['bottom']) for q in reference_bands],axis=0) if reference_bands else np.zeros(32)
        template-=template.mean();template/=max(np.linalg.norm(template),1e-6)
        similarities=[float(np.dot(wave(q['top'],q['bottom']),template)) for q in children]
        widths=[q['bottom']-q['top'] for q in children]
        if min(widths)<.6*typical or min(similarities)<.45 or np.median(similarities)<.65:
            output.append(band);reviewed.append(dict(band_id=band['id'],height=height,
                local_typical_height=typical,decision='keep_thick_child_structure_not_supported',
                child_template_similarities=similarities,child_heights=widths,boundaries=proof))
            continue
        for child,similarity in zip(children,similarities):
            child['thickness_child_similarity']=similarity
        output.extend(children)
        reviewed.append(dict(band_id=band['id'],height=height,local_typical_height=typical,
            decision='split_at_observed_internal_boundaries',children=len(children),boundaries=proof,
            child_template_similarities=similarities))
    output.sort(key=lambda b:b['center'])
    for i,b in enumerate(output):b.update(id=f'band_{i:05d}',index=i)
    return output,reviewed
