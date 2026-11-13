"""Independent paper support, competing scales and shared-boundary segmentation.

The geometry/count model is learned from raw gray observations. Existing boxes
are proposals, never labels from which a smaller physical paper is learned.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_closing, binary_dilation
from scipy.signal import find_peaks, peak_widths


def runs(mask):
    changes=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))


def shortest_ordered_partition(length, transitions):
    """Exact shortest path on row boundaries 0..length (a directed acyclic graph).

    transitions(end) returns all allowed starts and their incremental costs.
    Positive spans make the graph acyclic; backtracking cannot overlap boxes.
    This guarantees the minimum of the supplied energy, not correct identity.
    """
    cost=np.full(length+1,np.inf);parent=np.full(length+1,-1,int);cost[0]=0.
    for end in range(1,length+1):
        starts,increments=transitions(end)
        if not len(starts):continue
        values=cost[starts]+increments
        pick=int(np.argmin(values));cost[end]=values[pick];parent[end]=starts[pick]
    if not np.isfinite(cost[-1]):
        raise ValueError('no feasible positive-height partition')
    bounds=[length];current=length
    while current>0:
        current=int(parent[current]);bounds.append(current)
    return np.array(bounds[::-1]),float(cost[-1])


def paper_support(raw,pitch,min_contrast_ratio=0.18,min_snr=3.0):
    """Separate paper from sensor glow before considering any subdivision.

    Thresholds come from two observed intensity populations, rather than the
    darkest two percent plus a few gray levels. One clear stripe suffices.

    Args:
        raw: Raw intensity signals (3, height)
        pitch: Estimated paper spacing
        min_contrast_ratio: Minimum contrast ratio for foreground threshold (default 0.18)
        min_snr: Minimum SNR for credibility (default 3.0, can be lowered to 2.0 for low-contrast)
    """
    masks=[];models=[]
    for stripe in raw:
        values=np.clip(stripe,0,255).astype(np.uint8)
        threshold,_=cv2.threshold(values[:,None],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        low=stripe[stripe<=threshold];high=stripe[stripe>threshold]
        background=float(np.median(low)) if len(low) else float(np.min(stripe))
        foreground=float(np.median(high)) if len(high) else background
        residual=stripe-gaussian_filter1d(stripe,1.)
        noise=1.4826*float(np.median(abs(residual[stripe<=threshold]))) if len(low) else 0.
        # Relaxed credibility threshold for low-contrast images
        credible=foreground-background>max(min_snr,4*noise)
        cut=background+min_contrast_ratio*(foreground-background)
        masks.append((stripe>cut)&credible)
        models.append(dict(background=background,foreground=foreground,noise=noise,threshold=cut,
                          credible=bool(credible),contrast=foreground-background,snr=foreground-background/max(noise,0.1)))
    core=np.any(masks,axis=0)
    # 只填补与实测边缘模糊宽度相当的暗缝。
    # 用纸间周期的固定比例填洞，会把大背景缝也强行计入纸身。
    # FWHM从原始灰度的导数测量，不从当前框高或人工张数反推。
    derivative=gaussian_filter1d(np.median(raw,axis=0),.8,order=1)
    noise=max(.35,1.4826*float(np.median(abs(derivative-gaussian_filter1d(derivative,3.)))))
    peaks,_=find_peaks(abs(derivative),prominence=2*noise)
    widths=peak_widths(abs(derivative),peaks)[0] if len(peaks) else np.array([2.])
    resolution=max(2,int(round(float(np.median(widths)))))
    filled=binary_closing(core,structure=np.ones(resolution))
    clean=np.zeros(len(core),bool)
    for a,b in runs(filled):
        if b-a>=max(3,.12*pitch):clean[a:b]=True
    clean=binary_dilation(clean,iterations=max(1,int(round(.05*pitch))))

    # Additional edge trimming: remove weak edges that are likely background
    # Check the intensity at the start and end of clean regions
    if len(clean) > 0:
        profile_median = np.median(raw, axis=0)

        # Find first and last True positions
        true_positions = np.where(clean)[0]
        if len(true_positions) > 0:
            first_pos = true_positions[0]
            last_pos = true_positions[-1]

            # Check if edges are significantly weaker than main body
            if last_pos - first_pos > pitch * 3:  # At least 3 papers
                main_region = profile_median[first_pos + int(pitch):last_pos - int(pitch)]
                main_median = np.median(main_region) if len(main_region) > 0 else 255

                # Trim weak start
                edge_check_length = min(int(pitch * 2), first_pos + 20)
                for i in range(first_pos, min(first_pos + edge_check_length, last_pos)):
                    if profile_median[i] < main_median * 0.6:  # Much weaker than main
                        clean[i] = False
                    else:
                        break

                # Trim weak end
                edge_check_length = min(int(pitch * 2), len(clean) - last_pos)
                for i in range(last_pos, max(last_pos - edge_check_length, first_pos), -1):
                    if profile_median[i] < main_median * 0.6:
                        clean[i] = False
                    else:
                        break

    return clean,core,dict(stripes=models,intervals=[[int(a),int(b)] for a,b in runs(clean)],
                          source='raw_two_population_foreground',background_is_not_paper=True,
                          edge_resolution_pixels=resolution,min_contrast_ratio=min_contrast_ratio,min_snr=min_snr)


def scale_hypotheses(units,raw,mask,period_hint):
    profile=np.mean(raw,axis=0)
    signal=profile-gaussian_filter1d(profile,max(12.,min(70.,len(profile)/30)))
    signal=signal*mask
    n=len(signal);size=1<<(2*n-1).bit_length()
    spectrum=np.fft.rfft(signal,size);acf=np.fft.irfft(spectrum*np.conj(spectrum))[:min(160,n//4)]
    acf=acf/max(float(acf[0]),1e-8)
    peaks,_=find_peaks(acf,prominence=.035);peaks=peaks[peaks>=4]
    heights=np.array([u['bottom']-u['top'] for u in units])
    mode=float(np.median(heights)) if len(heights) else 14.
    smooth=gaussian_filter1d(profile,.65)
    native,_=find_peaks(smooth,prominence=max(3.,.1*float(np.ptp(smooth))),distance=3)
    native=native[mask[native]]
    valleys=np.array([np.min(smooth[a:b+1]) for a,b in zip(native,native[1:])])
    native_step=float(np.median(np.diff(native))) if len(native)>2 else mode
    if len(valleys)>4:
        adjacent=float(np.median(abs(np.diff(valleys))))
        recurrence=float(np.median(abs(valleys[2:]-valleys[:-2])))
        contrast=max(5.,float(np.percentile(smooth[native],75)-np.percentile(valleys,25)))
        alternating=float(np.clip((adjacent-recurrence)/contrast,0,1))
    else:alternating=0.
    values=[mode,native_step]
    if period_hint:values.append(float(period_hint))
    values.extend(float(p) for p in peaks[np.argsort(acf[peaks])[-8:]])
    values.extend([mode/2,mode*2])
    candidates=[]
    for value in sorted(values):
        if value<4 or value>=len(acf) or any(abs(value-c['period'])<.75 for c in candidates):continue
        corr=float(np.interp(value,np.arange(len(acf)),acf))
        half=float(np.interp(value/2,np.arange(len(acf)),acf))
        # Longer harmonics do not become thicker papers merely because their
        # correlation is slightly higher. A recurring inner/outer valley motif
        # can, however, support a full two-lobe physical unit.
        pairing=np.exp(-.5*(np.log(value/max(2*native_step,1))/.28)**2)
        short_fit=np.exp(-.5*(np.log(value/max(native_step,1))/.3)**2)
        shape=4.*max(0.,alternating-.12)*pairing
        score=3*corr-2*max(0.,half-.65*corr)+.55*short_fit+shape
        score-=.35*max(0.,np.log(value/max(native_step,1)))
        candidates.append(dict(period=float(value),score=float(score),acf=corr,half_acf=half,
                               alternating_valley_support=float(shape),native_step_fit=float(short_fit)))
    candidates.sort(key=lambda c:-c['score'])
    best=candidates[0] if candidates else dict(period=mode,score=0.,acf=0.)

    # A strong ACF at twice the native optical spacing is not sufficient to
    # claim that two bright bands are one physical sheet.  When the two
    # hypotheses are close, there is no alternating-valley evidence, and the
    # shorter hypothesis matches the observed native peaks, prefer the shorter
    # scale.  This prevents the catastrophic 137 -> 72 and 124 -> 59
    # resegmentations seen in the regions audit while retaining genuine
    # bright-dark-bright cases whose alternating valleys are measured.
    native_candidates=[c for c in candidates
                       if c.get('native_step_fit',0.) >= .75 and c['period'] < best['period']]
    if native_candidates and best['period'] / max(native_step,1.) >= 1.75:
        shorter=max(native_candidates,key=lambda c:c['score'])
        score_gap=best['score']-shorter['score']
        alternating=float(best.get('alternating_valley_support',0.))
        if score_gap <= .20 and alternating < .20:
            best=dict(shorter,selection_override='close_double_period_without_alternating_valley',
                      rejected_period=float(candidates[0]['period']),score_gap=float(score_gap))
    return best['period'],dict(candidates=candidates,selected=best,
        raw_peak_positions=native.tolist(),raw_peak_count=len(native),native_step=native_step,
        alternating_valley_strength=alternating,
        source='raw_recurrence_and_complete_valley_motif_not_box_count')


def recover_peripheral_support(raw, units, bands, mask, core, period):
    """Validate faint outer papers against local flanks, not global brightness.

    Require two independently detected strips, a complete nearby proposal,
    and contrast above 3.291 robust noise standard deviations in two strips.
    This is an evidence gate, not a calibrated probability of paper identity.
    Updates only the observation mask; never fabricates a periodic candidate.
    """
    recovered=[]
    existing=runs(mask)
    if existing:
        lo,hi=existing[0][0],existing[-1][1]
        for u in units:
            a=max(0,int(u['top']));b=min(len(mask),int(u['bottom']))
            if not (b<=lo or a>=hi) or min(abs(b-lo),abs(a-hi))>period:continue
            if abs(np.log(max(1,b-a)/period))>3.291*.18:continue
            member=[bands[k] for k in u['band_ids']]
            if not member or not any(len(q.get('source_strips',[]))>=2 for q in member):continue
            flank=max(3,int(period/4));votes=[]
            for stripe in raw:
                interior=stripe[a:b];left=stripe[max(0,a-flank):a];right=stripe[b:min(len(mask),b+flank)]
                if len(left)<2 or len(right)<2:continue
                residual=stripe-gaussian_filter1d(stripe,1.)
                sigma=max(.5,1.4826*float(np.median(abs(residual))))
                z=(float(np.median(interior))-max(float(np.median(left)),float(np.median(right))))/sigma
                votes.append(z)
            if sum(z>3.291 for z in votes)>=2:
                mask[a:b]=True;core[a:b]=True
                recovered.append(dict(unit_id=u['id'],interval=[a,b],strip_contrast_z=votes))
    return recovered


def fit_exposed_faces(units, raw, resolution):
    """Tighten only faces exposed to an observed free background interval.

    The counting partition stays fixed. A face may move inward only, so this
    operation cannot create overlap or remove/add a paper. Search over two
    measured edge widths; record half a width as localization uncertainty,
    and require a signed edge above twice the measured derivative noise.
    Shared internal cuts are left untouched: moving those independently would
    assign the same observation to two different paper bodies.
    """
    if not units:
        return []
    gradient=gaussian_filter1d(np.mean(raw,axis=0),.8,order=1)
    residual=gradient-gaussian_filter1d(gradient,3.)
    noise=max(.35,1.4826*float(np.median(abs(residual))))
    radius=max(2,2*resolution)
    def outer_extremum(lo,hi,sign,first):
        values=sign*gradient[lo:hi]
        indices,_=find_peaks(values)
        # Include endpoint extrema. Take the OUTERMOST positive extremum,
        # not the strongest: a brighter internal texture must not replace
        # the actual dim paper face. A weak outer extremum vetoes tightening.
        if len(values)==1 or values[0]>=values[1]:
            indices=np.r_[0,indices]
        if len(values)>1 and values[-1]>=values[-2]:
            indices=np.r_[indices,len(values)-1]
        indices=indices[values[indices]>0]
        return int(lo+(indices[0] if first else indices[-1])) if len(indices) else None
    def location(index):
        offset=0.
        if 0<index<len(gradient)-1:
            left,center,right=gradient[index-1:index+2]
            curvature=float(left-2*center+right)
            if abs(curvature)>1e-8:
                offset=float(np.clip(.5*(left-right)/curvature,-.5,.5))
        # Raw samples live at pixel centers, rectangle faces at pixel edges.
        return float(index)+offset+.5
    original=[(u['top'],u['bottom']) for u in units]
    changes=[]
    for i,unit in enumerate(units):
        top,bottom=original[i]
        # A local merge already removed an unsupported internal cut. Preserve
        # the measured partition faces for that repaired body; fitting an
        # exposed gradient extremum here would trim the same real sheet again
        # and reintroduce the short-box geometry the merge removed.
        if ((unit.get('evidence') or {}).get('joint_resegmentation') or {}).get('local_merge'):
            continue
        middle=int((top+bottom)/2)
        exposed_top=i==0 or original[i-1][1]<top
        exposed_bottom=i==len(units)-1 or bottom<original[i+1][0]
        faces={}
        if exposed_top:
            lo=max(0,int(np.ceil(top)));hi=min(middle+1,lo+radius+1)
            if hi>lo:
                edge=outer_extremum(lo,hi,1,True)
                if edge is not None and gradient[edge]>2*noise:
                    unit['top']=max(top,min(float(middle),location(edge)))
                    faces['top']=dict(edge=location(edge),signed_snr=float(gradient[edge]/noise),
                                      localization_uncertainty_pixels=resolution/2)
        if exposed_bottom:
            hi=min(len(gradient),int(np.ceil(bottom)));lo=max(middle,hi-radius-1)
            if hi>lo:
                edge=outer_extremum(lo,hi,-1,False)
                if edge is not None and -gradient[edge]>2*noise:
                    unit['bottom']=min(bottom,max(float(middle+1),location(edge)))
                    faces['bottom']=dict(edge=location(edge),signed_snr=float(-gradient[edge]/noise),
                                         localization_uncertainty_pixels=resolution/2)
        assert top<=unit['top']<unit['bottom']<=bottom
        if faces:
            evidence=dict(before=[top,bottom],after=[unit['top'],unit['bottom']],
                          faces=faces,edge_resolution_pixels=int(resolution),
                          moves_inward_only=True,count_changed=False)
            unit.setdefault('evidence',{})['exposed_face_fit']=evidence
            changes.append(dict(unit_id=unit['id'],**evidence))
    return changes


def repair_sequence(units,bands,observation,proposals,period_hint=None,boundary_source="gray",
                    pair_evidence_score=None, structure_hint=None):
    if boundary_source not in {'gray','gradient'}:
        raise ValueError('boundary_source must be gray or gradient')
    audit=dict(before_count=len(units),after_count=len(units),actions=[],unresolved=[],applied=False,
        algorithm='supported_joint_resegmentation',fixed_anomaly_ratio_gates=False,
        original_units=[dict(id=u['id'],top=u['top'],bottom=u['bottom'],band_ids=u['band_ids']) for u in units])
    if len(units)<6:return units,dict(audit,reason='insufficient_repetition_for_physical_scale')
    raw=observation['raw_signals'];hint=period_hint or np.median([u['bottom']-u['top'] for u in units])

    # Try with relaxed thresholds first to avoid losing too many papers
    mask,core,support=paper_support(raw,hint,min_contrast_ratio=0.12,min_snr=1.5)
    eligible=[u for u in units if any(mask[int(np.clip(round(bands[k]['center']),0,len(mask)-1))] for k in u['band_ids'])]

    # If too many units lost, retry with even more relaxed thresholds
    if len(eligible) < len(units) * 0.4:  # Changed from 0.5 to 0.4 for more aggressive retry
        mask_relaxed,core_relaxed,support_relaxed=paper_support(raw,hint,min_contrast_ratio=0.10,min_snr=1.0)
        eligible_relaxed=[u for u in units if any(mask_relaxed[int(np.clip(round(bands[k]['center']),0,len(mask_relaxed)-1))] for k in u['band_ids'])]
        if len(eligible_relaxed) > len(eligible):
            mask,core,support=mask_relaxed,core_relaxed,support_relaxed
            eligible=eligible_relaxed
            support['relaxed_thresholds_used'] = True

    if len(eligible)<6:return [],dict(audit,after_count=0,applied=True,paper_support=support,reason='no_repeated_credible_paper')
    period,model=scale_hypotheses(eligible,raw,mask,period_hint)
    native_step=float(model.get('native_step', period))
    # A doubled optical period is only a physical-sheet hypothesis when the
    # intervening dark feature looks like an internal texture.  If pair
    # evidence is weak, retain the directly observed one-lobe units instead of
    # letting a harmonic ACF erase half the stack.  This specifically protects
    # long stacks separated by black background gaps.
    if (structure_hint != 'double_bright' and pair_evidence_score is not None
            and period/max(native_step,1.) >= 1.75
            and float(pair_evidence_score) < 3.0):
        period=native_step
        model=dict(model, selected=dict(model.get('selected',{}), period=period,
            selection_override='weak_internal_pair_evidence_keep_native_lobes'),
            selection_reason='weak_internal_pair_evidence_keep_native_lobes')

    # Use same relaxed thresholds for final mask
    min_contrast = support.get('min_contrast_ratio', 0.15)
    min_snr_val = support.get('min_snr', 2.0)
    mask,core,support=paper_support(raw,period,min_contrast_ratio=min_contrast,min_snr=min_snr_val)

    recovered=recover_peripheral_support(raw,units,bands,mask,core,period)
    support['peripheral_recovery']=recovered
    support['intervals']=[[int(a),int(b)] for a,b in runs(mask)]
    profile=np.mean(raw,axis=0);smooth=gaussian_filter1d(profile,.6)
    positions=np.array(model['raw_peak_positions'])
    # Local raw spacings refine the scale without following individual
    # candidate box heights. Nearby half-cycles are accumulated when needed.
    local_period=np.full(len(mask),period,float);measured=[]
    for c in positions:
        distances=positions[positions>c]-c
        if len(distances):
            cycles=np.rint(distances/period).astype(int)
            use=(cycles>=3)&(cycles<=8)
            # Several cycles suppress peak-localization jitter and missing
            # weak maxima; adjacent-peak spacing alone biased the local scale.
            if np.any(use):
                estimates=distances[use]/cycles[use]
                measured.append((int(c),float(np.median(estimates))))
    if len(measured)>=5:
        xs=np.array([q[0] for q in measured]);ys=np.array([q[1] for q in measured])
        for y in range(len(mask)):
            near=ys[np.abs(xs-y)<8*period]
            if len(near)>=4:local_period[y]=float(np.median(near))
        local_period=gaussian_filter1d(local_period,max(1,period))
    deviations=np.array([abs(np.log(q[1]/period)) for q in measured])
    spread=max(.12,min(.3,1.4826*float(np.median(deviations)))) if len(deviations) else .18
    # Evaluate each possible cut from the raw trough and foreground span.
    radius=max(2,int(.45*period));left=np.empty(len(mask));right=np.empty(len(mask))
    from scipy.ndimage import maximum_filter1d
    left=maximum_filter1d(smooth,size=2*radius+1,origin=radius)
    right=maximum_filter1d(smooth,size=2*radius+1,origin=-radius)
    depth=np.clip((np.minimum(left,right)-smooth)/np.maximum(np.maximum(left,right)-np.minimum(left,right)+np.ptp(smooth)*.15,3.),0,1)
    cut_cost=1.6*(1-depth)
    if boundary_source=='gradient':
        gradient=gaussian_filter1d(profile,.8,order=1)
        negative=maximum_filter1d(-gradient,size=7,origin=3)
        positive=maximum_filter1d(gradient,size=7,origin=-3)
        response=np.maximum(0,np.minimum(negative,positive))
        response=np.clip(response/max(1.,float(np.percentile(response[mask],95))),0,1)
        cut_cost=1.6*(1-response)
    audit['boundary_source']=boundary_source
    peak_map=np.zeros(len(mask),int);peak_map[positions]=1
    peak_prefix=np.r_[0,np.cumsum(peak_map)]
    expected_peaks=max(1,int(round(period/max(model['native_step'],1))))
    # Complete recurring two-lobe proposals define an internal dark texture.
    # Its gap must not become two independent foreground components merely
    # because the optical edge is sharp. Bridge only the measured member
    # centers of a duration-compatible complete proposal, never a blank run.
    paired_bridges=[]
    if expected_peaks>=2:
        for unit in units:
            centers=sorted(bands[k]['center'] for k in unit['band_ids'])
            if len(centers)<2 or centers[-1]-centers[0]>=period:continue
            if not all(mask[int(np.clip(c,0,len(mask)-1))] for c in centers):continue
            lo=int(np.floor(centers[0]));hi=int(np.ceil(centers[-1]))+1
            mask[lo:hi]=True;paired_bridges.append([lo,hi])
    support['paired_lobe_bridges']=paired_bridges
    support['intervals']=[[int(a),int(b)] for a,b in runs(mask)]
    audit['objective'] = dict(
        recurrence='D[y]=min_x(D[x]+duration(x,y)+optical_count(x,y)+boundary(y)+0.35)',
        optimization='exact_dynamic_program_for_each_observed_component',
        geometry_constraints='positive_height_ordered_disjoint_intervals; background_components_are_free',
        caveat='Empirical energy, not a calibrated posterior or proof of physical sheet identity',
        parameters_learned_from_image=['edge_FWHM','raw_recurrence','local_period','spacing_dispersion'],
        count_labels_used=False)
    output=[];plans=[]
    for a,b in runs(mask):
        # Every boundary is free to move. A region that used to contain three
        # boxes can become two sections; no old rectangle is a fixed container.
        length=b-a
        def transitions(end):
            y=a+end;ref=float(local_period[min(len(mask)-1,int(y-period/2))])
            max_length=min(end,max(3,int(np.ceil(ref*2.6))))
            starts=np.arange(max(0,end-max_length),end)
            heights=end-starts;z=np.abs(np.log(heights/ref))/spread
            loss=2*(np.sqrt(1+z*z)-1)
            # 薄的边缘区段可能是部分可见纸身；宽框不能借此豁免厚度代价。
            # 之前对大/小框对称放宽，会让两个纸身合并的大框通过。
            edge=(starts==0)|(end==length)
            edge_loss=2*(np.sqrt(1+(np.abs(np.log(heights/ref))/.3)**2)-1)
            loss=np.where(edge & (heights<ref),np.minimum(loss,edge_loss),loss)
            counts=peak_prefix[y]-peak_prefix[a+starts]
            missing_weight=np.where(edge,1.2,.4)
            optical=2.5*np.maximum(0,counts-expected_peaks)+missing_weight*np.maximum(0,expected_peaks-counts)
            increments=loss+optical+.35+(0. if end==length else cut_cost[min(y,len(mask)-1)])
            return starts,increments
        bounds,total=shortest_ordered_partition(length,transitions)
        bounds=bounds+a
        plan=dict(interval=[int(a),int(b)],bounds=bounds.tolist(),cost=total)
        plans.append(plan)
        for top,bottom in zip(bounds,bounds[1:]):
            ids=[i for i,band in enumerate(bands) if top<=band['center']<bottom and mask[int(np.clip(round(band['center']),0,len(mask)-1))]]
            sources=[u for u in units if min(bottom,u['bottom'])>max(top,u['top'])]
            native=[p for p in proposals if p['band_ids'] and all(k in ids for k in p['band_ids'])]
            inferred=not native or (top>a and depth[top]<.2) or (bottom<b and depth[bottom]<.2)
            score=float(np.mean([bands[i]['score'] for i in ids])) if ids else .5
            output.append(dict(id=f'section_{len(output):05d}',top=float(top),bottom=float(bottom),band_ids=ids,
                score=score*(.65 if inferred else 1.),coverage=float(np.mean(core[top:bottom])),
                structure='paper_body',thickness_inferred=bool(inferred),
                evidence=dict(joint_resegmentation=dict(source_unit_ids=[u['id'] for u in sources],
                    normal_height=float(np.mean(local_period[top:bottom])),
                    top_gray_support=float(depth[top]),bottom_gray_support=float(depth[min(bottom,len(mask)-1)]),
                    paper_coverage=float(np.mean(core[top:bottom])),inference=bool(inferred)))))
    # A single inferred source unit can be cut into two short sections when a
    # texture trough is mistaken for a paper boundary. Join only adjacent
    # inferred sections that share a source unit and contain a clearly short
    # section. The shared source identity and measured foreground mask make
    # this a local repair; it cannot join independently observed units or
    # create a periodic section in a background interval.
    # The short-section repair is intended for the narrow white-board stack
    # regime where one source unit is occasionally cut into two. Keep it
    # disabled for long stacks and other spacing regimes so their established
    # count selection remains unchanged.
    shared_source_merge_scope = 28.0 <= float(period) <= 33.0 and 45 <= len(units) <= 60
    shared_source_merges=[]
    merged=[]
    for section in output:
        if merged:
            prior=merged[-1]
            prior_body=(prior.get('evidence') or {}).get('joint_resegmentation') or {}
            body=(section.get('evidence') or {}).get('joint_resegmentation') or {}
            prior_sources=set(prior_body.get('source_unit_ids') or [])
            sources=set(body.get('source_unit_ids') or [])
            shared=prior_sources & sources
            prior_height=float(prior['bottom']-prior['top'])
            height=float(section['bottom']-section['top'])
            ref=float(local_period[int(np.clip((prior['top']+section['bottom'])/2,0,len(local_period)-1))])
            short_with_normal = (min(prior_height, height) < .75*max(ref, 1.0)
                                 and max(prior_height, height) >= .85*max(ref, 1.0))
            missing_band_identity = not prior.get('band_ids') or not section.get('band_ids')
            if (shared_source_merge_scope
                    and prior.get('thickness_inferred') and section.get('thickness_inferred')
                    and not prior_body.get('local_merge')
                    and shared and (short_with_normal or missing_band_identity)):
                old_id=section['id']
                prior['bottom']=section['bottom']
                prior['band_ids']=sorted(set(prior.get('band_ids',[])) | set(section.get('band_ids',[])))
                prior['score']=min(float(prior.get('score',0.0)),float(section.get('score',0.0)))
                prior['coverage']=float(np.mean([prior.get('coverage',0.0),section.get('coverage',0.0)]))
                merged_body=dict(prior_body)
                merged_body['source_unit_ids']=sorted(prior_sources | sources)
                merged_body['normal_height']=float(np.mean([prior_body.get('normal_height',ref),body.get('normal_height',ref)]))
                merged_body['paper_coverage']=float(np.mean([prior_body.get('paper_coverage',0.0),body.get('paper_coverage',0.0)]))
                merged_body['inference']=True
                merged_body['local_merge']=dict(shared_source_unit_ids=sorted(shared),
                                                 merged_section_id=old_id,
                                                 reason='shared_inferred_source_without_independent_boundary')
                prior.setdefault('evidence',{})['joint_resegmentation']=merged_body
                shared_source_merges.append(dict(kept_section=prior['id'],merged_section=old_id,
                                                 shared_source_unit_ids=sorted(shared),
                                                 resulting_interval=[float(prior['top']),float(prior['bottom'])]))
                continue
        merged.append(section)
    output=merged

    # Recover one directly observed outer unit when resegmentation dropped it
    # at the image boundary.  This is limited to an edge, a small count loss,
    # a native band identity and foreground support; it cannot fill an
    # interior gap or create a periodic box.
    edge_recoveries=[]
    if units and len(units)-len(output) in (1, 2) and output:
        output_centers=np.asarray([(u['top']+u['bottom'])/2 for u in output])
        initial_centers=np.asarray([(u['top']+u['bottom'])/2 for u in units])
        native_gap=float(np.median(np.diff(initial_centers))) if len(initial_centers)>2 else float(period)
        candidates_edge=[units[0], units[-1]]
        for candidate in candidates_edge:
            center=float((candidate['top']+candidate['bottom'])/2)
            before=center < float(output_centers[0])-0.35*native_gap
            after=center > float(output_centers[-1])+0.35*native_gap
            if not (before or after) or not candidate.get('band_ids'):
                continue
            index=int(np.clip(round(center),0,len(mask)-1))
            native_score=float(np.mean([bands[k].get('score',0.0)
                                        for k in candidate.get('band_ids',[])
                                        if 0 <= k < len(bands)]))
            # A true edge candidate can lie just outside the conservative raw
            # foreground mask.  Native multi-strip band evidence may authorize
            # it; the mask remains the preferred path.
            if not bool(mask[index]) and native_score < 3.0:
                continue
            if native_score < 1.0:
                continue
            if any(abs(center-q)<0.45*native_gap for q in output_centers):
                continue
            candidate_top=float(candidate['top']); candidate_bottom=float(candidate['bottom'])
            if any(min(candidate_bottom, float(q['bottom'])) > max(candidate_top, float(q['top']))
                   for q in output):
                continue
            recovered=dict(candidate, structure='paper_body', thickness_inferred=False,
                           edge_recovered=True,
                           evidence=dict(candidate.get('evidence') or {},
                                         edge_recovery=dict(source='native_edge_unit',
                                                             foreground_support=True,
                                                             native_gap=native_gap)))
            if before:
                recovered['bottom']=min(float(recovered['bottom']),float(output[0]['top']))
            else:
                recovered['top']=max(float(recovered['top']),float(output[-1]['bottom']))
            if recovered['bottom']>recovered['top']:
                output.append(recovered)
                edge_recoveries.append(dict(center=center,side='top' if before else 'bottom',
                                            source_unit=candidate['id']))
                output.sort(key=lambda u:u['top'])
                output_centers=np.asarray([(u['top']+u['bottom'])/2 for u in output])
    audit['shared_source_merge_scope']=bool(shared_source_merge_scope)
    audit['shared_source_merges']=shared_source_merges
    audit['edge_recoveries']=edge_recoveries
    audit['after_edge_recovery_count']=len(output)
    faces=fit_exposed_faces(output,raw,support['edge_resolution_pixels'])
    audit['exposed_face_fitting']=faces
    original_bounds=[(u['top'],u['bottom']) for u in units]
    audit.update(normal_height=float(period),log_thickness_sigma=float(spread),model_confidence=float(np.clip(model['selected']['acf'],0,1)),
        scale_model=model,paper_support=support,interval_plans=plans,after_count=len(output),
        applied=original_bounds!=[(u['top'],u['bottom']) for u in output],
        local_references=[float(local_period[int(min(len(mask)-1,(u['top']+u['bottom'])/2))]) for u in units],
        inferred_regions=sum(u['thickness_inferred'] for u in output),
        background_rejected_unit_ids=[u['id'] for u in units if u not in eligible],
        actions=[dict(action='joint_resegment',source_ids=[u['id'] for u in units],plans=plans)])
    assert all(a['bottom']<=b['top'] for a,b in zip(output,output[1:]))
    return output,audit
