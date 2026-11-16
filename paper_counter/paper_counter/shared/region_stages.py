# -*- coding: utf-8 -*-
"""Image-region counting with a run-wide, explicit structural unit.

Bright bands are observations, not counts. In double_bright mode, only a
measured two-lobe region can become a counted box. Unmatched lobes remain
unresolved; there is no local fallback to counting a single lobe as a sheet.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter,maximum_filter1d,minimum_filter1d
from scipy.signal import find_peaks,peak_widths
from scipy.stats import trim_mean
from ..models import PaperCountResult
from ..preprocessing import to_gray,normalize_illumination
from ..geometry_alignment import align_columns
from ..period import estimate_period
from ..region_evidence import score_pair_phase,review_thickness,limit_to_observed_neighbors


from dataclasses import dataclass


@dataclass
class OpticalRegionEvidence:
    """Optical hypotheses shared as evidence, not final counted paper identities."""
    optical_units: list
    bands: list
    observation: dict | None
    proposals: list
    pitch: float | None
    pair_evidence_score: float | None
    structure: str
    identity: dict
    aligned_raw: np.ndarray
    valid: np.ndarray


@dataclass
class RegionDecision:
    units: list
    structure: str
    audit: dict | None


def vertical_iou(a,b):
    """IoU in the aligned image; all proposals span the same crop width."""
    overlap=max(0.,min(a['bottom'],b['bottom'])-max(a['top'],b['top']))
    return overlap/max(a['bottom']-a['top']+b['bottom']-b['top']-overlap,1e-9)


def region_nms(candidates,threshold=.65):
    """Remove duplicate regions, not two different lobes of a structure."""
    kept=[];suppressed=[]
    for item in sorted(candidates,key=lambda c:(-c['score'],c['top'],c['bottom'])):
        owner=next((k for k in kept if vertical_iou(item,k)>threshold),None)
        if owner is None:kept.append(item)
        else:suppressed.append(dict(candidate=item['id'],owner=owner['id'],reason='duplicate_region_iou'))
    return sorted(kept,key=lambda c:c['top']),suppressed


def select_nonconflicting(candidates):
    """Weighted interval selection: one observed lobe cannot belong to two units.

    A free background interval has no cost. We never demand a tiled stack,
    insert a periodic box, or favor a desired total count.
    """
    ordered=sorted(candidates,key=lambda c:(max(c['band_ids']),min(c['band_ids'])))
    values=[0.];solutions=[[]]
    for i,candidate in enumerate(ordered):
        first=min(candidate['band_ids'])
        predecessor=next((j for j in range(i-1,-1,-1) if max(ordered[j]['band_ids'])<first),-1)
        take=values[predecessor+1]+candidate['score']
        if take>values[-1]:values.append(take);solutions.append(solutions[predecessor+1]+[candidate])
        else:values.append(values[-1]);solutions.append(solutions[-1])
    return sorted(solutions[-1],key=lambda c:c['top'])


def extract_bands(aligned,raw,valid,max_period):
    """Multiscale gray-ridge proposals, validated by real column contrast."""
    h,w=aligned.shape;profile=trim_mean(aligned,.1,axis=1).astype(np.float32)
    centered=profile-gaussian_filter1d(profile,15.)
    pitch,strength,_=estimate_period(centered,3,max_period,.12,.45)
    scale=pitch or 14.
    window=max(15,int(6*scale)|1)
    amplitude=maximum_filter1d(profile,window)-minimum_filter1d(profile,window)
    noise=1.4826*median_filter(np.abs(profile-gaussian_filter1d(profile,.65)),size=window)
    raw_background=float(np.percentile(raw,2))
    raw_floor=raw_background+max(2.,.03*float(np.percentile(raw,95)-raw_background))
    columns=np.arange(w);proposals=[]
    for sigma in (.45,.8,1.2):
        signal=gaussian_filter1d(profile,sigma)
        peaks,props=find_peaks(signal,prominence=np.maximum(1.5,np.maximum(2.5*noise,.1*amplitude)),distance=2)
        widths,_,lefts,rights=peak_widths(signal,peaks,rel_height=.65)
        for index,(peak,left,right,width,prom) in enumerate(zip(peaks,lefts,rights,widths,props['prominences'])):
            if width<.7:continue
            # Only compare to this band's flanks; a neighboring bright band
            # must not turn a real bright-dark-bright structure into a failure.
            a=max(0,int(np.floor(left)));b=min(h-1,int(np.ceil(right)))
            radius=max(1,min(int(.2*scale),max(1,(b-a)//2)))
            lo=max(a,int(peak)-radius);hi=min(b,int(peak)+radius)
            rows=lo+np.argmax(aligned[lo:hi+1],axis=0)
            left_row=np.maximum(0,np.minimum(rows-1,a));right_row=np.minimum(h-1,np.maximum(rows+1,b))
            contrast=raw[rows,columns]-np.maximum(raw[left_row,columns],raw[right_row,columns])
            observed=(contrast>1.0)&(raw[rows,columns]>raw_floor)&valid[rows,columns]
            coverage=float(observed.mean())
            if coverage<.3:continue
            proposals.append(dict(id=f'band_proposal_{len(proposals)}',top=float(left),bottom=float(right),
                center=float(peak),prominence=float(prom),coverage=coverage,
                score=float(coverage*np.log1p(prom/max(1.5,noise[peak]))),scale=float(sigma),
                signal_source='normalized_gray'))

    # The normalized image is good at suppressing illumination, but it can
    # flatten a genuinely bright paper face when the surrounding texture is
    # also bright.  Add a second proposal channel from raw local contrast.
    # Raw candidates still pass the same original-gray column test, so this
    # channel can recover a missed sheet without turning texture into a count.
    raw_profile=trim_mean(raw,.1,axis=1).astype(np.float32)
    raw_scale=max(8.,1.5*scale)
    raw_centered=raw_profile-gaussian_filter1d(raw_profile,raw_scale)
    raw_noise=max(.25,1.4826*float(np.median(np.abs(
        raw_centered-gaussian_filter1d(raw_centered,1.0)))))
    raw_amplitude=maximum_filter1d(raw_profile,window)-minimum_filter1d(raw_profile,window)
    for sigma in (.6,1.0):
        signal=gaussian_filter1d(raw_centered,sigma)
        threshold=np.maximum(1.0,np.maximum(2.0*raw_noise,.06*raw_amplitude))
        peaks,props=find_peaks(signal,prominence=threshold,distance=2)
        widths,_,lefts,rights=peak_widths(signal,peaks,rel_height=.65)
        for peak,left,right,width,prom in zip(peaks,lefts,rights,widths,props['prominences']):
            if width<.7:continue
            a=max(0,int(np.floor(left)));b=min(h-1,int(np.ceil(right)))
            radius=max(1,min(int(.2*scale),max(1,(b-a)//2)))
            lo=max(a,int(peak)-radius);hi=min(b,int(peak)+radius)
            rows=lo+np.argmax(aligned[lo:hi+1],axis=0)
            left_row=np.maximum(0,np.minimum(rows-1,a));right_row=np.minimum(h-1,np.maximum(rows+1,b))
            contrast=raw[rows,columns]-np.maximum(raw[left_row,columns],raw[right_row,columns])
            observed=(contrast>1.0)&(raw[rows,columns]>raw_floor)&valid[rows,columns]
            coverage=float(observed.mean())
            if coverage<.3:continue
            proposals.append(dict(id=f'band_proposal_{len(proposals)}',top=float(left),bottom=float(right),
                center=float(peak),prominence=float(prom),coverage=coverage,
                score=float(coverage*np.log1p(prom/max(1.0,raw_noise))),scale=float(sigma),
                signal_source='raw_local_contrast'))
    bands,suppressed=region_nms(proposals,.55)
    # Different widths around the same gray maximum can have small IoU. They
    # still describe one band and must not later form a false two-lobe unit.
    unique=[]
    for band in sorted(bands,key=lambda c:-c['score']):
        if not any(abs(band['center']-q['center'])<max(1.5,.18*min(band['bottom']-band['top'],q['bottom']-q['top'])) for q in unique):
            unique.append(band)
    bands=sorted(unique,key=lambda c:c['center'])
    for i,band in enumerate(bands):band.update(id=f'band_{i:05d}',index=i)
    return bands,profile,pitch,float(strength),suppressed


def complete_regions(bands,profile,raw,structure,pitch):
    """Propose complete measured units; structure never changes locally."""
    if structure=='single_bright':
        return [dict(id=f'unit_{i:05d}',top=b['top'],bottom=b['bottom'],band_ids=[i],
            score=b['score'],coverage=b['coverage'],structure=structure,
            evidence=dict(bright_lobes=1)) for i,b in enumerate(bands)],dict(ambiguous_pair_phase=False)
    if len(bands)<2:return [],dict(ambiguous_pair_phase=True)
    distances=np.diff([b['center'] for b in bands])
    # Learn the shorter recurring lobe separation without assuming that every
    # pair of adjacent bands belongs together. Large background gaps can be skipped.
    ordinary=distances[(distances>=2)&(distances<=np.percentile(distances,75))]
    inner=float(np.median(ordinary)) if len(ordinary) else float(np.median(distances))
    candidates=[]
    for i,(left,right) in enumerate(zip(bands,bands[1:])):
        distance=right['center']-left['center']
        if not .45*inner<=distance<=2.5*inner:continue
        a=int(round(left['center']));b=int(round(right['center']))
        if b-a<3:continue
        valley=a+int(np.argmin(profile[a:b+1]));low=float(profile[valley])
        depth=min(float(profile[a]),float(profile[b]))-low
        if depth<max(1.5,.12*min(left['prominence'],right['prominence'])):continue
        # Full unit validation is two measured bright lobes separated by an
        # observed darker interior, not a globally halved peak count.
        ra=raw[a];rb=raw[b];rv=raw[valley]
        support=float(np.mean((np.minimum(ra,rb)-rv)>1.))
        if support<.25:continue
        inner_gap=max(0.,right['top']-left['bottom'])
        median_width=float(np.median([q['bottom']-q['top'] for q in bands]))
        # Explicitly refuse a large empty interval between two isolated sheets.
        if inner_gap>max(8.,6*median_width):continue
        balance=min(left['prominence'],right['prominence'])/max(left['prominence'],right['prominence'])
        gap_fit=float(np.exp(-abs(np.log(distance/max(inner,1e-6)))))
        score=float((left['score']+right['score'])*(.5+.5*np.sqrt(balance))*(.5+.5*gap_fit)*(.5+.5*support))
        candidates.append(dict(id=f'unit_{len(candidates):05d}',top=left['top'],bottom=right['bottom'],
            band_ids=[i,i+1],score=score,coverage=min(left['coverage'],right['coverage']),structure=structure,
            evidence=dict(bright_lobes=2,left_center=left['center'],right_center=right['center'],
                internal_dark_y=valley,dark_depth=depth,two_lobe_column_fraction=support,
                measured_lobe_distance=distance,learned_inner_distance=inner)))
    # Near-equal alternating gaps cannot establish which lobe belongs to which
    # physical paper. Report that ambiguity instead of inventing an identity.
    spread=float(np.percentile(distances,75)-np.percentile(distances,25))/max(inner,1.)
    phase_model=score_pair_phase(candidates,bands,profile,raw)
    return candidates,dict(learned_inner_distance=inner,ambiguous_pair_phase=spread<.18,
        pair_phase_model=phase_model,
        physical_sheet_identity='unverified_until_complete_structure_annotation')


def infer_structure(bands,requested):
    """Choose one image-wide grammar, not a peak-dependent counting rule.

    Recurring short/long/short gaps support a two-lobe motif. Check local
    triples so a missing lobe or a background break does not invert the phase
    of every subsequent observation. Equal spacing alone cannot distinguish
    identical half-cycles from individual sheets; preserve that ambiguity.
    """
    if requested!='auto':return requested,dict(source='explicit_configuration',resolved=requested)
    gaps=np.diff([b['center'] for b in bands])
    if len(gaps)<5:
        return 'single_bright',dict(source='insufficient_repetition',resolved='single_bright',
            physical_unit_ambiguous=True,alternating_gap_fraction=0.)
    triples=np.lib.stride_tricks.sliding_window_view(gaps,3)
    a,b,c=triples.T
    recurrence=np.abs(np.log(a/c))<.2
    alternating=(np.log(a/b)*np.log(c/b)>0)&(np.minimum(np.abs(np.log(a/b)),np.abs(np.log(c/b)))>np.log(1.18))
    evidence=recurrence&alternating
    fraction=float(np.mean(evidence))
    # Nearly equal lobe spacing can still hide a stable wide/narrow or
    # strong/weak pair. Compare shape recurrence too; a gap-only decision
    # would miss a bright-dark-bright unit with similar half-cycle lengths.
    shape_fractions={}
    for name,values in (
        ('width',[b['bottom']-b['top'] for b in bands]),
        ('prominence',[b['prominence'] for b in bands])):
        z=np.log(np.maximum(values,1e-6));left=z[:-2];middle=z[1:-1];right=z[2:]
        recurring=(np.abs(left-right)<.2)&((left-middle)*(right-middle)>0)
        different=np.minimum(np.abs(left-middle),np.abs(right-middle))>np.log(1.18)
        shape_fractions[name]=float(np.mean(recurring&different))
    support=max(fraction,*shape_fractions.values())
    resolved='double_bright' if support>=.55 and support*(len(bands)-3)>=4 else 'single_bright'
    return resolved,dict(source='whole_image_repeating_gap_motif',resolved=resolved,
        alternating_gap_fraction=fraction,evidence_triples=int(np.count_nonzero(evidence)),
        alternating_shape_fractions=shape_fractions,
        physical_unit_ambiguous=resolved=='single_bright',
        note='Equal-spaced identical lobes cannot establish physical sheet identity; explicit unit remains available.')


def run_region_stages(image,cfg,segmenter,semantic_warnings,validator):
    gray=to_gray(image);h,w=gray.shape
    if h<8 or w<16:raise ValueError('Region detection needs at least 8 rows and 16 columns')
    raw=gray.astype(np.float32)*(255./np.iinfo(gray.dtype).max)
    normalized=normalize_illumination(gray,cfg.illumination_sigma,cfg.epsilon)
    mapping,xs,valid=align_columns(raw)
    aligned=cv2.remap(normalized,xs,mapping,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    aligned_raw=cv2.remap(raw,xs,mapping,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    bands,profile,pitch,strength,band_nms=extract_bands(aligned,aligned_raw,valid,cfg.max_period)

    # ========== ADAPTIVE: NARROW GAP DETECTION AND BAND SPLITTING ==========
    adaptive_split_debug = {'enabled': False}
    if pitch is not None and pitch > 5:
        try:
            from ..region_adaptive import split_bands_at_narrow_gaps
            modified_bands, split_debug = split_bands_at_narrow_gaps(bands, profile, pitch)
            if split_debug['splits_performed'] > 0:
                bands = modified_bands
                adaptive_split_debug = {
                    'enabled': True,
                    'splits_performed': split_debug['splits_performed'],
                    'split_details': split_debug['split_details'],
                    'gap_analysis': split_debug['gap_analysis']
                }
        except Exception as e:
            # Graceful degradation
            adaptive_split_debug = {'enabled': False, 'error': str(e)}
    # ========== END ADAPTIVE BAND SPLITTING ==========

    bands,boundary_review=limit_to_observed_neighbors(bands,profile)
    structure,selection=infer_structure(bands,cfg.region_structure)
    observation=None;body_plan=None;unit_selection=None;thickness_postprocess=None
    if cfg.region_signal_mode=='three_strip':
        from ..region_joint import observe_strips,select_units,joint_boundaries
        # The whole-width profile defines only the run-wide structure grammar.
        # Detection evidence is retained independently in all three strips.
        bands,profile,observation=observe_strips(aligned,aligned_raw,valid,cfg.max_period,extract_bands)
        bands,boundary_review=limit_to_observed_neighbors(bands,profile)
    thickness_review=[]
    if structure=='single_bright' and observation is None:
        bands,thickness_review=review_thickness(bands,profile,aligned_raw,valid)
    validation_raw=aligned_raw
    if observation is not None:
        active=[(a,b) for k,(a,b) in enumerate(observation['bounds']) if np.mean(observation['weights'][k])>.03]
        if active:validation_raw=np.concatenate([aligned_raw[:,a:b] for a,b in active],axis=1)
        observation['inactive_strips_do_not_vote_against_body']=True
    # Evaluate the alternative two-lobe interpretation once for the global
    # scale decision.  It is evidence only; the requested image-wide grammar
    # still determines the actual proposals.
    pair_evidence_score=None
    if cfg.region_structure=='auto' and len(bands)>=4:
        pair_candidates,_=complete_regions(bands,profile,validation_raw,'double_bright',pitch)
        if pair_candidates:
            pair_evidence_score=float(np.median([c['score'] for c in pair_candidates]))
    candidates,identity=complete_regions(bands,profile,validation_raw,structure,pitch)
    identity['pair_evidence_score']=pair_evidence_score
    identity['structure_selection']=selection
    eligible=[c for c in candidates if not c.get('phase_rejected',False)]
    if observation is not None:
        # Do not use bright-contour IoU as a proxy for full-body overlap.
        # Alternative bodies remain available to the joint ordered decision.
        unique={};nms=[]
        for candidate in sorted(eligible,key=lambda c:-c['score']):
            key=tuple(candidate['band_ids'])
            if key not in unique:unique[key]=candidate
            else:nms.append(dict(candidate=candidate['id'],owner=unique[key]['id'],reason='same_observed_unit'))
        proposals=list(unique.values())
    else:proposals,nms=region_nms(eligible,.65)
    if observation is not None:
        for c in proposals:
            members=[bands[i] for i in c['band_ids']]
            c['evidence']['strip_support']=[dict(band_id=b['id'],source_strips=b['source_strips'],
                observations=[dict(strip=o['strip'],center=o['center'],score=o['score'],
                    reliability=o['reliability'],coverage=o['coverage']) for o in b['strip_observations']]) for b in members]
        selected,unit_selection=select_units(proposals,bands)
        selected,body_plan=joint_boundaries(selected,bands,observation)
        initial_selected_count = len(selected)
    else:
        selected=select_nonconflicting(proposals)
        initial_selected_count = len(selected)
    evidence=OpticalRegionEvidence(selected,bands,observation,proposals,pitch,pair_evidence_score,
        structure,identity,aligned_raw,valid)
    decision=segmenter(evidence)
    selected,structure,thickness_postprocess=decision.units,decision.structure,decision.audit
    used={i for q in selected for i in q['band_ids']}
    boxes=[]
    for i,unit in enumerate(selected):
        top=max(0,int(np.floor(unit['top'])));bottom=min(h-1,int(np.ceil(unit['bottom'])))
        # Global mode retains the historical curved envelope for comparison.
        # Joint mode exports shared horizontal sections, never independent envelopes.
        if observation is not None:
            # Shared horizontal cuts use the center-column coordinate system.
            # Curved boundary samples are exported separately; do not expand
            # each rectangle independently back into overlapping envelopes.
            y1=max(0.,unit['top']);y2=min(float(h),unit['bottom'])
        else:
            y1=max(0.,float(np.min(mapping[top])));y2=min(float(h),float(np.max(mapping[bottom]))+1.)
        boxes.append(dict(id=f'region_{i:05d}',number=i+1,detection_xyxy=[0.,y1,float(w),y2],
            aligned_xyxy=[0.,unit['top'],float(w),unit['bottom']],score=unit['score'],
            structure=structure,band_ids=unit['band_ids'],evidence=unit['evidence'],
            physical_sheet_identity='thickness_inferred' if unit.get('thickness_inferred',False) else 'unverified',
            thickness_inferred=bool(unit.get('thickness_inferred',False)),partial=False))
        if observation is not None:
            columns=[(a+b-1)//2 for a,b in observation['bounds']]
            boxes[-1]['boundary_samples']={name:[dict(x=int(x),y=float(np.interp(row,np.arange(h),mapping[:,x]))) for x in columns]
                for name,row in [('top',unit['top']),('bottom',unit['bottom'])]}
            boxes[-1]['rectangle_interpretation']='shared_horizontal_body_sections; curved boundaries stored separately'

    # ========== ADAPTIVE: ADJUST BOX BOUNDARIES FOR TILT ==========
    # DISABLED: Caused overlapping boxes issues
    # TODO: Need better non-overlapping constraint enforcement
    adaptive_tilt_debug = {'enabled': False, 'reason': 'disabled_due_to_overlap_issues'}
    # ========== END ADAPTIVE TILT ADJUSTMENT ==========

    if observation is not None:
        assert all(a['detection_xyxy'][3]<=b['detection_xyxy'][1]+1e-8 for a,b in zip(boxes,boxes[1:]))

    unit_label='complete_bright_dark_bright_structure' if structure=='double_bright' else 'complete_single_bright_band'
    if structure=='paper_body':unit_label='supported_physical_body_sections'
    warnings=['Boxes count explicit image structures; legacy edge-count ground truth is not automatically compatible.']
    warnings.extend(semantic_warnings(identity,selection,thickness_postprocess))
    unresolved=[b for i,b in enumerate(bands) if i not in used]
    alternatives=[]
    if observation is not None and used:
        genuine_unresolved=[]
        for band in unresolved:
            owner=min(used,key=lambda i:abs(bands[i]['center']-band['center']))
            distance=abs(bands[owner]['center']-band['center'])
            if distance<=1.25*observation['matching_gate']:
                alternatives.append(dict(band_id=band['id'],owner_band_id=bands[owner]['id'],
                    reason='alternative_lobe_localization_not_an_additional_sheet',distance=float(distance)))
            else:genuine_unresolved.append(band)
        unresolved=genuine_unresolved
    if unresolved:warnings.append(f'{len(unresolved)} candidate lobe observations were not selected; this is not a missing-sheet count.')
    inferred_count=sum(b['thickness_inferred'] for b in boxes)
    if inferred_count:warnings.append(f'{inferred_count} regions use thickness-led subdivision; their physical sheet identity is inferred, not independently observed.')
    # The intensity ACF may already describe a complete two-lobe cycle. Do
    # not multiply it by two again just because the selected grammar is double.
    unit_centers=[(u['top']+u['bottom'])/2 for u in selected]
    period=float(np.median(np.diff(unit_centers))) if len(unit_centers)>=3 else None
    centers=[int(round((b['detection_xyxy'][1]+b['detection_xyxy'][3])/2)) for b in boxes]
    confidence=float(np.mean([min(1.,b['score']/6) for b in boxes])) if boxes else 0.
    debug=dict(algorithm='narrow_regions',config=cfg.to_dict(),image_shape=list(image.shape),image_dtype=str(image.dtype),
        roi=dict(x1=0,y1=0,x2=w,y2=h),gradient_polarity='region',period=period,
        median_local_period=None,count=len(boxes),confidence=confidence,status='review_required',warnings=warnings,
        mean_peak_spacing=float(np.mean(np.diff(centers))) if len(centers)>1 else None,
        raw_candidate_count=len(candidates),polarity_scores={p:dict(score=strength) for p in ('positive','negative')},
        adaptive_narrow_gap_splitting=adaptive_split_debug,
        adaptive_tilt_adjustment=adaptive_tilt_debug,
        region_detection=dict(unit=unit_label,unit_locked_for_image=True,structure=structure,requested_structure=cfg.region_structure,
            count_source='selected_complete_region_boxes',identity=identity,bands=bands,candidates=candidates,
            thickness_review=thickness_review,
            boundary_review=boundary_review,
            signal_mode=cfg.region_signal_mode,
            three_strip_observation={k:v for k,v in observation.items() if k not in {'signals','raw_signals','weights'}} if observation is not None else None,
            joint_body_boundaries=body_plan,joint_unit_selection=unit_selection,
            thickness_postprocess=thickness_postprocess,
            selected_ids=[u['id'] for u in selected],unresolved_bands=unresolved,band_nms=band_nms,region_nms=nms,
            alternative_localizations=alternatives,
            period_source='median_spacing_of_observed_complete_boxes',gray_acf_period_hint=pitch,
            background_gaps_are_free=True,period_creates_boxes=False,aligned_iou_threshold=None if observation is not None else .65),
        count_breakdown=dict(confirmed=0,uncertain=len(boxes)-inferred_count,thickness_inferred=inferred_count,estimated_total=len(boxes)),region_boxes=boxes)

    # ========== POST-VALIDATION: CORRECT THICKNESS INFERENCE ERRORS ==========
    postvalidation_debug = {'enabled': False}
    if boxes and observation is not None:
        try:
            from ..region_postvalidation import check_physical_constraints

            # Validate and correct boxes
            corrected_boxes, validation_debug = validator(boxes, bands, observation, debug)

            # Always retain the validation audit, even when the conservative
            # validator decides that an outlier should remain for review.
            postvalidation_debug = validation_debug
            if validation_debug['corrections_applied']:
                boxes = corrected_boxes
                # Update counts in debug
                debug['count'] = len(boxes)
                centers=[int(round((b['detection_xyxy'][1]+b['detection_xyxy'][3])/2)) for b in boxes]
                debug['mean_peak_spacing']=float(np.mean(np.diff(centers))) if len(centers)>1 else None
                inferred_count=sum(b['thickness_inferred'] for b in boxes)
                debug['count_breakdown']=dict(confirmed=0,uncertain=len(boxes)-inferred_count,
                                            thickness_inferred=inferred_count,estimated_total=len(boxes))
                debug['region_boxes']=boxes

                # Re-check physical constraints
                constraint_check = check_physical_constraints(boxes)
                postvalidation_debug['constraint_check'] = constraint_check

                if not constraint_check['passed']:
                    # If corrections introduced violations, warn but don't revert
                    postvalidation_debug['constraint_violations'] = constraint_check['violations']
                else:
                    postvalidation_debug['validated'] = True
        except Exception as e:
            import traceback
            postvalidation_debug = {'enabled': False, 'error': str(e), 'traceback': traceback.format_exc()}

    debug['post_validation'] = postvalidation_debug
    # ========== END POST-VALIDATION ==========

    # Do not delete inferred boxes merely because a score is below a fixed
    # threshold.  Thickness and score are review evidence; deletion requires
    # an independently supported background decision in a later model.
    debug['final_filter'] = {
        'enabled': False,
        'removed_count': 0,
        'policy': 'retain_uncertain_boxes',
        'reason': 'fixed score deletion removed valid sheets in prior runs'
    }

    # A stage ledger makes detection loss auditable instead of hiding it in a
    # final count.  Counts are deliberately named by their semantic stage.
    thickness_info = thickness_postprocess or {}
    validation_info = postvalidation_debug or {}
    debug['stage_counts'] = {
        'raw_profile_bands': int(len(bands)),
        'complete_structure_proposals': int(len(proposals)),
        'selected_initial_units': int(initial_selected_count),
        'before_thickness_resegmentation': int(thickness_info.get('before_count', len(selected))),
        'after_thickness_resegmentation': int(thickness_info.get('after_count', len(selected))),
        'postvalidation_count': int(len(boxes)),
        'postvalidation_removed': int(len(validation_info.get('removed_boxes', [])))
            if isinstance(validation_info, dict) else 0,
        'final_count': int(len(boxes)),
        'final_filter_removed': 0,
    }

    artifacts=dict(roi=image,previews=dict(gray=gray,normalized=normalized),region_aligned=aligned,
        region_profile=profile,region_mapping=mapping)
    if observation is not None:artifacts['three_strip']={k:observation[k] for k in ('signals','raw_signals','weights')}
    return PaperCountResult(len(boxes),centers,[],[],[],period,'region',confidence,debug,artifacts,region_boxes=boxes)
