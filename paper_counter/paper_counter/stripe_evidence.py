"""Complementary RAW stripe observations and measured weak periodic boundaries."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d, grey_opening, grey_closing
from scipy.signal import find_peaks
from .body_evidence import measure_body, attach_body_evidence
from .models import StripPeakCandidate
from .ordered_association import associate_directional
from .period import estimate_period
from dataclasses import replace


def stripe_responses(line, period):
    """亮带顶帽/暗缝底帽去掉缓慢背景；增强响应只产生待核验的位置。"""
    smooth=gaussian_filter1d(np.asarray(line,float),.6)
    size=max(3,int(round(period))|1)
    bright=smooth-grey_opening(smooth,size=size)
    dark=grey_closing(smooth,size=size)-smooth
    return smooth,bright,dark


def stripe_proposals(raw, strips, slopes, offset, period, preferred, cfg):
    if period is None:
        return [],{'proposals':0,'reason':'no_measured_scale','period_creates_positions':False}
    rows={'positive':[],'negative':[]}; diagnostics=[]
    for s,line in enumerate(raw):
        smooth,bright_response,dark_response=stripe_responses(line,period)
        highpass=smooth-gaussian_filter1d(smooth,max(3.,period))
        measured,acf,_=estimate_period(highpass,cfg.min_period,cfg.max_period,
                                      cfg.min_acf_score,cfg.acf_harmonic_preference,cfg.epsilon)
        periodic=measured is not None and acf>=.3
        minimum=cfg.strip_sequence_raw_min_contrast*(.5 if periodic else 1.)
        peaks,props=find_peaks(smooth,prominence=minimum)
        bright_peaks,bright_props=find_peaks(bright_response,prominence=minimum)
        prominence_by_y={int(y):float(p) for y,p in zip(peaks,props['prominences'])}
        for y,p in zip(bright_peaks,bright_props['prominences']):
            prominence_by_y[int(y)]=max(prominence_by_y.get(int(y),0.),float(p))
        gradient=np.gradient(smooth)
        valleys,_=find_peaks(-smooth)
        dark_peaks,_=find_peaks(dark_response,prominence=minimum)
        valleys=np.unique(np.concatenate((valleys,dark_peaks)))
        by_polarity={'positive':[],'negative':[]}
        bands={}
        for peak,prominence in sorted(prominence_by_y.items()):
            left=valleys[valleys<peak];right=valleys[valleys>peak]
            a=int(left[-1]) if len(left) else 0
            b=int(right[0]) if len(right) else len(line)-1
            upper=a+int(np.argmax(gradient[a:peak+1]))
            lower=peak+int(np.argmin(gradient[peak:b+1]))
            for polarity,y in [('positive',upper),('negative',lower)]:
                proof=measure_body(line,y,polarity,measured or period,minimum)
                if proof['state']!='confirmed':continue
                if proof.get('pair_y') is not None:
                    a0,b0=sorted((proof['boundary_y']+offset,proof['pair_y']+offset))
                    key=(a0,b0)
                    bands[key]={'upper':a0,'lower':b0,'contrast':proof['contrast'],
                        'body_level':proof['body_level'],'outside_level':proof['outside_level'],
                        'snr':proof['snr'],'weak':proof['contrast']<cfg.strip_sequence_raw_min_contrast}
                # Weak boundaries need a measured repeat in this RAW strip;
                # filtered oscillations and phase-only predictions are excluded.
                weak=proof['contrast']<cfg.strip_sequence_raw_min_contrast
                if weak and not periodic:continue
                if any(abs(p.y-(y+offset))<=2 for p in by_polarity[polarity]):continue
                p=StripPeakCandidate(s,float(y+offset),float(abs(gradient[y])),
                    float(proof['contrast']),float(proof['width']),minimum,'strong',
                    local_prominence_reference=max(minimum,float(prominence)),
                    observation_source='period_assisted_RAW_band' if weak else 'RAW_bright_band')
                by_polarity[polarity].append(p)
        # A body truncated at the first/last row still has one measured edge.
        for polarity,direction in [('negative',-1),('positive',1)]:
            response=gradient*direction
            edge_peaks,_=find_peaks(response,prominence=minimum*.5)
            for y in edge_peaks:
                if min(y,len(line)-1-y)>1.25*period:continue
                proof=measure_body(line,int(y),polarity,period,minimum)
                if proof['state']=='confirmed' and proof['partial'] and not any(abs(p.y-(y+offset))<=2 for p in by_polarity[polarity]):
                    by_polarity[polarity].append(StripPeakCandidate(s,float(y+offset),float(response[y]),
                        proof['contrast'],proof['width'],minimum,'strong',local_prominence_reference=proof['contrast'],
                        observation_source='RAW_visible_partial_body'))
        for polarity in rows:rows[polarity].append(sorted(by_polarity[polarity],key=lambda p:p.y))
        diagnostics.append({'strip':s,'raw_period':measured,'acf':acf,'periodic_weak_search':periodic,
            'bright_band_proposals':len(prominence_by_y),'dark_gap_observations':len(dark_peaks),
            'measured_bands':list(bands.values())})
    proposals=[]
    for polarity in (preferred,'negative' if preferred=='positive' else 'positive'):
        tracks,_=associate_directional(rows[polarity],strips,slopes,offset,period,cfg)
        for i,track in enumerate(tracks):
            track.boundary_polarity=polarity;track.track_id=f'body_{polarity}_{i:05d}'
            attach_body_evidence([track],raw,offset,lambda y:period,polarity,cfg)
            if track.local_evidence['paper_body']['verified']:
                track.local_evidence['stripe_observation']=True
                track.local_evidence['period_assisted']=any(p.observation_source=='period_assisted_RAW_band' for p in track.peak_candidates)
                track.candidate_class='strong';proposals.append(track)
    return proposals,{'strips':diagnostics,'proposals':len(proposals),
                       'extractor':'RAW extrema + bright top-hat + dark bottom-hat',
                       'period_creates_positions':False,'contrast_gain_is_not_body_evidence':True}


def same_body(a,b,preferred):
    ea=a.local_evidence.get('paper_body',{});eb=b.local_evidence.get('paper_body',{})
    pa=a.boundary_polarity or preferred;pb=b.boundary_polarity or preferred
    common=set(a.strip_positions)&set(b.strip_positions)
    if not common:return False
    widths=[v['width'] for e in (ea,eb) for v in e.get('samples',[]) if v.get('width')]
    scale=float(np.median(widths)) if widths else 8.
    # 相同观测行不能同时代表两张纸，即使两次提取标记了不同极性。
    matched={s for s in common if abs(a.strip_positions[s]-b.strip_positions[s]) <= 2}
    proof_a={v['strip']:v for v in ea.get('samples',[]) if v.get('state')=='confirmed' and v.get('pair_y') is not None}
    proof_b={v['strip']:v for v in eb.get('samples',[]) if v.get('state')=='confirmed' and v.get('pair_y') is not None}
    for s in (set(proof_a)&set(proof_b) if pa!=pb else []):
        p,q=proof_a[s],proof_b[s]
        a0,a1=sorted((p['boundary_y'],p['pair_y']))
        b0,b1=sorted((q['boundary_y'],q['pair_y']))
        overlap=max(0.,min(a1,b1)-max(a0,b0))
        if overlap>=.8*min(a1-a0,b1-b0):
            matched.add(s)
    if pa==pb:
        anchors=set(eb.get('confirmed_strips',[])) or set(ea.get('confirmed_strips',[]))
        for s in common:
            if (not anchors or s in anchors) and abs(a.strip_positions[s]-b.strip_positions[s])<=max(3.,.3*scale):
                matched.add(s)
    else:
        for s in common:
            tolerance=max(3.,.3*scale)
            if s in proof_b and abs(proof_b[s]['pair_y']-a.strip_positions[s])<=tolerance:matched.add(s)
            elif s in proof_a and abs(proof_a[s]['pair_y']-b.strip_positions[s])<=tolerance:matched.add(s)
    # 短片段的一处身份锚点足以归属已有纸身。两条宽覆盖轨迹若只有
    # 一处吻合，属于局部错连冲突，不能把整张相邻纸直接合并掉。
    if min(a.support_count,b.support_count)>=6:
        return len(matched)>=max(2,.6*len(common))
    return bool(matched)


def fuse_body_proposals(tracks,proposals,preferred,cfg,signal_model=None):
    selected=list(tracks);decisions=[]
    for candidate in sorted(proposals,key=lambda t:-t.total_score):
        duplicate=next((t for t in selected if same_body(t,candidate,preferred)
            or (signal_model is not None and signal_model.same_identity(t,candidate))),None)
        if duplicate:
            decisions.append({'id':candidate.track_id,'decision':'same_measured_body','existing_id':duplicate.track_id})
            continue
        if candidate.smoothness_score<cfg.min_track_smoothness:continue
        # Confirmed local existence is sufficient; full-width coverage measures
        # uncertainty rather than vetoing a short but clear paper segment.
        candidate.source='recovered';candidate.sequence_decision='rescued'
        selected.append(candidate)
        decisions.append({'id':candidate.track_id,'decision':'added_measured_body','support':candidate.support_count})
    return sorted(selected,key=lambda t:t.center_y),decisions


def deduplicate_body_identities(tracks, preferred, signal_model=None):
    """主流程与补充流程均独占实测纸身；宽支持不能覆盖身份重复。"""
    kept=[];rejected=[];decisions=[]
    # 补充的局部亮带不能先替换已选主轨迹，再让终判删除整个纸层。
    for candidate in sorted(tracks,key=lambda t:(not t.track_id.startswith('body_'),t.total_score,t.support_count),reverse=True):
        owner=next((t for t in kept if independent_identity_duplicate(t,candidate,preferred,signal_model)),None)
        if owner is None:
            kept.append(candidate)
        else:
            candidate.reason='duplicate_measured_body';candidate.sequence_decision='suppressed'
            rejected.append(candidate)
            decisions.append({'id':candidate.track_id,'existing_id':owner.track_id,'decision':'same_measured_body'})
    return sorted(kept,key=lambda t:t.center_y),rejected,decisions


def independent_identity_duplicate(a,b,preferred,signal_model=None):
    if getattr(a,'local_evidence',{}).get('composite_structure',{}).get('verified') or getattr(b,'local_evidence',{}).get('composite_structure',{}).get('verified'):
        from .composite_layers import distinct_composite_observations
        if distinct_composite_observations(a,b):return False
    if (a.track_id.startswith('seed_') and b.track_id.startswith('seed_')
            and (getattr(a,'boundary_polarity','') or preferred)==(getattr(b,'boundary_polarity','') or preferred)):
        # 宽覆盖的原生轨迹局部相遇后又分开，是归属冲突，不能整条合并。
        common=set(a.strip_positions)&set(b.strip_positions)
        matched=[s for s in common if abs(a.strip_positions[s]-b.strip_positions[s])<=2]
        if matched and len(matched)<len(common):return False
        if len(common)>=2 and len(matched)==len(common):return True
    return same_body(a,b,preferred) or (signal_model is not None and signal_model.same_identity(a,b))


def independent_gap_observations(candidate,left,right,model):
    """新增一张纸需要同一对锚点间的实测位置，局部亮带本身不够。"""
    a,b,c=(model.canonical_track(t) for t in (left,right,candidate))
    period=model.local.at(c.center_y)
    common=set(a.strip_positions)&set(b.strip_positions)&set(c.strip_positions)
    votes=[]
    for s in common:
        x,y,z=a.strip_positions[s],c.strip_positions[s],b.strip_positions[s]
        if x<y<z and .35*period<=y-x<=1.65*period and .35*period<=z-y<=1.65*period and z-x>=1.55*period:
            votes.append(s)
    # 支持范围窄不等于证据弱。一条带若实测三段不同纸身以及其间
    # 两处暗槽，也能证明确有独立纸层；只有亮斑或周期位置则不能。
    single_complete=[]
    if len(votes)==1 and hasattr(model,'bindings') and hasattr(model,'pair_evidence'):
        s=votes[0]
        bindings=[model.bindings(t) for t in (left,candidate,right)]
        if all(s in binding for binding in bindings):
            ids=[binding[s] for binding in bindings]
            bands=[model.bands[s][i] for i in ids]
            separated=(len(set(ids))==3 and bands[0]['lower']<bands[1]['upper']
                and bands[1]['lower']<bands[2]['upper'])
            gaps=[model.pair_evidence(x,y) for x,y in ((left,candidate),(candidate,right))]
            measured_gaps=all(any(p['strip']==s and p['separate_bodies'] for p in g['samples']) for g in gaps)
            # 一处窄高亮纹理即使被两处低谷夹住，也不是整张纸身。
            # 单带特例须有与局部层间尺度相容的实测纸身宽度。
            body_width=bands[1]['lower']-bands[1]['upper']
            if separated and measured_gaps and body_width>=.35*period and model.strong_bands[s][ids[1]]:
                single_complete.append(s)
    return {'verified':len(votes)>=2 or bool(single_complete),'strips':sorted(votes),
        'single_strip_complete_identity':single_complete,
        'source':'observed_independent_slot_not_local_brightness_only'}


def select_body_sequence(tracks, local, preferred, cfg, regularity=None):
    """补充候选与主轨迹接受同一几何终选；正负边界先对应到同一纸身侧。

    临时坐标只用于身份/顺序比较。正式输出保持原来的实测极性和坐标，
    没有观测到的配对边界不能插值成新证据。
    """
    from .vertical_sequence import optimize_vertical_sequence
    if getattr(local,'signal_gap_model',None) is not None:
        result=optimize_vertical_sequence(tracks,local,cfg,regularity)
        return result.selected,result.suppressed,{'selected_ids':[t.track_id for t in result.selected],
            'suppressed_ids':[t.track_id for t in result.suppressed],
            'coordinate_comparison':'measured_same_body_side_with_RAW_gap_evidence','score':result.total_sequence_score}
    canonical = []
    originals = {t.track_id:t for t in tracks}
    for t in tracks:
        polarity = t.boundary_polarity or preferred
        evidence = t.local_evidence.get('paper_body',{})
        proofs = {p['strip']:p for p in evidence.get('samples',[]) if p['state']=='confirmed'}
        positions = dict(t.strip_positions)
        if polarity != preferred:
            positions = {s:p['pair_y'] for s,p in proofs.items() if p.get('pair_y') is not None}
            # 真实图像截断纸身只有一侧边界，保留独立观测作端点比较。
            if not positions and any(p.get('partial') for p in proofs.values()):
                positions = dict(t.strip_positions)
            if not positions:
                continue
        else:
            positions.update({s:p['boundary_y'] for s,p in proofs.items() if p.get('boundary_y') is not None})
        canonical.append(replace(t, center_y=float(np.median(list(positions.values()))),
            strip_positions=positions, curve_geometry={}, local_evidence=dict(t.local_evidence)))
    result = optimize_vertical_sequence(canonical,local,cfg,regularity)
    selected = [originals[t.track_id] for t in result.selected]
    rejected = [originals[t.track_id] for t in result.suppressed]
    for t in rejected:
        t.reason='body_sequence_competition';t.sequence_decision='suppressed'
    return sorted(selected,key=lambda t:t.center_y), rejected, {'selected_ids':[t.track_id for t in selected],
        'suppressed_ids':[t.track_id for t in rejected], 'coordinate_comparison':'measured_same_body_side',
        'score':result.total_sequence_score}


def reconcile_signal_regions(tracks, proposals, signal_model, cfg, regularity=None, native_pool=()):
    """Preserve the mature main hypothesis; solve measured anomaly windows.

    RAW evidence does not authorize a fresh whole-image count. The anchors
    outside each affected region constrain local competition and recovery.
    """
    from .vertical_sequence import optimize_vertical_sequence
    preferred=signal_model.preferred
    native,duplicates,duplicate_decisions=deduplicate_body_identities(tracks,preferred,signal_model)
    native=sorted(native,key=lambda t:signal_model.canonical_track(t).center_y)
    checks=signal_model.diagnostics(native)
    intervals=[];admissible=[];decisions=[]
    for i,check in enumerate(checks):
        crowded=(check['gap_period_ratio']<1-cfg.sequence_spacing_scale_ratio and
            check['raw_period_quality']>=.3 and not any(p['separate_bodies'] for p in check['samples']))
        if check['state'] in {'same_measured_body','crowded_unresolved','observed_bands_inside_gap'} or crowded:
            intervals.append((max(0,i-1),min(len(native)-1,i+2)))
    selected_ids={t.track_id for t in native}
    native_candidates=[t for t in native_pool if t.track_id not in selected_ids
        and t.candidate_class in {'strong','marginal'} and t.local_evidence.get('paper_body',{}).get('state')!='absent'
        and signal_model.transition_evidence(t)['verified']]
    native_candidate_ids={t.track_id for t in native_candidates}
    for t in native_candidates:t.local_evidence['raw_transition']=signal_model.transition_evidence(t)
    for p in list(proposals)+native_candidates:
        owner=next((t for t in native if independent_identity_duplicate(t,p,preferred,signal_model)),None)
        if owner is not None:
            decisions.append({'id':p.track_id,'decision':'same_measured_body','existing_id':owner.track_id})
            continue
        y=signal_model.canonical_track(p).center_y
        inside=next((i for i,(a,b) in enumerate(zip(native,native[1:]))
            if signal_model.canonical_track(a).center_y<y<signal_model.canonical_track(b).center_y),None)
        eligible=(inside is not None and (checks[inside]['state']=='observed_bands_inside_gap' or
            (checks[inside]['state']=='unresolved_large_gap' and (p.track_id in native_candidate_ids
                or p.local_evidence.get('compressed_layers',{}).get('verified')))))
        if eligible:
            proof=independent_gap_observations(p,native[inside],native[inside+1],signal_model)
            p.local_evidence['independent_gap_identity']=proof
            if not proof['verified']:
                decisions.append({'id':p.track_id,'decision':'local_band_not_independent_paper','proof':proof})
                continue
            admissible.append(p)
            p.local_evidence['allocated_signal_gap']={'left':native[inside].track_id,
                'right':native[inside+1].track_id,'kind':checks[inside]['state']}
            intervals.append((max(0,inside-1),min(len(native)-1,inside+2)))
            decisions.append({'id':p.track_id,'decision':'observed_gap_candidate','left':native[inside].track_id,'right':native[inside+1].track_id})
        else:
            # Unallocated endpoint fragments and bright bands inside an
            # ordinary slot are hypotheses, not automatic additional papers.
            decisions.append({'id':p.track_id,'decision':'no_independent_unoccupied_gap'})
    merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
        else:merged.append((lo,hi))
    selected=list(native);rejected=list(duplicates);windows=[]
    for lo,hi in merged:
        base=native[lo:hi+1]
        left=signal_model.canonical_track(base[0]).center_y
        right=signal_model.canonical_track(base[-1]).center_y
        new=[p for p in admissible if left<signal_model.canonical_track(p).center_y<right]
        pool,more_duplicates,_=deduplicate_body_identities(base+new,preferred,signal_model)
        for p in new:p.source='recovered';p.sequence_decision='rescued'
        required={base[0].track_id,base[-1].track_id}
        # If an anchor was physically duplicated, its representative takes
        # over the requirement; do not fabricate an anchor at a predicted row.
        if not required<={p.track_id for p in pool}:
            pool=base+new
        surviving={p.track_id for p in pool}
        new=[p for p in new if p.track_id in surviving]
        # 分锚点间隙求解，而不是把成熟主轨迹整体再选择一遍。
        # 同一纸层的代表是否应删除仍交给身份终判；补检不能先把它挤掉。
        additions=[];suppressed=[]
        for a,b in zip(base,base[1:]):
            lo_y,hi_y=signal_model.canonical_track(a).center_y,signal_model.canonical_track(b).center_y
            inner=[p for p in new if lo_y<signal_model.canonical_track(p).center_y<hi_y]
            if not inner:continue
            result=optimize_vertical_sequence([a]+inner+[b],signal_model.local,cfg,regularity,require_endpoints=True)
            if result.selected:
                additions.extend(p for p in result.selected if p.track_id not in {a.track_id,b.track_id})
                suppressed.extend(result.suppressed)
            else:suppressed.extend(inner)
        if not additions:
            # Contradictory endpoint identities cannot delete an entire valid
            # main hypothesis merely to avoid a gap penalty; request review.
            windows.append({'left':base[0].track_id,'right':base[-1].track_id,'decision':'kept_native_no_independent_addition'})
            continue
        replaced={t.track_id for t in base}
        selected=[t for t in selected if t.track_id not in replaced]+base+additions
        surviving_ids={t.track_id for t in base+additions}
        rejected.extend(t for t in more_duplicates+suppressed if t.track_id not in surviving_ids)
        windows.append({'left':base[0].track_id,'right':base[-1].track_id,
            'selected_ids':[t.track_id for t in base+additions],
            'suppressed_ids':[t.track_id for t in suppressed],
            'new_candidates':[t.track_id for t in new],'decision':'measured_window_competition'})
    signal_model.set_candidates(selected)
    return sorted(selected,key=lambda t:t.center_y),rejected,{'mode':'measured_anomaly_windows',
        'identity_duplicates':duplicate_decisions,'decisions':decisions,'windows':windows,
        'added':sum(t.track_id.startswith('body_') for t in selected)}
