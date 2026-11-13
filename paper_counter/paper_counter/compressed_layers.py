"""压实纸堆：沿实测方向积累 RAW 弱层纹，不要求清晰的双侧纸身。

周期只确定检查尺度；每个候选位置都必须是实测响应的局部极大值。
三个互不重叠的横向窗口用于排除单列颗粒，不能当成三张纸计数。
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .models import StripPeakCandidate
from .ordered_association import associate_directional


def accumulated_profiles(gray, strips, period, cfg, directions=None):
    """同一检测行沿 RAW 切线采样，返回三个横向子窗口的独立平均值。"""
    image=gray.astype(np.float32)
    if np.issubdtype(gray.dtype,np.integer):image*=255/np.iinfo(gray.dtype).max
    gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)
    # 邻近层共同形变提供方向；不搜索使周期相位吻合的行移位。
    sigma_y=max(1.,min(8.,.15*period))
    xx=cv2.GaussianBlur(gx*gx,(0,0),3.,sigmaY=sigma_y)
    yy=cv2.GaussianBlur(gy*gy,(0,0),3.,sigmaY=sigma_y)
    xy=cv2.GaussianBlur(gx*gy,(0,0),3.,sigmaY=sigma_y)
    coherence=np.sqrt((xx-yy)**2+4*xy*xy)/np.maximum(xx+yy,cfg.epsilon)
    tangent=-xy/np.maximum(yy,cfg.epsilon)
    tangent[(coherence<cfg.orientation_min_coherence)|(np.abs(tangent)>cfg.orientation_max_slope)]=0
    rows=np.arange(len(image),dtype=float)
    profiles=[];diagnostics=[]
    centers=np.array([(a+b-1)/2 for a,b in strips])
    for s,(a,b) in enumerate(strips):
        center=(a+b-1)/2
        # 弱信号需要充分利用像素；不再只抽取少数列，避免丢掉积累收益。
        columns=np.arange(a,b,dtype=int)
        traced={};max_shift=0.
        # 从中心向两侧各积分一次，避免为每列重复跟踪同一段路径。
        for targets in (columns[columns<center][::-1],columns[columns>=center]):
            y=rows.copy();previous=center
            for target in targets:
                steps=np.linspace(previous,target,max(2,int(abs(target-previous)/3)+1))
                for left,right in zip(steps,steps[1:]):
                    x=int(round((left+right)/2))
                    y+=np.interp(y,rows,tangent[:,x])*(right-left)
                # 弱纹自身方向不稳定时，借用邻近强边界的共同形变。
                # 位移仍来自 RAW 方向积分，不由周期相位搜索产生。
                neighbor=s-1 if target<center else s+1
                motion=getattr(directions,'motion',{}).get((s,neighbor))
                reliability=getattr(directions,'reliability',{}).get((s,neighbor))
                sampling_y=y.copy()
                if motion is not None and reliability is not None:
                    delta=np.asarray(motion)*(target-center)/(centers[neighbor]-center)
                    usable=np.asarray(reliability)&(1+np.gradient(delta)>.5)
                    sampling_y[usable]=rows[usable]+delta[usable]
                traced[target]=(np.interp(sampling_y,rows,image[:,target]),(sampling_y>=0)&(sampling_y<=len(rows)-1))
                max_shift=max(max_shift,float(np.max(np.abs(y-rows))));previous=target
        sampled=[traced[x][0] for x in columns];available=[traced[x][1] for x in columns]
        sampled=np.asarray(sampled);available=np.asarray(available)
        groups=np.array_split(np.arange(len(columns)),3)
        values=[]
        for indices in groups:
            block=np.where(available[indices],sampled[indices],np.nan)
            missing=np.all(~np.isfinite(block),axis=0)
            block[:,missing]=0.  # 缺失行仍由 available 标为不可观测，不是背景证据。
            # 只削弱极端颗粒；平均值保留低于单像素灰度步长的共同变化。
            lo=np.nanpercentile(block,10,axis=0);hi=np.nanpercentile(block,90,axis=0)
            clipped=np.clip(block,lo,hi)
            values.append(np.nansum(clipped,axis=0)/np.maximum(1,np.sum(np.isfinite(clipped),axis=0)))
        profiles.append(values)
        diagnostics.append({'sampled_columns':len(columns),'direction_shift_max':max_shift,
            'fully_observed_rows':np.all(available,axis=0),
            'view_available_rows':np.asarray([np.all(available[indices],axis=0) for indices in groups])})
    return np.asarray(profiles),diagnostics


def measure_weak_layers(groups, period, polarity, cfg, available=None):
    """从连续弱响应提议位置，再核验横向一致性和局部重复结构。"""
    groups=np.asarray(groups,float)
    smooth=gaussian_filter1d(groups,.6,axis=1)
    mean=smooth.mean(axis=0)
    highpass=smooth-gaussian_filter1d(smooth,max(3.,period),axis=1)
    derivatives=np.gradient(smooth,axis=1)
    signed=derivatives*(1 if polarity=='positive' else -1)
    response=signed.mean(axis=0)
    # 噪声来自横向窗口之间的差异，不把真实层纹自身的导数当成噪声。
    disagreement=signed-response
    noise=gaussian_filter1d(np.mean(disagreement**2,axis=0),max(2.,period))**.5/np.sqrt(3)
    noise=np.maximum(noise,.03)
    peaks,_=find_peaks(response,distance=max(2,int(.6*period)),prominence=.03)
    accepted=[];rejected=[]
    radius=max(8,int(3*period))
    for y in peaks:
        lo=max(0,y-radius);hi=min(len(mean),y+radius+1)
        if y<period or len(mean)-1-y<period:continue
        if available is not None and not np.all(available[max(0,y-2):min(len(mean),y+3)]):continue
        # 压实区两侧仍是纸面灰度；大面积黑区和明暗阶跃不进入弱纹分支。
        flank=int(max(2,round(.2*period)))
        # 已经横向去除极端颗粒；这里使用均值，保留仅一像素宽的弱暗缝。
        # 中位数会把短窗口里那一个暗像素再次抹掉。
        low=float(np.mean(mean[y-flank:y]));high=float(np.mean(mean[y+1:y+flank+1]))
        contrast=(high-low)*(1 if polarity=='positive' else -1)
        level=float(np.median(mean[lo:hi]))
        # 压实分支仅处理连续纸面上的弱调制。普通强纸缝邻域中的
        # 小纹理也可能满足“局部对比低”，不能因此套用弱纹身份模型。
        region_lo=max(0,y-int(.5*period));region_hi=min(len(mean),y+int(.5*period)+1)
        levels=np.median(smooth[:,region_lo:region_hi],axis=1)
        continuous=bool(np.all(np.min(smooth[:,region_lo:region_hi],axis=1)>=.5*levels))
        modulation=float(np.ptp(highpass[:,region_lo:region_hi].mean(axis=0)))
        votes=sum(np.max(signed[s,max(0,y-1):y+2])>=max(.03,noise[y]) for s in range(3))
        lags=range(max(2,int(np.floor(.8*period))),max(3,int(np.ceil(1.2*period)))+1)
        signal=highpass[:,lo:hi]
        correlations=[]
        for lag in lags:
            if signal.shape[1]<=lag+4:continue
            lhs=signal[:,:-lag];rhs=signal[:,lag:]
            denom=np.sqrt(np.sum(lhs*lhs,axis=1)*np.sum(rhs*rhs,axis=1))
            correlations.append((float(np.median(np.sum(lhs*rhs,axis=1)/np.maximum(denom,cfg.epsilon))),lag))
        quality,measured=max(correlations,default=(0.,period))
        short_correlations=[]
        for lag in range(max(2,int(.4*period)),max(3,int(.6*period))+1):
            lhs=signal[:,:-lag];rhs=signal[:,lag:]
            denom=np.sqrt(np.sum(lhs*lhs,axis=1)*np.sum(rhs*rhs,axis=1))
            short_correlations.append(float(np.median(np.sum(lhs*rhs,axis=1)/np.maximum(denom,cfg.epsilon))))
        short_quality=max(short_correlations,default=0.)
        # 局部模板来自实际相邻响应峰，不按周期生成两条参考纸边。
        before=peaks[(peaks<y)&(np.abs((y-peaks)-measured)<=.25*measured)]
        after=peaks[(peaks>y)&(np.abs((peaks-y)-measured)<=.25*measured)]
        neighbors=[int(before[-1]),int(after[0])] if len(before) and len(after) else []
        shape=[];half=max(2,int(.25*measured))
        for other in neighbors:
            if min(y,other)<half or max(y,other)+half>=len(mean):continue
            lhs=signed[:,y-half:y+half+1];rhs=signed[:,other-half:other+half+1]
            lhs=lhs-lhs.mean(axis=1,keepdims=True);rhs=rhs-rhs.mean(axis=1,keepdims=True)
            denom=np.sqrt(np.sum(lhs*lhs,axis=1)*np.sum(rhs*rhs,axis=1))
            shape.append(float(np.median(np.sum(lhs*rhs,axis=1)/np.maximum(denom,cfg.epsilon))))
        local_shape=min(shape) if len(shape)==2 else 0.
        reason=('not_continuous_compressed_region' if not continuous or modulation>.3*max(level,1.)
            else 'not_compressed_paper_interior' if min(low,high)<4*cfg.strip_sequence_raw_min_contrast or contrast>.15*max(level,1.)
            else 'insufficient_RAW_contrast' if contrast<max(.3,3*noise[y])
            else 'windows_disagree' if votes<3 or response[y]<3*noise[y]
            else 'no_local_repeat' if quality<.3
            else 'ambiguous_half_period' if short_quality>=max(.3,quality-.05)
            else None)
        proof={'y':float(y),'contrast':contrast,'response':float(response[y]),'noise':float(noise[y]),
            'horizontal_votes':int(votes),'repeat_quality':quality,'period':float(measured),
            'region_modulation_ratio':modulation/max(level,1.),
            'half_period_quality':short_quality,
            'neighbor_cycles_y':neighbors,'local_template_correlation':local_shape,
            'verified':reason is None,'reason':reason,
            'source':'three_disjoint_direction_accumulated_RAW_windows'}
        (accepted if reason is None else rejected).append(proof)
    return accepted,{'accepted':accepted,'rejected':rejected}


def compressed_proposals(gray,strips,slopes,offset,period,polarity,cfg):
    profiles,alignment=accumulated_profiles(gray,strips,period,cfg,slopes)
    rows=[];records=[];details=[]
    for s,groups in enumerate(profiles):
        proofs,detail=measure_weak_layers(groups,period,polarity,cfg,alignment[s]['fully_observed_rows'])
        nodes=[]
        for proof in proofs:
            y=proof['y']+offset
            records.append({'strip':s,**proof,'y':y,
                'neighbor_cycles_y':[v+offset for v in proof['neighbor_cycles_y']]})
            nodes.append(StripPeakCandidate(s,y,proof['response'],proof['contrast'],1.,3*proof['noise'],
                'weak',local_prominence_reference=max(proof['contrast'],cfg.epsilon),
                observation_source='direction_accumulated_weak_RAW_layer'))
        rows.append(nodes)
        details.append({'strip':s,'sampled_columns':alignment[s]['sampled_columns'],
            'direction_shift_max':alignment[s]['direction_shift_max'],**detail})
    tracks,_=associate_directional(rows,strips,slopes,offset,period,cfg)
    proposals=[]
    for i,t in enumerate(tracks):
        t.boundary_polarity=polarity;t.track_id=f'compressed_{i:05d}';t.candidate_class='marginal'
        proofs=[p for p in records if p['strip'] in t.strip_positions and abs(p['y']-t.strip_positions[p['strip']])<=1]
        # 至少两个条带共享层纹；一个横向颗粒不能变成独立纸层。
        if len(proofs)<2:continue
        t.local_evidence['compressed_layers']={'verified':True,'samples':proofs,
            'complete_body_required':False,'is_terminal_identity_exemption':False}
        t.local_evidence['raw_transition']={'verified':True,'samples':proofs,
            'source':'direction_accumulated_weak_RAW_layer'}
        proposals.append(t)
    return proposals,{'enabled':True,'proposals':len(proposals),'strips':details,
        'input_period':period,'polarity':polarity,
        'period_creates_positions':False,'terminal_identity_exemption':False}


def compressed_identity_context(track,tracks,cfg):
    """在共同实测条带上检查左右邻纸，避免不同覆盖范围的中位坐标误判拥挤。"""
    proof=track.local_evidence.get('compressed_layers',{})
    samples=[p for p in proof.get('samples',[]) if p.get('verified') and p.get('local_template_correlation',0)>=.3]
    if not proof.get('verified') or not track.local_evidence.get('allocated_signal_gap') or len(samples)<2:
        return {'verified':False,'reason':'needs_allocated_gap_and_two_local_templates'}
    witnesses=[]
    for p in samples:
        s=p['strip'];y=track.strip_positions[s];period=p['period']
        positions=[(other.strip_positions[s],other.track_id) for other in tracks
            if other.track_id!=track.track_id and s in other.strip_positions
            and (other.boundary_polarity or cfg.gradient_polarity)==(track.boundary_polarity or cfg.gradient_polarity)]
        before=[q for q in positions if q[0]<y];after=[q for q in positions if q[0]>y]
        if not before or not after:continue
        a=max(before);b=min(after)
        left=(y-a[0])/period;right=(b[0]-y)/period
        if .55<=left<=1.45 and .55<=right<=1.45:
            witnesses.append({'strip':s,'left':a[1],'right':b[1],
                'left_period_ratio':left,'right_period_ratio':right,'local_template_correlation':p['local_template_correlation']})
    from collections import Counter
    counts=Counter((p['left'],p['right']) for p in witnesses)
    pair,count=counts.most_common(1)[0] if counts else (None,0)
    matched=[p for p in witnesses if (p['left'],p['right'])==pair]
    return {'verified':count>=2 and count>=.6*len(samples),'samples':matched,
        'reason':'shared_observation_spacing_and_local_weak_template'}


def fragment_relation(candidate,other,strips=None,directions=None,offset=0,cfg=None):
    """共享实测点或可靠双向方向延续意味着不能直接声明独立新纸层。"""
    if cfg is not None and (candidate.boundary_polarity or cfg.gradient_polarity)!=(other.boundary_polarity or cfg.gradient_polarity):
        periods=[p['period'] for p in candidate.local_evidence.get('compressed_layers',{}).get('samples',[])]
        limit=1.25*float(np.median(periods)) if periods else cfg.max_period
        paired=[p for p in other.local_evidence.get('paper_body',{}).get('samples',[])
            if p.get('state')=='confirmed' and p.get('pair_y') is not None
            and p.get('width',float('inf'))<=limit and p['strip'] in candidate.strip_positions
            and abs(p['pair_y']-candidate.strip_positions[p['strip']])<=2]
        return 'already_observed_opposite_boundary' if len(paired)>=2 else None
    common=set(candidate.strip_positions)&set(other.strip_positions)
    collisions=[s for s in common if abs(candidate.strip_positions[s]-other.strip_positions[s])<=2]
    if collisions:
        return 'already_observed' if len(collisions)==len(common) else 'existing_identity_conflict'
    if strips is None or not getattr(directions,'motion',None):return None
    for target,y in candidate.strip_positions.items():
        supports=0
        for source,old_y in other.strip_positions.items():
            if source==target or abs(source-target)>cfg.max_skip_strips+1:continue
            forward=directions.motion.get((source,target));reverse=directions.motion.get((target,source))
            valid_f=directions.reliability.get((source,target));valid_r=directions.reliability.get((target,source))
            if forward is None or reverse is None or valid_f is None or valid_r is None:continue
            a=int(np.clip(round(old_y-offset),0,len(forward)-1));b=int(np.clip(round(y-offset),0,len(reverse)-1))
            if valid_f[a] and valid_r[b] and abs(old_y+forward[a]-y)<=2 and abs(y+reverse[b]-old_y)<=2:
                supports+=1
        if supports>=2:return 'possible_existing_fragment'
    return None


def prepare_compressed_candidates(native,proposals,cfg,strips=None,directions=None,offset=0):
    """保留成熟序列，只在同一对实测纸边之间提议弱层，不能替换原轨迹。"""
    selected=[];decisions=[]
    for t in sorted(proposals,key=lambda p:p.total_score,reverse=True):
        duplicate=False
        for other in native+selected:
            common=set(t.strip_positions)&set(other.strip_positions)
            relation=fragment_relation(t,other,strips,directions,offset,cfg)
            if relation:
                duplicate=True
                if relation=='already_observed' and any(other is original for original in native):
                    matched=[p for p in t.local_evidence['compressed_layers']['samples']
                        if p['strip'] in other.strip_positions and abs(p['y']-other.strip_positions[p['strip']])<=2]
                    if len(matched)>=max(2,int(np.ceil(.5*other.support_count))):
                        other.local_evidence['compressed_layers']={**t.local_evidence['compressed_layers'],'samples':matched}
                        other.local_evidence['allocated_signal_gap']={'source':'existing_measured_layer_recheck'}
                decisions.append({'id':t.track_id,'decision':relation,'existing_id':other.track_id})
                break
        if duplicate:
            continue
        t.local_evidence['allocated_signal_gap']={'source':'shared_observation_brackets'}
        context=compressed_identity_context(t,native,cfg)
        t.local_evidence['compressed_identity']=context
        if not context['verified']:
            decisions.append({'id':t.track_id,'decision':'insufficient_independent_gap_evidence'});continue
        t.source='recovered';t.sequence_decision='rescued'
        selected.append(t)
        decisions.append({'id':t.track_id,'decision':'candidate_for_terminal_check','samples':context['samples']})
    return selected,decisions


def recheck_compressed_identity(track,feature,tracks,cfg):
    """仅以充分的局部弱纹证据重评身份，保留背景/纸面支持/明确无纸拒绝。"""
    from .track_identity import conservative_identity_decision
    context=compressed_identity_context(track,tracks,cfg)
    track.local_evidence['compressed_identity']=context
    previous=feature.proposed_decision
    families=feature.evidence_families or []
    if context['verified'] and not {'background','paper_support'} & set(families) and previous!='REJECT_MEASURED_NON_PAPER':
        feature.evidence_families=[v for v in families if v not in {'template','negative_shape','strip_consistency','competition'}]
        feature.false_family,feature.proposed_decision=conservative_identity_decision(
            feature.evidence_families,feature.needs_final_verification,feature.identity_z_score,cfg)
        context['used_for_identity']=True
    else:context['used_for_identity']=False
    return {'id':track.track_id,'previous_decision':previous,'new_decision':feature.proposed_decision,
        'used_local_identity':context['used_for_identity']}
