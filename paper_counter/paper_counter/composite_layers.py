"""Learn local complete RAW layer signatures without a bright-interior rule.

Three disjoint horizontal views retain different textures. References come
from existing measured tracks; only maxima of measured patch agreement can
propose new observations. No count or image name enters this module.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
import numpy as np
from scipy.signal import find_peaks
from .compressed_layers import accumulated_profiles
from .ordered_association import associate_directional


class CompositeLayerModel:
    def __init__(self,gray,strips,directions,offset,period,local,cfg):
        self.views,self.alignment=accumulated_profiles(gray,strips,period,cfg,directions)
        self.strips=strips;self.directions=directions;self.offset=offset
        self.period=period;self.local=local;self.cfg=cfg
        self.relative=np.linspace(-.75,.75,31)
        self.references=[];self.decisions=[];self.templates=[]
        self.statistics=defaultdict(int)

    def patches(self,s,ys,period):
        ys=np.asarray(ys,float);coordinates=ys[:,None]-self.offset+self.relative*period
        available=(coordinates[:,0]>=0)&(coordinates[:,-1]<self.views.shape[2])
        arrays=np.stack([np.interp(coordinates.ravel(),np.arange(len(line)),line).reshape(coordinates.shape)
            for line in self.views[s]],axis=1)
        amplitudes=np.percentile(arrays,90,axis=2)-np.percentile(arrays,10,axis=2)
        alignment=getattr(self,'alignment',[])
        if alignment:
            masks=alignment[s]['view_available_rows']
            valid_views=np.stack([np.all(np.interp(coordinates.ravel(),np.arange(len(mask)),mask.astype(float),left=0,right=0).reshape(coordinates.shape)>=.999,axis=1) for mask in masks],axis=1)
            amplitudes=np.where(valid_views,amplitudes,0.)
            available&=np.sum(valid_views,axis=1)>=2
        intensity=arrays-np.mean(arrays,axis=2,keepdims=True)
        derivative=np.gradient(arrays,axis=2)
        def unit(a):return a/np.maximum(np.linalg.norm(a,axis=2,keepdims=True),self.cfg.epsilon)
        vectors=unit(np.concatenate((unit(intensity),unit(derivative)),axis=2))
        return vectors,amplitudes,available

    def template(self,s,y,exclude=()):
        period=float(self.local.at(y)) if self.local is not None else self.period
        refs=[t for t in self.references if t.track_id not in exclude and s in t.strip_positions
            and abs(t.strip_positions[s]-y)<=8*period]
        refs=sorted(refs,key=lambda t:abs(t.strip_positions[s]-y))[:8]
        if len(refs)<3 or not any(t.strip_positions[s]<y for t in refs) or not any(t.strip_positions[s]>y for t in refs):return None
        vectors,amplitude,valid=self.patches(s,[t.strip_positions[s] for t in refs],period)
        # 横向积累保留亚灰度级的共同变化；不能再次用单像素亮带阈值
        # 抹掉它。弱幅度仍须通过多邻层结构一致性和独立窗口匹配。
        raw_floor=max(.03,.15*self.cfg.strip_sequence_raw_min_contrast)
        valid&=np.sum(amplitude>=raw_floor,axis=1)>=2
        if np.count_nonzero(valid)<3:return None
        refs=[t for t,ok in zip(refs,valid) if ok];vectors=vectors[valid];amplitude=amplitude[valid]
        template=np.median(vectors,axis=0)
        template/=np.maximum(np.linalg.norm(template,axis=1,keepdims=True),self.cfg.epsilon)
        agreement=np.sum(vectors*template,axis=2)
        reliability=np.median(agreement,axis=0)
        if np.count_nonzero(reliability>=.55)<2:return None
        # 若半个层间距已能复现同样结构，当前尺度无法区分内部纹理
        # 和独立纸层；保持未知，不把每条重复亮纹都补成一张纸。
        shifted,_,shift_valid=self.patches(s,[t.strip_positions[s]+.5*period for t in refs],period)
        half=np.sum(shifted*template,axis=2)
        half_score=float(np.median(np.mean(np.sort(half[shift_valid],axis=1)[:,-2:],axis=1))) if np.any(shift_valid) else 0.
        full_score=float(np.median(np.mean(np.sort(agreement,axis=1)[:,-2:],axis=1)))
        if half_score>=max(.65,full_score-.05):return None
        cutoff=float(np.clip(np.median(np.sort(agreement,axis=1)[:,-2:])-.2,.55,.85))
        return {'period':period,'vector':template,'cutoff':cutoff,'amplitude':np.median(amplitude,axis=0),
            'reference_ids':[t.track_id for t in refs],'view_reliability':reliability.tolist()}

    def score(self,s,ys,template):
        vectors,amplitude,available=self.patches(s,ys,template['period'])
        correlations=np.sum(vectors*template['vector'],axis=2)
        # Texture may change across x: compare each view with its own local
        # reference, then use two agreeing views rather than a uniform band.
        observable=amplitude>=np.maximum(max(.03,.15*self.cfg.strip_sequence_raw_min_contrast),.2*template['amplitude'])
        correlations=np.where(observable,correlations,-1.)
        scores=np.mean(np.sort(correlations,axis=1)[:,-2:],axis=1)
        valid=available&(np.sum(correlations>=template['cutoff'],axis=1)>=2)
        return scores,valid,correlations

    def reconcile(self,tracks):
        self.references=[t for t in tracks if t.source=='detected' and t.support_ratio>=.6
            and (t.boundary_polarity or self.cfg.gradient_polarity)==self.cfg.gradient_polarity]
        ordered=sorted(self.references,key=lambda t:t.center_y)
        proposals=[];windows=[]
        from .models import StripPeakCandidate
        for left,right in zip(ordered,ordered[1:]):
            rows=[[] for _ in self.strips];proofs={}
            for s in set(left.strip_positions)&set(right.strip_positions):
                a,b=left.strip_positions[s],right.strip_positions[s]
                template=self.template(s,(a+b)/2)
                if template is None:continue
                period=template['period']
                if not 1.55*period<=b-a<=5*period:continue
                ys=np.arange(int(np.ceil(a+.45*period)),int(np.floor(b-.45*period))+1)
                if len(ys)<3:continue
                scores,valid,views=self.score(s,ys,template)
                peaks,_=find_peaks(scores,prominence=.08,distance=max(2,int(.45*period)))
                for i in peaks:
                    if not valid[i]:continue
                    y=float(ys[i]);proof={'strip':s,'y':y,'score':float(scores[i]),
                        'view_correlations':views[i].tolist(),'cutoff':template['cutoff'],
                        'reference_ids':template['reference_ids'],'period':period,'left':left.track_id,'right':right.track_id,
                        'left_y':a,'right_y':b,'source':'complete_RAW_patch_not_bright_band'}
                    proofs[s,y]=proof
                    rows[s].append(StripPeakCandidate(s,y,float(scores[i]),float(scores[i]),1.,template['cutoff'],
                        'strong',local_prominence_reference=template['cutoff'],observation_source='composite_RAW_patch'))
            if not any(rows):continue
            candidates,_=associate_directional(rows,self.strips,self.directions,self.offset,self.period,self.cfg)
            for t in candidates:
                samples=[proofs[s,y] for s,y in t.strip_positions.items()]
                if len(samples)<2:continue
                # Same measured anchor pair in multiple strips; a local patch
                # match by itself does not establish an independent paper.
                t.track_id=f'composite_{len(proposals):05d}'
                t.source='recovered';t.sequence_decision='rescued';t.candidate_class='marginal'
                t.boundary_polarity=self.cfg.gradient_polarity
                t.local_evidence['composite_structure']={'verified':True,'samples':samples,
                    'bounded_by_references':True,'left':left.track_id,'right':right.track_id,
                    'complete_bright_body_required':False,'period_creates_positions':False}
                t.local_evidence['allocated_signal_gap']={'left':left.track_id,'right':right.track_id,'kind':'measured_composite_structure'}
                t.local_evidence['raw_transition']={'verified':True,'source':'multi_view_RAW_structure','samples':samples}
                proposals.append(t)
        selected=list(tracks);added=[]
        for t in sorted(proposals,key=lambda t:(-t.support_count,-np.mean([p['score'] for p in t.local_evidence['composite_structure']['samples']]))):
            period=np.median([p['period'] for p in t.local_evidence['composite_structure']['samples']])
            collision=None
            for existing in selected:
                common=set(t.strip_positions)&set(existing.strip_positions)
                gaps=[abs(t.strip_positions[s]-existing.strip_positions[s]) for s in common]
                if gaps and np.median(gaps)<.45*period:
                    collision=existing;break
            if collision:
                self.decisions.append({'id':t.track_id,'decision':'occupied_measured_layer','owner':collision.track_id});continue
            proof=t.local_evidence['composite_structure']
            parents={r.track_id:r for r in ordered}
            left=parents[proof['left']];right=parents[proof['right']]
            fragment=next((other for other in selected if same_composite_slot(t,other,left,right,period)),None)
            if fragment is not None:
                # 横向不均匀会留下不相交的片段。以共同上下纸层作坐标，
                # 同一实测槽位的两段合为一张，不能因为没有共同条带多计。
                from .tracking import make_track
                by_strip={p.strip_index:p for p in fragment.peak_candidates}
                by_strip.update({p.strip_index:p for p in t.peak_candidates})
                fresh=make_track(list(by_strip.values()),period,self.cfg.num_strips,self.cfg,fragment.source)
                merged=replace(fresh,track_id=fragment.track_id,candidate_class=fragment.candidate_class,
                    sequence_decision=fragment.sequence_decision,boundary_polarity=fragment.boundary_polarity,
                    local_evidence={**fragment.local_evidence,'composite_structure':proof})
                selected=[merged if r.track_id==fragment.track_id else r for r in selected]
                self.decisions.append({'id':t.track_id,'decision':'merged_measured_disjoint_fragment',
                    'owner':fragment.track_id,'measured_strips':sorted(merged.strip_positions)})
                continue
            selected.append(t);added.append(t)
        # Existing tracks can also carry composite proof. Never use failure of
        # this supplementary model as a reason to remove an existing track.
        for t in selected:
            if t.local_evidence.get('composite_structure',{}).get('verified'):continue
            if (t.boundary_polarity or self.cfg.gradient_polarity)!=self.cfg.gradient_polarity:continue
            samples=[]
            for s,y in t.strip_positions.items():
                template=self.template(s,y,(t.track_id,))
                if template is None:continue
                score,valid,views=self.score(s,[y],template)
                if valid[0]:samples.append({'strip':s,'y':y,'score':float(score[0]),
                    'period':template['period'],'view_correlations':views[0].tolist(),'reference_ids':template['reference_ids']})
            if len(samples)>=2:
                t.local_evidence['composite_structure']={'verified':True,'samples':samples,
                    'bounded_by_references':True,'complete_bright_body_required':False,'period_creates_positions':False}
        return sorted(selected,key=lambda t:t.center_y),{'enabled':True,'references':[t.track_id for t in self.references],
            'proposed':len(proposals),'added_ids':[t.track_id for t in added],
            'decisions':self.decisions,'evidence':'three_disjoint_RAW_views_and_local_complete_layer_templates',
            'period_creates_positions':False,'no_failed_match_deletion':True}

    def plot_evidence(self,tracks,output):
        from pathlib import Path
        import matplotlib.pyplot as plt
        chosen=[t for t in tracks if t.local_evidence.get('composite_structure',{}).get('verified')]
        chosen.sort(key=lambda t:(not t.track_id.startswith('composite_'),t.local_evidence.get('paper_body',{}).get('verified',False)))
        fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=True,constrained_layout=True)
        label='No complete multi-view template evidence in this image'
        for t in chosen:
            proof=t.local_evidence['composite_structure']['samples'][0];s=proof['strip'];y=proof['y']
            template=self.template(s,y,(t.track_id,))
            if template is None:continue
            period=template['period'];coordinates=y-self.offset+self.relative*period
            refs=[r for r in self.references if r.track_id in template['reference_ids']]
            for v,ax in enumerate(axes):
                actual=np.interp(coordinates,np.arange(self.views.shape[2]),self.views[s,v])
                reference=np.median([np.interp(r.strip_positions[s]-self.offset+self.relative*period,
                    np.arange(self.views.shape[2]),self.views[s,v]) for r in refs],axis=0)
                ax.plot(self.relative,actual,label='Measured RAW view')
                ax.plot(self.relative,reference,'--',label='Median neighboring-layer RAW reference')
                ax.axvline(0,color='black',lw=.6);ax.set_ylabel(f'View {v+1}: gray')
                ax.grid(alpha=.2);ax.legend(fontsize=8)
            label=f'{t.track_id}; strip {s+1}; detection y={y:.1f}; full structure, no bright-interior requirement'
            break
        axes[-1].set_xlabel('Offset / local spacing (0 = measured boundary position)')
        fig.suptitle(label);fig.savefig(Path(output)/'composite_layer_evidence.png',dpi=145);plt.close(fig)


def distinct_composite_observations(a,b):
    proofs=[t.local_evidence.get('composite_structure',{}) for t in (a,b)]
    if not any(p.get('verified') for p in proofs):return False
    samples=[p for proof in proofs for p in proof.get('samples',[])]
    period=float(np.median([p['period'] for p in samples])) if samples else 0.
    common=set(a.strip_positions)&set(b.strip_positions)
    return period>0 and sum(abs(a.strip_positions[s]-b.strip_positions[s])>=.45*period for s in common)>=2


def same_composite_slot(candidate,other,left,right,period):
    if other.track_id in {left.track_id,right.track_id}:return False
    common=set(candidate.strip_positions)&set(other.strip_positions)
    if common and max(abs(candidate.strip_positions[s]-other.strip_positions[s]) for s in common)>2:return False
    def phases(track):
        values=[];gaps=[]
        for s in set(track.strip_positions)&set(left.strip_positions)&set(right.strip_positions):
            a,b=left.strip_positions[s],right.strip_positions[s];y=track.strip_positions[s]
            if a<y<b:
                values.append((y-a)/(b-a));gaps.append(b-a)
        return values,gaps
    a,ag=phases(candidate);b,bg=phases(other)
    return len(a)>=2 and len(b)>=2 and abs(np.median(a)-np.median(b))*np.median(ag+bg)<=.2*period
