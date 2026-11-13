"""Reject dim exterior reflection clusters against actual broad paper anchors."""
from __future__ import annotations
import numpy as np


def apply_background_region_guard(tracks,features,cfg,verifier_debug):
    diagnostic={'enabled':cfg.evidence_model,'reference_ids':[],'decisions':[]}
    if not cfg.evidence_model:return tracks,[],diagnostic
    anchors=[t for t in tracks if t.source=='detected' and t.support_ratio>=.75
        and t.sequence_decision!='rescued' and t.track_id in features
        and getattr(features[t.track_id],'complete_identity_strips',0)>=2
        and features[t.track_id].above_raw_level>features[t.track_id].below_raw_level]
    if len(anchors)<5:return tracks,[],diagnostic
    paper=float(np.median([features[t.track_id].above_raw_level for t in anchors]))
    contrast=float(np.median([features[t.track_id].raw_signed_contrast for t in anchors]))
    # 宽覆盖也可能来自反光；只有接近本图可靠纸面亮度/对比的锚点定义区域。
    anchors=[t for t in anchors if features[t.track_id].above_raw_level>=.35*paper
             and features[t.track_id].raw_signed_contrast>=.25*contrast]
    if len(anchors)<5:return tracks,[],diagnostic
    anchor_ids={t.track_id for t in anchors}
    diagnostic.update(reference_ids=[t.track_id for t in anchors],paper_reference=paper,contrast_reference=contrast)
    rejected=[]
    for t in tracks:
        f=features.get(t.track_id)
        if f is None or t.track_id in anchor_ids:continue
        level=f.above_raw_level;edge_contrast=f.raw_signed_contrast;sample_source='complete_aligned_profile'
        body_samples=[p for p in getattr(t,'local_evidence',{}).get('paper_body',{}).get('samples',[])
                      if p.get('state')=='confirmed' and p.get('body_level') is not None]
        partial=(getattr(f,'partial_endpoint',None) or {}).get('samples',[])
        if len(body_samples)>=2:
            # 完整大窗口可能只剩少数列可用，不能让这些列代替所有实测纸身。
            level=float(np.median([p['body_level'] for p in body_samples]))
            edge_contrast=float(np.median([p['contrast'] for p in body_samples]))
            sample_source='observed_bounded_RAW_bodies'
        elif not getattr(f,'complete_identity_strips',0):
            if len(partial)<2:continue
            level=float(np.median([p['paper_median'] for p in partial]))
            edge_contrast=float(np.median([p['contrast'] for p in partial]))
            sample_source='available_partial_RAW_samples'
        # 分离的暗纸也可能灰度低；宽覆盖且自身有清晰纸面/暗缝对比时保留。
        if t.support_ratio>=.75 and edge_contrast>=max(cfg.strip_sequence_raw_min_contrast,.4*level):continue
        outside=[]
        for s,y in t.strip_positions.items():
            rows=[a.strip_positions[s] for a in anchors if s in a.strip_positions]
            if rows:
                outside.append(y<min(rows)-2*f.local_spacing or y>max(rows)+2*f.local_spacing)
        # 空间上脱离可信纸堆，且真实灰度和边界对比都远低于本图纸面。
        # 真实分离的暗纸若有宽覆盖，仍是锚点；端部不可观测不是零灰度。
        low=(level<.20*paper and edge_contrast<.15*contrast)
        if len(outside)>=2 and sum(outside)>=.8*len(outside) and low:
            t.reason='exterior_dim_reflection_not_paper';rejected.append(t)
            diagnostic['decisions'].append({'id':t.track_id,'y':t.center_y,'outside_strips':sum(outside),
                'observed_strips':len(outside),'raw_paper_level':level,'raw_contrast':edge_contrast,'sample_source':sample_source})
    ids={t.track_id for t in rejected};kept=[t for t in tracks if t.track_id not in ids]
    if ids:
        verifier_debug['count_after']=len(kept)
        for name in ('proposed_reject_track_ids','applied_reject_track_ids'):
            target=verifier_debug.setdefault(name,[]);target.extend(sorted(ids-set(target)))
        verifier_debug['proposed_reject_count']=len(verifier_debug['proposed_reject_track_ids'])
        verifier_debug['applied_reject_count']=len(verifier_debug['applied_reject_track_ids'])
        for row in verifier_debug.get('track_decisions',[]):
            if row['track_id'] in ids:row.update(terminal_decision='REJECT_EXTERIOR_DIM_REFLECTION',applied=True)
    verifier_debug['background_region_guard']=diagnostic
    return kept,rejected,diagnostic
