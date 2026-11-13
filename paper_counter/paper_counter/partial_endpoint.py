"""Measured one-sided identity when the symmetric identity window leaves the image."""
import numpy as np


def partial_endpoint_evidence(track, spacing, raw, offset, polarity, cfg):
    radius=cfg.identity_profile_extent_ratio*spacing
    direction=1 if polarity=='positive' else -1
    complete=0; votes=[]; samples=[]; observable={}
    for s,y in sorted(track.strip_positions.items()):
        center=float(y)-offset
        if not 0<=int(s)<len(raw) or not 0<=center<len(raw[0]):
            continue
        line=np.asarray(raw[int(s)],float)
        complete+=int(center-radius>=0 and center+radius<=len(line)-1)
        paper=center+direction*np.linspace(.10,.45,12)*spacing
        outside=center-direction*np.linspace(.05,.20,8)*spacing
        # Missing pixels are unknown, never zeros or clamped boundary samples.
        valid=(paper>=0)&(paper<=len(line)-1)
        exterior=(outside>=0)&(outside<=len(line)-1)
        observable[int(s)]=bool(valid.sum()>=9 and exterior.sum()>=3)
        if valid.sum()<9 or exterior.sum()<3:
            votes.append(False);continue
        body=np.interp(paper[valid],np.arange(len(line)),line)
        gap=np.interp(outside[exterior],np.arange(len(line)),line)
        lo=max(0,int(center-2*spacing));hi=min(len(line),int(center+2*spacing)+1)
        local=line[lo:hi]
        background,bright=np.percentile(local,[10,90])
        contrast=float(np.median(body)-np.median(gap))
        supported=bool(contrast>=cfg.strip_sequence_raw_min_contrast and
                       np.median(body)>background+.25*(bright-background) and
                       np.mean(body>np.median(gap)+cfg.strip_sequence_raw_min_contrast)>=.75)
        votes.append(supported)
        samples.append(dict(strip=int(s),y=float(y),paper_median=float(np.median(body)),
                            gap_median=float(np.median(gap)),contrast=contrast,supported=supported))
    coverage=sum(votes)/max(1,cfg.num_strips)
    measured=[s for s,valid in observable.items() if valid]
    longest_missing_run=0;run=0
    if measured:
        for s in range(min(measured),max(measured)+1):
            run=0 if observable.get(s,False) else run+1
            longest_missing_run=max(longest_missing_run,run)
    # One bad strip does not split a sheet; a multi-strip loss of visibility
    # cannot establish that the reappearing edge belongs to the same sheet.
    continuous_visibility=longest_missing_run<2
    confirmed=(complete<.5*max(1,len(track.strip_positions)) and
               coverage>=cfg.min_track_support_ratio and
               continuous_visibility and
               track.continuity_ratio>=cfg.min_continuity_ratio and
               track.smoothness_score>=cfg.min_track_smoothness)
    return dict(confirmed=bool(confirmed),complete_identity_strips=int(complete),
                body_support_ratio=coverage,continuous_visibility=continuous_visibility,
                longest_unobservable_run=longest_missing_run,polarity=polarity,samples=samples,
                policy='observed_RAW_paper_and_gap_with_correct_polarity_no_extrapolation')
