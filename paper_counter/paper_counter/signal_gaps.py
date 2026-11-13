"""RAW signal evidence for crowded tracks and large inter-track gaps.

The spacing scale is measured independently of proposed tracks. A large empty
black interval is a stack break; a gap containing bounded measured bands asks
for observed candidates. Neither case authorizes a virtual paper position.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .local_period import LocalPeriod
from .period import estimate_period
from .boundary_direction import predicted_shift
from .body_evidence import measure_body


class SignalGapModel:
    def __init__(self, raw, stripe_debug, offset, initial, preferred, cfg,
                 strips=None, directions=None, primary_raw=None):
        self.raw = np.asarray(raw, float)
        self.primary_raw=np.asarray(primary_raw,float) if primary_raw is not None else self.raw
        self.offset = offset
        self.initial = initial
        self.preferred = preferred
        self.cfg = cfg
        self.strips = strips
        self.directions = directions
        self.bands = []
        self.centers = []
        self.uppers = []
        self.lowers = []
        self.strong_bands = []
        self.max_widths = []
        self.dark_prefix = []
        self.dark_cutoff = []
        self._bindings = {}
        self._canonical = {}
        self._pairs = {}
        self._available = {}
        records = {r['strip']:r for r in stripe_debug.get('strips', [])}
        for s, line in enumerate(self.raw):
            candidates = [p for p in records.get(s, {}).get('measured_bands', [])
                if p['lower']-p['upper'] <= cfg.strip_sequence_pair_max_ratio*initial]
            # A small bright pit inside a stronger complete band is texture,
            # not an independent identity. Adjacent nonoverlapping bands stay.
            ordered = sorted(candidates, key=lambda p:(p['contrast'],p['lower']-p['upper']), reverse=True)
            kept = []
            for p in ordered:
                width = p['lower']-p['upper']
                nested = any(q['upper'] <= p['upper'] and p['lower'] <= q['lower']
                    and q['lower']-q['upper'] > width and q['contrast'] >= p['contrast'] for q in kept)
                if not nested:
                    kept.append(p)
            kept.sort(key=lambda p:(p['upper']+p['lower'])/2)
            self.bands.append(kept)
            self.centers.append(np.array([(p['upper']+p['lower'])/2 for p in kept]))
            self.uppers.append(np.array([p['upper'] for p in kept]))
            self.lowers.append(np.array([p['lower'] for p in kept]))
            self.strong_bands.append(np.array([p['snr']>=6 and p['contrast']>=4*cfg.strip_sequence_raw_min_contrast for p in kept],bool))
            self.max_widths.append(max((p['lower']-p['upper'] for p in kept),default=0.))
            # This low RAW cutoff detects actual black, not a dark paper's
            # failure to reach the stack-wide bright-paper reference.
            cutoff = float(np.percentile(line,1)+max(.5,cfg.strip_sequence_raw_min_contrast*.5))
            self.dark_cutoff.append(cutoff)
            self.dark_prefix.append(np.r_[0,np.cumsum(line <= cutoff)])
        self.local, self.period_quality = self._measure_period()
        self.local.signal_gap_model = self

    def _measure_period(self):
        height = self.raw.shape[1]
        step = max(8, int(2*self.initial))
        radius = max(step, int(self.cfg.regularity_acf_window_periods*self.initial/2))
        knots = list(range(0,height,step))
        if knots[-1] != height-1:
            knots.append(height-1)
        values, deviations, qualities = [], [], []
        # A local texture repeat must not silently replace the established
        # paper scale by a shorter harmonic. The global RAW/native consensus
        # handles harmonic changes; this field follows bounded local drift.
        tolerance=self.cfg.sequence_spacing_scale_ratio
        low = max(self.cfg.min_period,int(np.ceil(max(self.cfg.regularity_acf_min_ratio,1-tolerance)*self.initial)))
        high = min(self.cfg.max_period,int(np.floor(min(self.cfg.regularity_acf_max_ratio,1+tolerance)*self.initial)))
        detrended = gaussian_filter1d(self.raw,.6,axis=1)-gaussian_filter1d(self.raw,max(3.,self.initial),axis=1)
        for y in knots:
            votes, strengths = [], []
            for line in detrended:
                segment = line[max(0,y-radius):min(height,y+radius+1)]
                if len(segment)<3*self.initial or np.ptp(segment)<self.cfg.strip_sequence_raw_min_contrast*.5:
                    continue
                period, quality, _ = estimate_period(segment,low,high,.3,
                    self.cfg.acf_harmonic_preference,self.cfg.epsilon)
                if period is not None:
                    votes.append(period);strengths.append(quality)
            if len(votes)>=2:
                median=float(np.median(votes))
                agreeing=np.abs(np.asarray(votes)-median)<=.2*median
                if agreeing.sum()>=2:
                    values.append(float(np.median(np.asarray(votes)[agreeing])))
                    deviations.append(float(np.median(np.abs(np.asarray(votes)[agreeing]-median))))
                    qualities.append(float(np.median(np.asarray(strengths)[agreeing]))*float(np.mean(agreeing)))
                    continue
            values.append(self.initial);deviations.append(0.);qualities.append(0.)
        local=LocalPeriod(np.asarray(knots,float)+self.offset,np.asarray(values),np.asarray(deviations),self.initial)
        return local,np.asarray(qualities)

    def quality_at(self,y):
        return float(np.interp(y,self.local.y,self.period_quality))

    def _key(self, track):
        return (track.track_id,track.total_score,track.candidate_class,
                track.boundary_polarity or self.preferred,tuple(sorted(track.strip_positions.items())),
                track.local_evidence.get('sequence_coordinates_polarity'),
                bool(track.local_evidence.get('raw_transition',{}).get('verified')))

    def bindings(self, track):
        key=self._key(track)
        if key in self._bindings:
            return self._bindings[key]
        polarity=track.local_evidence.get('sequence_coordinates_polarity') or track.boundary_polarity or self.preferred
        side='upper' if polarity=='positive' else 'lower'
        result={}
        for s,y in track.strip_positions.items():
            if not 0<=s<len(self.bands):continue
            period=self.local.at(y)
            options=[]
            radius=self.max_widths[s]/2+max(2.,.15*period)
            first=int(np.searchsorted(self.centers[s],y-radius,side='left'))
            last=int(np.searchsorted(self.centers[s],y+radius,side='right'))
            for i in range(first,last):
                band=self.bands[s][i]
                distance=abs(y-band[side])
                contains=band['upper']<=y<=band['lower']
                if distance<=max(2.,.15*period) or contains:
                    options.append((not contains,distance,-band['contrast'],i))
            if options:
                result[s]=min(options)[-1]
        self._bindings[key]=result
        return result

    def canonical_track(self, track):
        if track.local_evidence.get('sequence_coordinates_polarity')==self.preferred:
            return track
        key=self._key(track)
        if key in self._canonical:return self._canonical[key]
        polarity=track.boundary_polarity or self.preferred
        positions=dict(track.strip_positions)
        if polarity != self.preferred:
            side='upper' if self.preferred=='positive' else 'lower'
            proofs=track.local_evidence.get('paper_body',{}).get('samples',[])
            positions={p['strip']:p['pair_y'] for p in proofs
                if p.get('state')=='confirmed' and p.get('pair_y') is not None}
            bindings=self.bindings(track)
            positions.update({s:self.bands[s][i][side] for s,i in bindings.items() if s not in positions})
            if not positions:positions=dict(track.strip_positions)
        copy=replace(track,center_y=float(np.median(list(positions.values()))),strip_positions=positions,
            curve_geometry={},local_evidence={**track.local_evidence,'sequence_coordinates_polarity':self.preferred})
        self._canonical[key]=copy
        return copy

    def same_identity(self,a,b):
        aa,bb=self.bindings(a),self.bindings(b)
        common=set(aa)&set(bb)
        matched=sum(aa[s]==bb[s] for s in common)
        if common:
            ac,bc=self.canonical_track(a),self.canonical_track(b)
            observed_common=set(ac.strip_positions)&set(bc.strip_positions)
            gaps=[abs(ac.strip_positions[s]-bc.strip_positions[s]) for s in observed_common]
            if gaps and np.median(gaps)>=self.cfg.sequence_short_gap_ratio*self.local.at((ac.center_y+bc.center_y)/2):
                return False
            wide=min(a.support_count,b.support_count)>=6
            observed_common=set(a.strip_positions)&set(b.strip_positions)
            required=max(2,int(np.ceil(.6*len(observed_common)))) if wide else 1
            return matched>=required
        # Disjoint fragments need a mutually consistent measured direction and
        # both actual boundaries, not proximity of their median row alone.
        if not aa or not bb or self.strips is None or self.directions is None:return False
        pairs=[(abs(s-t),s,t) for s in aa for t in bb if abs(s-t)<=self.cfg.max_skip_strips+1]
        if not pairs:return False
        _,s,t=min(pairs)
        centers=np.array([(x1+x2-1)/2 for x1,x2 in self.strips])
        p,q=self.bands[s][aa[s]],self.bands[t][bb[t]]
        for side in ('upper','lower'):
            dy=predicted_shift(self.directions,centers,s,t,p[side],self.offset)
            reverse=predicted_shift(self.directions,centers,t,s,q[side],self.offset)
            if abs(p[side]+dy-q[side])>2 or abs(q[side]+reverse-p[side])>2:return False
        return True

    def pair_evidence(self, left, right):
        a,b=self.canonical_track(left),self.canonical_track(right)
        key=(self._key(a),self._key(b))
        if key in self._pairs:return self._pairs[key]
        midpoint=(a.center_y+b.center_y)/2
        period=self.local.at(midpoint)
        common=set(a.strip_positions)&set(b.strip_positions)
        ba,bb=self.bindings(a),self.bindings(b)
        samples=[]
        for s in sorted(common):
            low,high=a.strip_positions[s],b.strip_positions[s]
            if high<=low:continue
            left_band=self.bands[s][ba[s]] if s in ba else None
            right_band=self.bands[s][bb[s]] if s in bb else None
            start=max(low+2,left_band['lower'] if left_band else low+2)
            end=min(high-2,right_band['upper'] if right_band else high-2)
            centers=self.centers[s]
            first=int(np.searchsorted(centers,start,side='right'))
            last=int(np.searchsorted(centers,end,side='left')) if end>start else first
            inside=(self.uppers[s][first:last]>=start-2)&(self.lowers[s][first:last]<=end+2)
            available=self._available_masks[s][first:last] if hasattr(self,'_available_masks') else np.zeros(len(inside),bool)
            inner_count=int(np.count_nonzero(inside))
            available_count=int(np.count_nonzero(inside&available))
            observed_edges=getattr(self,'_available_edges',{}).get(s,[])
            edge_ids={name for y,name in observed_edges if name not in {left.track_id,right.track_id}
                and low+.25*period<y<high-.25*period}
            available_count=max(available_count,len(edge_ids))
            lo=max(0,int(np.ceil(start-self.offset)));hi=min(len(self.raw[s]),int(np.floor(end-self.offset))+1)
            dark=(self.dark_prefix[s][hi]-self.dark_prefix[s][lo])/max(1,hi-lo) if hi>lo else 0.
            anchor_levels=[p['body_level'] for p in (left_band,right_band) if p]
            observable=hi>lo and len(anchor_levels)==2 and min(anchor_levels)>2*self.dark_cutoff[s]
            separate=False
            if left_band and right_band and left_band['lower']<right_band['upper'] and hi>lo:
                # Separation is relative to these measured bodies. Requiring
                # absolute black would reject compressed or reflective sheets
                # whose real gap is still gray in RAW intensity.
                cutoff=min(left_band['outside_level'],right_band['outside_level'])+.3*min(left_band['contrast'],right_band['contrast'])
                separate=bool(np.min(self.raw[s,lo:hi])<=cutoff)
            samples.append({'strip':s,'gap':high-low,'same_band':s in ba and s in bb and ba[s]==bb[s],
                'inner_bands':inner_count,'strong_inner':bool(np.any(inside&self.strong_bands[s][first:last])),
                'available_inner_candidates':available_count,
                'black_fraction':float(dark),'observable_black_test':observable,'separate_bodies':separate})
        same=sum(p['same_band'] for p in samples)
        missing=[p['inner_bands'] for p in samples if p['inner_bands']]
        available_missing=[p['available_inner_candidates'] for p in samples if p['available_inner_candidates']]
        strong_missing=any(p['strong_inner'] for p in samples)
        black=[p for p in samples if p['observable_black_test'] and p['black_fraction']>=.6 and p['inner_bands']==0]
        ratio=(b.center_y-a.center_y)/max(period,self.cfg.epsilon)
        state='ordinary'
        if same>=max(2,int(np.ceil(.6*len(samples)))):
            state='same_measured_body' if ratio<self.cfg.sequence_short_gap_ratio else 'unresolved_band_identity'
        elif ratio<self.cfg.sequence_short_gap_ratio:
            state='close_distinct_bodies' if sum(p['separate_bodies'] for p in samples)>=2 else 'crowded_unresolved'
        elif ratio>self.cfg.sequence_large_gap_anomaly_ratio:
            if len(missing)>=2 or strong_missing:state='observed_bands_inside_gap'
            elif len(black)>=max(2,int(np.ceil(.6*len(samples)))):state='measured_background_break'
            elif available_missing:state='observed_transitions_inside_gap'
            else:state='unresolved_large_gap'
        result={'state':state,'gap_period_ratio':ratio,'raw_period_quality':self.quality_at(midpoint),
            'observed_missing_band_count':int(np.median(missing)) if missing else 0,
            'available_missing_candidate_count':int(np.median(available_missing)) if available_missing else 0,
            'samples':samples,'period_creates_positions':False}
        self._pairs[key]=result
        return result

    def geometry_tracks(self,left,right):
        return self.canonical_track(left),self.canonical_track(right)

    def transition_evidence(self,track):
        """Verify a measured signed RAW edge without requiring a complete band.

        Native candidates use the same broad-strip median as their gradient;
        a shifted narrow-column sample cannot prove that this edge is absent.
        """
        direction=1 if (track.boundary_polarity or self.preferred)=='positive' else -1
        samples=[]
        for s,y in track.strip_positions.items():
            if not 0<=s<len(self.primary_raw):continue
            line=self.primary_raw[s];center=y-self.offset;period=self.local.at(y)
            distance=np.linspace(1,max(2.,.25*period),7)
            paper=center+direction*distance;outside=center-direction*distance
            if min(paper.min(),outside.min())<0 or max(paper.max(),outside.max())>=len(line):continue
            inside=np.interp(paper,np.arange(len(line)),line)
            exterior=np.interp(outside,np.arange(len(line)),line)
            lo=max(0,int(center-.5*period));hi=min(len(line),int(center+.5*period)+1)
            derivatives=np.diff(line[lo:hi]);noise=max(.5,1.4826*float(np.median(np.abs(derivatives-np.median(derivatives)))))
            contrast=float(np.percentile(inside,75)-np.median(exterior))
            minimum=self.cfg.strip_sequence_raw_min_contrast*(.5 if self.quality_at(y)>=.3 else 1.)
            samples.append({'strip':s,'contrast':contrast,'noise':noise,'supported':contrast>=max(minimum,2*noise)})
        votes=[p for p in samples if p['supported']]
        strong=any(p['contrast']>=4*self.cfg.strip_sequence_raw_min_contrast and p['contrast']/p['noise']>=6 for p in votes)
        return {'verified':len(votes)>=2 or strong,'samples':samples,
            'source':'signed_RAW_transition_at_actual_native_observations_not_virtual_positions'}

    def protect_observed_transitions(self,tracks):
        for t in tracks:
            body=t.local_evidence.get('paper_body',{})
            if body.get('state')=='absent':
                proof=self.transition_evidence(t)
                polarity=t.boundary_polarity or self.preferred
                paired=[measure_body(self.primary_raw[s],y-self.offset,polarity,self.local.at(y),
                    self.cfg.strip_sequence_raw_min_contrast) for s,y in t.strip_positions.items()]
                bodies=[p for p in paired if p['state']=='confirmed']
                bounded=(len(bodies)>=2 or any(p['snr']>=6 and p['contrast']>=4*self.cfg.strip_sequence_raw_min_contrast for p in bodies))
                if proof['verified'] and bounded:
                    body['state']='unknown'
                    body['absence_conflict']='native_RAW_transition_exists_narrow_window_not_total_absence'
                    t.local_evidence['raw_transition']=proof

    def set_candidates(self,tracks):
        # Missing RAW structure with no usable node is a review finding, not a
        # reason to discard its two valid anchors to avoid a spacing penalty.
        self._available={}
        self._available_edges={}
        self._pairs={}
        for t in tracks:
            for s,i in self.bindings(t).items():
                self._available.setdefault((s,i),set()).add(t.track_id)
            if t.local_evidence.get('raw_transition',{}).get('verified'):
                for s,y in t.strip_positions.items():self._available_edges.setdefault(s,[]).append((y,t.track_id))
        self._available_masks=[np.array([(s,i) in self._available for i in range(len(bands))],bool)
            for s,bands in enumerate(self.bands)]

    def annotate(self,tracks):
        for track in tracks:
            bindings=self.bindings(track)
            side='upper' if (track.boundary_polarity or self.preferred)=='positive' else 'lower'
            samples=[{'strip':s,'band_index':i,**self.bands[s][i],
                'on_boundary':abs(track.strip_positions[s]-self.bands[s][i][side])<=2}
                for s,i in bindings.items()]
            measured=[p for p in samples if p['on_boundary']]
            boundary_verified=(len(measured)>=max(2,int(np.ceil(.6*track.support_count))))
            track.local_evidence['signal_identity']={'samples':samples,
                'boundary_verified':boundary_verified,'evidence_is_not_terminal_exemption':True}

    def diagnostics(self,tracks):
        ordered=sorted(tracks,key=lambda t:self.canonical_track(t).center_y)
        return [{'left_track_id':a.track_id,'right_track_id':b.track_id,**self.pair_evidence(a,b)}
            for a,b in zip(ordered,ordered[1:])]

    def to_dict(self):
        return {'spacing_source':'local_RAW_autocorrelation_not_candidate_density',
            'period_samples':[{**p,'raw_quality':float(q)} for p,q in zip(self.local.to_samples(),self.period_quality)],
            'measured_band_counts':[len(v) for v in self.bands],'period_creates_positions':False}
