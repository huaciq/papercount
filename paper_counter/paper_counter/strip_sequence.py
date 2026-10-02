"""Independent RAW-supported strip sequences and ordered evidence fusion.

No count is generated from a period, missing strip, interpolated curve or
opposite-polarity agreement. Every counted observation has an actual peak
and a same-sign transition in the unnormalized input image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.signal import find_peaks, correlate

from .config import CounterConfig
from .local_detection import detect_strip_peaks
from .local_period import LocalPeriod
from .models import StripPeakCandidate, PaperEdgeTrack
from .profiles import normalize_and_fuse
from .raw_mask import StackState
from .tracking import make_track
from .vertical_sequence import SequenceResult


@dataclass
class Observation:
    peak: StripPeakCandidate
    polarity: str
    quality: float
    contrast: float
    raw_strength: float
    spacing: float
    pair_y: float | None = None
    pair_score: float = 0.0
    pair_source: str = "none"
    sources: list[float] = field(default_factory=list)
    raw_significance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**self.peak.to_dict(), 'polarity': self.polarity,
                'local_quality': self.quality, 'raw_contrast_8bit': self.contrast,
                'raw_strength': self.raw_strength, 'local_spacing': self.spacing,
                'opposite_pair_y': self.pair_y, 'pair_score': self.pair_score,
                'pair_source': self.pair_source,
                'scale_sources': self.sources, 'raw_significance': self.raw_significance}


def quality_profiles(gray: np.ndarray, strips: list[tuple[int, int]], cfg: CounterConfig) -> np.ndarray:
    """Pre-peak local observability, independent of whether a peak was found.

    Saturation and local RAW contrast are measured per strip/window. Weights
    are bounded away from zero; a absent candidate is never automatically
    reclassified as an unobservable area to inflate its neighbors' support.
    """
    from scipy.ndimage import uniform_filter1d
    depth = float(np.iinfo(gray.dtype).max)
    result = np.empty((len(strips), len(gray)), np.float32)
    window = min(len(gray), cfg.strip_sequence_period_window_rows)
    for s, (a, b) in enumerate(strips):
        raw = gray[:, a:b].astype(np.float32) * (255/depth)
        med = np.median(raw, axis=1)
        saturation = uniform_filter1d(np.mean(raw >= 254, axis=1).astype(np.float32), size=window)
        energy = uniform_filter1d(np.abs(np.gradient(med)), size=window)
        mean = uniform_filter1d(med, size=window)
        variance = np.maximum(0, uniform_filter1d(med*med, size=window)-mean*mean)
        contrast = np.sqrt(variance)
        q = (0.15+0.85*np.minimum(1, contrast/12)) * (0.25+0.75*np.minimum(1, energy/2))
        q *= 1-0.7*saturation
        result[s] = np.clip(q, 0.05, 1) if cfg.strip_sequence_quality_enabled else 1
    return result


def period_hypotheses(signal: np.ndarray, cfg: CounterConfig) -> list[dict]:
    """Keep multiple measured ACF explanations, never insert their layers."""
    values = np.asarray(signal, float)-float(np.mean(signal))
    energy = float(np.dot(values, values))
    if energy <= cfg.epsilon:
        return []
    limit = min(cfg.max_period, len(values)-2)
    acf = correlate(values, values, mode='full', method='fft')[len(values)-1:len(values)+limit]/energy
    peaks, _ = find_peaks(acf)
    peaks = [int(p) for p in peaks if cfg.min_period <= p <= limit and acf[p] >= cfg.min_acf_score]
    return [{'period': p, 'acf_score': float(acf[p])} for p in sorted(peaks, key=lambda p: -acf[p])[:4]]


def _local_spacing(observations: list[Observation], default: float, cfg: CounterConfig) -> None:
    strong = [o for o in observations if o.peak.strength_type == 'strong' and
              o.raw_significance >= cfg.strip_sequence_min_raw_significance]
    ys = np.array([o.peak.y for o in strong])
    gaps = np.diff(ys)
    mid = (ys[:-1]+ys[1:])/2
    for obs in observations:
        nearby = gaps[np.abs(mid-obs.peak.y) <= cfg.strip_sequence_period_window_rows/2]
        nearby = nearby[(nearby >= cfg.min_period) & (nearby <= cfg.max_period)]
        if len(nearby) >= 3:
            obs.spacing = float(np.median(nearby))
        else:
            obs.spacing = default


def decode_strip_sequences(data: dict, raw_intensity: np.ndarray, raw_gradient: dict,
                           quality: np.ndarray, offset: int, period: float,
                           mask, cfg: CounterConfig) -> tuple[dict, dict, dict]:
    """Detect at each scale first, deduplicate candidates, then check RAW.

    Positive and negative have independent proposals but share one physical
    RAW image. Pairing is optional evidence, never two independent votes.
    """
    sequences, candidates, diagnostics = {}, {}, {}
    for polarity in ('positive', 'negative'):
        d = data[polarity]
        hypotheses = [period_hypotheses(row, cfg) for row in d['native_profiles']]
        # Proposal thresholds need a window size, not a committed global T.
        bootstrap = (cfg.min_period+cfg.max_period)/2
        levels = d.get('independent_scale_profiles', [(0.0, d['native_profiles'])])
        detected = []
        stats = None
        for scale, level in levels:
            rows, stats = detect_strip_peaks(level, bootstrap, cfg, offset)
            detected.append((scale, rows))
        sequences[polarity], candidates[polarity] = [], []
        rejected_raw, duplicate_scales = 0, 0
        sign = 1 if polarity == 'positive' else -1
        for s in range(len(raw_intensity)):
            raw = raw_intensity[s]
            proposals = sorted([(p, scale) for scale, rows in detected for p in rows[s]
                                if p.reason is None], key=lambda item: -item[0].prominence)
            merged: list[Observation] = []
            for peak, scale in proposals:
                y = int(round(peak.y))-offset
                radius = min(cfg.multiscale_raw_support_radius, max(1, cfg.min_period//3))
                a, b = max(0, y-radius), min(len(raw), y+radius+1)
                # Locate a same-sign native RAW maximum inside a tiny corridor.
                native = raw_gradient[polarity][s, a:b]
                if not len(native) or native.max() <= cfg.epsilon:
                    rejected_raw += 1
                    continue
                raw_y = a+int(np.argmax(native))
                before = float(np.mean(raw[max(0, raw_y-3):raw_y])) if raw_y else float(raw[0])
                after = float(np.mean(raw[raw_y+1:min(len(raw), raw_y+4)])) if raw_y+1 < len(raw) else float(raw[-1])
                contrast = sign*(after-before)
                if contrast < cfg.strip_sequence_raw_min_contrast or mask.state_at(raw_y+offset) == StackState.INVALID:
                    rejected_raw += 1
                    continue
                close = next((o for o in merged if abs(o.peak.y-(raw_y+offset)) <= radius), None)
                if close is not None:
                    close.sources = sorted(set(close.sources+[float(scale)]))
                    duplicate_scales += 1
                    continue
                peak.y = float(raw_y+offset)
                peak.response_scale = float(scale)
                peak.raw_support = float(native.max())
                peak.observation_source = 'unnormalized_raw_supported_scale_candidate'
                merged.append(Observation(peak, polarity, float(quality[s, raw_y]), float(contrast),
                                          float(native.max()), period, sources=[float(scale)]))
            merged.sort(key=lambda o: o.peak.y)
            for obs in merged:
                nearby = [o.contrast for o in merged if abs(o.peak.y-obs.peak.y) <= cfg.strip_sequence_period_window_rows/2]
                reference = max(cfg.strip_sequence_raw_min_contrast, float(np.percentile(nearby, 75)))
                obs.raw_significance = obs.contrast/reference
            _local_spacing(merged, period, cfg)
            sequences[polarity].append(merged)
            candidates[polarity].append([o.peak for o in merged])
        diagnostics[polarity] = {'strip_period_hypotheses': hypotheses,
                                'raw_unsupported_proposals': rejected_raw,
                                'cross_scale_duplicates': duplicate_scales}
    # RAW paper-body morphology, not just a nearest opposite peak.
    for s, (positive, negative) in enumerate(zip(sequences['positive'], sequences['negative'])):
        raw = raw_intensity[s]
        for obs in positive+negative:
            others = negative if obs.polarity == 'positive' else positive
            viable = [o for o in others if o.raw_significance >= cfg.strip_sequence_min_raw_significance and
                      1 <= ((o.peak.y-obs.peak.y) if obs.polarity == 'positive' else (obs.peak.y-o.peak.y))
                      <= cfg.strip_sequence_pair_max_ratio*obs.spacing]
            if viable:
                paired = min(viable, key=lambda o: abs(o.peak.y-obs.peak.y))
                paired_y, paired_contrast = paired.peak.y-offset, paired.contrast
                pair_source = 'opposite_scale_candidate'
            else:
                # The other polarity's normalized proposal may be absent. A
                # RAW opposite peak can verify a body, but is not manufactured
                # into a primary observation or counted as a second paper.
                y = int(round(obs.peak.y))-offset
                extent = int(np.ceil(cfg.strip_sequence_pair_max_ratio*obs.spacing))
                a, b = (y+2, min(len(raw), y+extent)) if obs.polarity == 'positive' else (max(0, y-extent), y-1)
                response = raw_gradient['negative' if obs.polarity == 'positive' else 'positive'][s, a:b]
                if not len(response) or response.max() <= cfg.epsilon:
                    continue
                paired_y = a+int(np.argmax(response))
                before = float(np.mean(raw[max(0, paired_y-3):paired_y])) if paired_y else float(raw[0])
                after = float(np.mean(raw[paired_y+1:min(len(raw), paired_y+4)])) if paired_y+1 < len(raw) else float(raw[-1])
                paired_contrast = (before-after) if obs.polarity == 'positive' else (after-before)
                if paired_contrast < cfg.strip_sequence_raw_min_contrast:
                    continue
                pair_source = 'unnormalized_raw_opposite_peak'
            upper, lower = sorted([int(round(obs.peak.y))-offset, int(round(paired_y))])
            if lower-upper < 2:
                continue
            body = float(np.mean(raw[upper+1:lower]))
            outside = min(float(np.mean(raw[max(0, upper-3):upper+1])),
                          float(np.mean(raw[lower:min(len(raw), lower+4)])))
            body_contrast = body-outside
            if body_contrast < cfg.strip_sequence_raw_min_contrast:
                continue
            strength = min(obs.contrast, paired_contrast, body_contrast)
            obs.pair_y = float(paired_y+offset)
            obs.pair_source = pair_source
            obs.pair_score = float(np.clip(strength/max(4, 0.5*np.ptp(raw[max(0, upper-20):min(len(raw), lower+20)])), 0, 1))
    for p in diagnostics:
        diagnostics[p]['strips'] = [
            {'strip': s, 'raw_supported_count': len(rows),
             'paired_count': sum(o.pair_y is not None for o in rows),
             'observations': [{**o.to_dict(), 'raw_significance': o.raw_significance,
                               'local_state': 'candidate' if o.raw_significance >= cfg.strip_sequence_min_raw_significance else 'weak_raw_transition_pending'} for o in rows]}
            for s, rows in enumerate(sequences[p])]
    # Local classification is based on RAW contrast, not normalized peak
    # height or required periodicity. Weak interior transitions remain saved
    # for diagnosis, rather than becoming extra layers through many-strip votes.
    sequences = {p: [[o for o in rows if o.raw_significance >= cfg.strip_sequence_min_raw_significance]
                     for rows in strips] for p, strips in sequences.items()}
    return sequences, candidates, {'polarities': diagnostics, 'count_policy': 'positions_and_order_not_count_vote',
                                    'period_missing_observations': 0}


def ordered_match(reference: np.ndarray, measured: np.ndarray,
                  gates: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    """Monotone one-to-one edit alignment; unmatched edges stay explicit."""
    n, m = len(reference), len(measured)
    dp = np.empty((n+1, m+1), np.float64)
    move = np.zeros((n+1, m+1), np.uint8)
    dp[:, 0], dp[0, :] = np.arange(n+1), np.arange(m+1)
    move[1:, 0], move[0, 1:] = 1, 2
    for i in range(1, n+1):
        distances = np.abs(measured-reference[i-1])
        for j in range(1, m+1):
            match = dp[i-1, j-1]+distances[j-1]/max(1, gates[i-1]) if distances[j-1] <= gates[i-1] else np.inf
            delete, insert = dp[i-1, j]+1, dp[i, j-1]+1
            if match <= delete and match <= insert:
                dp[i, j], move[i, j] = match, 0
            elif delete <= insert:
                dp[i, j], move[i, j] = delete, 1
            else:
                dp[i, j], move[i, j] = insert, 2
    pairs, new = [], []
    i, j = n, m
    while i or j:
        step = move[i, j]
        if step == 0:
            pairs.append((i-1, j-1)); i -= 1; j -= 1
        elif step == 1:
            i -= 1
        else:
            new.append(j-1); j -= 1
    return list(reversed(pairs)), list(reversed(new))


def fuse_polarity(rows: list[list[Observation]], polarity: str, strips: list,
                  slopes: list[np.ndarray], quality: np.ndarray, offset: int,
                  period: float, cfg: CounterConfig) -> tuple[list, list, dict]:
    """Fuse observed ordered sequences before any node-exclusive path choice."""
    centers = np.array([(a+b-1)/2 for a, b in strips])
    # Quality, not count or ACF regularity, selects the initial reference.
    anchor = int(np.argmax([np.mean(quality[s]) if row else -1 for s, row in enumerate(rows)]))
    groups = [[o] for o in rows[anchor]]
    matching = []
    order = sorted((s for s in range(len(rows)) if s != anchor), key=lambda s: abs(s-anchor))
    for s in order:
        predictions, spacing = [], []
        for group in groups:
            nearest = min(group, key=lambda o: abs(o.peak.strip_index-s))
            previous = nearest.peak.strip_index
            y = int(np.clip(round(nearest.peak.y)-offset, 0, quality.shape[1]-1))
            slope = float((slopes[previous][y]+slopes[s][y])/2)
            predictions.append(nearest.peak.y+slope*(centers[s]-centers[previous]))
            spacing.append(nearest.spacing)
        permutation = np.argsort(predictions)
        groups = [groups[i] for i in permutation]
        predictions = np.asarray(predictions)[permutation]
        gates = np.maximum(2, np.asarray(spacing)[permutation]*cfg.strip_sequence_match_ratio)
        pairs, new = ordered_match(predictions, np.array([o.peak.y for o in rows[s]]), gates)
        for i, j in pairs:
            groups[i].append(rows[s][j])
        groups.extend([[rows[s][j]] for j in new])
        matching.append({'strip': s, 'matched': len(pairs), 'new_fragments': len(new)})
    accepted, pending, decisions = [], [], []
    for group in groups:
        spacing = float(np.median([o.spacing for o in group]))
        peak_rows = [o.peak for o in group]
        track = make_track(peak_rows, spacing, len(rows), cfg)
        track.boundary_polarity = polarity
        track.track_id = f'{polarity}_local_{len(decisions):05d}'
        known = {o.peak.strip_index: o for o in group}
        reference = max(group, key=lambda o: o.quality)
        y0 = int(np.clip(round(reference.peak.y)-offset, 0, quality.shape[1]-1))
        q_all = []
        for s in range(len(rows)):
            predicted = reference.peak.y+float(slopes[reference.peak.strip_index][y0])*(centers[s]-centers[reference.peak.strip_index])
            y = int(np.clip(round(predicted)-offset, 0, quality.shape[1]-1))
            q_all.append(float(quality[s, y]))
        support = sum(q_all[s] for s in known)/max(cfg.epsilon, sum(q_all))
        pair_score = float(np.median([o.pair_score for o in group]))
        contrast = float(np.median([o.contrast for o in group]))
        strong = sum(o.peak.strength_type == 'strong' for o in group)
        # A single strip is permitted only for strong RAW body+two-edge evidence
        # when other strips independently have poor local observability.
        single = (len(group) == 1 and strong and pair_score >= cfg.strip_sequence_single_pair_score
                  and reference.quality >= 0.5 and support >= 0.5)
        enough = len(group) >= cfg.strip_sequence_min_observations and support >= cfg.strip_sequence_min_weighted_support
        confirmed = bool((enough or single) and strong and contrast >= cfg.strip_sequence_raw_min_contrast
                         and track.smoothness_score >= cfg.min_track_smoothness)
        track.local_evidence = {'weighted_support': float(support), 'observed_quality': [o.quality for o in group],
                                'expected_quality': q_all, 'paired_score': pair_score, 'raw_contrast_8bit': contrast,
                                'spacing_from_local_candidates': spacing, 'single_strip_confirmation': bool(single),
                                'raw_observation_count': len(group), 'opposite_positions': {
                                    str(o.peak.strip_index): o.pair_y for o in group if o.pair_y is not None}}
        # Full-width completeness remains reported, but is not an existence gate.
        track.candidate_class = 'strong' if confirmed else 'marginal'
        track.sequence_decision = 'selected' if confirmed else 'suppressed'
        track.reason = None if confirmed else 'local_evidence_pending'
        track.total_score = float(np.clip(0.45*support+0.3*track.gradient_score+0.15*pair_score+0.1*track.smoothness_score, 0, 1))
        (accepted if confirmed else pending).append(track)
        decisions.append({'track_id': track.track_id, 'center_y': track.center_y,
                          'decision': 'confirmed' if confirmed else 'pending', **track.local_evidence})
    return sorted(accepted, key=lambda t: t.center_y), pending, {
        'anchor_strip': anchor, 'matching': matching, 'track_decisions': decisions}


def opposite_duplicates(a: PaperEdgeTrack, b: PaperEdgeTrack) -> bool:
    """Only an observed body pair may establish same-sheet identity."""
    common = set(a.strip_positions) & set(b.strip_positions)
    if not common:
        return False
    pairs = a.local_evidence.get('opposite_positions', {})
    matches = [abs(float(pairs[str(s)])-b.strip_positions[s]) <= 3
               for s in common if str(s) in pairs]
    return len(matches) >= 2 and sum(matches) >= 0.6*len(matches)


def augment_observed_tracks(existing: list[PaperEdgeTrack], proposals: list[PaperEdgeTrack],
                            cfg: CounterConfig, chosen: str) -> tuple[list[PaperEdgeTrack], dict]:
    """Add only strong observed body identities before the terminal verifier.

    Independent replacement remains available for research, but regression
    showed interior textures survive local sequence fusion. The production
    experiment keeps the established sequence and uses local evidence only
    for unrepresented bodies, without extrapolating any missing boundaries.
    """
    selected = list(existing)
    decisions = []
    for candidate in sorted(proposals, key=lambda t: -t.total_score):
        evidence = candidate.local_evidence
        spacing = evidence['spacing_from_local_candidates']
        reason = None
        if evidence['paired_score'] < 0.65 or candidate.support_count < 3:
            reason = 'insufficient_raw_body_pair'
        else:
            for other in selected:
                common = set(candidate.strip_positions) & set(other.strip_positions)
                differences = [abs(candidate.strip_positions[s]-other.strip_positions[s]) for s in common]
                same_polarity = (candidate.boundary_polarity or chosen) == (other.boundary_polarity or chosen)
                if same_polarity and differences and np.median(differences) < 0.55*spacing:
                    reason = 'existing_boundary_or_close_competition'
                    break
                # An opposite boundary needs observed body-pair exclusion even
                # when established tracks predate per-track polarity metadata.
                if not same_polarity and common:
                    opposite = evidence.get('opposite_positions', {})
                    paired = [abs(float(opposite[str(s)])-other.strip_positions[s])
                              for s in common if str(s) in opposite]
                    if paired and np.median(paired) <= 3:
                        reason = 'existing_opposite_boundary_same_body'
                        break
        if reason is None:
            candidate.source, candidate.sequence_decision = 'recovered', 'rescued'
            candidate.local_evidence['augmentation'] = True
            selected.append(candidate)
        decisions.append({'track_id': candidate.track_id, 'center_y': candidate.center_y,
                          'decision': 'added_observed_body' if reason is None else 'diagnostic_only',
                          'reason': reason})
    return sorted(selected, key=lambda t: t.center_y), {
        'mode': 'augment', 'baseline_sequence_count': len(existing),
        'added': len(selected)-len(existing), 'decisions': decisions,
        'after_verifier_recovery': False}


def choose_strip_spacing(diagnostic: dict, chosen: str, initial: float,
                        cfg: CounterConfig) -> tuple[float, dict]:
    """Correct a harmonic prior only with native ACF and RAW body observations.

    This changes a search/association scale, never adds a peak or paper. A
    conservative absolute strip quorum protects against choosing the period
    of one localized reflection. Non-harmonic or ambiguous disagreements
    retain the original scale and are reported for review.
    """
    part = diagnostic['polarities'][chosen]
    votes, evidence = [], []
    for s, row in enumerate(part['strips']):
        observed = [o for o in row['observations'] if
                    o['raw_significance'] >= cfg.strip_sequence_min_raw_significance
                    and o['pair_score'] >= 0.25]
        if len(observed) < 6:
            continue
        gaps = np.diff([o['y'] for o in observed])
        gaps = gaps[(gaps >= cfg.min_period) & (gaps <= cfg.max_period)]
        if len(gaps) < 4:
            continue
        measured = float(np.median(gaps))
        mad = float(np.median(np.abs(gaps-measured)))
        if mad/max(measured, cfg.epsilon) > 0.25:
            continue
        choices = [h for h in part['strip_period_hypotheses'][s]
                   if abs(h['period']-measured) <= 0.25*measured]
        if not choices:
            continue
        native = max(choices, key=lambda h: h['acf_score'])
        votes.append(float(native['period']))
        evidence.append({'strip': s, 'raw_gap_median': measured, 'raw_gap_mad': mad,
                         'native_acf_period': native['period'], 'native_acf_score': native['acf_score']})
    required = max(2, int(np.ceil(cfg.min_track_support_ratio*cfg.num_strips)))
    diagnostic = {'original_scale': initial, 'raw_acf_strip_evidence': evidence,
                  'required_strips': required, 'changed': False, 'period_generated_paper_count': 0}
    if len(votes) < required:
        return initial, {**diagnostic, 'reason': 'insufficient_raw_acf_quorum'}
    consensus = float(np.median(votes))
    agreeing = sum(abs(v-consensus) <= 0.20*consensus for v in votes)
    ratio = initial/consensus
    harmonic = int(round(ratio))
    if agreeing >= required and 2 <= harmonic <= 4 and abs(ratio-harmonic) <= 0.15*harmonic:
        return consensus, {**diagnostic, 'changed': True, 'selected_scale': consensus,
                           'harmonic': harmonic, 'reason': 'native_acf_and_raw_body_gap_consensus'}
    return initial, {**diagnostic, 'selected_scale': initial, 'alternative_consensus': consensus,
                     'reason': 'non_harmonic_or_ambiguous_disagreement_review_only'}


def attach_raw_body_confirmation(tracks: list[PaperEdgeTrack], diagnostic: dict,
                                 chosen: str, cfg: CounterConfig) -> None:
    """Record direct body evidence that may conflict with a global template.

    This is not a new candidate and does not run after a verifier deletion.
    A wide, spatially distributed set of strong RAW transitions is required.
    Conflicts are retained for review, not labelled definitely correct.
    """
    for track in tracks:
        polarity = track.boundary_polarity or chosen
        part = diagnostic['polarities'][polarity]['strips']
        evidence = []
        for s, y in track.strip_positions.items():
            matches = [o for o in part[s]['observations'] if abs(o['y']-y) <= 2]
            if not matches:
                continue
            observation = min(matches, key=lambda o: abs(o['y']-y))
            if observation['raw_significance'] >= 0.6 and observation['pair_score'] >= 0.65:
                evidence.append({'strip': s, 'y': observation['y'], 'pair_y': observation['opposite_pair_y'],
                                 'raw_significance': observation['raw_significance'],
                                 'pair_score': observation['pair_score']})
        spatial_groups = {min(2, int(e['strip']*3/cfg.num_strips)) for e in evidence}
        required = max(3, int(np.ceil(cfg.num_strips*.5)))
        track.local_evidence['raw_body_confirmation'] = {
            'supported': len(evidence) >= required and len(spatial_groups) >= 2,
            'required_strips': required, 'spatial_groups': sorted(spatial_groups),
            'observations': evidence, 'interpretation': 'direct_evidence_not_accuracy_probability'}


def build_strip_consensus(data, raw_intensity, raw_gradient, gray, strips,
                          slopes, offset, period, mask, chosen, cfg):
    quality = quality_profiles(gray, strips, cfg)
    sequences, candidates, debug = decode_strip_sequences(data, raw_intensity, raw_gradient,
                                                          quality, offset, period, mask, cfg)
    branch, pending, fusion = {}, [], {}
    for polarity in ('positive', 'negative'):
        branch[polarity], rejected, fusion[polarity] = fuse_polarity(
            sequences[polarity], polarity, strips, slopes, quality, offset, period, cfg)
        pending.extend(rejected)
    selected = list(branch[chosen])
    fallback, suppressed = [], []
    opposite = 'negative' if chosen == 'positive' else 'positive'
    if cfg.strip_sequence_dual_enabled:
        for candidate in branch[opposite]:
            duplicate = next((t for t in selected if opposite_duplicates(t, candidate)
                              or opposite_duplicates(candidate, t)), None)
            if duplicate is not None:
                candidate.reason = 'opposite_boundary_same_sheet'
                candidate.sequence_decision = 'suppressed'
                suppressed.append(candidate)
                continue
            # Weak/single-sided opposite tracks are diagnostic, never additions.
            if candidate.local_evidence['paired_score'] < 0.5 or candidate.support_count < 3:
                candidate.reason = 'opposite_unpaired_pending'
                candidate.sequence_decision = 'suppressed'
                pending.append(candidate)
                continue
            # A primary fragment matching this body means existence is observed,
            # but identity ambiguity remains: do not add the opposite as a second paper.
            fragment = next((t for t in pending if t.boundary_polarity == chosen and
                (opposite_duplicates(t, candidate) or opposite_duplicates(candidate, t))), None)
            if fragment is not None:
                if (fragment.support_count >= 2 and fragment.local_evidence['weighted_support'] >= 0.2
                        and candidate.local_evidence['paired_score'] >= 0.65):
                    fragment.source, fragment.sequence_decision = 'recovered', 'rescued'
                    fragment.reason = None
                    fragment.candidate_class = 'strong'
                    fragment.local_evidence['joint_confirmed_by'] = candidate.track_id
                    selected.append(fragment)
                    pending.remove(fragment)
                    fallback.append(fragment.track_id)
                candidate.reason = 'opposite_matches_primary_pending'
                candidate.sequence_decision = 'suppressed'
                pending.append(candidate)
                continue
            candidate.source = 'recovered'
            candidate.sequence_decision = 'rescued'
            selected.append(candidate)
            fallback.append(candidate.track_id)
    selected.sort(key=lambda t: t.center_y)
    # Local scale comes from measured strip candidates before final selection.
    samples = sorted((o.peak.y, o.spacing) for rows in sequences[chosen] for o in rows
                     if o.peak.strength_type == 'strong')
    ys = np.array([r[0] for r in samples], np.float32)
    vals = np.array([r[1] for r in samples], np.float32)
    if len(ys):
        unique = np.unique(ys)
        vals = np.array([np.median(vals[ys == y]) for y in unique], np.float32)
        ys = unique
    local = LocalPeriod(ys, vals, np.zeros_like(vals), period)
    debug.update(fusion=fusion, independent_counts={p: len(branch[p]) for p in branch},
                 opposite_fallback_track_ids=fallback, opposite_same_sheet_suppressed=len(suppressed),
                 confirmed_before_verifier=len(selected), pending_count=len(pending),
                 quality_source='raw_pre_peak_local_windows', quality_is_accuracy=False,
                 local_spacing_source='raw_supported_strip_candidates_before_global_selection')
    sequence = SequenceResult(selected=selected, suppressed=pending+suppressed,
                              rescued=[t for t in selected if t.track_id in fallback])
    sequence.decisions = [{'track_id': t.track_id, 'center_y': t.center_y,
                           'decision': t.sequence_decision, 'unary_score': t.total_score}
                          for t in selected+pending+suppressed]
    return selected, pending+suppressed, local, sequence, candidates[chosen], debug, quality
