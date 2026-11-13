"""Evaluation-only, verified count/curve/background GT and stage attribution.

Coordinates are detection-image pixels after the recorded rotation, never PNG
overlay coordinates. Partial truth cannot estimate precision. Pending labels
never contribute to metrics. No detector imports this module.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def validate_ground_truth(data: dict) -> dict:
    if data.get('schema_version') != 1 or data.get('coordinate_space') != 'detection_image':
        raise ValueError('Expected GT v1 in detection_image coordinates')
    if not isinstance(data.get('images'), dict):
        raise ValueError('images must be keyed by dataset/relative image path')
    for name, entry in data['images'].items():
        if not re.fullmatch('[a-f0-9]{64}', entry.get('image_sha256', '')):
            raise ValueError(f'{name}: SHA256 must bind the original image file')
        if entry.get('rotation_ccw_degrees') not in (0, 90):
            raise ValueError(f'{name}: record rotation 0 or 90')
        count = entry.get('count')
        if count and (type(count.get('value')) is not int or count['value'] < 0):
            raise ValueError(f'{name}: count must be a nonnegative integer')
        if entry.get('track_completeness', 'partial') not in ('partial', 'complete'):
            raise ValueError(f'{name}: invalid track_completeness')
        if entry.get('track_completeness') == 'complete' and (
                entry.get('track_annotation_status') != 'verified' or any(
                    t.get('status') != 'verified' for t in entry.get('tracks', []))):
            raise ValueError(f'{name}: complete GT must be fully verified')
        ids = set()
        for track in entry.get('tracks', []):
            if track.get('id') in ids or not track.get('id'):
                raise ValueError(f'{name}: track IDs must be nonempty and unique')
            ids.add(track['id'])
            if track.get('status') != 'verified':
                continue
            if not track.get('source'):
                raise ValueError(f'{name}: verified track provenance required')
            if track.get('endpoint') not in ('top', 'bottom', 'internal'):
                raise ValueError(f'{name}: track endpoint must be truth-labelled')
            if not np.isfinite(track.get('tolerance_px', 3)) or track.get('tolerance_px', 3) <= 0:
                raise ValueError('Positive finite fixed truth tolerance required')
            if type(track.get('min_shared_observations',2)) is not int or track.get('min_shared_observations',2) < 2:
                raise ValueError('At least two actual shared observations required')
            if not 0 < track.get('min_span_fraction', .5) <= 1:
                raise ValueError('Truth curve span fraction must be in (0,1]')
            for boundary in track.get('boundaries', []):
                points = np.asarray(boundary.get('points', []), float)
                if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2
                        or not np.all(np.isfinite(points)) or np.any(np.diff(points[:,0]) <= 0)):
                    raise ValueError(f'{name}: boundary points need increasing X and finite XY')
                if boundary.get('polarity') not in ('positive', 'negative'):
                    raise ValueError(f'{name}: boundary polarity required')
            if not track.get('boundaries'):
                raise ValueError(f'{name}: verified tracks need actual boundary points')
        region_ids = set()
        for region in entry.get('background_regions', [])+entry.get('ignore_regions', []):
            if not region.get('id') or region['id'] in region_ids:
                raise ValueError('Region IDs must be nonempty and unique')
            region_ids.add(region['id'])
            if region.get('status') != 'verified':
                continue
            if not region.get('source'):
                raise ValueError(f'{name}: verified region provenance required')
            box = np.asarray(region.get('bbox', []), float)
            if box.shape != (4,) or not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f'{name}: background bbox must be x1,y1,x2,y2 exclusive')
            if not 0 < region.get('min_observation_fraction', .5) <= 1:
                raise ValueError('Background overlap fraction must be in (0,1]')
    return data


def load_stage_ground_truth(path: str | Path) -> dict:
    return validate_ground_truth(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def observation_points(track: dict, debug: dict) -> np.ndarray:
    strips = debug.get('strip_bounds_roi_x', [])
    offset = debug.get('roi', {}).get('x1', 0)
    points = []
    for strip, y in track.get('strip_positions', {}).items():
        index = int(strip)
        if index < len(strips):
            a, b = strips[index]
            points.append((offset + (a+b-1)/2, float(y)))
    return np.asarray(sorted(points), float).reshape(-1, 2)


def curve_distance(truth: dict, track: dict, debug: dict) -> float | None:
    points = observation_points(track, debug)
    polarity = track.get('boundary_polarity') or debug.get('gradient_polarity')
    options = []
    for boundary in truth['boundaries']:
        if boundary['polarity'] != polarity or not len(points):
            continue
        annotated = np.asarray(boundary['points'], float)
        usable = points[(points[:,0] >= annotated[0,0]) & (points[:,0] <= annotated[-1,0])]
        if len(usable) < truth.get('min_shared_observations', 2):
            continue
        # Require the predicted observation extent to cover the annotated span.
        extent = (usable[-1,0]-usable[0,0])/max(1, annotated[-1,0]-annotated[0,0])
        if extent < truth.get('min_span_fraction', .5):
            continue
        distances = np.abs(usable[:,1]-np.interp(usable[:,0], annotated[:,0], annotated[:,1]))
        error = float(np.percentile(distances, 90)) / truth.get('tolerance_px', 3)
        if error <= 1:
            options.append(error)
    return min(options) if options else None


def match_tracks(truth: list[dict], tracks: list[dict], debug: dict) -> list[dict]:
    if not truth or not tracks:
        return []
    # Dummy columns allow unmatched truth. Penalty > all possible real costs
    # ensures maximum cardinality before minimum geometric distance.
    cost = np.full((len(truth), len(tracks)+len(truth)), 1e9)
    cost[:, len(tracks):] = len(truth)+1
    for i, gt in enumerate(truth):
        for j, prediction in enumerate(tracks):
            value = curve_distance(gt, prediction, debug)
            if value is not None:
                cost[i,j] = value
    rows, columns = linear_sum_assignment(cost)
    return [{'truth_id': truth[i]['id'], 'track_id': tracks[j]['track_id'],
             'endpoint': truth[i]['endpoint'], 'normalized_p90_error': float(cost[i,j])}
            for i,j in zip(rows,columns) if j < len(tracks) and cost[i,j] <= 1]


def saved_stages(debug: dict) -> dict[str, list[dict]]:
    if 'evaluation_stages' in debug:
        return debug['evaluation_stages']
    known = {t['track_id']:t for t in debug.get('rejected_tracks', [])+debug.get('paper_edge_tracks', [])}
    pre = debug.get('pre_verifier_sequence', {}).get('track_ids', [])
    if any(t not in known for t in pre):
        raise ValueError('Historical pre-verifier stage cannot be reconstructed completely')
    return {'pre_verifier': [known[t] for t in pre], 'final': debug.get('paper_edge_tracks', [])}


def evaluate_stage(tracks: list[dict], entry: dict, debug: dict) -> dict:
    all_tracks = tracks
    ignored = []
    for track in all_tracks:
        points = observation_points(track,debug)
        for region in entry.get('ignore_regions', []):
            if region.get('status') != 'verified' or not len(points):
                continue
            x1,y1,x2,y2 = region['bbox']
            inside = ((points[:,0]>=x1)&(points[:,0]<x2)&(points[:,1]>=y1)&(points[:,1]<y2))
            if np.mean(inside)>=region.get('min_observation_fraction',.5):
                ignored.append(track['track_id'])
                break
    tracks = [t for t in all_tracks if t['track_id'] not in ignored]
    truth = [t for t in entry.get('tracks', []) if t.get('status') == 'verified']
    matches = match_tracks(truth, tracks, debug)
    endpoint_truth = [t for t in truth if t['endpoint'] in ('top','bottom')]
    endpoint_tp = sum(m['endpoint'] in ('top','bottom') for m in matches)
    background = [r for r in entry.get('background_regions', []) if r.get('status') == 'verified']
    false_ids, regions_hit = set(), set()
    for region in background:
        x1,y1,x2,y2 = region['bbox']
        for track in tracks:
            points = observation_points(track, debug)
            if not len(points):
                continue
            inside = ((points[:,0] >= x1)&(points[:,0] < x2)&(points[:,1] >= y1)&(points[:,1] < y2))
            if np.mean(inside) >= region.get('min_observation_fraction', .5):
                false_ids.add(track['track_id'])
                regions_hit.add(region['id'])
    complete = entry.get('track_completeness') == 'complete' and entry.get('track_annotation_status') == 'verified'
    precision = len(matches)/len(tracks) if complete and tracks else (1.0 if complete and not truth else None)
    return {'prediction_count': len(all_tracks), 'evaluated_track_count':len(tracks),
            'ignored_track_ids':ignored, 'verified_track_gt': len(truth), 'matches': matches,
            'matched': len(matches), 'recall': len(matches)/len(truth) if truth else None,
            'precision': precision, 'partial_gt_precision_available': False if not complete else True,
            'endpoint_gt': len(endpoint_truth), 'endpoint_tp': endpoint_tp,
            'endpoint_recall': endpoint_tp/len(endpoint_truth) if endpoint_truth else None,
            'background_gt_regions': len(background), 'background_fp': len(false_ids) if background else None,
            'background_fp_track_ids': sorted(false_ids), 'background_regions_hit': len(regions_hit),
            'background_region_fp_rate': len(regions_hit)/len(background) if background else None}


def evaluate_debug(debug: dict, entry: dict, image_sha256: str) -> dict:
    if entry['image_sha256'] != image_sha256:
        raise ValueError('GT image hash mismatch')
    if entry['rotation_ccw_degrees'] != debug.get('rotation_ccw_degrees', 0):
        raise ValueError('GT rotation mismatch; overlay/old polarity truth cannot be silently reused')
    stages = {name: evaluate_stage(tracks, entry, debug) for name,tracks in saved_stages(debug).items()}
    count = entry.get('count')
    expected = count['value'] if count and count.get('status') == 'verified' else None
    attribution = []
    previous = None
    for name, values in stages.items():
        matched = {m['truth_id'] for m in values['matches']}
        attribution.append({'stage': name, 'newly_matched_truth_ids': sorted(matched-(previous or set())),
                            'lost_truth_ids': sorted(previous-matched) if previous is not None else []})
        previous = matched
    return {'count_gt': expected, 'final_count': debug['count'],
            'count_error': debug['count']-expected if expected is not None else None,
            'stages': stages, 'attribution': attribution,
            'track_annotation_status': entry.get('track_annotation_status'),
            'metrics_unavailable_reason': 'No verified endpoint/background GT' if not any(
                s['endpoint_gt'] or s['background_gt_regions'] for s in stages.values()) else None}


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get('count_error') is not None]
    names = list(dict.fromkeys(name for row in rows for name in row['stages']))
    stages = {}
    for name in names:
        values = [r['stages'][name] for r in rows if name in r['stages']]
        denominator = sum(v['endpoint_gt'] for v in values)
        tp = sum(v['endpoint_tp'] for v in values)
        negatives = sum(v['background_gt_regions'] for v in values)
        stages[name] = {'endpoint_gt': denominator, 'endpoint_tp': tp,
                        'endpoint_recall': tp/denominator if denominator else None,
                        'background_gt_regions': negatives,
                        'background_fp': sum(v['background_fp'] or 0 for v in values) if negatives else None,
                        'background_regions_hit': sum(v['background_regions_hit'] for v in values),
                        'background_region_fp_rate': sum(v['background_regions_hit'] for v in values)/negatives if negatives else None}
    return {'images': len(rows), 'count_gt_images': len(scored),
            'exact': sum(r['count_error']==0 for r in scored) if scored else None,
            'MAE': float(np.mean([abs(r['count_error']) for r in scored])) if scored else None,
            'signed_error': sum(r['count_error'] for r in scored) if scored else None,
            'stages': stages, 'unlabelled_predictions_are_not_ground_truth': True}
