"""E1.5: read-only count reevaluation and evidence-bounded diagnosis.

This script never imports or runs the detector. Human source files and saved
predictions are versioned by SHA256 and copied/read without modifying them.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('Baseline E0', 'Dual E0', 'Baseline E1', 'Dual E1')
STATES = ('verified', 'unscorable', 'pending')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def decode_text(path):
    raw = Path(path).read_bytes()
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        return raw.decode('utf-16'), 'utf-16'
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise ValueError('Cannot decode human GT: ' + str(path))


def classify_gt(text):
    """Full-line counts only: an approximate 170 is never verified as 170."""
    value = text.strip()
    if re.fullmatch(r'\d+\s*张?', value):
        return 'verified', int(re.match(r'\d+', value)[0]), 'explicit_integer_count'
    if not value or re.search(r'pending|待标|待确认|未标|未知', value, re.I):
        return 'pending', None, 'no_confirmed_count'
    if re.search(r'unscorable|成像|无法|不可评|不能评|左右|大约|约\d|看不清', value, re.I):
        return 'unscorable', None, 'human_quality_or_approximation_note'
    return 'pending', None, 'unrecognized_annotation_requires_review'


def capture_sort_key(path):
    # Timestamp order, then duplicate-capture copy number, not OS directory order.
    match = re.search(r'Pic_(\d{4})_(\d{2})_(\d{2})_(\d{6})_', path.name)
    if not match:
        raise ValueError('Ordinal table needs timestamped filenames: ' + path.name)
    duplicate = re.search(r'\((\d+)\)$', path.stem)
    return (*match.groups(), int(duplicate[1]) if duplicate else 0, path.name)


def pair_table(images, lines):
    """Preserve empty interior rows. Only one qualitative line may broadcast."""
    ordered = sorted(images, key=capture_sort_key)
    if not lines or all(not line.strip() for line in lines):
        return [(image, '', None, 'empty_table_pending') for image in ordered]
    if len(lines) == 1 and classify_gt(lines[0])[0] != 'verified':
        return [(image, lines[0], 1, 'folder_level_qualitative_note') for image in ordered]
    if len(lines) != len(ordered):
        raise ValueError(f'GT table/image cardinality mismatch: {len(lines)} vs {len(ordered)}')
    return [(image, line, i + 1, 'timestamp_order_then_duplicate_index')
            for i, (image, line) in enumerate(zip(ordered, lines))]


def normalized_stem(stem):
    return re.sub(r'[\s_\-—–]+', '', stem.removesuffix('真值')).lower()


def metadata(key):
    parts = key.split('/')
    stem = Path(parts[-1]).stem
    if len(parts) > 2:
        material = re.sub(r'^\d{6}-?', '', parts[-2])
        duplicate = re.search(r'(\d+)\s*\((\d+)\)$', stem)
        shot = re.search(r'(\d+)-(\d+)$', stem)
        if duplicate:
            pile = 'duplicate_capture/stack_' + duplicate[1]
            source = 'filename_duplicate_template_inferred'
        elif shot:
            pile = 'standard_capture/stack_' + shot[1]
            source = 'filename_last_two_numbers_inferred'
        else:
            pile, source = 'unknown/' + stem, 'no_stack_metadata'
    else:
        material = re.sub(r'[-\d]+$', '', stem) or '旧编号' + stem.split('-')[0]
        # Old validation filenames do not identify a physical stack reliably.
        pile, source = 'unknown/' + stem, 'legacy_image_separate_no_verified_stack_id'
    return {'dataset': parts[0], 'material': material,
            'pile_group': '/'.join((parts[0], material, pile)), 'pile_group_source': source}


def collect_gt(data_root, e0, output):
    entries = {}
    sources = {}
    folders = defaultdict(list)
    for key, record in e0['inputs'].items():
        image = data_root / key
        if not image.exists():
            raise ValueError('Missing frozen input: ' + str(image))
        if sha(image) != record['sha256']:
            raise ValueError('Input changed since E0: ' + key)
        folders[image.parent].append((key, image))

    def register(path):
        if path not in sources:
            text, encoding = decode_text(path)
            relative = path.relative_to(data_root)
            copy = output / 'gt_sources' / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            digest = sha(path)
            shutil.copy2(path, copy)
            if sha(copy) != digest:
                raise ValueError('GT snapshot copy failed')
            sources[path] = {'source': str(path), 'sha256': digest, 'encoding': encoding,
                             'snapshot': str(copy), 'mtime_ns': path.stat().st_mtime_ns, 'text': text}
        return sources[path]

    def add(key, image, text, source, line, mapping):
        status, count, reason = classify_gt(text)
        entries[key] = {'image_sha256': e0['inputs'][key]['sha256'],
                        'status': status, 'count': count, 'raw_annotation': text,
                        'status_reason': reason, 'source': source['source'] if source else None,
                        'source_sha256': source['sha256'] if source else None,
                        'source_line': line, 'mapping_method': mapping, **metadata(key)}

    for folder, records in sorted(folders.items()):
        table = folder / '真值表.txt'
        if table.exists():
            source = register(table)
            mapped = pair_table([image for _, image in records], source['text'].splitlines())
            keys = {image: key for key, image in records}
            for image, text, line, method in mapped:
                add(keys[image], image, text, source, line, method)
        else:
            available = list(folder.glob('*.txt'))
            for key, image in records:
                direct = [image.with_suffix('.txt'), image.with_name(image.stem + '真值.txt')]
                exact = next((p for p in direct if p.exists()), None)
                method = 'exact_image_stem'
                if exact is None:
                    candidates = [p for p in available if normalized_stem(p.stem) == normalized_stem(image.stem)]
                    if len(candidates) > 1:
                        raise ValueError('Ambiguous GT filename mapping: ' + key)
                    exact = candidates[0] if candidates else None
                    method = 'unique_punctuation_normalized_stem' if exact else 'missing_count_source'
                source = register(exact) if exact else None
                add(key, image, source['text'].strip() if source else '', source,
                    1 if source else None, method)
    version = 'e1_5_count_gt_' + canonical_hash(entries)[:16]
    snapshot = {'schema': 'e1_5_count_gt_v1', 'version': version, 'images': entries,
                'status_policy': 'explicit integer -> verified; quality/approximation -> unscorable; absent -> pending',
                'ordinal_mapping_assumption': 'timestamp order, duplicate copy number, no OS order',
                'human_GT_not_modified': True}
    write_json(output / 'gt_snapshot.json', snapshot)
    source_rows = [{k: v for k, v in source.items() if k != 'text'} for source in sources.values()]
    write_json(output / 'gt_source_manifest.json', source_rows)
    write_csv(output / 'gt_mapping.csv', [{'image': key, **entry} for key, entry in entries.items()])
    return snapshot, source_rows


def geometry_hash(tracks):
    return canonical_hash({t['track_id']: {'strip_positions': {str(s): float(y) for s, y in t['strip_positions'].items()},
                                           'boundary_polarity': t.get('boundary_polarity', ''),
                                           'total_score': t['total_score']} for t in tracks})


def stages_from_saved(debug):
    if 'evaluation_stages' in debug:
        return debug['evaluation_stages'], 'native_instant_snapshots'
    ids = debug.get('pre_verifier_sequence', {}).get('track_ids')
    if ids is None:
        raise ValueError('Historical prediction missing pre-verifier IDs')
    pool = {}
    for key in ('paper_edge_tracks', 'rejected_tracks', 'track_candidate_pool', 'strong_tracks', 'marginal_tracks'):
        for track in debug.get(key, []):
            pool.setdefault(track['track_id'], track)
    if any(tid not in pool for tid in ids):
        raise ValueError('Historical pre-verifier geometry cannot be reconstructed')
    return {'pre_verifier': [pool[tid] for tid in ids], 'final': debug['paper_edge_tracks']}, 'only_pre_and_final_reconstructed'


def compact_prediction(debug):
    stages, provenance = stages_from_saved(debug)
    final_ids = sorted(t['track_id'] for t in debug['paper_edge_tracks'])
    if len(final_ids) != debug['count'] or len(set(final_ids)) != len(final_ids):
        raise ValueError('Count does not equal unique final track count')
    classified = stages.get('classified', [])
    pre = stages['pre_verifier']
    period = debug.get('period')
    local = debug.get('median_local_period')
    ratio = period / local if period and local and local > 0 else None
    crossing = sum(bool(t.get('crossing_hard_reject')) for t in debug.get('vertical_sequence', {}).get('rejected_transitions', []))
    active = debug.get('active_roi_diagnostics', {})
    roi = debug.get('roi', {})
    fg_top, fg_bottom = active.get('foreground_top'), active.get('foreground_bottom')
    outside = bool(fg_top is not None and fg_bottom is not None and
                   (fg_top < roi.get('y1', 0) or fg_bottom >= roi.get('y2', float('inf'))))
    verifier = debug.get('final_track_verifier', {})
    endpoints = verifier.get('endpoint_identity_v2', {}).get('tracks', [])
    return {'count': debug['count'], 'stage_counts': {k: len(v) for k, v in stages.items()},
            'stage_provenance': provenance, 'pre_geometry_sha256': geometry_hash(pre),
            'final_ids': final_ids, 'pre_ids': sorted(t['track_id'] for t in pre),
            'period': period, 'median_local_period': local, 'period_to_local_ratio': ratio,
            'roi': roi, 'active_roi_reason': active.get('reason'), 'raw_foreground_outside_roi': outside,
            'sequence_suppressed_count': len(debug.get('sequence_suppressed_tracks', [])),
            'sequence_suppressed_ids': [t['track_id'] for t in debug.get('sequence_suppressed_tracks', [])],
            'crossing_hard_reject_count': crossing,
            'large_gap_anomaly_count': debug.get('large_gap_anomaly_count', 0),
            'legacy_kept_ids': sorted(verifier.get('legacy_terminal', {}).get('kept_track_ids', [])),
            'endpoints': endpoints, 'applied_reject_ids': verifier.get('applied_reject_track_ids', []),
            'rotation_ccw_degrees': debug.get('rotation_ccw_degrees', 0),
            'image_shape': debug.get('image_shape'),
            'classified_ids': [t['track_id'] for t in classified]}


def collect_predictions(e0_root, e1_run, output):
    e0 = read_json(e0_root / 'e0/manifest.json')
    rows_path = e1_run / 'rows.json'
    old_rows = read_json(rows_path)
    predictions = {name: {} for name in MODELS}
    sources = []
    controls = [e0_root / 'e0/manifest.json', rows_path, e1_run / 'manifest.json',
                e0_root / 'e0/ground_truth.json']
    jobs = []
    for item in e0['outputs']:
        snapshot = Path(item['snapshot'])
        controls.append(snapshot)
        model = 'Baseline E0' if item['variant'] == 'baseline' else 'Dual E0'
        for rec in read_json(snapshot):
            jobs.append((model, rec['image'], Path(rec['debug']), rec['count'], rec['sha256']))
    for row in old_rows:
        model = 'Baseline E1' if row['variant'] == 'baseline' else 'Dual E1'
        jobs.append((model, row['image'], Path(row['output']) / 'debug_info.json', row['e1']['final_count'], None))
    for index, (model, key, path, count, frozen_hash) in enumerate(jobs):
        digest = sha(path)
        if frozen_hash and digest != frozen_hash:
            raise ValueError('Frozen E0 prediction changed: ' + str(path))
        debug = read_json(path)
        if debug['count'] != count:
            raise ValueError('Saved prediction differs from frozen index count: ' + key)
        if key in predictions[model]:
            raise ValueError('Duplicate prediction key')
        rec = compact_prediction(debug)
        rec['path'], rec['sha256'] = str(path), digest
        predictions[model][key] = rec
        sources.append({'model': model, 'image': key, 'path': str(path), 'sha256': digest})
        if (index + 1) % 50 == 0:
            print(f'PREDICTIONS {index + 1}/{len(jobs)}', flush=True)
    control_hashes = {str(path): sha(path) for path in controls}
    write_json(output / 'prediction_manifest.json', {'predictions': sources, 'controls': control_hashes})
    return e0, predictions, sources, control_hashes


def metric(errors):
    n = len(errors)
    return {'n': n, 'exact': sum(e == 0 for e in errors),
            'exact_accuracy': sum(e == 0 for e in errors) / n if n else None,
            'MAE': sum(abs(e) for e in errors) / n if n else None,
            'signed_error': sum(errors) if n else None,
            'mean_signed_error': sum(errors) / n if n else None,
            'absolute_error_sum': sum(abs(e) for e in errors),
            'overcount_images': sum(e > 0 for e in errors), 'undercount_images': sum(e < 0 for e in errors),
            'overcount_sheets': sum(e for e in errors if e > 0),
            'undercount_sheets': sum(-e for e in errors if e < 0)}


def change(before, after, gt):
    a, b = abs(before - gt), abs(after - gt)
    return 'improved' if b < a else 'worsened' if b > a else 'unchanged'


def paired_metrics(keys, predictions, entries, variant):
    a, b = ('Baseline E0', 'Baseline E1') if variant == 'baseline' else ('Dual E0', 'Dual E1')
    outcomes = Counter(change(predictions[a][k]['count'], predictions[b][k]['count'], entries[k]['count']) for k in keys)
    return {'n': len(keys), 'improved': outcomes['improved'], 'worsened': outcomes['worsened'],
            'unchanged': outcomes['unchanged'],
            'count_unchanged': sum(predictions[a][k]['count'] == predictions[b][k]['count'] for k in keys),
            'equal_absolute_error_but_count_changed': sum(predictions[a][k]['count'] != predictions[b][k]['count'] and
                 change(predictions[a][k]['count'], predictions[b][k]['count'], entries[k]['count']) == 'unchanged' for k in keys),
            'exact_to_wrong': sum(predictions[a][k]['count'] == entries[k]['count'] and predictions[b][k]['count'] != entries[k]['count'] for k in keys),
            'wrong_to_exact': sum(predictions[a][k]['count'] != entries[k]['count'] and predictions[b][k]['count'] == entries[k]['count'] for k in keys)}


def audit_endpoint_pair(old, new, gt_entry):
    if old['pre_geometry_sha256'] != new['pre_geometry_sha256']:
        raise ValueError('E0/E1 upstream geometry differs')
    if old['final_ids'] != new['legacy_kept_ids']:
        raise ValueError('E1 legacy selection differs from frozen E0')
    endpoints = {e['track_id']: e for e in new['endpoints']}
    added = set(new['final_ids']) - set(old['final_ids'])
    removed = set(old['final_ids']) - set(new['final_ids'])
    if (added | removed) - set(endpoints):
        raise ValueError('E1 changed a non-endpoint track')
    records = []
    verified = gt_entry['status'] == 'verified'
    for direction, ids in [('added', added), ('removed', removed)]:
        for tid in sorted(ids):
            e = endpoints[tid]
            counterfactual = new['count'] - (1 if direction == 'added' else -1)
            gt = gt_entry['count']
            effect = change(counterfactual, new['count'], gt) if verified else 'not_evaluable'
            records.append({'track_id': tid, 'selection_change': direction,
                            'old_count': old['count'], 'new_count': new['count'],
                            'gt_status': gt_entry['status'], 'gt_count': gt,
                            'individual_count_effect': effect, 'counterfactual_without_this_change': counterfactual,
                            'count_effect_status': 'confirmed_counterfactual' if verified else 'unknown',
                            'track_identity_status': 'unknown_no_verified_track_GT',
                            'background_FP_status': 'unknown_no_verified_background_GT',
                            'center_y': e['center_y'], 'side': e['side'],
                            'support_strips': e['observed_strips'], 'raw_body_support': e['inward_raw_body_support'],
                            'foreground_support': e['foreground_support'], 'evidence_score': e['evidence_score'],
                            'background_negative_evidence': e['background_negative_evidence'],
                            'legacy_false_family': e['legacy_false_family'], 'decision': e['decision'],
                            'gates': e['gates'], 'failed_gates': e['failed_gates']})
    if new['count'] - old['count'] != len(added) - len(removed):
        raise ValueError('Endpoint set delta does not explain count delta')
    return records


def diagnosis_rows(image, model, gt, pred, old=None):
    """Count operations can be confirmed; true-edge loss needs spatial GT."""
    error = pred['count'] - gt
    if error == 0:
        return []
    common = {'image': image, 'model': model, 'gt': gt, 'prediction': pred['count'], 'error': error,
              'stage_provenance': pred['stage_provenance'], 'prediction_path': pred['path'],
              'spatial_root_cause_status': 'unknown_no_verified_track_GT'}
    result = []

    def add(category, status, scope, evidence, note):
        result.append({**common, 'category': category, 'status': status, 'evidence_scope': scope,
                       'evidence': evidence, 'interpretation': note})

    add('少计' if error < 0 else '多计', 'confirmed', 'aggregate_count', {'signed_error': error},
        '只确认最终总数差；不定位具体漏边或伪边。')
    ratio = pred['period_to_local_ratio']
    harmonic = ratio is not None and 1.75 <= ratio <= 2.25
    add('倍周期', 'suspected' if harmonic else 'unknown', 'predicted_spacing_comparison',
        {'global_period': pred['period'], 'median_local_period': pred['median_local_period'], 'ratio': ratio},
        '全局/局部预测间距接近2可疑；没有独立纸层间距GT，不确认真实倍周期漏检。' if harmonic else
        '没有可靠倍周期证据；禁止由总误差推定隔层漏检。')
    add('ROI', 'suspected' if pred['raw_foreground_outside_roi'] else 'unknown', 'saved_raw_foreground_envelope',
        {'roi': pred['roi'], 'reason': pred['active_roi_reason'], 'raw_foreground_outside_roi': pred['raw_foreground_outside_roi']},
        '保存的RAW前景超出ROI，仅作为裁剪疑点，非纸边GT。' if pred['raw_foreground_outside_roi'] else
        '缺ROI外真实纸边标注，不能确认裁剪漏纸；靠近ROI边缘本身不是证据。')
    stages = pred['stage_counts']
    associated, classified = stages.get('associated'), stages.get('classified')
    candidate = error < 0 and classified is not None and classified < gt
    add('候选', 'suspected' if candidate else 'unknown', 'early_track_count',
        {'associated': associated, 'classified': classified, 'GT': gt},
        '可用轨迹在序列前已少于GT；候选、掩码、分类或关联均可能导致，不能确定漏检位置。' if candidate else
        '候选数或纸身份不足以确定候选漏检/纹理误检，历史输出没有即时早期快照。')
    cross = pred['crossing_hard_reject_count']
    add('关联', 'suspected' if cross else 'unknown', 'saved_crossing_rejections', {'crossing_hard_reject_count': cross},
        '记录了交叉约束拒绝，可疑错连/几何冲突；缺同纸曲线GT。' if cross else
        '缺逐条带同纸对应GT，不能确认错连、断连或合并。')
    sequence_loss = classified is not None and stages.get('sequence_pass1') is not None and stages['sequence_pass1'] < classified
    sequence_suspect = error < 0 and (sequence_loss or pred['sequence_suppressed_count'] > 0)
    add('序列', 'suspected' if sequence_suspect else 'unknown', 'saved_sequence_operations',
        {'classified': classified, 'sequence_pass1': stages.get('sequence_pass1'),
         'suppressed_count': pred['sequence_suppressed_count'], 'suppressed_ids': pred['sequence_suppressed_ids']},
        '确认存在抑制操作，但被抑制对象是否真纸未知；不把数量差等同真实漏纸数。' if sequence_suspect else
        '没有足以确认序列错误的身份标注或阶段证据。')
    transition_counts = []
    for first, last in [('sequence_pass1', 'after_recovery'), ('after_recovery', 'pre_verifier')]:
        if first in stages and last in stages:
            transition_counts.append({'from': first, 'to': last, 'count_before': stages[first],
                                      'count_after': stages[last],
                                      'absolute_error_delta': abs(stages[last]-gt)-abs(stages[first]-gt)})
    worsened_transition = any(t['absolute_error_delta'] > 0 for t in transition_counts)
    changed_transition = any(t['count_before'] != t['count_after'] for t in transition_counts)
    add('恢复/融合区间', 'confirmed' if worsened_transition else 'suspected' if changed_transition else 'unknown',
        'aggregate_between_snapshot_count_transition', {'transitions': transition_counts},
        '只确认采样阶段之间总误差变化；区间含恢复、融合、几何或后续序列，不能指定子操作或真假纸边。')
    before = stages['pre_verifier']
    if old is not None and old['count'] != pred['count']:
        impact = change(old['count'], pred['count'], gt)
        add('终判', 'confirmed', 'E0_E1_counterfactual_count_effect',
            {'E0_count': old['count'], 'E1_count': pred['count'], 'impact': impact,
             'same_pre_verifier_geometry': old['pre_geometry_sha256'] == pred['pre_geometry_sha256'],
             'legacy_selection_equal': old['final_ids'] == pred['legacy_kept_ids']},
            '只确认E1端部覆写对总误差的影响；每条新增轨迹是否背景、纹理或同纸重复仍未知。')
    elif abs(pred['count'] - gt) > abs(before - gt):
        add('终判', 'confirmed', 'aggregate_terminal_count_transition',
            {'pre_verifier': before, 'final': pred['count'], 'applied_reject_ids': pred['applied_reject_ids']},
            '保存证据确认该次终判使总计数误差增大；不证明具体被拒轨迹是真纸。')
    else:
        add('终判', 'suspected' if before != pred['count'] else 'unknown', 'saved_terminal_operations',
            {'pre_verifier': before, 'final': pred['count'], 'applied_reject_ids': pred['applied_reject_ids']},
            '终判去留操作可观测，真实身份和误差根因无法由总数确认。')
    add('空间根因', 'unknown', 'count_GT_only', {},
        '本轮无verified曲线/背景负区域；不能确定具体漏检Y、纸张或伪边类型。')
    return result


def build_tables(entries, predictions):
    complete = set.intersection(*(set(predictions[name]) for name in MODELS))
    common = sorted(k for k in complete if entries[k]['status'] == 'verified')
    if not common:
        raise ValueError('No common verified images')
    metric_rows, distribution, changes, group_rows = [], [], [], []
    for scope, keys in [('all_common_verified', common)] + [
            (ds, [k for k in common if entries[k]['dataset'] == ds]) for ds in sorted({entries[k]['dataset'] for k in common})]:
        for model in MODELS:
            errors = [predictions[model][k]['count'] - entries[k]['count'] for k in keys]
            metric_rows.append({'scope': scope, 'model': model, **metric(errors)})
            distribution.extend({'scope': scope, 'model': model, 'signed_error': e, 'images': n}
                                for e, n in sorted(Counter(errors).items()))
        for variant in ('baseline', 'dual'):
            changes.append({'group_type': 'scope', 'group': scope, 'variant': variant,
                            **paired_metrics(keys, predictions, entries, variant)})
    for group_type, field in [('material', 'material'), ('pile', 'pile_group')]:
        groups = defaultdict(list)
        for k in entries:
            groups[(entries[k]['dataset'], entries[k][field])].append(k)
        for (dataset, group), all_keys in sorted(groups.items()):
            keys = [k for k in all_keys if k in common]
            statuses = Counter(entries[k]['status'] for k in all_keys)
            for model in MODELS:
                group_rows.append({'group_type': group_type, 'dataset': dataset, 'group': group,
                                   'total_images': len(all_keys), **{s + '_images': statuses[s] for s in STATES},
                                   'verified_missing_any_model': sum(entries[k]['status'] == 'verified' and k not in complete for k in all_keys),
                                   'verified_GT_values': sorted({entries[k]['count'] for k in keys}),
                                   'model': model, **metric([predictions[model][k]['count'] - entries[k]['count'] for k in keys])})
            for variant in ('baseline', 'dual'):
                changes.append({'group_type': group_type, 'dataset': dataset, 'group': group, 'variant': variant,
                                **paired_metrics(keys, predictions, entries, variant)})
    per_image, errors, excluded, endpoint_audit, diagnoses = [], [], [], [], []
    for k, entry in sorted(entries.items()):
        comparable = k in common
        missing = [model for model in MODELS if k not in predictions[model]]
        row = {'image': k, **entry, 'four_way_evaluable': comparable, 'missing_models': missing}
        for model in MODELS:
            prediction = predictions[model].get(k)
            row[model + '_count'] = prediction['count'] if prediction else None
            row[model + '_error'] = prediction['count'] - entry['count'] if comparable else None
        for variant, old_model, new_model in [('baseline', 'Baseline E0', 'Baseline E1'), ('dual', 'Dual E0', 'Dual E1')]:
            old, new = predictions[old_model].get(k), predictions[new_model].get(k)
            row[variant + '_E1_vs_E0'] = change(old['count'], new['count'], entry['count']) if comparable else 'not_evaluable'
            if old and new:
                for record in audit_endpoint_pair(old, new, entry):
                    endpoint_audit.append({'image': k, 'variant': variant, 'material': entry['material'],
                                           'pile_group': entry['pile_group'], 'E1_prediction_path': new['path'], **record})
        per_image.append(row)
        if comparable:
            if any(row[model + '_error'] != 0 for model in MODELS):
                errors.append(row)
            for model in MODELS:
                old_model = model.replace(' E1', ' E0') if model.endswith('E1') else None
                diagnoses.extend(diagnosis_rows(k, model, entry['count'], predictions[model][k],
                                 predictions[old_model][k] if old_model else None))
        else:
            excluded.append({'image': k, 'dataset': entry['dataset'], 'material': entry['material'],
                             'pile_group': entry['pile_group'], 'gt_status': entry['status'], 'gt_count': entry['count'],
                             'raw_annotation': entry['raw_annotation'], 'missing_models': missing,
                             'exclusion_reason': 'GT_' + entry['status'] if entry['status'] != 'verified' else 'verified_GT_but_missing_predictions',
                             'source': entry['source'], 'source_line': entry['source_line']})
    return {'common': common, 'complete': sorted(complete), 'metrics': metric_rows, 'distribution': distribution,
            'changes': changes, 'groups': group_rows, 'per_image': per_image, 'errors': errors,
            'excluded': excluded, 'endpoints': endpoint_audit, 'diagnoses': diagnoses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e0-root', type=Path, required=True)
    parser.add_argument('--e1-run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / '数据')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--as-of', required=True, help='User-facing date, YYYY-MM-DD')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    detector_files = [p for folder in ('paper_counter', 'configs') for p in (ROOT / folder).rglob('*')
                      if p.is_file() and p.suffix in ('.py', '.yaml')]
    detector_files.append(ROOT / 'main.py')
    detector_hashes = {str(p): sha(p) for p in detector_files}
    print('OUTPUT:', output, flush=True)
    e0 = read_json(args.e0_root / 'e0/manifest.json')
    print('PHASE: version human GT and verify input hashes', flush=True)
    snapshot, gt_sources = collect_gt(args.data_root.resolve(), e0, output)
    print('GT STATUS:', dict(Counter(e['status'] for e in snapshot['images'].values())), flush=True)
    previous_path = args.e0_root / 'e0/ground_truth.json'
    previous = read_json(previous_path)
    shutil.copy2(previous_path, output / 'gt_previous_e0_snapshot.json')
    gt_changes = []
    for key, entry in snapshot['images'].items():
        count = previous['images'][key].get('count')
        old_status = count['status'] if count else 'pending'
        old_value = count['value'] if count and count['status'] == 'verified' else None
        if (old_status, old_value) != (entry['status'], entry['count']):
            gt_changes.append({'image': key, 'old_status': old_status, 'old_count': old_value,
                               'new_status': entry['status'], 'new_count': entry['count'],
                               'latest_source': entry['source'], 'latest_source_line': entry['source_line'],
                               'latest_raw_annotation': entry['raw_annotation']})
    write_csv(output / 'gt_version_changes.csv', gt_changes)
    print('PHASE: read frozen predictions; detector is not executed', flush=True)
    _, predictions, pred_sources, controls = collect_predictions(args.e0_root.resolve(), args.e1_run.resolve(), output)
    tables = build_tables(snapshot['images'], predictions)
    for name, key in [('metrics', 'metrics'), ('error_cases', 'errors'), ('case_diagnosis', 'diagnoses'),
                      ('per_image', 'per_image'), ('group_metrics', 'groups'), ('e1_vs_e0', 'changes'),
                      ('unscorable_images', 'excluded'), ('endpoint_track_audit', 'endpoints'),
                      ('error_distribution', 'distribution')]:
        write_csv(output / (name + '.csv'), tables[key])
    write_json(output / 'analysis.json', {'gt_version': snapshot['version'], **tables})
    write_json(output / 'diagnostic_evidence.json', predictions)
    print('PHASE: integrity audit of GT, predictions, algorithms and experiment controls', flush=True)
    changed_gt = [s['source'] for s in gt_sources if sha(s['source']) != s['sha256']]
    changed_pred = [s['path'] for s in pred_sources if sha(s['path']) != s['sha256']]
    changed_core = [p for p, digest in detector_hashes.items() if sha(p) != digest]
    changed_controls = [p for p, digest in controls.items() if sha(p) != digest]
    if changed_gt or changed_pred or changed_core or changed_controls:
        raise ValueError('Read-only audit failed')
    audit = {'as_of': args.as_of, 'gt_version': snapshot['version'],
             'gt_snapshot_sha256': sha(output / 'gt_snapshot.json'),
             'common_verified_images': len(tables['common']), 'four_way_prediction_images': len(tables['complete']),
             'GT_images': len(snapshot['images']), 'prediction_files_read': len(pred_sources),
             'GT_source_files': len(gt_sources), 'changed_human_GT': changed_gt,
             'changed_predictions': changed_pred, 'changed_detector_and_config': changed_core,
             'changed_historical_controls': changed_controls, 'detector_executed': False,
             'all_paired_images_same_upstream_and_endpoint_only_selection_delta': True,
             'source_hashes': detector_hashes, 'historical_control_hashes': controls,
             'evaluation_source_sha256': {str(p): sha(p) for p in
                (Path(__file__), ROOT/'resources/e1_5_report.py', ROOT/'tests/test_e1_5_evaluation.py')},
             'previous_e0_GT_snapshot_sha256': sha(output/'gt_previous_e0_snapshot.json'),
             'GT_status_or_count_changes_since_E0': len(gt_changes),
             'ordinal_mapping': 'timestamp_then_duplicate_index; filename-free human tables require this documented assumption'}
    write_json(output / 'integrity_audit.json', audit)
    render_report(output, args, snapshot, tables, audit)
    print(json.dumps({'output': str(output), 'gt_version': snapshot['version'],
                      'metrics': [m for m in tables['metrics'] if m['scope'] == 'all_common_verified'],
                      'changes': [m for m in tables['changes'] if m['group'] == 'all_common_verified']}, ensure_ascii=False, indent=2), flush=True)


def render_report(output, args, snapshot, tables, audit):
    # Report generation is kept separate from count and evidence calculations.
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('e1_5_report', ROOT / 'resources/e1_5_report.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.render_report(output, args, snapshot, tables, audit)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
