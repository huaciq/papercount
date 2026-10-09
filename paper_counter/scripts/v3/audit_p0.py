"""P0: immutable, count-only baseline and GT audit. Never executes a detector."""
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import replace
import csv
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper_counter.config import CounterConfig

OUT = ROOT / 'output/v3_p0_p1_20261009_01'
OLD = ROOT / 'output/optimization_20261009/delivery_02'


def sha(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def metrics(rows):
    errors = [int(r['count']) - int(r['gt_count']) for r in rows if r['gt_status'] == 'verified']
    n = len(errors)
    return dict(n=n, exact=sum(e == 0 for e in errors), exact_accuracy=sum(e == 0 for e in errors) / n if n else None,
                MAE=sum(abs(e) for e in errors) / n if n else None,
                signed_error_sum=sum(errors) if n else None, signed_error_mean=sum(errors) / n if n else None,
                over_images=sum(e > 0 for e in errors), over_sheets=sum(max(e, 0) for e in errors),
                under_images=sum(e < 0 for e in errors), under_sheets=sum(max(-e, 0) for e in errors))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    out = OUT / 'p0'
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((OLD / 'delivery_manifest.json').read_text(encoding='utf-8'))
    checked = [{ 'path': str(OLD / k), 'expected': v, 'actual': sha(OLD / k)} for k, v in manifest['files'].items()]
    for name in ('GT_VERSION.json', 'per_image.csv', 'group_metrics.csv', 'integrity_audit.json'):
        assert name in manifest['files'] and sha(OLD / name) == manifest['files'][name], name
    truth_path = OLD / 'GT_VERSION.json'
    truth = json.loads(truth_path.read_text(encoding='utf-8'))
    original = json.loads((OLD / 'integrity_audit.json').read_text(encoding='utf-8'))
    human = [{**r, 'actual_now': sha(Path(r['path']))} for r in original['human_GT']]
    assert all(r['actual_now'] == r['expected'] for r in human), 'Human GT changed: stop, do not relabel'
    shutil.copyfile(truth_path, out / 'GT_VERSION.json')
    shutil.copyfile(OLD / 'per_image.csv', out / 'historical_per_image.csv')
    with (OLD / 'per_image.csv').open(encoding='utf-8-sig', newline='') as f:
        all_rows = list(csv.DictReader(f))
    schemes = ('Baseline E0', 'Dual E0', 'Baseline E1', 'Dual E1')
    rows = [r for r in all_rows if r['model'] in schemes]
    keys = sorted({r['image'] for r in rows})
    assert len(keys) == 99
    assert all(sum(r['image'] == k for r in rows) == 4 for k in keys)
    for r in rows:
        gt = truth['images'][r['image']]
        assert gt['status'] == r['gt_status'] and (str(gt['count']) if gt['count'] is not None else '') == r['gt_count']
        assert gt['material'] == r['material'] and gt['pile_group'] == r['pile_group']
    a0 = [r for r in rows if r['model'] == 'Dual E0']
    write_csv(out / 'A0_DUAL_E0_COUNTS.csv', a0)
    summary = [dict(model=m, **metrics([r for r in rows if r['model'] == m])) for m in schemes]
    write_csv(out / 'baseline_metrics.csv', summary)
    grouped = []
    for column in ('dataset', 'material', 'pile_group'):
        for group in sorted({r[column] for r in rows}):
            for model in schemes:
                rr = [r for r in rows if r[column] == group and r['model'] == model]
                grouped.append(dict(grouping=column, group=group, model=model, **metrics(rr)))
    write_csv(out / 'group_metrics.csv', grouped)
    with (OLD / 'group_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        old_groups = list(csv.DictReader(f))
    group_mismatches = []
    for g in old_groups:
        if g['model'] not in schemes:
            continue
        matches = [r for r in grouped if r['grouping'] == g['grouping'] and r['group'] == g['group'] and r['model'] == g['model']]
        assert len(matches) == 1, g
        for col in ('n', 'exact', 'MAE', 'signed_error_sum'):
            expected, actual = g[col], matches[0][col]
            if actual is None:
                good = expected == ''
            else:
                good = abs(float(expected) - actual) < 1e-9
            if not good:
                group_mismatches.append(dict(group=g['group'], model=g['model'], column=col, old=expected, new=actual))
    assert not group_mismatches
    mappings = []
    image_hashes = {}
    for key, e in truth['images'].items():
        image = ROOT.parent / '数据' / key
        actual_image = sha(image)
        image_hashes[str(image)] = actual_image
        source = Path(e['source']) if e.get('source') else None
        line = None
        if source and source.is_file() and e.get('source_line'):
            try:
                text = source.read_text(encoding='utf-8-sig')
            except UnicodeDecodeError:
                text = source.read_text(encoding='gb18030')
            lines = text.splitlines()
            index = e['source_line'] - 1
            if index < len(lines):
                line = lines[index].strip()
        mappings.append(dict(image=key, status=e['status'], count=e['count'], source=str(source) if source else '',
                             source_line=e.get('source_line'), raw_annotation=e['raw_annotation'], source_line_text=line,
                             raw_text_matches=line == e['raw_annotation'].strip() if line is not None else None,
                             mapping_method=e['mapping_method'], mapping_semantically_verified=False if 'timestamp' in e['mapping_method'] or 'order' in e['mapping_method'] else e['mapping_method'] == 'exact_image_stem',
                             image_exists=image.is_file(), image_sha_matches=actual_image == e['image_sha256'],
                             pile_group=e['pile_group'], pile_group_source=e['pile_group_source'], partition='audit_only'))
    assert all(r['image_sha_matches'] for r in mappings), 'Image content differs from frozen snapshot'
    assert not any(r['raw_text_matches'] is False for r in mappings), 'GT source line mismatch'
    write_csv(out / 'gt_mapping_audit.csv', mappings)
    write_csv(out / 'non_evaluable.csv', [r for r in mappings if r['status'] != 'verified' or r['image'] not in keys])
    cfg = CounterConfig.from_yaml(ROOT / 'configs/experiments/strip_sequence_dual.yaml')
    disable = {name: False for name in ('endpoint_identity_v2_enabled', 'raw_period_consensus_enabled',
                'strip_sequence_adaptive_body_dedup_enabled', 'strip_sequence_best_boundary_enabled',
                'raw_polarity_consensus_enabled', 'oriented_association_enabled', 'shared_raw_body_dedup_enabled',
                'raw_final_observation_guard_enabled')}
    assert all(hasattr(cfg, k) for k in disable)
    cfg = replace(cfg, **disable)
    write_json(out / 'A0_effective_config.json', cfg.to_dict())
    shutil.copyfile(ROOT / 'configs/experiments/strip_sequence_dual.yaml', out / 'A0_original_config.yaml')
    historical = {}
    for name, values in original.items():
        if isinstance(values, list) and values and isinstance(values[0], dict) and 'path' in values[0]:
            historical[name] = [{'path': r['path'], 'expected': r.get('expected'), 'actual_now': sha(Path(r['path']))} for r in values]
    protected = {}
    for folder in (ROOT / 'paper_counter', ROOT / 'configs/experiments'):
        for p in folder.glob('*'):
            if p.is_file():
                protected[str(p)] = sha(p)
    for r in human:
        protected[r['path']] = r['actual_now']
    for r in checked:
        if r['actual'] is not None:
            protected[r['path']] = r['actual']
    write_json(out / 'protected_before.json', protected)
    audit = dict(version='V3-P0-1', gt_version=truth['version'], gt_sha256=sha(truth_path),
                 gt_status_counts=dict(Counter(e['status'] for e in truth['images'].values())),
                 common_status_counts=dict(Counter(truth['images'][k]['status'] for k in keys)),
                 primary='Dual E0 archived count CSV; no reconstructed historical stages', baseline_metrics=summary,
                 delivery_file_checks=checked, human_gt_checks=human, historical_path_checks=historical,
                 source_line_mismatches=0, image_sha_mismatches=0, grouping_metric_mismatches=group_mismatches,
                 group_table_order=list(schemes), real_partition='audit_only; synthetic development only; physical stack IDs not verified',
                 old_stage_paths_missing=sum(r['actual_now'] is None for v in historical.values() for r in v),
                 config=cfg.to_dict(), input_image_sha256=image_hashes)
    write_json(out / 'audit.json', audit)
    print(json.dumps({k: audit[k] for k in ('gt_version', 'gt_status_counts', 'common_status_counts', 'baseline_metrics', 'old_stage_paths_missing')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
