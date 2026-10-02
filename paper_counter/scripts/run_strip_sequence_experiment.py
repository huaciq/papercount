"""Reproducible independent-strip ablation; count truth is not edge truth."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_counter import CounterConfig, PaperCounter
from paper_counter.preprocessing import read_image
from paper_counter.ground_truth import load_count_text
from paper_counter.visualization import save_overlay, save_debug
from scripts.evaluate_txt_ground_truth import find_truth


def configs():
    baseline = CounterConfig.from_yaml(ROOT/'configs/experiments/multiscale_positive.yaml')
    return {'baseline': baseline,
            'roi': replace(baseline, stack_roi_mode='foreground_envelope'),
            'association': replace(baseline, stack_roi_mode='foreground_envelope', track_validate_before_claim=True),
            'single': CounterConfig.from_yaml(ROOT/'configs/experiments/strip_sequence_single.yaml'),
            'dual': CounterConfig.from_yaml(ROOT/'configs/experiments/strip_sequence_dual.yaml'),
            'oriented': CounterConfig.from_yaml(ROOT/'configs/experiments/strip_sequence_oriented.yaml'),
            'replacement': replace(CounterConfig.from_yaml(ROOT/'configs/experiments/strip_sequence_dual.yaml'), strip_sequence_mode='replace')}


def run_one(path, source, destination, cfg, rotate, complete_debug):
    raw = read_image(path)
    original_shape = list(raw.shape)
    image = np.rot90(raw, 1).copy() if rotate else raw
    start = time.perf_counter()
    result = PaperCounter(cfg).process(image)
    elapsed = time.perf_counter()-start
    result.debug_info.update(input_image=str(path), input_original_shape=original_shape,
                             rotation_ccw_degrees=90 if rotate else 0,
                             coordinate_system='rotated detection image; x2/y2 exclusive')
    target = destination/'images'/path.relative_to(source)
    target.mkdir(parents=True, exist_ok=False)
    # Matplotlib is not thread safe. Complete plots are emitted separately.
    metadata = save_overlay(result, target, cfg)
    result.debug_info['render_metadata'] = metadata
    (target/'render_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    (target/'debug_info.json').write_text(json.dumps(result.debug_info, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if cfg.strip_sequence_enabled:
        arrays = {'quality': result.artifacts['strip_quality'],
                  'raw_intensity_8bit': result.artifacts['strip_raw_intensity_profiles']}
        for polarity, data in result.artifacts['polarities'].items():
            arrays[polarity] = data['profiles']
            arrays[polarity+'_normalized_native'] = data['native_profiles']
            arrays[polarity+'_unnormalized_original_gradient'] = result.artifacts['strip_raw_gradient_profiles'][polarity]
        arrays['original_y'] = np.arange(len(image[result.debug_info['roi']['y1']:result.debug_info['roi']['y2']]))+result.debug_info['roi']['y1']
        np.savez_compressed(target/'strip_sequences.npz', **arrays)
    truth, match, warning = find_truth(path, list(path.parent.glob('*.txt')))
    truth_count = load_count_text(truth) if truth else None
    if complete_debug:
        save_debug(result, target, cfg)
    return {'image': str(path.relative_to(source)), 'count': result.count, 'seconds': elapsed,
            'truth': truth_count, 'truth_path': str(truth) if truth else None,
            'truth_match': match, 'truth_warning': warning,
            'error': result.count-truth_count if truth_count is not None else None,
            'roi': result.debug_info['roi'], 'output': str(target),
            'pre_verifier_count': result.debug_info['pre_verifier_sequence']['count'],
            'pending': result.debug_info.get('strip_consensus', {}).get('pending_count'),
            'opposite_fallback': len(result.debug_info.get('strip_consensus', {}).get('opposite_fallback_track_ids', []))}


def summarize(rows):
    truth = [r for r in rows if r.get('error') is not None]
    return {'images': len(rows), 'truth_images': len(truth),
            'exact': sum(r['error'] == 0 for r in truth) if truth else None,
            'exact_rate': float(np.mean([r['error'] == 0 for r in truth])) if truth else None,
            'MAE': float(np.mean([abs(r['error']) for r in truth])) if truth else None,
            'signed_error': sum(r['error'] for r in truth) if truth else None,
            'mean_seconds': float(np.mean([r['seconds'] for r in rows])) if rows else None,
            'count_truth_not_edge_truth': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT/'output')
    parser.add_argument('--variants', nargs='+', choices=list(configs()), default=['baseline','roi','single','dual','oriented'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--rotate-ccw', action='store_true')
    parser.add_argument('--images', nargs='*', default=None, help='Optional exact image filename subset')
    parser.add_argument('--full-debug', action='store_true', help='Generate every diagnostic plot (serial execution)')
    args = parser.parse_args()
    source = args.input.resolve()
    if args.workers < 1 or not source.is_dir():
        parser.error('Valid input directory and positive workers required')
    images = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in {'.bmp','.png','.jpg','.jpeg','.tif','.tiff'})
    if args.images is not None:
        images = [p for p in images if p.name in args.images]
    if not images:
        parser.error('No selected images')
    now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    destination = args.output_root.resolve()/('experiment_'+now)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'input': str(source), 'rotation_ccw_degrees': 90 if args.rotate_ccw else 0,
                'revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                'git_diff_sha256': hashlib.sha256(subprocess.check_output(['git','diff','HEAD'], cwd=ROOT)).hexdigest(),
                'command_arguments': vars(args) | {'input': str(source), 'output_root': str(args.output_root)},
                'workers': 1 if args.full_debug else args.workers,
                'runtime_scope': 'processing excludes reads, rotation and artifact saving; concurrent timings not isolated benchmarks',
                'data_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in images},
                'configs': {name: configs()[name].to_dict() for name in args.variants},
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for folder in ('paper_counter','scripts') for p in (ROOT/folder).glob('*.py')},
                'unlabelled_is_diagnostic_only': True}
    (destination/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('OUTPUT:', destination, flush=True)
    all_rows, summary = {}, {}
    for variant in args.variants:
        folder = destination/variant
        folder.mkdir()
        cfg = configs()[variant]
        rows = []
        with ThreadPoolExecutor(max_workers=1 if args.full_debug else args.workers) as pool:
            jobs = [pool.submit(run_one, p, source, folder, cfg, args.rotate_ccw, args.full_debug) for p in images]
            for job in as_completed(jobs):
                row = job.result()
                rows.append(row)
                print(json.dumps({'variant': variant, 'progress': f'{len(rows)}/{len(images)}', **row}, ensure_ascii=True), flush=True)
                (folder/'summary.json').write_text(json.dumps(sorted(rows,key=lambda r:r['image']), ensure_ascii=False, indent=2), encoding='utf-8')
        rows.sort(key=lambda r:r['image'])
        all_rows[variant], summary[variant] = rows, summarize(rows)
        (destination/'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Independent strip experiment', '',
             'Count truth reports exact image count and MAE only. Missing edge truth means no Precision/Recall estimate.', '',
             '| Variant | Truth images | Exact | MAE | Signed error | Mean processing s |',
             '|---|---:|---:|---:|---:|---:|']
    for name, metrics in summary.items():
        lines.append(f"| {name} | {metrics['truth_images']} | {metrics['exact']} | {metrics['MAE']} | {metrics['signed_error']} | {metrics['mean_seconds']:.2f} |")
    lines += ['', '| Image | GT | '+' | '.join(args.variants)+' |', '|---|---:|'+'---:|'*len(args.variants)]
    for i, path in enumerate(images):
        name = str(path.relative_to(source))
        truth = next((r['truth'] for r in all_rows[args.variants[0]] if r['image'] == name), None)
        counts = [str(next(r['count'] for r in all_rows[v] if r['image'] == name)) for v in args.variants]
        lines.append('| '+name+' | '+str(truth)+' | '+' | '.join(counts)+' |')
    (destination/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('DONE:', destination, flush=True)
    manifest['source_unchanged_during_run'] = all(
        hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
        for path, digest in manifest['source_sha256'].items())
    (destination/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
