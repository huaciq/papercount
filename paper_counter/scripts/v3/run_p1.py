"""P1 standalone synthetic and fixed scanline ablation, no final-count writes."""
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
import argparse
import csv
import json
import sys
import time
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper_counter.v3_shadow import ShadowConfig, analyze
from paper_counter.v3_shadow.gaussian import gaussian, ridge_filter
from paper_counter.v3_shadow.bigaussian import bi_gaussian, bi_ridge_filter
from paper_counter.v3_shadow.synthetic import suite, evidence_metrics
from paper_counter.preprocessing import read_image, to_gray, normalize_illumination, crop_roi
from paper_counter.active_roi import detect_active_stack_roi
from paper_counter.config import CounterConfig
from audit_p0 import OUT, sha, write_json, write_csv, metrics

plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                     'figure.dpi': 110, 'savefig.dpi': 160})


def save_result(folder, result, metadata):
    folder.mkdir(parents=True, exist_ok=False)
    arrays = dict(result['arrays'])
    kernels = []
    for j, k in enumerate(result['kernels']):
        arrays[f'kernel_{j}_x'], arrays[f'kernel_{j}_weights'] = k['x'], k['weights']
        kernels.append({a: b for a, b in k.items() if a not in ('x', 'weights')})
    np.savez_compressed(folder / 'responses.npz', **arrays)
    write_json(folder / 'candidates.json', {**metadata, **{k: v for k, v in result.items() if k not in ('arrays', 'kernels')}, 'kernels': kernels})


def summarize(rows, scope, stage, source):
    verified = [r for r in rows if r['gt_status'] == 'verified']
    mm = metrics(verified)
    mm['exact_count_accuracy'] = mm.pop('exact_accuracy')
    n = len(rows)
    row = dict(scope=scope, stage=stage, source=source, executed=True, images=n, **mm,
               review_rate=sum(bool(r.get('review_required')) for r in rows) / n if n and stage != 'A0' else None,
               runtime_mean_seconds=float(np.mean([r['seconds'] for r in rows])) if n and stage != 'A0' else None,
               runtime_scope='standalone_width_filter_peak_verify; excludes 2D preprocessing/IO/plot' if stage != 'A0' else 'archived runtime unavailable',
               gt_candidate_recall=None, weak_sheet_recall=None, ridge_duplicate_rate=None, texture_false_positive=None,
               track_association_accuracy=None, cross_layer_link_rate=None, physical_sheet_precision=None,
               physical_sheet_recall=None, metric_unavailable_reason='no verified real curve/texture GT; no 2D association in P1',
               count_semantics='archived whole-image edge count' if stage == 'A0' else '1D single-scanline shadow center-count proxy')
    if scope.startswith('synthetic'):
        candidate_truth = sum(r['pre_gt_layers'] for r in rows)
        weak = sum(r['pre_weak_layers'] for r in rows)
        count = sum(r['post_candidate_count'] for r in rows)
        truth = sum(r['post_gt_layers'] for r in rows)
        texture_objects = sum(r['post_texture_objects'] for r in rows)
        row.update(gt_candidate_recall=sum(r['pre_tp'] for r in rows) / candidate_truth if candidate_truth else None,
                   weak_candidate_recall=sum(r['pre_matched_weak'] for r in rows) / weak if weak else None,
                   weak_sheet_recall=sum(r['post_matched_weak'] for r in rows) / weak if weak else None,
                   ridge_duplicate_rate=sum(r['post_duplicate_candidates'] for r in rows) / count if count else None,
                   texture_false_positive=sum(r['post_texture_fp'] for r in rows),
                   texture_objects=texture_objects,
                   texture_false_positive_per_object=sum(r['post_texture_fp'] for r in rows) / texture_objects if texture_objects else None,
                   physical_sheet_precision=sum(r['post_tp'] for r in rows) / count if count else None,
                   physical_sheet_recall=sum(r['post_tp'] for r in rows) / truth if truth else None,
                   metric_unavailable_reason='track metrics N/A: synthetic 1D localization only',
                   count_semantics='synthetic physical bodies; independent interval localization GT')
    return row


def plot_kernels(out):
    sigma, rho = 4., .3
    x = np.linspace(-20, 20, 2401)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(x, gaussian(x, sigma), label='Gaussian')
    ax[0, 0].plot(x, bi_gaussian(x, sigma, rho), label='Piecewise Bi-Gaussian')
    ax[0, 0].set_title('Analytic kernels (no unit-area rescaling)')
    ax[0, 1].plot(x, ridge_filter(x, sigma), label='-sigma^2 G second derivative')
    ax[0, 1].plot(x, bi_ridge_filter(x, sigma, rho), label='-sigma^2 bg second derivative')
    ax[0, 1].axvline(-sigma, color='grey', ls=':'); ax[0, 1].axvline(sigma, color='grey', ls=':')
    ax[0, 1].set_title('Scale-normalized analytic ridge filters')
    from paper_counter.v3_shadow.gaussian import sampled_kernel as gk
    from paper_counter.v3_shadow.bigaussian import sampled_kernel as bk
    for label, kernel in [('Gaussian', gk(sigma)), ('Bi-Gaussian', bk(sigma, rho))]:
        xx, ww, dc = kernel
        ax[1, 0].plot(xx, ww, '.-', label=label)
        ax[1, 1].plot(np.fft.rfftfreq(4096), np.abs(np.fft.rfft(ww, 4096)), label=label)
    ax[1, 0].set_title('Pixel-bin integrated weights; DC-corrected')
    ax[1, 1].set_title('Discrete frequency magnitude')
    for a in ax.flat:
        a.legend(fontsize=8); a.grid(alpha=.2)
    fig.suptitle('P1 kernel reproduction: sigma=4 px, rho=0.3, gamma=1')
    fig.tight_layout(); fig.savefig(out / 'kernel_comparison.png'); plt.close(fig)
    write_csv(out / 'analytic_kernels.csv', [dict(x=float(t), gaussian=float(gaussian(t, sigma)),
             bi_gaussian=float(bi_gaussian(t, sigma, rho)), gaussian_ridge=float(ridge_filter(t, sigma)),
             bi_ridge=float(bi_ridge_filter(t, sigma, rho))) for t in x])


def comparison_plot(path, signal, results, title, truths=None, yoffset=0):
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    y = np.arange(len(signal)) + yoffset
    axes[0].plot(y, signal, color='black', lw=.8, label='RAW single scanline')
    if truths:
        for t in truths:
            axes[0].axvline(t['center'], color='green', alpha=.3)
    colors = {'A1': '#4878a8', 'A2': '#e38232', 'A3': '#4c9c70', 'A4': '#d45c70'}
    for stage, r in results.items():
        if not r['arrays']:
            continue
        response = r['arrays']['response']
        axes[1].plot(y, response, lw=.8, label=f"{stage} response", color=colors[stage])
        axes[1].scatter([c['y'] + yoffset for c in r['candidates']], [c['strength'] for c in r['candidates']], s=8, color=colors[stage])
        axes[2].scatter([c['y'] + yoffset for c in r['identified']], [int(stage[1])]*len(r['identified']), marker='|', s=85, color=colors[stage], label=f'{stage} identified')
    a4 = results['A4']
    if a4['arrays']:
        for i, level in enumerate(a4['arrays']['scale_responses']):
            axes[3].plot(y, level, lw=.7, alpha=.5, label=f"B sigma={a4['arrays']['sigmas'][i]:.2f}")
        axes[3].plot(y, a4['arrays']['valley'], 'k--', lw=.8, label='independent valley')
        axes[2].scatter([c['y'] + yoffset for c in a4['candidates']], [4.35]*len(a4['candidates']), marker='|', s=95, color='purple', label='A4 verified')
        axes[2].scatter([c['y'] + yoffset for c in a4['rejected']], [4.65]*len(a4['rejected']), marker='x', s=14, color='red', label='A4 rejected')
    for a in axes:
        a.legend(loc='upper right', ncol=4, fontsize=7); a.grid(alpha=.2)
    axes[1].axhline(0, color='grey', lw=.5); axes[3].axhline(0, color='grey', lw=.5)
    axes[2].set_ylim(.8, 5.1); axes[2].set_yticks([1, 2, 3, 4, 4.35, 4.65], ['A1', 'A2', 'A3', 'A4 pre', 'A4 post', 'reject'])
    axes[3].set_xlabel('Native scanline row (px)')
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=OUT / 'p1')
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    configs = {stage: ShadowConfig(**json.loads((ROOT / f'configs/v3/p1_{stage.lower()}.json').read_text(encoding='utf-8')))
               for stage in ('A1', 'A2', 'A3', 'A4')}
    write_json(out / 'configs.json', {k: v.to_dict() for k, v in configs.items()})
    (out / 'figures').mkdir()
    plot_kernels(out / 'figures')
    rows, summary = [], []
    synthetic = suite()
    for case in synthetic:
        results = {}
        for mode in ('metrology', 'raw_estimated'):
            for stage, cfg in configs.items():
                result = analyze(case['signal'], config=cfg, measured_scales=(case['scales'], case['sigma_v']) if mode == 'metrology' else None, intensity_ceiling=255)
                pre = evidence_metrics(result['identified'], case['truths'], case['texture_intervals'])
                post = evidence_metrics(result['candidates'], case['truths'], case['texture_intervals'])
                row = dict(scope=f'synthetic_{mode}', source='RAW', stage=stage, image=case['name'],
                           dataset='synthetic', material=case['kind'], pile_group=case['name'],
                           gt_status='verified', gt_count=case['count'], count=len(result['candidates']),
                           error=len(result['candidates']) - case['count'], review_required=result['review_required'],
                           status=result['status'], seconds=result['seconds'], **{'pre_' + k: v for k, v in pre.items()},
                           **{'post_' + k: v for k, v in post.items()})
                rows.append(row)
                save_result(out / 'synthetic' / mode / case['name'] / stage, result,
                            dict(case=case['name'], truths=case['truths'], seed=case['seed'], texture_intervals=case['texture_intervals']))
                if mode == 'metrology':
                    results[stage] = result
        if case['name'].endswith('_0') or case['kind'] in ('flat', 'linear_drift', 'black_band_only'):
            comparison_plot(out / 'figures' / (case['name'] + '.png'), case['signal'], results, 'Synthetic: ' + case['name'], case['truths'])
    # Verification thresholds are undisclosed in the paper. This sensitivity
    # probe is frozen before any real-GT metric is observed, not used to pick
    # an optimal real-image threshold.
    sensitivity = []
    for area_ratio, peak_ratio in ((0., 0.), (.05, .05), (.1, .1), (.2, .2), (.1, .2), (.2, .1)):
        rr = []
        for case in synthetic:
            result = analyze(case['signal'], config=replace(configs['A4'], region_area_ratio=area_ratio, region_peak_ratio=peak_ratio),
                             measured_scales=(case['scales'], case['sigma_v']), intensity_ceiling=255)
            post = evidence_metrics(result['candidates'], case['truths'], case['texture_intervals'])
            rr.append(dict(case=case['name'], area_ratio=area_ratio, peak_ratio=peak_ratio, error=len(result['candidates'])-case['count'], **post))
        sensitivity.extend(rr)
    write_csv(out / 'threshold_sensitivity.csv', sensitivity)
    write_csv(out / 'synthetic_cases.csv', [{k: v for k, v in c.items() if k not in ('signal', 'scales', 'truths', 'texture_intervals')} for c in synthetic])
    print('SYNTHETIC DONE: 53 cases, two scale-source modes, four stages', flush=True)
    with (OUT / 'p0/A0_DUAL_E0_COUNTS.csv').open(encoding='utf-8-sig', newline='') as f:
        a0 = list(csv.DictReader(f))
    gt = json.loads((OUT / 'p0/GT_VERSION.json').read_text(encoding='utf-8'))
    prep = CounterConfig(**json.loads((OUT / 'p0/A0_effective_config.json').read_text(encoding='utf-8')))
    input_metadata, changes = [], []
    for index, baseline in enumerate(a0):
        key = baseline['image']
        image_path = ROOT.parent / '数据' / key
        image_hash = sha(image_path)
        assert image_hash == gt['images'][key]['image_sha256']
        started = time.perf_counter()
        original = read_image(image_path)
        rotated = np.rot90(original).copy()
        requested, b = crop_roi(rotated, prep.roi)
        fullgray = to_gray(requested)
        top, bottom, info = detect_active_stack_roi(fullgray, None, prep, original=requested)
        gray = fullgray[top:bottom]
        roi = dict(x1=b['x1'], x2=b['x2'], y1=b['y1'] + top, y2=b['y1'] + bottom)
        x = gray.shape[1] // 2
        norm = normalize_illumination(gray, prep.illumination_sigma, prep.epsilon)
        ceiling = np.iinfo(gray.dtype).max
        raw_signal = gray[:, x].astype(float) * (255. / ceiling)
        norm_signal = norm[:, x].astype(float)
        k = max(1, int(np.ceil(len(fullgray) * configs['A4'].reference_end_fraction)))
        reference = np.r_[fullgray[:k, x], fullgray[-k:, x]].astype(float) * (255. / ceiling)
        metadata = dict(image=key, image_sha256=image_hash, roi_rotated=roi, scanline_rotated_x=int(x + roi['x1']),
                        original_shape=list(original.shape), rotation_ccw_degrees=90,
                        inverse_coordinate_formula='original_x=original_width-1-rotated_y; original_y=rotated_x',
                        profile_extraction='one center column at native resolution; no strip or whole-image averaging',
                        preprocessing_seconds=time.perf_counter()-started, legacy_edge_count=int(baseline['count']),
                        legacy_stage_evidence_available=False, roi_source='current unchanged legacy foreground_envelope function; not archived A0 stage',
                        gt_version=gt['version'], source_dtype=str(gray.dtype))
        input_metadata.append(metadata)
        raw_results = {}
        for source, signal in (('RAW', raw_signal), ('normalized', norm_signal)):
            for stage, cfg in configs.items():
                result = analyze(signal, raw_signal, cfg, reference_values=reference, intensity_ceiling=255)
                for c in result['candidates'] + result['identified'] + result['rejected']:
                    c['y_rotated'] = c['y'] + roi['y1']
                    c['x_rotated'] = x + roi['x1']
                    c['x_original'] = int(original.shape[1] - 1 - c['y_rotated'])
                    c['y_original'] = int(c['x_rotated'])
                row = dict(scope='real_common', stage=stage, source=source, image=key, dataset=baseline['dataset'],
                           material=baseline['material'], pile_group=baseline['pile_group'], gt_status=baseline['gt_status'],
                           gt_count=int(baseline['gt_count']) if baseline['gt_status']=='verified' else None,
                           count=len(result['candidates']), legacy_edge_count=int(baseline['count']),
                           error=len(result['candidates'])-int(baseline['gt_count']) if baseline['gt_status']=='verified' else None,
                           review_required=result['review_required'], status=result['status'], seconds=result['seconds'],
                           identified_count=len(result['identified']), rejected_count=len(result['rejected']),
                           sigma_min=result['widths'].get('sigma_min'), sigma_max=result['widths'].get('sigma_max'), sigma_v=result['widths'].get('sigma_v'),
                           image_sha256=image_hash, gt_version=gt['version'])
                rows.append(row)
                save_result(out / 'real' / key / source / stage, result, metadata)
                if source == 'RAW':
                    raw_results[stage] = result
                if stage == 'A4':
                    before = {c['y'] for c in result['identified']}
                    after = {c['y'] for c in result['candidates']}
                    for c in result['rejected']:
                        changes.append(dict(image=key, source=source, observation='ridge_center', y_rotated=c['y_rotated'],
                                            change='rejected', reason=c['reason'], physical_identity_status='unknown_no_curve_GT',
                                            strength=c['strength']))
                    for y in sorted(after - before):
                        raise AssertionError('Verification invented a candidate')
        # Image names select REPORT FIGURES ONLY, never detector parameters.
        focus = any(t in key for t in ('200g4-1', '200g4-2', '7号纸3-2', '7号纸4-1', '7号纸4-2', '3-3-2', '9号纸1-2', '淋膜纸2'))
        if focus or key.startswith('验证集/'):
            target = out / 'figures/real' / (key + '.png')
            target.parent.mkdir(parents=True, exist_ok=True)
            text = f"{key}\nA0 whole image={baseline['count']}; GT={baseline['gt_count'] or baseline['gt_status']}; 1D RAW A1/A2/A3/A4=" + '/'.join(str(len(raw_results[s]['candidates'])) for s in configs)
            comparison_plot(target, raw_signal, raw_results, text, yoffset=roi['y1'])
        print(json.dumps(dict(progress=f'{index+1}/{len(a0)}', image=key, A0=int(baseline['count']),
                              RAW_A4=len(raw_results['A4']['candidates']), review=raw_results['A4']['review_required']), ensure_ascii=False), flush=True)
        del original, rotated, requested, fullgray, gray, norm
    for mode in ('metrology', 'raw_estimated'):
        for stage in configs:
            rr = [r for r in rows if r['scope'] == f'synthetic_{mode}' and r['stage'] == stage]
            summary.append(summarize(rr, f'synthetic_{mode}', stage, 'RAW'))
    summary.append(summarize(a0, 'real_common', 'A0', 'archived'))
    for source in ('RAW', 'normalized'):
        for stage in configs:
            rr = [r for r in rows if r['scope']=='real_common' and r['source']==source and r['stage']==stage]
            summary.append(summarize(rr, 'real_common', stage, source))
    for stage in ('A5', 'A6'):
        summary.append(dict(scope='real_common', stage=stage, executed=False, metric_unavailable_reason='outside authorized P0+P1 scope; not implemented'))
    write_csv(out / 'P1_RIDGE_VALLEY_ABLATION.csv', summary)
    write_csv(out / 'per_case.csv', rows)
    write_csv(out / 'candidate_changes.csv', changes, fields=['image','source','observation','y_rotated','change','reason','physical_identity_status','strength'])
    groups = []
    for grouping in ('dataset', 'material', 'pile_group'):
        for group in sorted({r[grouping] for r in rows if r['scope']=='real_common'}):
            for source in ('RAW', 'normalized'):
                for stage in configs:
                    rr = [r for r in rows if r['scope']=='real_common' and r[grouping]==group and r['source']==source and r['stage']==stage]
                    groups.append(dict(grouping=grouping, group=group, **summarize(rr, 'real_common', stage, source)))
    write_csv(out / 'group_metrics.csv', groups)
    write_json(out / 'input_manifest.json', input_metadata)
    protected = json.loads((OUT / 'p0/protected_before.json').read_text(encoding='utf-8'))
    checks = [{ 'path': p, 'before': h, 'after': sha(Path(p)), 'unchanged': sha(Path(p)) == h} for p, h in protected.items()]
    assert all(r['unchanged'] for r in checks), 'A protected legacy file changed'
    write_json(out / 'protected_after.json', checks)
    write_json(out / 'run_manifest.json', dict(algorithm_version='V3-P1-1', gt_version=gt['version'], gt_sha256=sha(OUT / 'p0/GT_VERSION.json'),
               config_sha256={k: sha(ROOT / f'configs/v3/p1_{k.lower()}.json') for k in configs},
               source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'paper_counter/v3_shadow').glob('*.py')},
               executed_stages=['A1', 'A2', 'A3', 'A4'], production_count_write=False, real_partition='audit_only',
               paper_doi='10.1109/TIM.2014.2366977', numerical_verification_thresholds_published=False,
               real_curve_gt_verified=False, stages_A5_A6_executed=False))
    print('P1 COMPLETE', str(out), flush=True)


if __name__ == '__main__':
    main()
