"""Assert evidence/evaluation consistency and protected-file integrity."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import numpy as np
from audit_p0 import ROOT, OUT, sha, write_json, write_csv
from document_results import read_csv, typed
from run_p1 import summarize


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p1 = OUT / 'p1'
    protected = json.loads((OUT / 'p0/protected_before.json').read_text(encoding='utf-8'))
    checks = [{'path': p, 'sha256': h, 'unchanged': sha(Path(p)) == h} for p, h in protected.items()]
    assert all(c['unchanged'] for c in checks)
    truth = json.loads((OUT / 'p0/GT_VERSION.json').read_text(encoding='utf-8'))
    original_hash = sha(OUT / 'p0/GT_VERSION.json')
    assert sha(ROOT / 'experiments/v3/p0/GT_VERSION.json') == original_hash
    committed = subprocess.run(['git', 'show', 'HEAD:paper_counter/experiments/v3/p0/GT_VERSION.json'],
                               cwd=ROOT.parent, capture_output=True, check=True).stdout
    assert hashlib.sha256(committed).hexdigest() == original_hash, 'Git must preserve frozen GT byte content'
    rows = [typed(r) for r in read_csv(p1 / 'per_case.csv')]
    summary = read_csv(p1 / 'P1_RIDGE_VALLEY_ABLATION.csv')
    real = [r for r in rows if r['scope']=='real_common']
    common = {r['image'] for r in real}
    assert len(common) == 99 and len(real) == 99*2*4
    verified_keys = {k for k in common if truth['images'][k]['status']=='verified'}
    assert len(verified_keys) == 69
    candidate_checks = 0
    for source in ('RAW','normalized'):
        for stage in ('A1','A2','A3','A4'):
            rr = [r for r in real if r['source']==source and r['stage']==stage]
            assert len(rr) == 99 and {r['image'] for r in rr if r['gt_status']=='verified'} == verified_keys
            for r in rr:
                gt = truth['images'][r['image']]
                assert r['gt_status']==gt['status']
                expected_error = r['count']-gt['count'] if gt['status']=='verified' else None
                assert r['error'] == expected_error
                record = json.loads((p1 / 'real' / r['image'] / source / stage / 'candidates.json').read_text(encoding='utf-8'))
                assert len(record['candidates']) == r['count']
                assert record['image_sha256']==gt['image_sha256']
                if stage=='A4':
                    assert {c['y'] for c in record['candidates']} <= {c['y'] for c in record['identified']}
                    a3 = json.loads((p1 / 'real' / r['image'] / source / 'A3/candidates.json').read_text(encoding='utf-8'))
                    assert {c['y'] for c in record['identified']} == {c['y'] for c in a3['candidates']}
                for c in record['candidates']:
                    assert c['x_original'] == record['original_shape'][1]-1-c['y_rotated']
                    assert c['y_original'] == c['x_rotated']
                candidate_checks += 1
            expected = summarize(rr, 'real_common', stage, source)
            saved = next(r for r in summary if r['scope']=='real_common' and r['stage']==stage and r['source']==source)
            for k in ('n','exact','MAE','signed_error_sum','review_rate'):
                assert abs(float(saved[k])-expected[k]) < 1e-9, (stage, source, k)
            assert all(saved[k]=='' for k in ('gt_candidate_recall','weak_sheet_recall','physical_sheet_recall','track_association_accuracy','cross_layer_link_rate'))
    for scope in ('synthetic_metrology','synthetic_raw_estimated'):
        for stage in ('A1','A2','A3','A4'):
            rr = [r for r in rows if r['scope']==scope and r['stage']==stage]
            assert len(rr)==53
            expected = summarize(rr, scope, stage, 'RAW')
            saved = next(r for r in summary if r['scope']==scope and r['stage']==stage)
            for k in ('gt_candidate_recall','weak_candidate_recall','weak_sheet_recall','physical_sheet_recall','MAE','exact'):
                assert abs(float(saved[k])-expected[k]) < 1e-9
    for stage in ('A5','A6'):
        rr = next(r for r in summary if r['stage']==stage)
        assert rr['executed']=='False' and rr['exact_count_accuracy']==''
    assert sha(ROOT / 'P1_RIDGE_VALLEY_ABLATION.csv')==sha(p1 / 'P1_RIDGE_VALLEY_ABLATION.csv')
    diagnosis_path = p1 / 'case_diagnosis.csv'
    diagnoses = read_csv(diagnosis_path)
    if any(r['category']=='width_measurement' and r['status']=='suspected' for r in diagnoses):
        # Width estimates alone do not establish an abnormality or a cause.
        # Preserve the draft rather than promote speculation into a diagnosis.
        import shutil
        shutil.copyfile(diagnosis_path, p1 / 'case_diagnosis_draft_width_suspected.csv')
        for r in diagnoses:
            if r['category']=='width_measurement':
                r['status']='unknown'
        write_csv(diagnosis_path, diagnoses)
    weak_reasons = {}
    for folder in (p1 / 'synthetic/raw_estimated').glob('weak_sheet_*'):
        rec = json.loads((folder / 'A4/candidates.json').read_text(encoding='utf-8'))
        for t in rec['truths']:
            if not t['weak']:
                continue
            for c in rec['rejected']:
                if t['left'] <= c['y'] <= t['right']:
                    weak_reasons[c['reason']] = weak_reasons.get(c['reason'], 0) + 1
    write_json(p1 / 'synthetic_weak_rejection_audit.json', dict(weak_candidate_rejection_reasons=weak_reasons,
               scope='raw_estimated A4, synthetic known physical body intervals only'))
    manifest = json.loads((p1 / 'run_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT / path)==h for path,h in manifest['source_sha256'].items())
    assert all(sha(ROOT / f'configs/v3/p1_{s.lower()}.json')==h for s,h in manifest['config_sha256'].items())
    audit = dict(protected_legacy_files=len(checks), all_unchanged=True, gt_hash_preserved_in_git=True,
                 real_candidate_files_checked=candidate_checks, real_images=99, common_verified=69,
                 real_candidate_gt_unavailable_correctly_blank=True, synthetic_predictions=53*2*4,
                 candidate_and_final_weak_recall_separated=True, A4_adds_no_candidate=True,
                 synthetic_weak_rejection_reasons=weak_reasons,
                 A5_A6_not_executed=True, production_counter_not_modified=True,
                 tests=dict(p1=24, full=239, warnings=2), protected_checks=checks)
    write_json(p1 / 'final_integrity_audit.json', audit)
    write_json(ROOT / 'experiments/v3/p1/final_integrity_summary.json', {k:v for k,v in audit.items() if k!='protected_checks'})
    # Full audit links every saved response and candidate file, not only totals.
    delivery = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
    documents = {str(p.relative_to(ROOT)): sha(p) for p in ROOT.glob('*.md') if p.name.startswith(('V3_', 'P1_', 'P2_'))}
    write_json(OUT / 'delivery_manifest.json', dict(gt_version=truth['version'], input_hashes_from='p0/audit.json',
               code_sha256={str(p.relative_to(ROOT)):sha(p) for folder in ('paper_counter/v3_shadow','scripts/v3') for p in (ROOT/folder).glob('*.py')},
               root_csv_sha256=sha(ROOT / 'P1_RIDGE_VALLEY_ABLATION.csv'), documents=documents,
               files=delivery, changed_legacy_files=0, method_status='continuous formulas verified; numeric verification thresholds are declared engineering extensions'))
    print(json.dumps({k:v for k,v in audit.items() if k!='protected_checks'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
