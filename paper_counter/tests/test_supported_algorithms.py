"""Guard the supported entry points after removing legacy/narrow."""
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from paper_counter import CounterConfig, PaperCounter, PaperDetector
from paper_counter.geometry_alignment import align_columns, center_window

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('retired', ['legacy', 'narrow', 'typo'])
def test_retired_algorithms_fail_explicitly(retired):
    with pytest.raises(ValueError, match='tracks or regions'):
        CounterConfig(algorithm=retired)


def test_mutated_configuration_cannot_fall_back_to_legacy():
    cfg = CounterConfig()
    cfg.algorithm = 'legacy'
    with pytest.raises(ValueError, match='tracks and regions'):
        PaperCounter(cfg).process(np.zeros((100, 96), np.uint8))


@pytest.mark.parametrize('algorithm', ['tracks', 'regions'])
def test_packaged_algorithm_alias_and_configuration(algorithm):
    detector = PaperDetector(preset=algorithm)
    assert detector.config.algorithm == algorithm
    filename = 'default.yaml' if algorithm == 'tracks' else 'regions.yaml'
    assert detector.config == CounterConfig.from_yaml(ROOT/'configs'/filename)


def test_shared_geometry_preserves_order_and_declared_crop_mapping():
    y, x = np.mgrid[:160, :300]
    raw = np.clip(125+100*np.cos((y-.04*x)/3), 0, 255).astype(np.uint8)
    mapping, columns, valid = align_columns(raw)
    assert mapping.shape == columns.shape == valid.shape == raw.shape
    np.testing.assert_array_equal(mapping[:, raw.shape[1]//2], np.arange(len(raw)))
    assert np.min(np.diff(mapping, axis=0)) >= .099
    crop, geometry = center_window(raw, 225, dict(input_to_detection=np.eye(3).tolist()))
    start = geometry['center_window']['x1']
    np.testing.assert_array_equal(crop, raw[:, start:start+225])
    np.testing.assert_allclose(np.asarray(geometry['input_to_detection']) @ [start, 50, 1], [0, 50, 1])


@pytest.mark.parametrize('algorithm', ['tracks', 'regions'])
def test_unified_batch_exports_both_algorithms_with_frozen_configuration(tmp_path, algorithm):
    folder = tmp_path/'inputs'
    folder.mkdir()
    ok, encoded = cv2.imencode('.png', np.zeros((120, 250), np.uint8))
    assert ok
    encoded.tofile(folder/'blank.png')
    output = tmp_path/'batch'
    config = ROOT/'configs'/('default.yaml' if algorithm == 'tracks' else 'regions.yaml')
    proc = subprocess.run([sys.executable, '-X', 'utf8=0', str(ROOT/'scripts/batch_detect.py'),
        '--input', str(folder), '--config', str(config), '--output', str(output), '--workers', '1'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, timeout=90)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    verification = json.loads((output/'audit/verification.json').read_text(encoding='utf-8'))
    assert verification['algorithm'] == algorithm
    assert verification['exported'] == verification['pngs_verified'] == 1
    assert verification['source_matches_current'] and verification['inputs_unchanged']
    assert (output/'images/blank.png_result.png').exists()
    debug = json.loads((output/'audit/details/blank.png/debug_info.json').read_text(encoding='utf-8'))
    assert debug['count'] == 0
