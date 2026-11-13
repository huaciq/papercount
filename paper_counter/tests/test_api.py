"""Public API contract and parity with the existing configured pipeline."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from paper_counter import CounterConfig, PaperCounter, PaperDetector, detect_papers
from test_counter import synthetic_papers


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("preset,filename", [("paper_layers_v4", "paper_layers_v4.yaml"),
                                              ("e3_3", "default.yaml")])
def test_preset_and_detection_match_existing_pipeline(preset, filename):
    image, _ = synthetic_papers(count=10)
    before = image.copy()
    config = CounterConfig.from_yaml(ROOT / "configs" / filename)
    detector = PaperDetector(preset=preset)
    assert detector.config.to_dict() == config.to_dict()
    expected = PaperCounter(config).process(image)
    actual = detector.detect(image)
    assert actual.count == expected.count > 0
    assert actual.raw_result.edge_positions == expected.edge_positions
    assert len(actual.tracks) == actual.count
    assert np.array_equal(image, before)
    json.dumps(actual.to_dict(), allow_nan=False)


def test_unicode_path_rotation_and_exports(tmp_path):
    image, _ = synthetic_papers(count=8)
    original = np.rot90(image, -1).copy()
    path = tmp_path / "纸张.png"
    cv2.imencode(".png", original)[1].tofile(path)
    result = detect_papers(path, rotate_ccw=True, output_dir=tmp_path / "result")
    reference = detect_papers(image)
    assert result.count == reference.count > 0
    assert result.original_shape == original.shape
    assert result.detection_shape == image.shape
    for track in result.tracks:
        for point in track["observations"]:
            assert point["input_x"] == original.shape[1] - 1 - point["detection_y"]
            assert point["input_y"] == point["detection_x"]
    assert (tmp_path / "result" / "result_overlay.png").is_file()
    coordinates = json.loads((tmp_path / "result" / "detection_coordinates.json").read_text("utf-8"))
    assert coordinates["count"] == result.count
    assert coordinates["tracks"][0]["observations"][0]["input_x"] == result.tracks[0]["observations"][0]["input_x"]


def test_custom_config_and_invalid_inputs(tmp_path):
    config = CounterConfig.from_yaml(ROOT / "configs" / "default.yaml")
    detector = PaperDetector(config=config)
    assert detector.preset == "custom"
    assert detector.config is not config
    assert PaperDetector(config=ROOT / "configs" / "default.yaml").config == config
    with pytest.raises(ValueError):
        PaperDetector(preset="typo")
    with pytest.raises(TypeError):
        detect_papers(None)
    with pytest.raises(ValueError):
        detect_papers(np.empty((0, 12), np.uint8))
    with pytest.raises(ValueError):
        detect_papers(np.zeros((200, 96), np.float32))
    with pytest.raises(FileNotFoundError):
        detect_papers(tmp_path / "missing.png")
    with pytest.raises(TypeError):
        detector.detect(np.zeros((200, 96), np.uint8), rotate_ccw=90)
