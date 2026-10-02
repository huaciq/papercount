from dataclasses import replace
import numpy as np
from paper_counter import CounterConfig
from paper_counter.active_roi import detect_active_stack_roi


def cfg():
    return replace(CounterConfig(), stack_roi_mode='foreground_envelope', foreground_roi_margin_rows=8)


def test_roi_contains_low_texture_paper_between_two_textured_regions():
    image = np.zeros((700, 100), np.uint8)
    image[80:590] = 220
    image[80:180:12] = 80
    image[480:590:12] = 80
    top, bottom, info = detect_active_stack_roi(image, None, cfg())
    assert top <= 80 and bottom >= 590
    assert top > 0 and bottom < 700
    assert info['reason'] == 'raw_foreground_envelope'


def test_roi_retains_internal_black_gap_and_rejects_narrow_border_scratch():
    image = np.zeros((700, 100), np.uint8)
    image[100:240] = 180
    image[360:580] = 180
    image[610:690, 3:5] = 250
    top, bottom, _ = detect_active_stack_roi(image, None, cfg())
    assert top <= 100 and bottom >= 580 and bottom < 610


def test_roi_preserves_first_and_last_low_texture_sheets():
    image = np.zeros((650, 100), np.uint8)
    image[0:500] = 160
    image[60:440:16] = 70
    top, bottom, _ = detect_active_stack_roi(image, None, cfg())
    assert top == 0 and bottom >= 500


def test_no_contrast_does_not_invent_foreground_bounds():
    image = np.full((150, 100), 128, np.uint8)
    assert detect_active_stack_roi(image, None, cfg())[:2] == (0, 150)


def test_color_foreground_and_uint16_are_supported():
    gray = np.zeros((500, 100), np.uint16)
    color = np.zeros((500, 100, 3), np.uint16)
    color[70:400, :, 0] = 22000
    top, bottom, info = detect_active_stack_roi(gray, None, cfg(), original=color)
    assert top <= 70 and bottom >= 400 and bottom < 500
    assert info['uses_color']
