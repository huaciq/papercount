import itertools

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from paper_counter.region_sequence import (
    paper_support, recover_peripheral_support, shortest_ordered_partition, fit_exposed_faces,
)


def test_partition_matches_exhaustive_search_not_greedy_merging():
    """All 128 partitions compete, including cuts crossing former boxes."""
    length = 8
    rng = np.random.default_rng(129)
    edges = rng.uniform(-1, 4, (length + 1, length + 1))

    def transitions(end):
        starts = np.arange(end)
        return starts, edges[starts, end]

    bounds, cost = shortest_ordered_partition(length, transitions)
    candidates = []
    for choices in itertools.product([False, True], repeat=length - 1):
        path = [0] + [i + 1 for i, use in enumerate(choices) if use] + [length]
        candidates.append(sum(edges[a, b] for a, b in zip(path, path[1:])))
    assert cost == pytest.approx(min(candidates))
    assert sum(edges[a, b] for a, b in zip(bounds, bounds[1:])) == pytest.approx(cost)
    assert np.all(np.diff(bounds) > 0)


def test_background_gap_wider_than_measured_blur_is_left_free():
    raw = np.full(500, 10.)
    for top in range(30, 450, 50):
        raw[top:top + 34] = 180
    raw = np.tile(gaussian_filter1d(raw, .7), (3, 1))
    mask, _, audit = paper_support(raw, 50)
    # A previous .4*pitch closing filled this clear 16px black gap.
    assert not mask[72]
    assert audit['edge_resolution_pixels'] < 8


def test_faint_outer_paper_needs_two_independent_contrast_observations():
    raw = np.zeros((3, 220))
    raw[:, 20:44] = 70
    raw[:, 50:190] = 200
    mask = np.zeros(220, bool)
    mask[50:190] = True
    core = mask.copy()
    units = [dict(id='faint', top=20., bottom=44., band_ids=[0])]
    bands = [dict(center=32., source_strips=[0, 1, 2])]
    recovered = recover_peripheral_support(raw, units, bands, mask, core, 24)
    assert [r['unit_id'] for r in recovered] == ['faint']
    assert mask[20:44].all()
    raw[:2, 20:44] = 0
    mask[:50] = False
    assert recover_peripheral_support(raw, units, bands, mask, core, 24) == []


def test_broad_background_glow_is_not_recovered_as_a_faint_outer_paper():
    raw = np.full((3, 260), 15.)
    raw[:, 70:220] = 180
    mask = np.zeros(260, bool)
    mask[70:220] = True
    units = [dict(id='glow', top=30., bottom=68., band_ids=[0])]
    bands = [dict(center=49., source_strips=[0, 1, 2])]
    assert recover_peripheral_support(raw, units, bands, mask, mask.copy(), 30) == []
    assert not mask[:70].any()


def test_exposed_face_fit_removes_halo_and_preserves_count_and_shared_cuts():
    raw=np.full(180,10.)
    raw[34:68]=180
    raw[100:145]=180
    raw=np.tile(gaussian_filter1d(raw,.8),(3,1))
    units=[dict(id='a',top=30.,bottom=72.),dict(id='b',top=96.,bottom=120.),
           dict(id='c',top=120.,bottom=150.)]
    before=[(u['top'],u['bottom']) for u in units]
    changes=fit_exposed_faces(units,raw,3)
    assert len(units)==3 and changes
    assert 32<=units[0]['bottom']-units[0]['top']<=38
    assert units[1]['bottom']==units[2]['top']==120
    assert all(a<=u['top']<u['bottom']<=b for u,(a,b) in zip(units,before))
    assert all(a['bottom']<=b['top'] for a,b in zip(units,units[1:]))


@pytest.mark.parametrize('bright_at_top', [True,False])
def test_brighter_internal_texture_cannot_replace_dim_outer_face(bright_at_top):
    raw=np.full(150,10.)
    raw[34:68]=70
    raw[34:51] = 220 if bright_at_top else 70
    raw[51:68] = 70 if bright_at_top else 220
    raw=np.tile(gaussian_filter1d(raw,.7),(3,1))
    # Wide enough resolution/search window to include the stronger inner edge.
    units=[dict(id='paper',top=30.,bottom=72.)]
    fit_exposed_faces(units,raw,12)
    assert units[0]['top']==pytest.approx(34,abs=1)
    assert units[0]['bottom']==pytest.approx(68,abs=1)
