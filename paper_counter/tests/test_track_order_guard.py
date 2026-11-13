from dataclasses import replace
from types import SimpleNamespace
import pytest
from paper_counter import CounterConfig
from paper_counter.track_order_guard import apply_track_order_guard


@pytest.mark.parametrize('positions,reject',[
    ([30,30,10,10],True),([30,30,30,10],False),([22,22,18,18],False),([10,10,10,10],False)])
def test_order_guard_uses_multiple_measured_positions_on_both_sides(positions,reject):
    anchor=SimpleNamespace(track_id='primary',boundary_polarity='positive',support_ratio=1,
        smoothness_score=1,strip_positions={s:20 for s in range(4)})
    aux=SimpleNamespace(track_id='aux',boundary_polarity='negative',support_ratio=1,
        smoothness_score=1,strip_positions=dict(enumerate(positions)))
    cfg=replace(CounterConfig(),auxiliary_order_guard_enabled=True)
    kept,rejected,debug=apply_track_order_guard([anchor,aux],'positive',cfg)
    assert bool(rejected)==reject
    assert anchor in kept
    if reject:assert debug['decisions'][0]['signed_differences']==[10,10,-10,-10]


@pytest.mark.parametrize('supported,matching,reject',[(True,6,True),(False,6,False),(True,2,False)])
def test_only_verified_same_body_pairs_remove_opposite_duplicate(supported,matching,reject):
    anchor=SimpleNamespace(track_id='primary',boundary_polarity='negative',support_ratio=1,
        smoothness_score=1,strip_positions={s:40 for s in range(6)})
    aux=SimpleNamespace(track_id='aux',boundary_polarity='positive',support_ratio=1,
        smoothness_score=1,strip_positions={s:15 for s in range(6)},local_evidence={
            'raw_body_confirmation':{'supported':supported,'observations':[
                {'strip':s,'pair_y':40 if s<matching else 20} for s in range(6)]}})
    cfg=replace(CounterConfig(),auxiliary_verified_body_dedup_enabled=True)
    kept,rejected,debug=apply_track_order_guard([anchor,aux],'negative',cfg)
    assert bool(rejected)==reject
    assert anchor in kept
