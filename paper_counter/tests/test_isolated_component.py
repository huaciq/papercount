import numpy as np
from paper_counter import CounterConfig
from paper_counter.isolated_component import recover_isolated_outer_components


def scene(width=8):
    raw=np.zeros((180,120),dtype=np.uint8)
    raw[20:60]=100
    for x in range(120):
        y=100+x//6
        raw[y:y+width,x]=100
    return raw,[(x,x+10) for x in range(0,120,10)]


def test_slanted_outer_sheet_has_measured_boundary_and_no_duplicate():
    raw,strips=scene()
    tracks,_,debug=recover_isolated_outer_components(raw,strips,[],10,0,'positive',CounterConfig())
    assert len(tracks)==1 and debug['added']==1
    for s,y in tracks[0].strip_positions.items():
        assert abs(y-(100+(s*10+5)//6))<=1
    again,_,debug=recover_isolated_outer_components(raw,strips,tracks,10,0,'positive',CounterConfig())
    assert len(again)==1 and debug['added']==0


def test_thick_component_or_absent_sheet_does_not_create_paper():
    raw,strips=scene(30)
    assert not recover_isolated_outer_components(raw,strips,[],10,0,'positive',CounterConfig())[0]
    raw[80:]=0
    assert not recover_isolated_outer_components(raw,strips,[],10,0,'positive',CounterConfig())[0]


def test_compact_export_retains_endpoint_proof(tmp_path,monkeypatch):
    import json
    from dataclasses import replace
    from types import SimpleNamespace
    from paper_counter import compact_output
    monkeypatch.setattr(compact_output,'save_overlay',lambda *args:{})
    # This unit test isolates endpoint JSON; coordinates need real render metadata
    # and are covered by test_detection_coordinates.py and test_api.py.
    from paper_counter import detection_coordinates
    monkeypatch.setattr(detection_coordinates,'export_detection_coordinates',lambda *args:None)
    proof={'enabled':True,'added':1,'decisions':[{'decision':'measured_outer_paper'}]}
    result=SimpleNamespace(count=0,paper_edge_tracks=[],rejected_tracks=[],debug_info={'outer_component':proof})
    compact_output.save_compact(result,tmp_path,replace(CounterConfig(),isolated_raw_component_enabled=True))
    assert json.loads((tmp_path/'debug_info.json').read_text(encoding='utf-8'))['endpoint_evidence']['outer_component']==proof
