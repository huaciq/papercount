from types import SimpleNamespace as NS
import csv,json
import numpy as np
from paper_counter.coordinates import build_render_metadata
from paper_counter.detection_coordinates import export_detection_coordinates


def fixture(rotation=90):
    debug={'roi':{'x1':100,'x2':200,'y1':200,'y2':300},'image_shape':[500,400],
        'input_original_shape':[400,500],'rotation_ccw_degrees':rotation,
        'strip_bounds_roi_x':[(10,30),(30,50)]}
    track=NS(track_id='seed_1',center_y=245.,strip_positions={0:240.,1:250.},
        boundary_polarity='negative',source='detected',sequence_decision='selected',
        curve_geometry={'positions':[240.,245.,250.]})
    signal=np.arange(100,dtype=float)
    result=NS(debug_info=debug,paper_edge_tracks=[track],rejected_tracks=[track],count=1,
        gradient_polarity='negative',artifacts={'polarities':{'negative':{'global':signal,'profiles':np.stack([signal,signal])}}})
    metadata=build_render_metadata(debug,50,50,760,166,116)
    return result,metadata


def test_coordinates_include_roi_scaling_title_and_inverse_rotation(tmp_path):
    result,metadata=fixture()
    data=export_detection_coordinates(result,tmp_path,metadata)
    p=data['tracks'][0]['observations'][0]
    assert (p['detection_x'],p['detection_y'])==(120.,240.)
    assert (p['overlay_x'],p['overlay_y'])==(10.,136.)
    assert (p['input_x'],p['input_y'])==(259.,120.)
    assert data['tracks'][0]['primary_global_response']==45.
    assert p['strip_response']==40.


def test_tables_only_number_final_tracks_and_real_observations(tmp_path):
    result,metadata=fixture(0)
    data=export_detection_coordinates(result,tmp_path,metadata)
    assert data['track_count']==1 and data['measured_point_count']==2
    assert len(data['tracks'][0]['observations'])==2
    assert result.paper_edge_tracks[0].strip_positions=={0:240.,1:250.}
    with (tmp_path/'detection_points.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2 and rows[0]['条带序号(从1起)']=='1'
    assert json.loads((tmp_path/'detection_coordinates.json').read_text(encoding='utf-8'))['count']==1
    p=data['tracks'][0]['observations'][0]
    assert (p['input_x'],p['input_y'])==(p['detection_x'],p['detection_y'])
