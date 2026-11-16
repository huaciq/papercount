"""Map existing exported geometry into counted-instance contracts."""
from .contracts import Evidence,PaperInstance
from .coordinates import detection_to_original


def exported_tracks(raw,original_shape,rotation_ccw_degrees=0):
    tracks=[]
    strips=raw.debug_info.get('strip_bounds_roi_x',[])
    for number,track in enumerate(sorted(raw.paper_edge_tracks,key=lambda t:(t.center_y,t.track_id)),1):
        observations=[]
        for strip,y in sorted(track.strip_positions.items()):
            left,right=strips[strip]
            x=raw.debug_info['roi']['x1']+(left+right)*.5
            ix,iy=detection_to_original(x,y,rotation_ccw_degrees,original_shape[1],original_shape[0],
                                        raw.debug_info.get('geometry_preprocessing'))
            observations.append(dict(strip_index=int(strip),detection_x=float(x),detection_y=float(y),
                                     input_x=float(ix),input_y=float(iy)))
        tracks.append(dict(number=number,track_id=str(track.track_id),center_detection_y=float(track.center_y),
                           source=track.source,observations=observations))
    return tracks


def counted_instances(count,unit,boxes,tracks):
    instances=[]
    for box in boxes:
        inferred=bool(box.get('thickness_inferred',False))
        instances.append(PaperInstance(f"region:{box['number']}",unit,'body_polygon',
            dict(polygon=box['input_polygon'],xyxy=box['input_xyxy']),
            (Evidence('body_section',box['id'],not inferred),),state='inferred' if inferred else 'unverified'))
    for track in tracks:
        points=[dict(x=p['input_x'],y=p['input_y'],strip_index=p['strip_index']) for p in track['observations']]
        instances.append(PaperInstance(f"track:{track['track_id'] or track['number']}",unit,'measured_boundary',
            dict(observations=points),(Evidence('boundary_track',track['track_id'],bool(points)),)))
    # Preserve count-only legacy responses without inventing a measured x,
    # a full-body box, or a second edge that the detector never observed.
    if not instances:
        instances=[PaperInstance(f'legacy:{i+1}',unit,'unlocalized_count',{},
                   (Evidence('legacy_count',f'legacy:{i+1}',False,'none'),),state='unlocalized',
                   coordinate_system='none') for i in range(count)]
    if len(instances)!=count:
        raise ValueError('Count does not match the exported counted instances')
    return instances


def instances_from_raw(raw):
    info=raw.debug_info['pipeline']
    original=raw.debug_info.get('input_original_shape',raw.debug_info['image_shape'])
    tracks=exported_tracks(raw,original,raw.debug_info.get('rotation_ccw_degrees',0))
    from .region_output import box_coordinates
    boxes=box_coordinates(raw) if raw.region_boxes else []
    return counted_instances(raw.count,info['count_unit'],boxes,tracks)
