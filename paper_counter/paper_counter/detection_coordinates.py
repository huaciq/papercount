"""Export final track identities and measured strip observations for debugging."""
from __future__ import annotations
import csv,json
from pathlib import Path
import cv2
import numpy as np
from .coordinates import transform_from_metadata,detection_to_original


def ordered_final_tracks(result):
    return sorted(result.paper_edge_tracks,key=lambda t:(t.center_y,t.track_id))


def export_detection_coordinates(result,directory,metadata):
    """Only terminal survivors receive numbers; interpolated points are excluded."""
    directory=Path(directory);debug=result.debug_info
    mapping=transform_from_metadata(metadata)
    strips=debug.get('strip_bounds_roi_x',[])
    tracks=[];points=[]
    data=result.artifacts.get('polarities',{})
    offset=debug['roi']['y1']
    def signal(polarity,y,strip=None):
        branch=data.get(polarity,{})
        values=branch.get('global') if strip is None else branch.get('profiles')
        if values is None:return None
        values=np.asarray(values) if strip is None else np.asarray(values)[strip]
        if not 0<=y-offset<len(values):return None
        return float(np.interp(y-offset,np.arange(len(values)),values))
    for number,t in enumerate(ordered_final_tracks(result),1):
        polarity=t.boundary_polarity or result.gradient_polarity
        observed=[]
        for strip,y in sorted(t.strip_positions.items()):
            # x is the midpoint of an aggregation strip, not an exact edge pixel.
            a,b=strips[strip];x=debug['roi']['x1']+(a+b)*.5
            ox,oy=mapping.detection_to_overlay(x,y)
            ix,iy=detection_to_original(x,y,metadata['rotation_ccw_degrees'],metadata['original_width'],metadata['original_height'],metadata.get('geometry_preprocessing'))
            point={'number':number,'track_id':t.track_id,'strip_number':strip+1,'strip_index':strip,
                'detection_x':float(x),'detection_y':float(y),'overlay_x':ox,'overlay_y':oy,
                'input_x':ix,'input_y':iy,'polarity':polarity,
                'strip_response':signal(polarity,y,strip),'observation_type':'measured_strip_position'}
            points.append(point);observed.append(point)
        ys=[p['detection_y'] for p in observed]
        tracks.append({'number':number,'track_id':t.track_id,'center_detection_y':float(t.center_y),
            'center_overlay_y':mapping.detection_to_overlay(0,t.center_y)[1],
            'observed_y_min':min(ys) if ys else None,'observed_y_max':max(ys) if ys else None,
            'support_count':len(observed),'polarity':polarity,'source':t.source,
            'decision':t.sequence_decision,'primary_global_response':signal(result.gradient_polarity,t.center_y),
            'observations':observed})
    assert not tracks or len(tracks)==result.count
    bundle={'count':result.count,'track_count':len(tracks),'measured_point_count':len(points),
        'coordinate_notes':{'detection':'full detection image after requested rotation; ROI offsets included',
            'overlay':'result_overlay.png including title and realized preview scaling',
            'input':'image supplied before requested rotation and optional geometry affine',
            'x':'representative aggregation-strip midpoint; not a precisely localized edge pixel',
            'center':'summary row coordinate, not a measured peak in the global curve',
            'points':'only measured strip_positions; no interpolated curve samples'},
        'render_metadata':metadata,'tracks':tracks}
    (directory/'detection_coordinates.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    track_fields={'number':'编号','track_id':'轨迹ID','center_detection_y':'检测图中心y','center_overlay_y':'结果图中心y',
        'observed_y_min':'实测最小y','observed_y_max':'实测最大y','support_count':'支持条带数',
        'polarity':'边界极性','source':'来源','decision':'保留方式','primary_global_response':'主极性全局信号在中心处的值'}
    point_fields={'number':'编号','track_id':'轨迹ID','strip_number':'条带序号(从1起)',
        'detection_x':'检测图x(条带中点)','detection_y':'检测图实测y','overlay_x':'结果图x','overlay_y':'结果图y',
        'input_x':'输入图x','input_y':'输入图y','polarity':'边界极性','strip_response':'本极性条带信号在实测y处的值'}
    for name,rows,fields in [('detection_tracks.csv',tracks,track_fields),('detection_points.csv',points,point_fields)]:
        with (directory/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(fields.values())
            writer.writerows([[row.get(k) for k in fields] for row in rows])
    (directory/'coordinates_README.md').write_text(
        '# 检测坐标说明\n\n编号按最终轨迹中心从上到下排序，与编号结果图和编号信号图一致；轨迹ID用于跨阶段追踪。\n\n'
        '- detection_tracks.csv：每条最终轨迹一行，中心y只是汇总位置，不保证是全局信号峰。\n'
        '- detection_points.csv：每个真实条带观测一行，不含插值点；x为条带中点，不是逐像素定位的边缘点。\n'
        '- detection_coordinates.json：相同数据、坐标含义及完整渲染变换。\n'
        '- result_overlay_numbered.png：在原最终结果图上增加编号，原结果图保留。\n'
        '- profile_numbered.png：完整调试模式导出，编号点表示轨迹中心投影，不表示全局寻峰结果。\n\n'
        '检测图坐标已包含ROI偏移；结果图坐标包含标题及缩放；输入图坐标已逆变换自动旋正、放缩及手动90度旋转。'
        '端部或混合极性候选的条带信号可能不明显，信号值不是独立纸层存在的证明。'
        '旧算法没有轨迹对象时坐标表可能为空，仍可查看其已有峰记录。\n',encoding='utf-8')
    overlay_path=directory/'result_overlay.png'
    if overlay_path.exists():
        canvas=cv2.imdecode(np.fromfile(overlay_path,np.uint8),cv2.IMREAD_COLOR)
        for track in tracks:
            obs=track['observations']
            if not obs:continue
            # 相邻轨迹交错选择标号位置，避免密集纸边的文字挤成一列。
            fraction=(.2,.5,.8)[(track['number']-1)%3]
            anchor=obs[round((len(obs)-1)*fraction)];x=int(anchor['overlay_x']);y=int(anchor['overlay_y'])
            label=f"#{track['number']}"
            (width,height),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.38,1)
            x=max(0,min(x,canvas.shape[1]-width-2));y=max(height+2,min(y,canvas.shape[0]-2))
            cv2.putText(canvas,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(canvas,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,0),1,cv2.LINE_AA)
        ok,encoded=cv2.imencode('.png',canvas)
        if not ok:raise OSError('Cannot encode numbered overlay')
        encoded.tofile(directory/'result_overlay_numbered.png')
    return bundle
