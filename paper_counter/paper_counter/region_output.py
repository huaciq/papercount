# -*- coding: utf-8 -*-
"""Box exports: explicit coordinate systems, no fictitious centerline points."""
from pathlib import Path
import csv,json
import numpy as np
from .coordinates import detection_to_original


def box_coordinates(result,metadata=None):
    debug=result.debug_info;shape=debug.get('input_original_shape',debug['image_shape'])
    geometry=debug.get('geometry_preprocessing');rotation=debug.get('rotation_ccw_degrees',0)
    output=[]
    for box in result.region_boxes:
        x1,y1,x2,y2=box['detection_xyxy']
        # Affine maps address pixel centers. Convert the exclusive outer edges
        # to half-pixel edge coordinates, transform, then return edge coordinates.
        corners=[(x1-.5,y1-.5),(x2-.5,y1-.5),(x2-.5,y2-.5),(x1-.5,y2-.5)]
        transformed=[detection_to_original(x,y,rotation,shape[1],shape[0],geometry) for x,y in corners]
        polygon=[[float(x+.5),float(y+.5)] for x,y in transformed]
        row=dict(box,input_polygon=polygon,
            input_xyxy=[min(p[0] for p in polygon),min(p[1] for p in polygon),
                        max(p[0] for p in polygon),max(p[1] for p in polygon)])
        if metadata:
            sx=metadata['preview_scale_x'];sy=metadata['preview_scale_y']
            ox=metadata['overlay_offset_x'];oy=metadata['overlay_offset_y']
            row['overlay_xyxy']=[ox+x1*sx,oy+y1*sy,ox+x2*sx,oy+y2*sy]
        output.append(row)
    return output


def save_regions(result,directory,cfg,full=False):
    from .visualization import save_overlay,write_png,display_uint8
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    metadata=save_overlay(result,directory,cfg);rows=box_coordinates(result,metadata)
    def write(name,value):
        (directory/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write('detection_boxes.json',dict(count=result.count,unit=result.debug_info['region_detection']['unit'],
        bounds='x2/y2 exclusive; input_polygon transforms the declared detection rectangle; curved boundaries are separate',
        physical_sheet_identity='unverified',boxes=rows,render_metadata=metadata))
    write('debug_info.json',result.debug_info);write('render_metadata.json',metadata)
    with (directory/'detection_boxes.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['编号','结构类型','检测x1','检测y1','检测x2','检测y2','得分','原图外接框x1','原图外接框y1','原图外接框x2','原图外接框y2'])
        for b in rows:writer.writerow([b['number'],b['structure'],*b['detection_xyxy'],b['score'],*b['input_xyxy']])
    (directory/'boxes_README.md').write_text(
        '每个框只计一个完整结构。double_bright为亮暗亮，single_bright为单亮带；auto整图选择一次，同图不切换。\n'
        '三条带模式的框是共同边界约束下的水平纸身区段，内部允许暗色，框之间不重叠；曲线边界另存boundary_samples。\n'
        '检测框坐标不含顶部标题；overlay_xyxy包含标题和显示缩放。input_polygon已经还原预处理和手动旋转。\n'
        'input_xyxy是该四边形的外接框，可能大于实际纸身。NMS在方向对齐图中执行，不能用原图大外接框重新合并相邻纸。\n'
        '框数量不是原边缘/亮线计数，旧真值不自动沿用或除二；物理纸张身份需要完整结构标注。\n',encoding='utf-8')
    if full:
        import matplotlib.pyplot as plt
        aligned=result.artifacts['region_aligned'];profile=result.artifacts['region_profile']
        write_png(directory/'center_input.png',result.artifacts['roi'])
        write_png(directory/'aligned_normalized.png',display_uint8(aligned))
        fig,ax=plt.subplots(figsize=(16,5),constrained_layout=True)
        ax.plot(profile,lw=.7,label='Normalized gray profile')
        for i,box in enumerate(result.region_boxes):
            a=box['aligned_xyxy'][1];b=box['aligned_xyxy'][3]
            ax.axvspan(a,b,color='green' if i%2 else 'cyan',alpha=.15)
            ax.text((a+b)/2,float(np.max(profile)),str(i+1),fontsize=6,rotation=90)
        ax.set(xlabel='Aligned center Y',ylabel='Normalized gray',title=f'{result.count} complete structure boxes')
        ax.legend();fig.savefig(directory/'region_profile.png',dpi=150);plt.close(fig)
        np.savez_compressed(directory/'region_arrays.npz',profile=profile,aligned=aligned,mapping=result.artifacts['region_mapping'])
        if 'three_strip' in result.artifacts:
            data=result.artifacts['three_strip']
            np.savez_compressed(directory/'strip_signals.npz',**data,fused=profile)
            fig,axes=plt.subplots(4,1,figsize=(16,10),sharex=True,constrained_layout=True)
            for i in range(3):
                axes[i].plot(data['signals'][i],lw=.6,label=f'Strip {i+1}: independent normalized gray')
                axes[i].legend(loc='upper right');axes[i].grid(alpha=.2)
            axes[3].plot(profile,lw=.7,label='Quality weighted gray (auxiliary)')
            axes[3].legend(loc='upper right');axes[3].set_xlabel('Aligned center-column Y')
            for ax in axes:
                for box in result.region_boxes:
                    ax.axvline(box['aligned_xyxy'][1],color='green',lw=.3,alpha=.4)
                    ax.axvline(box['aligned_xyxy'][3],color='red',lw=.3,alpha=.4)
            fig.savefig(directory/'strip_signals.png',dpi=150);plt.close(fig)
            fig,axes=plt.subplots(3,2,figsize=(16,8),sharex=True,constrained_layout=True)
            for i in range(3):
                axes[i,0].plot(data['raw_signals'][i],lw=.6,label=f'Strip {i+1}: raw gray')
                axes[i,1].plot(data['weights'][i],lw=.6,label=f'Strip {i+1}: reliability weight')
                for ax in axes[i]:ax.legend(loc='upper right');ax.grid(alpha=.2)
            fig.savefig(directory/'strip_raw_and_weights.png',dpi=150);plt.close(fig)
