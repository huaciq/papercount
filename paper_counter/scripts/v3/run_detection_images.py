"""Fresh P1 scanline tests and actual-photo overlays, plus legacy E0 replay.

The detector never sees image names/GT. Names choose review examples only.
No center tracks, joint evidence, or new final-count backend are implemented.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import csv
import gzip
import json
import sys
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper_counter import PaperCounter, CounterConfig
from paper_counter.preprocessing import read_image, crop_roi, to_gray, normalize_illumination
from paper_counter.active_roi import detect_active_stack_roi
from paper_counter.v3_shadow import ShadowConfig, analyze
from audit_p0 import OUT, sha, write_json, write_csv


def font(size):
    for name in ('msyh.ttc', 'simhei.ttf', 'arial.ttf'):
        path = Path('C:/Windows/Fonts') / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def rgb_display(array):
    # Display conversion only. Every detector receives native integer pixels.
    if array.dtype == np.uint16:
        array = np.clip(array / 257., 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array).convert('RGB')
    return Image.fromarray(cv2.cvtColor(array[:, :, :3], cv2.COLOR_BGR2RGB))


def title_canvas(photo, lines):
    w = max(900, photo.width)
    header = 150
    canvas = Image.new('RGB', (w, photo.height + header + 64), (25, 28, 34))
    canvas.paste(photo, (0, header))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((15, 12 + i * 31), line, font=font(21 if i else 25), fill=(240, 242, 245))
    return canvas, header


def candidate_overlay(roi_array, record, label, gt_label, max_height=5000):
    photo = rgb_display(roi_array)
    ratio = min(1., max_height / photo.height, 1500 / photo.width)
    if ratio < 1:
        photo = photo.resize((max(1, round(photo.width * ratio)), max(1, round(photo.height * ratio))), Image.Resampling.LANCZOS)
    sx, sy = photo.width / roi_array.shape[1], photo.height / roi_array.shape[0]
    roi = record['roi_rotated']
    canvas, header = title_canvas(photo, [label,
        f"候选 {len(record['candidates'])}  |  {gt_label}  |  冻结 Dual E0 {record['legacy_edge_count']}",
        'P1 Shadow：单扫描线中心观测；不是二维轨迹或正式计数',
        f"{Path(record['image']).name}  |  {record['status']}"])
    draw = ImageDraw.Draw(canvas)
    x = (record['scanline_rotated_x'] - roi['x1']) * sx
    draw.line([(x, header), (x, header + photo.height - 1)], fill=(40, 150, 250), width=1)
    last_label = -100
    for i, c in enumerate(record['candidates'], 1):
        y = (c['y_rotated'] - roi['y1']) * sy + header
        color = (60, 255, 85) if c['complete_support'] and c['observable'] else (255, 170, 45)
        draw.line([(x-22, y), (x+22, y)], fill=color, width=3)
        draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        if y - last_label >= 15:
            draw.text((x+29, y-9), str(i), font=font(13), fill=color)
            last_label = y
    foot = header + photo.height + 8
    draw.text((15, foot), '蓝：实际扫描线  绿：完整输入支持  橙：支持不完整（需复核）', font=font(18), fill=(230, 232, 235))
    draw.text((15, foot+27), '标记只表示观测位置，颜色不是正确概率；照片仅为显示缩放。', font=font(16), fill=(200, 205, 210))
    return canvas


def legacy_overlay(rotated, result, key, gt_label, archived_count):
    roi = result.debug_info['roi']
    native = rotated[roi['y1']:roi['y2'], roi['x1']:roi['x2']]
    photo = rgb_display(native)
    ratio = min(1., 5000 / photo.height, 1500 / photo.width)
    if ratio < 1:
        photo = photo.resize((max(1, round(photo.width*ratio)), max(1, round(photo.height*ratio))), Image.Resampling.LANCZOS)
    sx, sy = photo.width / native.shape[1], photo.height / native.shape[0]
    canvas, header = title_canvas(photo, ['Dual E0 原后端：当前源码重新检测的真实多条带轨迹',
        f'新运行计数 {result.count}  |  历史归档 {archived_count}  |  {gt_label}',
        'E1/E2扩展关闭；未修改检测参数或算法', Path(key).name])
    draw = ImageDraw.Draw(canvas)
    strips = result.debug_info['strip_bounds_roi_x']
    assert len(result.paper_edge_tracks) == result.count
    for i, t in enumerate(sorted(result.paper_edge_tracks, key=lambda t:t.center_y), 1):
        color = (255, 215, 30) if t.source=='recovered' or t.sequence_decision=='rescued' else (40, 255, 90)
        positions = sorted(t.strip_positions.items())
        points = []
        for s, y in positions:
            x = (strips[s][0]+strips[s][1]) * .5 * sx
            yy = (y-roi['y1']) * sy + header
            points.append((s, x, yy))
            draw.ellipse((x-2, yy-2, x+2, yy+2), fill=color)
        # Do not paint a continuous segment through missing strip observations.
        for left, right in zip(points, points[1:]):
            if right[0] == left[0] + 1:
                draw.line([(left[1],left[2]), (right[1],right[2])], fill=color, width=2)
        if points:
            draw.text((points[0][1]+4, points[0][2]-10), str(i), font=font(13), fill=color)
    draw.text((15, header+photo.height+8), '绿：最终保留轨迹  黄：既有恢复/救援轨迹；仅连接相邻有观测条带', font=font(17), fill=(230, 232, 235))
    draw.text((15, header+photo.height+34), '这是新运行的阶段证据，未覆盖历史实验或冒充历史轨迹。', font=font(17), fill=(200, 205, 210))
    return canvas


def save_image(canvas, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, compress_level=3)


def montage(panels, path):
    shown = []
    for panel in panels:
        scale = min(1., 1700/panel.height, 1000/panel.width)
        shown.append(panel.resize((round(panel.width*scale), round(panel.height*scale)), Image.Resampling.LANCZOS))
    canvas = Image.new('RGB', (sum(p.width for p in shown)+12*(len(shown)-1), max(p.height for p in shown)), (50,53,58))
    x=0
    for panel in shown:
        canvas.paste(panel,(x,0)); x += panel.width+12
    save_image(canvas, path)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT/'output/v3_detection_images_20261009_01')
    args=ap.parse_args()
    out=args.output
    out.mkdir(parents=True, exist_ok=False)
    protected=json.loads((OUT/'p0/protected_before.json').read_text(encoding='utf-8'))
    assert all(sha(Path(p))==h for p,h in protected.items())
    p1=OUT/'p1'
    truth=json.loads((OUT/'p0/GT_VERSION.json').read_text(encoding='utf-8'))
    cfg=CounterConfig(**json.loads((OUT/'p0/A0_effective_config.json').read_text(encoding='utf-8')))
    configs={s:ShadowConfig(**json.loads((ROOT/f'configs/v3/p1_{s.lower()}.json').read_text(encoding='utf-8'))) for s in ('A1','A2','A3','A4')}
    with (OUT/'p0/A0_DUAL_E0_COUNTS.csv').open(encoding='utf-8-sig', newline='') as f:
        baselines=list(csv.DictReader(f))
    # Priority/legacy subset is report selection only. All99 P1 cases use the
    # same signal-only detector and unchanged stage configuration.
    focus_tokens=('200g4-1','200g4-2','7号纸3-2','7号纸4-1','7号纸4-2','覆膜铜板纸3-3-2','9号纸1-2')
    def priority(row):
        key=row['image']
        if key=='验证集/淋膜纸2.bmp':return 0
        if any(t in key for t in focus_tokens):return 1
        if key.startswith('验证集/'):return 2
        return 3
    baselines.sort(key=lambda r:(priority(r),r['image']))
    rows, checks, legacy_keys=[], [], []
    write_json(out/'run_manifest.json',dict(scope='fresh P1 on99; unchanged legacy Dual E0 on16 validation +7 difficult',
        production_backend_changed=False, P2_not_implemented=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'paper_counter').glob('*.py')},
        p1_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'paper_counter/v3_shadow').glob('*.py')},
        renderer_sha256=sha(Path(__file__)), gt_version=truth['version'], baseline_config=cfg.to_dict()))
    for index,b in enumerate(baselines):
        key=b['image']; source_path=ROOT.parent/'数据'/key
        image_hash=sha(source_path)
        assert image_hash==truth['images'][key]['image_sha256']
        raw=read_image(source_path); rotated=np.rot90(raw).copy()
        requested,bounds=crop_roi(rotated,cfg.roi); fullgray=to_gray(requested)
        top,bottom,_=detect_active_stack_roi(fullgray,None,cfg,original=requested)
        native=requested[top:bottom]
        gray=fullgray[top:bottom]
        norm=normalize_illumination(gray,cfg.illumination_sigma,cfg.epsilon)
        x=gray.shape[1]//2; ceiling=np.iinfo(gray.dtype).max
        rawsignal=gray[:,x].astype(float)*(255./ceiling)
        normsignal=norm[:,x].astype(float)
        k=max(1,int(np.ceil(len(fullgray)*configs['A4'].reference_end_fraction)))
        reference=np.r_[fullgray[:k,x],fullgray[-k:,x]].astype(float)*(255./ceiling)
        roi=dict(x1=bounds['x1'],x2=bounds['x2'],y1=bounds['y1']+top,y2=bounds['y1']+bottom)
        gt=truth['images'][key]
        gt_label=f"GT {gt['count']} (verified)" if gt['status']=='verified' else f"GT {gt['status']}（不计准确率）"
        panels={}
        for source,signal in (('RAW',rawsignal),('normalized',normsignal)):
            for stage,settings in configs.items():
                previous=json.loads((p1/'real'/key/source/stage/'candidates.json').read_text(encoding='utf-8'))
                assert previous['roi_rotated']==roi and previous['scanline_rotated_x']==x+roi['x1']
                result=analyze(signal,rawsignal,settings,reference_values=reference,intensity_ceiling=255)
                assert [c['y'] for c in result['candidates']]==[c['y'] for c in previous['candidates']], (key,source,stage,'candidate mismatch')
                with np.load(p1/'real'/key/source/stage/'responses.npz') as z:
                    for name,array in result['arrays'].items():
                        np.testing.assert_allclose(array,z[name],rtol=1e-8,atol=1e-7,err_msg=f'{key}/{source}/{stage}/{name}')
                for c in result['candidates']+result['identified']+result['rejected']:
                    c.update(y_rotated=c['y']+roi['y1'],x_rotated=x+roi['x1'],
                             x_original=raw.shape[1]-1-(c['y']+roi['y1']),y_original=x+roi['x1'])
                    assert 0<=c['x_original']<raw.shape[1] and 0<=c['y_original']<raw.shape[0]
                record={**{a:v for a,v in previous.items() if a not in ('candidates','identified','rejected')},
                        **{a:result[a] for a in ('candidates','identified','rejected','seconds','status','review_required')},
                        'fresh_detection':True,'frozen_candidates_equal':True}
                write_json(out/'evidence'/key/source/(stage+'.json'),record)
                rows.append(dict(image=key,model=stage,source=source,count=len(result['candidates']),
                    gt_status=gt['status'],gt_count=gt['count'],error=len(result['candidates'])-gt['count'] if gt['status']=='verified' else None,
                    review_required=result['review_required'],seconds=result['seconds'],frozen_candidates_equal=True))
                checks.append(dict(image=key,stage=stage,source=source,candidate_equal=True,all_response_arrays_equal=True))
                # A4 every image; full A1--A4 visual comparison on selected23.
                if stage=='A4' or priority(b)<3:
                    canvas=candidate_overlay(native,record,f'{stage} / {source} 新运行：实际照片上的纸身中心',gt_label)
                    save_image(canvas,out/f'{stage}_{source}'/(key+'.png'))
                    if source=='RAW' or stage=='A4':panels[(source,stage)]=canvas
        if priority(b)<3:
            start=time.perf_counter()
            legacy=PaperCounter(cfg).process(rotated)
            seconds=time.perf_counter()-start
            legacy.debug_info.update(rotation_ccw_degrees=90,input_image_sha256=image_hash,fresh_v3_visual_test=True)
            target=out/'legacy_e0_evidence'/key
            target.mkdir(parents=True,exist_ok=False)
            with gzip.open(target/'debug_info.json.gz','wt',encoding='utf-8') as f:
                json.dump(legacy.debug_info,f,ensure_ascii=False,allow_nan=False)
            write_json(target/'tracks.json',dict(count=legacy.count,roi_rotated=legacy.debug_info['roi'],
                strip_bounds_roi_x=legacy.debug_info['strip_bounds_roi_x'],tracks=[t.to_dict() for t in legacy.paper_edge_tracks],
                archived_count=int(b['count']),fresh_matches_archive=legacy.count==int(b['count']),seconds=seconds,
                input_sha256=image_hash,gt_version=truth['version']))
            canvas=legacy_overlay(rotated,legacy,key,gt_label,b['count'])
            save_image(canvas,out/'Dual_E0_fresh'/(key+'.png'))
            montage([canvas,panels[('RAW','A4')],panels[('normalized','A4')]],out/'compare'/(key+'.png'))
            montage([panels[('RAW',s)] for s in configs],out/'A1_A4_compare'/(key+'.png'))
            rows.append(dict(image=key,model='Dual E0 fresh',source='legacy_multistrip',count=legacy.count,
                archived_count=int(b['count']),gt_status=gt['status'],gt_count=gt['count'],
                error=legacy.count-gt['count'] if gt['status']=='verified' else None,seconds=seconds,
                frozen_candidates_equal=None,fresh_count_matches_archive=legacy.count==int(b['count'])))
            legacy_keys.append(key)
            print(json.dumps(dict(progress=f'{index+1}/99',image=key,legacy_fresh=legacy.count,archived=int(b['count']),legacy_seconds=round(seconds,2)),ensure_ascii=False),flush=True)
            del legacy
        else:
            print(json.dumps(dict(progress=f'{index+1}/99',image=key,P1_A4_RAW=len(json.loads((out/'evidence'/key/'RAW/A4.json').read_text(encoding='utf-8'))['candidates'])) ,ensure_ascii=False),flush=True)
        write_csv(out/'fresh_results.csv',rows)
        del raw,rotated,requested,fullgray,native,gray,norm,panels
    assert len(legacy_keys)==23
    assert len(checks)==792
    assert all(sha(Path(p))==h for p,h in protected.items())
    write_json(out/'fresh_equivalence_audit.json',dict(real_images=99,p1_stage_source_cases=792,all_equal=True,
        legacy_fresh_images=len(legacy_keys),legacy_keys=legacy_keys,
        legacy_counts_matching_archive=sum(r['fresh_count_matches_archive'] for r in rows if r['model']=='Dual E0 fresh'),
        protected_legacy_files=len(protected),all_unchanged=True,checks=checks))
    indexlines=['# 实际检测图：新运行测试', '',
        'P1的99张图均重新检测，792套响应和候选与上轮保存结果一致。旧Dual E0重新运行23张（16验证+7重点困难），输出实际多条带轨迹。没有改动检测算法或历史结果。', '',
        '- `Dual_E0_fresh`：旧后端实际多条带轨迹，23张。',
        '- `A4_RAW`、`A4_normalized`：新P1单扫描线候选叠加原照片，每路99张。',
        '- `compare`：旧轨迹 / A4 RAW / A4 normalized 三图对照，23張。',
        '- `A1_A4_compare`：A1～A4 RAW候选图对照，23张。', '',
        'P1中心短刻度只是单扫描线观测，没有扩展为整条纸边或新增二维轨迹。彩色不代表校准的正确概率。标签GT只用于评价与图注，未输入检测器。', '',
        '| 图像 | Dual E0多条带轨迹 | 原图三方案对照 | A4 RAW | A4 normalized |', '|---|---|---|---|---|']
    for b in baselines:
        key=b['image']
        def link(folder):return f'[{folder}]({(out/folder/(key+".png")).as_posix()})'
        indexlines.append('| '+key+' | '+(link('Dual_E0_fresh') if key in legacy_keys else '本轮未重跑旧后端')+' | '+(link('compare') if key in legacy_keys else '—')+' | '+link('A4_RAW')+' | '+link('A4_normalized')+' |')
    (out/'DETECTION_IMAGE_INDEX.md').write_text('\n'.join(indexlines)+'\n',encoding='utf-8')
    write_json(out/'delivery_manifest.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}))
    print('ACTUAL DETECTION IMAGES COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
