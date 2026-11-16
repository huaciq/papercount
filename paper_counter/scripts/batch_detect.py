"""Batch tracks or regions with frozen source/config and separate image/audit output."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone,timedelta
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'数据'
EXTENSIONS={'.bmp','.png','.jpg','.jpeg','.tif','.tiff','.webp'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def run_one(item,output,rotate,full,ground_truth):
    path=Path(item['path']);key=item['image'];target=output/'audit/details'/key
    target.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(output/'audit/source/main.py'),
             '--image',str(path),'--recipe',str(output/'audit/recipe.yaml'),'--output',str(target)]
    if rotate:command.append('--rotate-ccw')
    if full:command.append('--full-debug')
    if ground_truth:command.extend(['--ground-truth',str(ground_truth)])
    row=dict(image=key,input_sha256=item['sha256'])
    try:
        if sha(path)!=item['sha256']:raise RuntimeError('Input changed before detection')
        with (target/'cli.log').open('w',encoding='utf-8') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                   env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        if process.returncode:raise RuntimeError(f'CLI exit {process.returncode}; see audit/details/{key}/cli.log')
        d=json.loads((target/'debug_info.json').read_text(encoding='utf-8'))
        algorithm='regions' if d['algorithm']=='narrow_regions' else 'tracks'
        geometry={}
        if algorithm=='regions':
            boxes=json.loads((target/'detection_boxes.json').read_text(encoding='utf-8'))['boxes']
            ordered=sorted(boxes,key=lambda b:b['detection_xyxy'][1])
            pairs=[dict(box_a=a['number'],box_b=b['number'],height=min(a['detection_xyxy'][3],b['detection_xyxy'][3])-max(a['detection_xyxy'][1],b['detection_xyxy'][1]))
                   for i,a in enumerate(ordered) for b in ordered[i+1:]
                   if min(a['detection_xyxy'][3],b['detection_xyxy'][3])>max(a['detection_xyxy'][1],b['detection_xyxy'][1])+1e-8]
            geometry=dict(positive_height=all(b['detection_xyxy'][3]>b['detection_xyxy'][1] for b in boxes),
                          count_matches_boxes=len(boxes)==d['count'],nonoverlap=not pairs,overlap_pairs=pairs)
            write(target/'batch_geometry_validation.json',geometry)
        destination=output/'images'/Path(key).parent/(Path(key).name+'_result.png')
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(target/'result_overlay.png',destination)
        if sha(path)!=item['sha256']:raise RuntimeError('Input changed during detection')
        row.update(status='exported',algorithm=algorithm,count=d['count'],detector_status=d['status'],
                   pipeline=d.get('pipeline',{}).get('id'),recipe_id=d.get('pipeline',{}).get('recipe_id'),
                   geometry=geometry,result_image=destination.relative_to(output).as_posix())
    except Exception as error:row.update(status='error',error=str(error))
    return row


def summarize(output,rows,total):
    rows=sorted(rows,key=lambda r:r['image']);write(output/'audit/results.json',rows)
    lines=['# 批量检测结果','',f'成功导出 {sum(r["status"]=="exported" for r in rows)}/{total} 张。',
           'images/保存结果图；audit/保存配置、源码快照、日志和坐标。',
           '导出成功不代表计数或几何检查通过。重叠问题单独记录，导出时不修改算法原始结果。','',
           '| 图片 | 算法 | 计数 | 重叠框对数 | 状态 |','| --- | --- | ---: | ---: | --- |']
    for r in rows:
        lines.append(f'| {r["image"]} | {r.get("algorithm","—")} | {r.get("count","—")} | {len(r.get("geometry",{}).get("overlap_pairs",[]))} | {r["status"]} |')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input',type=Path,help='Input folder, searched recursively')
    group.add_argument('--datasets',nargs='+',help='Folders relative to ../数据')
    parser.add_argument('--extra-image',action='append',default=[],help='Additional image relative to ../数据')
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--config',type=Path,default=ROOT/'configs/default.yaml')
    selection.add_argument('--recipe',help='Versioned recipe name or YAML path')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--rotate-ccw',action='store_true')
    parser.add_argument('--full-debug',action='store_true')
    parser.add_argument('--ground-truth',type=Path)
    parser.add_argument('--output',type=Path,help='New result folder; must not already exist')
    parser.add_argument('--output-root',type=Path,default=ROOT/'output')
    args=parser.parse_args(argv)
    if args.workers<1:parser.error('--workers must be positive')
    base=args.input.resolve() if args.input else DATA.resolve()
    folders=[base] if args.input else [(DATA/name).resolve() for name in args.datasets]
    if any(not p.is_dir() for p in folders):parser.error('Every input folder must exist')
    if not args.input and any(base not in p.parents and p!=base for p in folders):parser.error('Use --input for folders outside ../数据')
    paths={p for folder in folders for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS}
    paths.update((DATA/name).resolve() for name in args.extra_image)
    if not paths or any(not p.is_file() or p.suffix.lower() not in EXTENSIONS for p in paths):parser.error('No images or invalid --extra-image')
    sys.path.insert(0,str(ROOT))
    from paper_counter.recipe import Recipe,load_config_or_recipe
    import yaml
    recipe=Recipe.load(args.recipe) if args.recipe else load_config_or_recipe(args.config.resolve())
    cfg=recipe.to_counter_config()
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output=args.output.resolve() if args.output else args.output_root.resolve()/f'batch_{cfg.algorithm}_{stamp}'
    output.mkdir(parents=True,exist_ok=False);audit=output/'audit';audit.mkdir()
    source=audit/'source';source.mkdir()
    shutil.copytree(ROOT/'paper_counter',source/'paper_counter',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'main.py',source/'main.py')
    (audit/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(),allow_unicode=True,sort_keys=False),encoding='utf-8')
    (audit/'recipe.yaml').write_text(yaml.safe_dump(recipe.to_dict(),allow_unicode=True,sort_keys=False),encoding='utf-8')
    shutil.copy2(Path(__file__),audit/'batch_source.py')
    inventory=[dict(image=p.relative_to(base).as_posix(),path=str(p),sha256=sha(p)) for p in sorted(paths)]
    write(audit/'inventory.json',inventory);rows=[];summarize(output,rows,len(paths))
    print('OUTPUT: '+str(output),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run_one,r,output,args.rotate_ccw,args.full_debug,args.ground_truth.resolve() if args.ground_truth else None) for r in inventory]
        for job in as_completed(jobs):
            row=job.result();rows.append(row);summarize(output,rows,len(paths))
            print(json.dumps(dict(progress=f'{len(rows)}/{len(paths)}',**row),ensure_ascii=True),flush=True)
    snapshot={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    unchanged=all((ROOT/name).exists() and sha(ROOT/name)==digest for name,digest in snapshot.items())
    import cv2
    import numpy as np
    images=list((output/'images').rglob('*.png')) if (output/'images').exists() else []
    for path in images:
        if cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_UNCHANGED) is None:
            raise RuntimeError(f'Cannot decode result image: {path}')
    verification=dict(total=len(paths),exported=sum(r['status']=='exported' for r in rows),failed=sum(r['status']=='error' for r in rows),
        algorithm=cfg.algorithm,pipeline=recipe.pipeline,recipe_id=recipe.recipe_id,
        source_matches_current=unchanged,source_hashes=snapshot,pngs_verified=len(images),
        inputs_unchanged=all(sha(Path(r['path']))==r['sha256'] for r in inventory),
        images_with_overlap=sum(bool(r.get('geometry',{}).get('overlap_pairs')) for r in rows))
    write(audit/'verification.json',verification)
    if not unchanged:raise RuntimeError('Source changed while running; results refer to frozen snapshot')
    return 1 if verification['failed'] else 0


if __name__=='__main__':
    raise SystemExit(main())
