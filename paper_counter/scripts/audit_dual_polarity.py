"""Run independent positive/negative detection and save compact batch evidence."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_counter import CounterConfig
from paper_counter.preprocessing import read_image
from paper_counter.dual_polarity import process_dual


def run_one(job):
    path, source, output, cfg, rotate = job
    name = path.relative_to(source).as_posix()
    try:
        image = read_image(path)
        if rotate:
            image = np.rot90(image, 1).copy()
        negative, positive, audit = process_dual(image, cfg)
        target = output/'images'/name
        target.mkdir(parents=True)
        audit['rotation_ccw_degrees'] = 90 if rotate else 0
        audit['input_image'] = str(path.resolve())
        (target/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
        row = {k:v for k,v in audit.items() if k not in {'intervals','positive_tracks'}}
        row.update(image=name, status='ok')
        return row
    except Exception as exc:
        return dict(image=name, status='error', error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/final_verifier_active.yaml')
    parser.add_argument('--rotate-ccw', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    output = ROOT/'output'/('dual_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    cfg = CounterConfig.from_yaml(args.config)
    (output/'config.json').write_text(json.dumps(cfg.to_dict(), indent=2), encoding='utf-8')
    files = sorted(p for p in args.input.rglob('*') if p.suffix.lower() in {'.bmp','.png','.tif','.tiff','.jpg'})
    rows = []
    print(str(output), flush=True)
    jobs = [(p, args.input, output, cfg, args.rotate_ccw) for p in files]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run_one, jobs)):
            rows.append(row)
            (output/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'{i+1}/{len(files)} {row["image"]}: {row.get("negative_final_count")} / {row.get("positive_count")}',flush=True)
    lines = ['# 双梯度对照', '', '坐标为旋转后检测坐标。正梯度未启用负梯度专用末端校验；负梯度校验前数量用于对照。间隔一致率不是准确率。', '', '|图像|负梯度最终|负梯度校验前|正梯度|间隔一致率|', '|---|---:|---:|---:|---:|']
    for r in rows:
        if r['status']=='ok':
            agreement = r['interval_agreement']
            label = f'{agreement:.1%}' if agreement is not None else 'N/A'
            lines.append(f'|{r["image"]}|{r["negative_final_count"]}|{r["negative_pre_verifier_count"]}|{r["positive_count"]}|{label}|')
        else:
            lines.append(f'|{r["image"]}|ERROR||||')
    (output/'summary.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    main()
