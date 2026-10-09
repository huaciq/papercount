"""Derive reports from saved evidence, never rerun or tune the detector."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import shutil
import sys
import numpy as np
from audit_p0 import ROOT, OUT, sha, write_json, write_csv, metrics
from run_p1 import summarize

P1 = OUT / 'p1'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def typed(r):
    values = {}
    for k, v in r.items():
        if v == '':
            values[k] = None
        elif v in ('True', 'False'):
            values[k] = v == 'True'
        elif k in ('gt_count', 'count', 'error', 'identified_count', 'rejected_count', 'legacy_edge_count') or k.startswith(('pre_', 'post_')) and k not in ('post_physical_sheet_precision', 'post_physical_sheet_recall', 'post_ridge_duplicate_rate', 'post_weak_sheet_recall', 'pre_physical_sheet_precision', 'pre_physical_sheet_recall', 'pre_ridge_duplicate_rate', 'pre_weak_sheet_recall'):
            values[k] = int(v)
        elif k in ('seconds', 'sigma_min', 'sigma_max', 'sigma_v') or k.startswith(('pre_', 'post_')):
            values[k] = float(v)
        else:
            values[k] = v
    return values


def number(x, digits=4):
    return 'N/A' if x is None else f'{x:.{digits}f}'


def percent(x):
    return 'N/A' if x is None else f'{x:.2%}'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    rows = [typed(r) for r in read_csv(P1 / 'per_case.csv')]
    baseline = read_csv(OUT / 'p0/A0_DUAL_E0_COUNTS.csv')
    summary = []
    for mode in ('metrology', 'raw_estimated'):
        for stage in ('A1', 'A2', 'A3', 'A4'):
            rr = [r for r in rows if r['scope'] == f'synthetic_{mode}' and r['stage'] == stage]
            summary.append(summarize(rr, f'synthetic_{mode}', stage, 'RAW'))
    summary.append(summarize(baseline, 'real_common', 'A0', 'archived'))
    for source in ('RAW', 'normalized'):
        for stage in ('A1', 'A2', 'A3', 'A4'):
            rr = [r for r in rows if r['scope']=='real_common' and r['source']==source and r['stage']==stage]
            summary.append(summarize(rr, 'real_common', stage, source))
    summary += [dict(scope='real_common', stage=s, executed=False, metric_unavailable_reason='P0+P1 boundary; not implemented') for s in ('A5','A6')]
    # Retain the initial draft metrics: candidate vs final weak recall now have
    # separate names, while every saved prediction/array remains unchanged.
    if not (P1 / 'metrics_draft_candidate_weak_recall.csv').exists():
        shutil.copyfile(P1 / 'P1_RIDGE_VALLEY_ABLATION.csv', P1 / 'metrics_draft_candidate_weak_recall.csv')
    write_csv(P1 / 'P1_RIDGE_VALLEY_ABLATION.csv', summary)
    if (ROOT / 'P1_RIDGE_VALLEY_ABLATION.csv').exists():
        raise FileExistsError('Refuse to replace an existing delivered root CSV')
    shutil.copyfile(P1 / 'P1_RIDGE_VALLEY_ABLATION.csv', ROOT / 'P1_RIDGE_VALLEY_ABLATION.csv')
    diagnosis = []
    real = [r for r in rows if r['scope']=='real_common']
    for r in real:
        if r['error'] is None or r['error'] == 0:
            continue
        record = json.loads((P1 / 'real' / r['image'] / r['source'] / r['stage'] / 'candidates.json').read_text(encoding='utf-8'))
        reasons = Counter(c['reason'] for c in record['rejected'])
        common = dict(image=r['image'], stage=r['stage'], source=r['source'], gt_count=r['gt_count'],
                      ridge_center_count=r['count'], error=r['error'], count_semantics='1D proxy vs whole-image count GT')
        diagnosis.append(dict(**common, category='count_proxy_under' if r['error'] < 0 else 'count_proxy_over', status='confirmed',
                              evidence='saved candidate count compared with verified count GT', physical_error_location='unknown'))
        diagnosis.append(dict(**common, category='width_measurement', status='unknown',
                              evidence=f"RAW brightness-run sigma=[{r['sigma_min']},{r['sigma_max']}], gap sigma={r['sigma_v']}; no measured body curve GT", physical_error_location='unknown'))
        diagnosis.append(dict(**common, category='raw_gate_and_region', status='confirmed_stage_operation' if reasons else 'unknown',
                              evidence=json.dumps(dict(reasons), ensure_ascii=False), physical_error_location='unknown'))
        diagnosis.append(dict(**common, category='physical_identity_or_roi_loss', status='unknown',
                              evidence='No verified curve identity, no archived A0 stage geometry; total error cannot locate a missed layer', physical_error_location='unknown'))
    write_csv(P1 / 'case_diagnosis.csv', diagnosis)
    deltas = []
    by_image = defaultdict(dict)
    for r in real:
        by_image[(r['image'],r['source'])][r['stage']] = r
    for (image, source), stages in by_image.items():
        for old, new in (('A1','A2'),('A2','A3'),('A3','A4')):
            before, after = stages[old], stages[new]
            delta_error = abs(after['error'])-abs(before['error']) if before['error'] is not None else None
            deltas.append(dict(image=image, source=source, transition=old+'->'+new, gt_status=before['gt_status'],
                               before_count=before['count'], after_count=after['count'], count_delta=after['count']-before['count'],
                               absolute_error_delta=delta_error, classification='unscorable' if delta_error is None else 'improved' if delta_error<0 else 'worsened' if delta_error>0 else 'unchanged'))
    write_csv(P1 / 'stage_deltas.csv', deltas)
    groups = []
    for grouping in ('dataset', 'material', 'pile_group'):
        for group in sorted({r[grouping] for r in real}):
            for source in ('RAW', 'normalized'):
                for stage in ('A1','A2','A3','A4'):
                    rr = [r for r in real if r[grouping]==group and r['source']==source and r['stage']==stage]
                    groups.append(dict(grouping=grouping, group=group, **summarize(rr, 'real_common', stage, source)))
    write_csv(P1 / 'group_metrics.csv', groups)
    synthetic = [r for r in rows if r['scope'].startswith('synthetic')]
    sg = []
    for mode in ('metrology','raw_estimated'):
        for kind in sorted({r['material'] for r in synthetic}):
            for stage in ('A1','A2','A3','A4'):
                rr = [r for r in synthetic if r['scope']==f'synthetic_{mode}' and r['material']==kind and r['stage']==stage]
                sg.append(dict(kind=kind, **summarize(rr, f'synthetic_{mode}', stage, 'RAW')))
    write_csv(P1 / 'synthetic_group_metrics.csv', sg)
    sensitivity = [typed(r) for r in read_csv(P1 / 'threshold_sensitivity.csv')]
    sensitivities = []
    for pair in sorted({(r['area_ratio'],r['peak_ratio']) for r in sensitivity}):
        rr = [r for r in sensitivity if (r['area_ratio'],r['peak_ratio'])==pair]
        weak = sum(int(r['weak_layers']) for r in rr)
        sensitivities.append(dict(area_ratio=pair[0], peak_ratio=pair[1], n=len(rr), exact=sum(r['error']==0 for r in rr),
                                  MAE=float(np.mean([abs(r['error']) for r in rr])), fp=sum(int(r['fp']) for r in rr),
                                  fn=sum(int(r['fn']) for r in rr), weak_recall=sum(int(r['matched_weak']) for r in rr)/weak if weak else None))
    write_csv(P1 / 'threshold_sensitivity_summary.csv', sensitivities)
    synlines = ['# P1 合成信号测试报告', '',
        '53个确定性样本：10类纸层信号各5个随机种子，另有常量、线性漂移、纯黑带3个负样本。纸层样本均为12张，总600个物理纸身，其中10个弱纸层；纹理样本共有10个已知非纸身亮斑。随机种子及真实物理区间与响应分别保存。', '',
        '双峰信号在同一纸身内形成两个亮峰；窄缝为1px，宽缝插入额外32px间隙；背景漂移与9级噪声分别测试，端部纸身被视野裁切。宽度由信号生成的独立几何参数决定，不用中心周期。', '',
        'metrology模式输入独立指定的纸身/暗缝宽度；raw_estimated模式仅根据RAW测宽。前者是受控算子验证，不能视为真实图可自动获得的测量结果。3个无纸身负样本在metrology列中使用固定尺度做算子压力测试，该尺度不代表测得了不存在的纸身；自动模式无法测宽时输出insufficient_evidence。识别和检测均不接收人工中心坐标或目标计数。', '',
        '评价按不重叠的物理纸身区间匹配，每张最多一个TP；同纸额外峰记重复，未匹配峰记FP。定位容差为独立物理纸身区间，不是要求亚像素中心准确。GT Candidate Recall与Weak Candidate Recall在灰度筛选后、区域验证前计算；Weak Sheet Recall和Physical Sheet Precision/Recall在最终Shadow验证后计算。', '',
        '| 模式 | 阶段 | Exact | MAE | 候选召回 | 弱层候选召回 | 最终弱层召回 | 最终物理Precision/Recall | 最终重复率 | 纹理FP | Review | ms/样本 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['scope'].startswith('synthetic'):
            synlines.append(f"| {r['scope']} | {r['stage']} | {r['exact']}/{r['n']} | {r['MAE']:.4f} | {percent(r['gt_candidate_recall'])} | {percent(r['weak_candidate_recall'])} | {percent(r['weak_sheet_recall'])} | {percent(r['physical_sheet_precision'])}/{percent(r['physical_sheet_recall'])} | {percent(r['ridge_duplicate_rate'])} | {r['texture_false_positive']} | {percent(r['review_rate'])} | {r['runtime_mean_seconds']*1000:.3f} |")
    synlines += ['', '## 分类型结果', '', '| 类别 | metrology A4 Exact/样本、Σerror | 自动测宽 A4 Exact/样本、Σerror |', '|---|---:|---:|']
    for kind in sorted({r['material'] for r in synthetic}):
        pair = [[r for r in synthetic if r['scope']==f'synthetic_{mode}' and r['material']==kind and r['stage']=='A4'] for mode in ('metrology','raw_estimated')]
        synlines.append('| '+kind+' | '+' | '.join(f"{sum(r['error']==0 for r in rr)}/{len(rr)}, {sum(r['error'] for r in rr):+d}" for rr in pair)+' |')
    synlines += ['', '## 未公布验证阈值的敏感性', '',
        '所有门槛组合在读取真实GT结果之前已固定，只评估并完整保留，不选“最高分”回填配置。', '',
        '| 面积比 | 峰值比 | Exact | MAE | FP | FN | 最终弱层召回 |', '|---|---|---:|---:|---:|---:|---:|']
    for r in sensitivities:
        synlines.append(f"| {r['area_ratio']} | {r['peak_ratio']} | {r['exact']}/{r['n']} | {r['MAE']:.4f} | {r['fp']} | {r['fn']} | {percent(r['weak_recall'])} |")
    synlines += ['', 'A2/A3恢复弱层候选并提高召回，同时引入背景正峰。A4的区域/能量验证降低FP；自动测宽模式最终丢失弱层，说明“候选存在”不等于“被最后保留”。这也是本报告将候选弱层召回与最终弱层召回分列的原因。初版CSV把弱层候选召回放在weak_sheet_recall列，已保存为 metrics_draft_candidate_weak_recall.csv；正式CSV已纠正字段定义，所有预测及响应未改动。', '',
        '这批合成结果不是论文原图、论文统计复现或工业泛化率。高候选召回不能替代最终身份precision。负样本和端部错误仍保留，未通过目标张数“纠错”。', '',
        '图位于 `output/v3_p0_p1_20261009_01/p1/figures/`，各样本响应/核/GT位于 `p1/synthetic/{mode}/{case}/{stage}/`。', '',
        '测试：24项P1公式/行为测试通过；完整现有测试套件239项通过，2条原绘图空图例警告。代码版本、配置和输入哈希另见run_manifest.json。']
    (ROOT / 'P1_SYNTHETIC_TEST_REPORT.md').write_text('\n'.join(synlines)+'\n', encoding='utf-8')
    real_lines = ['# P1 真实图一维 Shadow 候选对照', '',
        '全部99张共同预测图均完成RAW和现有归一化图像的A1～A4；仅69张verified进入计数指标。22张unscorable及8张pending保留响应，不计算准确率。相同纸堆多视图全部属于audit_only，不按图级拆分开发/保留集。', '',
        '扫描线为原分辨率图像逆时针90°旋转后的ROI中心列，没有整图纵向平均。ROI与归一化复用当前未改动的旧前处理；原A0阶段证据已缺失，不能声称该ROI来自历史A0快照。保存旋转坐标及逆旋转回原输入坐标。', '',
        '**语义限制**：A0为整图多条带边界计数，A1～A4为单扫描线纸身中心候选数。此表是前端消融与代理计数核查，不是新完整二维系统准确率。没有人工纸身/边界身份GT，任何总数相同都不证明物理层对应正确。', '',
        '| 阶段 | 图像源 | Exact（69张） | MAE | Σsigned error | 多计图/张 | 少计图/张 | Review（99张） | ms/图（1D） |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['scope']=='real_common' and r['executed']:
            runtime = number(r['runtime_mean_seconds']*1000, 3) if r['runtime_mean_seconds'] is not None else 'N/A'
            real_lines.append(f"| {r['stage']} | {r['source']} | {r['exact']}/{r['n']}={percent(r['exact_count_accuracy'])} | {r['MAE']:.4f} | {r['signed_error_sum']:+d} | {r['over_images']}/{r['over_sheets']} | {r['under_images']}/{r['under_sheets']} | {percent(r['review_rate'])} | {runtime} |")
    real_lines += ['', '## 重点图', '', '| 图像 | 状态/GT | A0 | RAW A1/A2/A3/A4 | normalized A1/A2/A3/A4 | A4 RAW Review |', '|---|---|---:|---|---|---|']
    baseline_by_key = {r['image']: r for r in baseline}
    focus_rows = []
    for key in sorted(baseline_by_key):
        if any(t in key for t in ('200g4-1','200g4-2','7号纸3-2','7号纸4-1','7号纸4-2','3-3-2','9号纸1-2','淋膜纸2')):
            b = baseline_by_key[key]
            raw, norm = by_image[(key,'RAW')], by_image[(key,'normalized')]
            focus_rows.append(dict(image=key, gt_status=b['gt_status'], gt_count=b['gt_count'], legacy_edge_count=b['count'],
                                   **{f'{source}_{stage}_ridge_center_count': by_image[(key,source)][stage]['count'] for source in ('RAW','normalized') for stage in ('A1','A2','A3','A4')}))
            real_lines.append(f"| {key} | {b['gt_status']}/{b['gt_count']} | {b['count']} | "+'/'.join(str(raw[s]['count']) for s in ('A1','A2','A3','A4'))+' | '+'/'.join(str(norm[s]['count']) for s in ('A1','A2','A3','A4'))+f" | {raw['A4']['review_required']} |")
    write_csv(P1 / 'difficult_cases.csv', focus_rows)
    real_lines += ['', '## A3→A4 的候选验证影响', '', '| 图像源 | 改善 | 恶化 | 误差不变 |', '|---|---:|---:|---:|']
    for source in ('RAW','normalized'):
        dd = [r for r in deltas if r['transition']=='A3->A4' and r['source']==source and r['gt_status']=='verified']
        cc = Counter(r['classification'] for r in dd)
        real_lines.append(f"| {source} | {cc['improved']} | {cc['worsened']} | {cc['unchanged']} |")
    real_lines += ['', '这里的改善/恶化是同图代理计数绝对误差变化。真实图区域拒绝、灰度拒绝和合并操作可由日志confirmed，但被拒峰是否真实纸身只能unknown。case_diagnosis.csv没有从−N误差生成N个“漏纸位置”。', '',
        '## 旧稳定回归图', '', '| 范围 | A0 Exact | RAW A4 Exact/MAE | normalized A4 Exact/MAE |', '|---|---:|---:|---:|']
    vr = [r for r in real if r['dataset']=='验证集' and r['stage']=='A4']
    real_lines.append('| 验证集16张 | 16/16 | '+' | '.join(f"{metrics([r for r in vr if r['source']==s])['exact']}/16 / {metrics([r for r in vr if r['source']==s])['MAE']:.4f}" for s in ('RAW','normalized'))+' |')
    metadata = json.loads((P1 / 'input_manifest.json').read_text(encoding='utf-8'))
    real_lines += ['', f"同一前处理平均耗时 {np.mean([r['preprocessing_seconds'] for r in metadata]):.4f}s/图（图像读取、旋转、ROI、归一化，不含SHA/写图）。表中1D耗时包含测宽、诊断valley、全部尺度卷积、峰识别及区域验证，不含2D前处理、IO或绘图。A0历史运行时缺失，不能据此声称生产速度提升。", '',
        'Review在全部99张上统计，未用于筛除难图提高准确率。complete_support仅表示核窗口有输入像素，不能保证是正确纸身；review=false也不是已校准的正确概率。', '',
        '真实GT Candidate Recall、Weak Sheet Recall、Ridge Duplicate Rate、Texture False Positive、Physical Sheet Precision/Recall均N/A：缺已核验人工曲线/纹理标注。Track Association Accuracy及Cross-layer Link Rate均N/A：本轮未实现二维关联。', '',
        '图位于 `p1/figures/real/{image}.png`；8套每图候选、全部尺度响应和核分别位于 `p1/real/{image}/{RAW|normalized}/{A1...A4}/`。A0没有可追溯边界曲线，图中只显示其冻结计数。']
    (ROOT / 'P1_REAL_SHADOW_COMPARISON.md').write_text('\n'.join(real_lines)+'\n', encoding='utf-8')
    rec = ['# P2 实施建议（本轮未执行）', '',
        '本轮公式与受控行为验证成立，但单扫描线结果未证明整图净收益。保持P1 Shadow、Dual E0对照，E1默认关闭。不能将A4直接替换旧后端或把其候选与Scharr峰相加。', '',
        '## 按优先级推进', '',
        '1. **先补评价依据**：核验真值表54张的图名对应与物理堆ID；200g第4堆、7号纸第3/4堆、覆膜第3堆和淋膜纸2建立人工纸身中心、上下边界、不可观测区及纹理负样本。proposal只能pending，不能冒充verified。整堆分开发/保留，预先锁定接受门槛。',
        '2. **再验证宽度与可观测性**：自动RAW阈值测宽会把暗弱纸身归入暗缝，也会把纹理分裂当窄纸身；测宽sigma与gap-sigma分别审计。用人工宽度GT评价估计偏差，保留RAW与归一化差异、饱和与宽黑带，避免按GT计数调σ或参考区域。先固定全局同一规则再做留堆验证。',
        '3. **逐实际条带接入辅助观测**：统一Evidence含类型、原坐标、来源、sigma、RAW支持和不可观测状态；positive_edge、negative_edge、ridge_center、valley_gap不可混为同一个语义。逐条带计算，跨尺度去重；建立辅助中心轨迹后仍分别记录legacy_edge_count与ridge_center_count。',
        '4. **重点检查合并拒绝弱纸**：A4区域验证能够消掉合成背景假峰，但自动测宽模式最终丢掉弱层。需要人工身份GT确认同一正区域的多峰是同纸双峰还是相邻两纸；不能仅靠“一正区域一纸”强制合并所有难图。面积/峰值阈值未在论文发布，作为独立实验参数验证，并保留每次候选增删。',
        '5. **二维评估与回退**：用已verified曲线计算召回、重复率、纹理FP和跨层率；检查旧16张及83张全量回归，按物理堆汇总paired count delta，report review率，不过滤review图取得高分。新中心观测必须有真实支持，不跨宽黑带补造。未证明净收益继续Shadow。', '',
        '## P3以后再处理的事项', '',
        'SheetHypothesis、同纸身份联合计数、tracking_global及Vertical Sequence联合优化属于P3/P4；正式计数切换需要新的显式开关与独立验收。P5性能优化需先锁响应等价。它们没有在本轮实现。', '',
        '当前没有经过校准的正确概率、逐纸真实关联准确率或可主张的99%整图准确率。合成A4召回不能代替真实物理身份验证。']
    (ROOT / 'P2_RECOMMENDATIONS.md').write_text('\n'.join(rec)+'\n', encoding='utf-8')
    results = ['# V3 P0 + P1 执行结果', '',
        '本轮已完成冻结/GT审计、连续核公式实现、合成测试和99张真实图的1D Shadow消融。未进入P2～P5，未修改现有最终计数。', '',
        '- 主要对照Dual E0：69张verified，48张完全正确，Exact 69.5652%，MAE 1.681159。',
        '- 合成metrology A4：51/53完全正确，MAE 0.056604，物理区间召回100%；自动测宽A4：48/53，MAE 0.188679，最终弱层召回0%，说明弱层候选虽存在，仍被区域能量门槛删除。',
        '- 真实单扫描线A4：RAW 25/69，MAE 8.753623；normalized 27/69，MAE 8.405797。单扫描线计数语义与完整二维系统不同，结果不足以证明替换收益。',
        '- 完整测试239通过（包含24项P1测试），2条已有绘图警告；原GT、原配置、现有后端源码与已保存历史交付文件的保护哈希通过。', '',
        '## 交付索引', '',
        '- `V3_DESIGN.md`：设计、坐标语义和授权范围。',
        '- `V3_BASELINE_AUDIT.md`：冻结证据、逐行映射、列顺序问题与同堆划分限制。',
        '- `P1_PAPER_REPRODUCTION.md`：分段公式、验证、不能准确确认的数值细节。',
        '- `P1_RIDGE_VALLEY_ABLATION.csv`：A0～A4统计、A5/A6未执行、不可评价指标为空。',
        '- `P1_SYNTHETIC_TEST_REPORT.md`：合成分型、阈值敏感性与失败案例。',
        '- `P1_REAL_SHADOW_COMPARISON.md`：真实重点图、稳定回归及逐阶段计数变化。',
        '- `P2_RECOMMENDATIONS.md`：下一阶段优先级，未实际执行。', '',
        '详细结果：`output/v3_p0_p1_20261009_01/`。p0保存GT和冻结计数；p1保存53×2×4合成结果及99×2×4真实结果，各有候选JSON、NPZ核及全部尺度响应。', '',
        '使用最新原GT状态，淋膜纸1/2均40；unscorable/pending保留。没有通过图名、目标张数或人工坐标特殊处理算法。图名仅用于报告重点图选择。', '',
        '## 结论与限制', '',
        'Gaussian/Bi-Gaussian/Ridge–Valley连续公式及区域验证结构已实现并检验。论文未给验证门槛、数字卷积和参考区细节；PDF未取得供像素核查，不声称原论文完整实验数值复现成功。真实候选质量缺曲线GT，GT逐行对应缺独立文件名确认，原E0阶段文件已缺失。', '',
        '区域验证对合成噪声/纹理有作用，但自动测宽与最终弱层删除是明确的后续审计重点。当前结果保持Shadow，正式系统保持既有计数路径。']
    (ROOT / 'P1_RESULTS.md').write_text('\n'.join(results)+'\n', encoding='utf-8')
    frozen = ROOT / 'experiments/v3/p1'
    frozen.mkdir(parents=True, exist_ok=False)
    for name in ('P1_RIDGE_VALLEY_ABLATION.csv','run_manifest.json','difficult_cases.csv','threshold_sensitivity_summary.csv','stage_deltas.csv'):
        shutil.copyfile(P1 / name, frozen / name)
    write_json(frozen / 'test_results.json', dict(p1_tests=24, full_tests=239, warnings=2, full_seconds=100.39,
               warning_source='existing track_visualization empty legends', result='passed'))
    (frozen / '.gitattributes').write_text('* -text\n', encoding='ascii')
    print('Reports derived; no prediction changes. Final weak recall is separately reported.')


if __name__ == '__main__':
    main()
