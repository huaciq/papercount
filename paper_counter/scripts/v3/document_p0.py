"""Generate P0 reviewable records using audited named fields."""
from pathlib import Path
import json
import shutil
from audit_p0 import ROOT, OUT, OLD, sha, write_json


def main():
    p0 = OUT / 'p0'
    a = json.loads((p0 / 'audit.json').read_text(encoding='utf-8'))
    snapshot = json.loads((p0 / 'GT_VERSION.json').read_text(encoding='utf-8'))
    frozen = ROOT / 'experiments/v3/p0'
    frozen.mkdir(parents=True, exist_ok=False)
    for name in ('GT_VERSION.json', 'A0_DUAL_E0_COUNTS.csv', 'baseline_metrics.csv', 'A0_original_config.yaml'):
        shutil.copyfile(p0 / name, frozen / name)
    cfgdir = ROOT / 'configs/v3'
    cfgdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p0 / 'A0_effective_config.json', cfgdir / 'p0_dual_e0.json')
    write_json(frozen / 'manifest.json', dict(gt_version=a['gt_version'], gt_sha256=a['gt_sha256'],
        source_delivery=str(OLD), source_delivery_sha256=sha(OLD / 'delivery_manifest.json'),
        missing_historical_paths=a['old_stage_paths_missing'],
        files={p.name: sha(p) for p in frozen.glob('*') if p.is_file()},
        config_sha256=sha(cfgdir / 'p0_dual_e0.json'),
        pipeline_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'paper_counter').glob('*.py')},
        partition='real audit_only; synthetic development only'))
    lines = ['# V3 基线与 GT 审计（P0）', '',
        '本轮只执行 P0 + P1。主要对照为已归档的 Dual E0 整图计数；E1 独立保留且默认关闭。未重新生成或覆盖历史预测。', '',
        f"GT 版本 `{a['gt_version']}`，SHA256 `{a['gt_sha256']}`。", '',
        '## 文件与标注完整性', '',
        '105 张图像 SHA256 全部吻合 GT 快照；29 个原人工 GT 文件的哈希全部吻合上轮审计。源行文字与快照一致，未把“左右”等估计改成整数。105 张包含 verified 71、unscorable 22、pending 12。', '',
        '四方案同组 99 张：verified 69、unscorable 22、pending 8。其余 6 张五粮液未进入四方案配对评价。淋膜纸1、2当前 verified GT 均为40。', '',
        '旧 E0/E1 阶段目录及部分历史路径已缺失：本次扫描记录418个缺失历史路径。保存下来的 delivery_02 文件按其交付清单检查；本次使用的 GT_VERSION.json、per_image.csv、group_metrics.csv、integrity_audit.json 均通过 SHA256。', '',
        '**冻结边界**：可验证并冻结的是历史计数 CSV、GT 字节副本、配置与当前源码哈希，不能凭计数 CSV 恢复历史候选、轨迹、ROI或运行时。新的 Dual E0 显式配置关闭 E1/E2扩展，不能把当前源码重跑称作原实验原始阶段快照。', '',
        '本次 Git 保存独立的 experiments/v3/p0 副本与 configs/v3/p0_dual_e0.json；详细审计位于 output/v3_p0_p1_20261009_01/p0。原文件没有被替换。', '',
        '## 同组 verified 指标', '',
        '| 方案 | n | Exact | Exact Accuracy | MAE（张/图） | signed sum | signed mean | 多计图/张 | 少计图/张 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in a['baseline_metrics']:
        lines.append(f"| {r['model']} | {r['n']} | {r['exact']} | {r['exact_accuracy']:.4%} | {r['MAE']:.6f} | {r['signed_error_sum']:+d} | {r['signed_error_mean']:.6f} | {r['over_images']}/{r['over_sheets']} | {r['under_images']}/{r['under_sheets']} |")
    lines += ['', 'Exact 指整图计数完全相同。旧运行时与人工曲线级指标均没有可用的原始证据，不能置零或代推。', '',
        '## 逐行映射、分组与划分风险', '',
        '测试集真值表未逐行写图名，快照使用拍摄时间及同时间副本序号排序。机器审计确认文件存在、图像内容、源行原文、状态和计数一致；另以 audit_mapping_order.py 独立复核文件名排序及快照行号，详见 filename_order_audit.csv。**排序对应的人工语义仍是未经独立确认的映射假设**。这不会自动降低或改写原 verified 状态，但限制准确率解释。详见 gt_mapping_audit.csv。', '',
        '旧 group_metrics.csv 按 grouping/group/model 列名重算，与所有四方案的 n、exact、MAE、signed sum 一致。发现旧 resources/e1_5_report.py 的纸种 Markdown 表头顺序为 Baseline E0、Baseline E1、Dual E0、Dual E1，而生成单元格循环顺序为 Baseline E0、Dual E0、Baseline E1、Dual E1；这是报告生成器的列顺序问题。原历史 Markdown 文件已缺失，本次不覆盖历史报告，并在新表中显式按列名填充。', '',
        '纸堆ID来自文件名推定，旧验证集没有已核验物理堆ID。为避免多视图泄漏，本次所有真实图只作 audit_only，合成信号用于开发；不划分真实开发/保留集，也不声称这是独立泛化测试。P2前需人工确认图名对应与物理堆ID，再按整堆划分。', '',
        '## 重点图的 A0 计数', '', '| 图像 | 状态 | GT | Dual E0 | 总误差 |', '|---|---|---:|---:|---:|']
    import csv
    with (p0 / 'A0_DUAL_E0_COUNTS.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if any(s in r['image'] for s in ('200g4-1', '200g4-2', '7号纸3-2', '7号纸4-1', '7号纸4-2', '3-3-2', '9号纸1-2', '淋膜纸2')):
                lines.append(f"| {r['image']} | {r['gt_status']} | {r['gt_count']} | {r['count']} | {r['error']} |")
    lines += ['', '总误差只证明少计或多计，不证明具体漏纸Y位置。重点图的1D Shadow图将在P1独立输出，不据此修改GT或调单图参数。', '',
        '## 可复核入口', '',
        '- `python scripts/v3/audit_p0.py`：只读取原数据，输出目录须不存在，防止覆盖。',
        '- `p0/audit.json`：源哈希、配置、历史缺失路径及统计。',
        '- `p0/group_metrics.csv`：dataset/material/pile_group三类指标，列名固定。',
        '- `p0/gt_mapping_audit.csv`、`p0/non_evaluable.csv`：105行映射与不可评价清单。',
        '- `p0/protected_before.json`：后续阶段必须逐文件验证未变化。']
    (ROOT / 'V3_BASELINE_AUDIT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
