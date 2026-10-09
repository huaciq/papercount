# V3 P0 + P1 执行结果

本轮已完成冻结/GT审计、连续核公式实现、合成测试和99张真实图的1D Shadow消融。未进入P2～P5，未修改现有最终计数。

- 主要对照Dual E0：69张verified，48张完全正确，Exact 69.5652%，MAE 1.681159。
- 合成metrology A4：51/53完全正确，MAE 0.056604，物理区间召回100%；自动测宽A4：48/53，MAE 0.188679，最终弱层召回0%，说明弱层候选虽存在，仍被区域能量门槛删除。
- 真实单扫描线A4：RAW 25/69，MAE 8.753623；normalized 27/69，MAE 8.405797。单扫描线计数语义与完整二维系统不同，结果不足以证明替换收益。
- 完整测试239通过（包含24项P1测试），2条已有绘图警告；原GT、原配置、现有后端源码与已保存历史交付文件的保护哈希通过。

## 交付索引

- [V3_DESIGN.md](F:/纸张/paper_counter/V3_DESIGN.md)：设计、坐标语义和授权范围。
- [V3_BASELINE_AUDIT.md](F:/纸张/paper_counter/V3_BASELINE_AUDIT.md)：冻结证据、逐行映射、列顺序问题与同堆划分限制。
- [P1_PAPER_REPRODUCTION.md](F:/纸张/paper_counter/P1_PAPER_REPRODUCTION.md)：分段公式、验证、不能准确确认的数值细节。
- [P1_RIDGE_VALLEY_ABLATION.csv](F:/纸张/paper_counter/P1_RIDGE_VALLEY_ABLATION.csv)：A0～A4统计、A5/A6未执行、不可评价指标为空。
- [P1_SYNTHETIC_TEST_REPORT.md](F:/纸张/paper_counter/P1_SYNTHETIC_TEST_REPORT.md)：合成分型、阈值敏感性与失败案例。
- [P1_REAL_SHADOW_COMPARISON.md](F:/纸张/paper_counter/P1_REAL_SHADOW_COMPARISON.md)：真实重点图、稳定回归及逐阶段计数变化。
- [P2_RECOMMENDATIONS.md](F:/纸张/paper_counter/P2_RECOMMENDATIONS.md)：下一阶段优先级，未实际执行。

详细结果：[output/v3_p0_p1_20261009_01](F:/纸张/paper_counter/output/v3_p0_p1_20261009_01/)。p0保存GT和冻结计数；p1保存53×2×4合成结果及99×2×4真实结果，各有候选JSON、NPZ核及全部尺度响应。[滤波核对照图](F:/纸张/paper_counter/output/v3_p0_p1_20261009_01/p1/figures/kernel_comparison.png)；[真实困难图响应目录](F:/纸张/paper_counter/output/v3_p0_p1_20261009_01/p1/figures/real/)。

使用最新原GT状态，淋膜纸1/2均40；unscorable/pending保留。没有通过图名、目标张数或人工坐标特殊处理算法。图名仅用于报告重点图选择。

## 结论与限制

Gaussian/Bi-Gaussian/Ridge–Valley连续公式及区域验证结构已实现并检验。论文未给验证门槛、数字卷积和参考区细节；PDF未取得供像素核查，不声称原论文完整实验数值复现成功。真实候选质量缺曲线GT，GT逐行对应缺独立文件名确认，原E0阶段文件已缺失。

区域验证对合成噪声/纹理有作用，但自动测宽与最终弱层删除是明确的后续审计重点。当前结果保持Shadow，正式系统保持既有计数路径。
