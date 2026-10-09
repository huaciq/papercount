# P1 实际检测图与重跑测试

本次重新运行检测并将结果标到实物照片，输出独立目录，没有覆盖历史实验或修改算法。照片采用既有逆时针90°检测方向和ROI裁切；显示缩放不参与检测。

- 完整测试：239项通过，2条既有空图例警告，耗时109.44秒。
- P1：99张照片、RAW/归一化两路、A1～A4共792套候选及响应，全部与冻结结果一致；坐标逆映射通过。
- 旧Dual E0：重跑16张验证图及7张重点困难图，23/23计数与历史归档一致。这是归档计数一致性，不是23张都计数正确。
- 188个受保护的旧源码、配置、GT和历史交付文件保持原哈希。未进入P2～P5。

## 直接看图

[全部检测图索引](F:/纸张/paper_counter/output/v3_detection_images_20261009_01/DETECTION_IMAGE_INDEX.md)

| 目录 | 内容 | 图数 |
|---|---|---:|
| [Dual_E0_fresh](F:/纸张/paper_counter/output/v3_detection_images_20261009_01/Dual_E0_fresh/) | 旧后端实际多条带轨迹 | 23 |
| [A4_RAW](F:/纸张/paper_counter/output/v3_detection_images_20261009_01/A4_RAW/) | P1 RAW中心候选叠加照片 | 99 |
| [A4_normalized](F:/纸张/paper_counter/output/v3_detection_images_20261009_01/A4_normalized/) | P1归一化中心候选叠加照片 | 99 |
| [compare](F:/纸张/paper_counter/output/v3_detection_images_20261009_01/compare/) | 旧轨迹 / P1 RAW / P1归一化三图并排 | 23 |
| [A1_A4_compare](F:/纸张/paper_counter/output/v3_detection_images_20261009_01/A1_A4_compare/) | A1～A4 RAW实际候选四图并排 | 23 |

另有重点图及验证图的A1、A2、A3两路单图，共138张；本次总计405张PNG。

P1短刻度仅表示扫描线上的中心观测，未扩展为二维纸边。绿表示完整输入支持，橙表示支持不完整，蓝为扫描线；颜色不是正确概率。旧后端只连接相邻有观测条带，黄表示既有恢复/救援轨迹。GT只出现在评价与图注中。

## 示例

[淋膜纸2三图对照](F:/纸张/paper_counter/output/v3_detection_images_20261009_01/compare/验证集/淋膜纸2.bmp.png)：旧Dual E0为40，P1 RAW为30，归一化为34，GT为40。

[200g4-1旧后端轨迹图](F:/纸张/paper_counter/output/v3_detection_images_20261009_01/Dual_E0_fresh/测试集/260921-200g铜板/Pic_2026_09_21_140511_6-200g4-1.bmp.png)：旧后端137，GT170；保留真实少计结果，未根据GT增画纸层。

运行入口：`scripts/v3/run_detection_images.py`；输出目录须不存在，防止覆盖。结果目录内包含新运行候选JSON、旧轨迹及压缩阶段证据、`fresh_results.csv`、`fresh_equivalence_audit.json`、`test_results.json`和文件SHA256清单。
