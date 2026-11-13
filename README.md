# 纸张计数项目

对外调用与安装：见 [Python API 使用说明](paper_counter/API_README.md)。新接口 `from paper_counter import detect_papers, PaperDetector` 默认使用最新显式 v4 配置，支持路径/内存图像与可选结果导出；`preset="e3_3"` 可复现原命令行默认配置。

## `tracks` 水平纸堆预处理变体

当前稳定 `tracks` 默认仍保持 E3.3，历史结果可复现。新增 `tracks_preprocessed`（[配置](paper_counter/configs/tracks_preprocessed.yaml)）用于纸边接近水平、层间暗缝较清楚的图像：沿 x 方向抑制纹理、低增益增强长水平暗缝，并在终端清理共享实测观测的近重复轨迹。RAW 掩膜、纸身证据和身份终判继续读取原始灰度；因此增强图只负责提出候选，不会凭空增加纸张。需要自动小角度旋正时，可在该配置中打开 `auto_geometry_enabled`，并使用输出里的几何仿射回映原图坐标。

显式调用：

```powershell
conda activate pytorch
python -X utf8=0 main.py --image "实际图像路径.bmp" --config configs/tracks_preprocessed.yaml --rotate-ccw --output output/tracks_preprocessed
```

## 横向不均匀的复合明暗结构（2026-10-15，局部实验）

新增 `configs/composite_layers.yaml`：每条带沿局部方向保留三个独立RAW窗口，学习相邻纸层的完整灰度与导数结构，允许纸内暗纹和亮分界。新增位置来自实际结构匹配响应的峰；半周期有歧义时不补层。左右不相交片段若位于同一上下参考槽位，只合并观测，不增加张数。原v4配置和默认入口没有切换。

按用户要求只测试 `数据/测试集/260921-200g铜板` 的7张：6张有可用人工真值，1张未核验。完全准确仍为3/6；总绝对误差22→18，平均绝对误差3.6667→3.0000。200g4-1由159→161，200g4-2由161→163，其余verified计数不变，未出现新的计数回归。未核验200g2-1由158→160，不纳入收益。73项相关单元测试通过；没有进行全量99张回归，不能声称全数据集提升。

- [7张前后对照、完整图表及坐标](paper_counter/output/composite_layers_20261015/index.html)
- [实验结果与限制](paper_counter/output/composite_layers_20261015/REPORT.md)
- [实现原理与运行入口](paper_counter/docs/COMPOSITE_LAYERS.md)

200g3-1在输入目录中已预先旋转，本次不再重复旋转；已核对历史检测区域RAW预览逐像素一致。其他6张仍逆时针旋转90°，原图和人工总张数没有改写。

## 按用户实际误差修复梯度流程（2026-10-14，显式 v4）

沿用12条带梯度前端，修正端部越界零占位被当黑背景、低亮外部反光误计，以及纸身补检替换成熟主轨迹的问题。补充候选须有独立纸层证据；只有一条带清楚时，允许用三段实测纸身和两处分隔暗槽证明存在，不能把几像素亮斑直接增加为一张纸。

99张全部重新检测成功，83张有人工真值的图片完全正确由70/83提高到75/83（84.34%→90.36%），每图平均绝对误差由0.7349降到0.4578。7张误差改善、2张变大；原13张误差图中6张完全纠正、1张改善，剩余误差图共8张。325项测试通过，普通命令行入口复测通过。原图和冻结GT逐张核对，未用文件名、用户坐标或GT参与检测判断。

- [全量99张修改前后检测图](paper_counter/output/gradient_fix_20261014/full_v3/index.html)
- [用户28处标记的局部对照](paper_counter/output/gradient_fix_20261014/marked_positions/index.html)
- [结果与剩余问题报告](paper_counter/output/gradient_fix_20261014/REPORT.md)
- [实现原理和运行方法](paper_counter/docs/GRADIENT_EVIDENCE_REPAIRS.md)

图片输出到 `paper_counter/output/gradient_fix_20261014/full_v3/images/`，逐图目录含 `result_overlay.png`、中文对照说明和完整诊断。默认配置文件未切换；这批指标只对应显式 `--config configs/paper_layers_v4.yaml`。200g压实/弯曲纸边和灰色淋膜纸的部分关联漏检仍未解决；验证集淋膜纸2由40变41，长条淋膜纸2-1由109变106，两处回归保留在报告中。该批数据参与开发，不是独立保留集的泛化证明。

## 压实弱层纹改进（2026-10-14，显式试验配置）

新增沿原始图像局部方向的横向积累、三个独立窗口的弱层纹核验、邻近实测层纹模板，以及片段归属与同纸去重。只增加一个开关，不按周期生成虚拟纸层；背景和明确无纸的拒绝仍保留。

99 张全量检测成功。与上一轮 v4 相比，83 张有人工计数的图片中 1 张改善、0 张退步、82 张计数不变；完全正确仍为 70/83，MAE 从 0.7349 降到 0.7229。200g3-1 从 168 变成 169，人工数为 170，仍少一张。收益有限，尚未解决压实纸层的全部困难。315 项测试通过，默认仍为 E3.3。

- [99 张结果图和逐图中文说明](paper_counter/output/compressed_20261014/full_v10/index.html)
- [实际结果报告](paper_counter/output/compressed_20261014/REPORT.md)
- [代码原理与使用方法](paper_counter/docs/COMPRESSED_LAYERS.md)

结果图位于 `paper_counter/output/compressed_20261014/full_v10/images/`，每张图片的目录有 `result_overlay.png`、`检测说明.html` 和完整诊断。使用新分支时显式指定 `--config configs/compressed_layers.yaml`。

## 纸身证据与方向关联改造（2026-10-13，显式试验配置）

新增 RAW 纸身三态判定、亮带顶帽/暗缝底帽候选、局部方向预测与保序关联、实测弱周期条纹，以及清晰局部锚点的存在证据。输出单独报告已确认和待确认的纸身证据，确认有亮带不能代替独立纸张的身份判定。参数改为分组入口，停用七项没有消费者的参数，历史平铺 YAML 保持兼容。

新增拥挤轨迹、漏层和真实黑色间隔的 RAW 信号核验。修正上一轮亮带加分和终判豁免造成的退化后，`gap_full_v11` 已全量检测 99 张，0 运行失败；83 张 verified 中完全正确 70 张（84.34%），MAE 0.7349。原方案为 69/83、MAE 1.0241；上一轮失败模型 full_v18 为 48/83、MAE 1.1084。与原方案相比，10 张误差改善、8 张退步、65 张不变；296 项测试通过。

本轮保留稳定默认 E3.3。试验改进模型必须增加 `--config "configs/paper_layers_v4.yaml"`。稳定默认已重新检测 99 张，全部计数与修改前一致。新模型会导出未解决信号间隙并提示复核，不能将局部亮带或计数吻合 GT 当成全宽纸张身份已经验证。

- [新模型设计与代码导读](paper_counter/docs/PAPER_LAYERS_V4.md)
- [拥挤轨迹、缺层与黑色间隔的处理](paper_counter/docs/SIGNAL_GAP_REVIEW.md)
- [本轮完整回归报告](paper_counter/output/revision_20261013/REPORT.md)
- [样本与 RAW 证据图](paper_counter/output/revision_20261013/index.html)
- [所有 verified 图的修改前后计数](paper_counter/output/revision_20261013/comparison.csv)
- [本轮全量 99 张检测图](paper_counter/output/revision_20261013/gap_full_v11/index.html)

全部结果图位于 `paper_counter/output/revision_20261013/gap_full_v11/images/`，按原数据集、纸种及原图片文件名建目录；每个目录包含 `result_overlay.png`、完整诊断和较小的 `signal_review.json`。历史失败模型 full_v18 的逐图统计、回归报告和阶段诊断保留用于对照，当前本地已没有它的大体积图像/源码目录。

```powershell
python -X utf8=0 main.py --image "实际图像路径.bmp" --rotate-ccw --config "configs/paper_layers_v4.yaml" --output "output/current_v4" --full-debug
```

## 当前默认：E3.3 端部证据与轨迹关联修复（2026-10-10）

默认配置已启用首末 RAW 纸身验证、孤立暗反光拒绝、辅助轨迹交叉与同纸去重、端部碎片重新关联。最近新增修复长条淋膜纸1-2：117→118，只重新关联已有且具备RAW纸身证据的候选，不生成虚拟纸层。使用同一最新 GT，83 张 verified 的完全正确数由原E2的60提升到69（72.29%→83.13%），MAE 1.1446→1.0241；测试集53/67正确，验证集16/16正确。10张verified误差改善、0张退步、73张不变。另有7张unscorable计数变化，不计入收益。99张全部重新检测，256项测试通过。

GT版本：`e1_5_count_gt_7ed99c5259a30b2d`。最新两项人工修正（长条淋膜纸1-1=117、覆膜铜板纸3-2-1=133）已纳入；原人工GT未修改。初始无约束放宽方案未采用，目前启用带可观测性检查的端部验证。用户标注15张中11张计数正确（9张算法修复、2张GT修正）；灰色淋膜纸1-2、白底白板4-2、7号纸3-1/4-1仍待解决。总体仍有14张verified计数错误，未达到99%。

- [15张重点图：修改前后与逐图诊断](paper_counter/output/endpoint_fix_20261010/index.html)
- [最新99张检测图](paper_counter/output/endpoint_fix_20261010/reassociation_full/index.html)
- [修复与回归报告](paper_counter/output/endpoint_fix_20261010/REPORT.md)
- [本次完整指标](paper_counter/output/endpoint_fix_20261010/reassociation_full/metrics.csv)

旧配置和预测继续保存在 `output/best/`、`output/retest_20261010_updated_gt/`；本次快照和预测在 `output/endpoint_fix_20261010/reassociation_full/`。

## 上一轮正式重跑（2026-10-10）

已读取灰色淋膜纸及长条淋膜纸更新的人工真值表，使用 E2 guarded 原配置重新检测测试集 83 张与验证集 16 张，全部成功。99 张计数均与历史正式预测一致。更新后 83 张 verified、16 张 unscorable、0 张 pending；仅 verified 参与准确率。

| 范围 | 总图 | verified | 完全正确 | Exact Accuracy | MAE | 多计 | 少计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 测试集 | 83 | 67 | 42 | 62.69% | 1.4478 | 11 | 14 |
| 验证集 | 16 | 16 | 16 | 100.00% | 0.0000 | 0 | 0 |
| 总体 | 99 | 83 | 58 | 69.88% | 1.1687 | 11 | 14 |

最新 GT 版本：`e1_5_count_gt_ecd260a96276c89c`。新纳入 14 张 verified，其中 6 张完全正确；原 69 张 verified 仍为 52/69（75.36%），因此汇总变化来自评价范围扩大。原人工标注、配置及历史结果均未覆盖。

- [最新检测图总览](paper_counter/output/retest_20261010_updated_gt/index.html)
- [完整统计报告](paper_counter/output/retest_20261010_updated_gt/REPORT.md)、[指标 CSV](paper_counter/output/retest_20261010_updated_gt/metrics.csv)、[25 张错误图清单](paper_counter/output/retest_20261010_updated_gt/error_cases.csv)
- [纸种统计](paper_counter/output/retest_20261010_updated_gt/group_metrics.csv)、[GT 变化](paper_counter/output/retest_20261010_updated_gt/gt_changes.csv)、[最新 GT 快照](paper_counter/output/retest_20261010_updated_gt/GT_VERSION.json)

下面 `output/best/` 的统计为 2026-10-09 冻结记录；最新结果在 `output/retest_20261010_updated_gt/`。

此前默认运行 **E2 guarded**；现默认配置在其基础上启用上述E3.3修复。算法仍为 Scharr 正负梯度、多条带轨迹、RAW 纸身证据、Vertical Sequence、Identity 终判，并启用 RAW 周期/极性共识和孤立弱轨迹约束。E1 端部增强默认关闭。

统一配置：[paper_counter/configs/default.yaml](paper_counter/configs/default.yaml)。该文件选择稳定预设，完整历史有效参数在 `paper_counter/presets/e3_3.yaml`；无需选择历史实验配置。共用的预处理、关联、身份验证与评价模块及回归测试继续保留，冻结历史结果不变。Python API 使用 `PaperCounter(CounterConfig.from_yaml("configs/default.yaml"))`；仅构造 `CounterConfig()` 仍是模块测试使用的基础参数。

## 历史冻结效果与检测图

2026-10-09 冻结 GT 版本为 `e1_5_count_gt_b15afeb631ca4021`。在保留的 99 张预测中，69 张为 verified，22 张 unscorable，8 张 pending。仅 verified 参与准确率：

| 完全正确 | Exact Accuracy | MAE | 平均 signed error | 多计图 | 少计图 |
|---:|---:|---:|---:|---:|---:|
| 52 / 69 | 75.36% | 0.609 | -0.261 | 8 | 9 |

尚未达到 99%。这是现有数据集的评价结果，不能解释为独立保留集的泛化准确率。内部 confidence 是未校准的一致性评分，不能解释成正确概率。

- [99 张检测图浏览页](paper_counter/output/best/index.html)
- 检测图目录：`paper_counter/output/best/images/`，按原数据目录和图像文件名组织，每张图有 `result_overlay.png` 与精简 `debug_info.json`。
- [逐图结果](paper_counter/output/best/results.csv)、[总体指标](paper_counter/output/best/metrics.csv)、[17 张错误图](paper_counter/output/best/error_cases.csv)、[纸种指标](paper_counter/output/best/group_metrics.csv)、[评价排除项](paper_counter/output/best/excluded_images.csv)。
- [GT 版本及逐图映射](paper_counter/output/best/GT_VERSION.json)、[配置/输入/源码审计](paper_counter/output/best/manifest.json)。GT 保留全部 105 张记录，另有 6 张没有本方案预测。两张淋膜纸的人工计数均为 40。

保留图由已保存的 E2 guarded 最终轨迹重新渲染，没有重新检测或修改计数；绿色为选中轨迹，黄色为恢复轨迹，只显示计入计数的轨迹。历史完整中间日志已删除。此前 99 张上游新鲜运行及 21 张完整 guarded 运行已核对一致，当前保留最终轨迹和来源哈希。部分 GT 表按时间戳及副本序号逐行映射，纸堆分组由文件名推断，仍需人工确认；没有逐边 GT 时，不能从计数误差推断漏检位置。

## 运行

在 `F:\纸张\paper_counter` 打开 PowerShell，使用 `pytorch` conda 环境：

```powershell
conda activate pytorch
python -X utf8=0 main.py --image "..\数据\1.png"
```

默认输出到 `paper_counter/output/current/`，只生成检测图及精简 JSON，可重复运行覆盖当前结果，减少文件增长。已有 `output/best/` 是保留的评价结果，日常运行请使用 `output/current/` 或另设输出目录。

验证集、测试集的横向拍摄图需逆时针旋转 90°，与当前评价一致：

```powershell
python -X utf8=0 main.py --image "实际图像路径.bmp" --rotate-ccw --output "output/current"
```

需要完整候选、阶段图和中间信号时增加 `--full-debug`。批量检测仅在明确调用时生成一个带时间戳的结果目录：

```powershell
python -X utf8=0 scripts/batch_detect.py --input "..\数据\验证集" --rotate-ccw --workers 2
python -X utf8=0 scripts/evaluate_best.py --output "output/reevaluation"
python -X utf8=0 -m pytest -q tests
```

`evaluate_best.py` 默认读取保留结果；新结果必须通过 `--predictions` 指定与 GT 文件名键一致的目录，不会模糊匹配或猜测 GT。输出目录必须是尚不存在的目录。阶段级评价保留 `scripts/evaluate_stages.py` 与 GT schema，但需要完整诊断输出及人工轨迹 GT。`evaluate_e1_5.py` 保留人工 GT 状态/映射与评价工具，其历史四方案运行入口依赖已删除的旧实验，不作为当前运行入口。

## 保留范围

保留原始 `数据/` 的全部内容及人工标注、核心代码、必要评价工具、回归测试、依赖说明和一套最佳配置/结果。清理旧实验配置、V3 Shadow 模块、重复源码/缓存、旧输出、实验驱动脚本和过时报告。不创建备份；Git 仓库及其正常历史保留。

2026-10-09 清理验证：215 项回归测试通过；200g4-1、7号纸3-2、淋膜纸2 的计数与完整轨迹与保留结果一致。235 个数据/标注文件的哈希保持不变。约释放 7.68 GB，不创建备份。验证记录见 `paper_counter/output/best/cleanup_verification.json`。测试重新生成的 4 个小型缓存目录因自动审批拒绝删除而保留（工具仅返回 blocked by policy）。
