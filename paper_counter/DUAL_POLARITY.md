# 正负梯度对照

单张图增加 `--dual-polarity`：负梯度结果保存在指定输出目录，独立正梯度结果保存在其 `positive` 子目录。最终计数继续使用负梯度，`debug_info.json` 中的 `dual_polarity` 提供两路对照。

```powershell
python main.py --image IMAGE --config configs/final_verifier_active.yaml --rotate-ccw --dual-polarity --output output/unique_run
python scripts/audit_dual_polarity.py --input DATA_DIRECTORY --rotate-ccw --workers 4
```

批量对照脚本将报告写入 `output/dual_时间戳`，保存每张图的正梯度轨迹及分歧区间，不生成整套调试图。需要看两路标注图时使用单图命令。

- 正梯度独立进行候选检测、关联、序列选择和弱轨迹恢复。
- 正梯度禁用负梯度专用末端身份校验，允许依据正梯度证据恢复。
- 报告同时保留负梯度最终计数、负梯度校验前计数和正梯度计数。
- 负梯度保留原有正梯度恢复辅助，两路因此并非完全对称实验。
- 分歧区间统计：相邻负梯度轨迹中心之间有零条、一条还是多条正梯度轨迹。边缘有相位差，因此不把两路轨迹直接合并。
- 区间坐标是逆时针旋转后的检测坐标，不能直接当作带标题栏的PNG坐标。
- 一条正梯度/区间的一致率不是准确率；零条或多条只能作为审核线索。曲率、边界相位、纸内纹理都可能造成分歧。
- 共用ROI使两路可能同时丢失被截掉的纸；计数相同也可能发生漏检和误检相互抵消。不能自动取平均、最大值或选较高Confidence。

此阶段增加诊断证据，尚未以双路融合替换最终计数。局部人工标注用于后续评估配对、补检和去重；没有标注的位置不能默认正确。
