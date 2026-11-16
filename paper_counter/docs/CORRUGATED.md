# 瓦楞纸框检测

迁移后推荐 `PaperDetector(recipe="corrugated_side_v1")` 或命令行 `--recipe corrugated_side_v1`。
瓦楞纸策略独立管理准备、分段与验证，数值算法保持原行为；旧配置与预设仍兼容。见 [流水线说明](PIPELINES.md)。

瓦楞纸使用 `configs/corrugated.yaml` 或 `PaperDetector(preset="corrugated")`。
它属于 `regions`，不是第三套算法。`region_material: generic` 使用通用分段流程；
`region_material: corrugated` 使用瓦楞纸的实测横向缝隙分段。

```powershell
conda activate pytorch
python scripts/batch_detect.py --input "F:\纸张\数据\261112瓦楞纸" --config configs/corrugated.yaml
```

自动几何先通过亮度双群体提取纸堆主体，排除远离主体的孤立亮夹具；邻近较暗纸层必须有灰度边缘证据才能扩展 ROI。
窄区选择后，通过最终仿射矩阵直接采样原图，并保留到输入坐标的逆变换。

瓦楞分段保留每列的原始灰度。只有存在左右纸身支撑的实测灰度凹槽才能成为切分候选。
开放空腔较多时，横穿空腔的亮色瓦楞筋会否决内部切线；模糊、压实的接触缝允许使用较弱的实测灰度对比。
厚度先验只比较已经检测到的缝隙，无法生成等距切点，也没有强制最大框高。
没有新缝隙的宽纸身保留为一个待复核框，记录在 `unresolved_contact_regions`，可能仍包含多张相贴纸。

实测全暗间隔保留为空，不计作纸身。完整重复结构外侧、仅含单个亮带且远薄于主体的残片，
记录到 `excluded_partial_edges` 后不计入完整纸身。内部薄层不会仅凭厚度被删除。

`debug_info.region_detection.thickness_postprocess` 记录：

- `observed_seams`：横向灰度、列覆盖、空腔筋证据、背景间隔；
- `selected_seams` / `rejected_seams`：实际采用及舍弃的候选；
- `excluded_partial_edges` / `background_intervals_excluded`：外侧残片与空背景；
- `unresolved_contact_regions`：缺少进一步缝隙证据的宽纸身。

规则没有使用图片编号、人工张数或本批图片的预期计数。缺少逐层真值时，只能检查框位置、
背景误检和切分证据，不能把计数变化当作准确率提升。
