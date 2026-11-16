# 保留的两套算法

`CounterConfig.algorithm` 保留 `regions` 与 `tracks` 两个方法族的兼容选择。
实际调度由 `DetectionEngine` 和策略注册表负责；`PaperCounter.process()` 是旧配置适配入口。
新配方与三个策略的对应关系见 [流水线迁移说明](PIPELINES.md)。
没有隐藏的 legacy 回退路径。结果诊断中的历史标签 `narrow_regions`、`multi_strip_tracks`
为了兼容坐标导出和既有结果格式而保留，不代表额外的算法入口。

## regions

配置为 `configs/regions.yaml`，打包时同时提供 `presets/regions.yaml`。

1. `paper_geometry.py` 定位主体、判断方向、旋正并决定缩放，选择约225像素宽的窄区。
2. `geometry_alignment.py` 保存窄区裁剪和局部切线对齐工具。这些函数从旧 narrow 文件原样迁出。
3. `shared/region_stages.py`、`region_joint.py` 提取三个条带的灰度观测，形成光学候选并融合；`region_counter.py` 保留兼容导入。
4. `region_sequence.py`、`region_thickness.py` 处理重复尺度、分段与纸身厚度证据。
5. `region_adaptive.py`、`region_postvalidation.py`、`box_content_verification.py` 保留自适应与验证实现。
   后验证现在只把厚度异常标为复核；固定分数删除已取消，只有重叠这种硬物理约束允许自动裁切。
6. 每张图的 `debug_info.stage_counts` 记录候选、初始单元、厚度重分割、后验证和最终框数量，
   可区分“没有提取到”和“后面被合并/删除”。
7. `region_output.py` 输出矩形框、原图坐标、多边形映射、信号诊断。

共享字段 `narrow_width` 的名称为兼容配置保留，表示 regions 检测窄区的目标宽度；
独立 narrow 算法的 `narrow_feature`、`narrow_signal_mode` 已移除。

regions 通过裁窄和方向对齐减轻全宽弯曲关联困难，并在三条带候选融合后分别评估光学周期和物理纸厚。
归一化灰度和原始局部对比都会提出候选；两倍周期只有在内部暗槽和纸身支持足够时才允许参与物理重分割。
它仍不是逐像素纸身分割。
灰度亮带、纸身内部暗纹和纸间间隔的身份判断仍需证据；不能把计数相符当成逐框位置都正确。

### 瓦楞纸预设

`configs/corrugated.yaml` 同属 regions，使用实测横向缝隙，不允许仅凭厚度创建切点。
详见 [瓦楞纸说明](CORRUGATED.md)。

## tracks

默认配置 `configs/default.yaml` 通过包内 `e3_3` 预设展开。
`configs/paper_layers_v4.yaml` 与 `configs/tracks_variants/` 是同一 tracks 算法的可选增强配置。

1. `track_counter.py` 编排原始灰度、梯度、条带候选与后续处理。
2. `local_detection.py`、`profiles.py` 生成和检测独立条带观测。
3. `tracking.py`、`ordered_association.py` 等建立轨迹并判断几何关系。
4. `strip_sequence.py`、`vertical_sequence.py`、`track_identity.py` 等处理保序、恢复和验证。
5. `compressed_layers.py`、`composite_layers.py` 是可开关的弱层纹/复合结构增强。
6. `visualization.py`、`compact_output.py`、`detection_coordinates.py` 输出轨迹及坐标。

`adaptive_structure.py`、`enhanced_edge_detection.py` 也属于 tracks 增强模块，清理时没有删除或改写它们。

### tracks 的水平纸堆变体

`configs/tracks_preprocessed.yaml`（API 名称 `tracks_preprocessed`）针对纸边接近水平、层间暗缝清楚的拍摄条件。默认保留调用方的图像方向（横向拍摄仍使用 `rotate_ccw=True`），随后 `preprocess_tracks_signal()` 沿 x 方向抑制纹理，并用低增益宽水平黑帽增强长暗缝；需要自动小角度旋正时可把该配置的 `auto_geometry_enabled` 打开，tracks 会保留完整图像范围并记录仿射映射。RAW 掩膜、纸身证据和身份终判仍读取原始灰度，因此增强结果只能提出候选，不能单独制造计数。

终端 `track_postprocessing.py` 只清理共享多个真实条带观测、间距极小且质量明显较弱的近重复轨迹；不按周期插入轨迹，也不删除没有共享观测的紧邻物理层。每一步的参数、几何仿射、黑帽核和清理决定都写入 `debug_info`，便于与原 `tracks` 基线逐图比较。

## 共用内容

`config.py` 管理配置；`models.py` 管理结果数据结构；`preprocessing.py` 管理读取和基础图像处理；
`coordinates.py` 管理显示与输入坐标变换；`api.py` 提供可安装接口；`ground_truth.py` 与
`stage_evaluation.py` 保留标注和离线评价工具。

原先的 legacy 信号融合/补峰流程、独立 narrow 检测器及其专用实验信号模块不再提供。
