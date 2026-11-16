# 材料策略与配方：迁移后的使用方式

2026-11-16 第一轮迁移。检测数值逻辑保持原样，调度、参数组织、材料判定与运行记录已经分离。

## 调用入口

```python
from paper_counter import PaperDetector

detector = PaperDetector(recipe="corrugated_side_v1")
result = detector.detect("图片.bmp", output_dir="output/current")
print(result.count, result.count_unit, result.pipeline_id)
for instance in result.instances:
    print(instance.instance_id, instance.geometry, instance.evidence)
```

```powershell
conda activate pytorch
python main.py --image "图片.bmp" --recipe corrugated_side_v1 --output output/current
python scripts/batch_detect.py --input "图片目录" --recipe corrugated_side_v1
python infer.py --recipe corrugated_side_v1
```

`infer.py` 继续使用文件头部的输入、输出路径。`--recipe` 选择新配方；未指定时继续使用原 `CONFIG_PATH`。
单图和批量入口的 `--config` 与 `--recipe` 互斥。旧平铺 YAML、旧预设和 `PaperCounter(CounterConfig(...))` 继续有效。

| 配方 | 策略 | 方法族 | 计数单位 |
| --- | --- | --- | --- |
| `regions_generic_v1` | `GenericRegionsPipeline` | regions | optical_unit，明确可见的光学结构 |
| `corrugated_side_v1` | `CorrugatedBodyPipeline` | regions | board，候选整块瓦楞纸身 |
| `tracks_existing_v1` | `ExistingTracksPipeline` | tracks | paper_edge，原有计数边界 |

单位是算法声明的观测/计数语义，不代表物理身份已获真值确认。历史置信分数继续作为一致性分数，不能跨策略按概率比较。
轨迹增强版本仍可使用 `preset="tracks_preprocessed"`、`preset="paper_layers_v4"` 等旧别名。

## 参数与资源

规范配方放在包内 `paper_counter/recipes/materials/`，成像记录放在 `paper_counter/recipes/capture/`，安装包也包含这些文件。
可以传入配方名称、独立 YAML 路径或 `Recipe` 对象。

```yaml
schema_version: 1
recipe_id: my_corrugated_v1
material_family: corrugated
count_unit: board
pipeline: corrugated_body_v1
capture_profile: unspecified_v1
parameters:
  preparation:
    automatic_geometry: true
    roi_policy: bright_core_with_dim_edges
    target_spacing_px: 14.0
    sampling_width_px: 225
  observation:
    illumination_sigma: 15.0
    epsilon: 1.0e-6
    max_period_px: 50
runtime:
  export: compact
```

通用框检测另有自己的 `structure`、`signal_mode` 参数；瓦楞纸参数不接收这些通用结构开关或轨迹开关。
轨迹参数分为 `profile`、`preparation`、受限的 `overrides`，数值验证暂复用旧轨迹配置实现。
`runtime` 管理显示尺寸、调试块大小及导出模式，不参与纸身身份判定。
未知键、材料/计数单位与策略不匹配、非法参数、重复选择入口都会明确报错。

成像配置记录 `capture_id`、`camera`、`lighting`、`pixel_size_mm` 和可序列化元数据。
未标定的相机信息保持为空。第一轮迁移仅分离和记录这些信息，保持原有图像判断行为；没有新增自动曝光控制或成像拒绝阈值。

旧配置经适配后保留在展开配方的 `compatibility.config` 中，避免丢失历史非默认参数。
新规范配方不需要该区块。展开配方可以重新加载；参数区块与兼容区块冲突会拒绝加载。

## 策略边界

- `engine.py`：输入隔离、策略调度、结果基础契约和配方溯源。
- `registry.py`：显式注册策略、参数类型、适用结构族、计数单位。
- `pipelines/*/pipeline.py`：各自决定图像准备、材料分段和验证政策。
- `pipelines/*/parameters.py`：策略自己的参数类型。
- `shared/region_stages.py`：复用当前区域算法的光学证据、候选及结果组装阶段；分段和验证由策略回调提供。
- `shared/preparation.py`：组合几何工具，ROI 定位器由调用策略明确选择。
- `region_counter.py`、`counter.py`：原调用方式的兼容入口。

瓦楞数值实现仍在 `corrugated_regions.py`，通用区域序列实现仍在 `region_sequence.py`。
共享区域阶段中的旧光学候选适配暂时保留，以保证迁移前后结果一致；候选不作为最终纸张身份。
新的材料策略可以直接组合其他算子，无需使用这套区域阶段，也不需要在引擎中添加材料判断。

新增策略需实现 `run(InputFrame, Recipe) -> PipelineExecution`，提供独立参数解析类型，并注册 `PipelineSpec`。
在第一轮兼容输出协议中，参数类型还需提供 `to_config(runtime)` 供旧绘图/导出适配器使用，
返回的 `PaperCountResult` 必须保留状态、坐标和计数所需数据；旧数值配置类尚未从底层算法中删除。

## 输出与坐标

原有检测图、框/轨迹坐标和诊断文件继续输出，并新增：

- `resolved_recipe.yaml`：完整配方、成像记录及必要的历史兼容参数；
- `instances.json`：统一计数实例、证据来源、原图几何和计数单位；
- `debug_info.pipeline`：策略与配方标识。

API 的 `instances` 使用原图坐标。原 `detection_x/y`、`detection_xyxy` 等字段保持原含义，不在同名字段下换坐标系。
轨迹只输出实际观测点，不补造第二条边或纸身矩形；缺少定位信息的历史计数显式标为 `unlocalized_count`。
`PreparedFrame.to_input(..., space="aligned")` 支持局部切线映射与几何仿射的组合，普通 prepared 坐标也可单独映射。

批量输出在 `audit/recipe.yaml` 中冻结本轮配方，在 `audit/source/` 中冻结源码。
每个子进程重放同一份展开配方，验证记录包含 `pipeline` 和 `recipe_id`。

## 验证范围

架构迁移核对旧 API 字段、数量、几何、关键证据和检测图像素/字节，新增配方与实例输出单独验证。
这些检查只证明行为等价，不代表算法准确率提高；没有真值时不将既有计数视为正确标注。

本轮基线及验证记录放在 `output/migration_material_pipelines_20261116/`。
已完成验证：完整测试 435 项通过；72 项真实图片检测与迁移前一致，其中包括全部 60 张瓦楞纸，检测图 PNG 字节也一致。
安装包在独立目录中验证了三套配方及成像资源加载。两条历史 Matplotlib 图例警告仍保留，不影响测试通过。
详细记录见该目录的 `PARITY.md`、`PACKAGE_VALIDATION.json` 和 `tests_final.log`；迁移后的瓦楞纸结果图在 `validated_corrugated/images/`。
实验输出继续保留，算法最终定版后再清理，并保留报告引用的基线和最终结果。
