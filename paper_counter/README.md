# 纸张计数：regions 与 tracks

项目现在只提供两套算法。`legacy`、`narrow` 已移除，传入这些名称会明确报错。
历史实验脚本、旧说明和重复源码已备份到项目外；原始数据不修改，历史检测结果保留在 `output/`。

| 算法 | 配置 | 用途与输出 |
| --- | --- | --- |
| `regions` | `configs/regions.yaml` | 旋正、缩放、约225像素窄区、3条带灰度观测、纸身分段，输出矩形框 |
| `tracks` | `configs/default.yaml` | 多条带梯度观测、轨迹关联、恢复与验证，输出轨迹 |
| `tracks_preprocessed` | `configs/tracks_preprocessed.yaml` | 在 tracks 前端增加横向抑噪、暗缝增强和终端近重复清理；可选自动旋正 |

命令行默认仍为 **tracks**，没有借清理改变默认检测方案。
`paper_layers_v4` 及压实/复合层纹是 tracks 的配置与增强模块，不是额外算法。
`tracks_preprocessed` 是可独立回归的 tracks 变体；它保留原图做 RAW 纸身证据，只把增强图用于梯度候选。

## 单张图片

简易推理入口为 `infer.py`。先修改文件开头的 `INPUT_PATH`（单张图片或目录）、
`OUTPUT_PATH`、`CONFIG_PATH` 和 `ROTATE_CCW`，然后运行：

```powershell
conda activate pytorch
python infer.py
python infer.py --all
```

默认在输出目录的 `compact/` 下保存检测图、坐标和基本数据；`--all` 在 `all/` 下保存全部中间过程图和诊断数据。
两种模式均有 `summary.json`，包含计数、状态、提示及耗时。目录输入会递归处理图片，逐图保留相对路径与扩展名；
同一模式再次运行会覆盖同名结果，其他历史文件保留。默认使用 tracks 配置；需要框检测时将 `CONFIG_PATH` 改为 `configs/regions.yaml`。

本机默认使用 `pytorch` 环境。先执行 `conda activate pytorch`。

```powershell
python main.py --image "F:\纸张\数据\新测试\1.bmp" --config configs/regions.yaml --output output/regions_single
python main.py --image "图片路径" --config configs/default.yaml --rotate-ccw --output output/tracks_single
```

regions 配置启用自动方向判断；tracks 的输入需要纸层沿 y 方向排列，横向拍摄时使用 `--rotate-ccw`。
精简导出包含结果图和坐标；加 `--full-debug` 可导出完整中间信号。

## 文件夹与多组数据

统一使用一个批量程序：

```powershell
python scripts/batch_detect.py --input "F:\纸张\数据\新测试" --config configs/regions.yaml
python scripts/batch_detect.py --datasets "测试集/260921-覆膜铜板纸3" "新测试" --config configs/regions.yaml --workers 2
```

`--datasets` 的路径相对于项目旁的 `数据/`。任意绝对目录用 `--input`。
每轮生成独立的 `output/batch_算法_时间/`：

- `images/`：最终结果图。
- `audit/`：冻结源码、展开后的配置、逐图日志和坐标、验证结果。
- `REPORT.md`：逐图计数和运行状态。

若算法输出了重叠框，批处理会记录问题并原样导出，不会偷偷改框或把导出成功当成检测正确。
旧的 `scripts/run_three_strip_batch.py` 仅保留一个很短的 regions 兼容入口，共用上述程序。

## Python 调用

```python
from paper_counter import PaperDetector

regions = PaperDetector(preset="regions")
result = regions.detect("图片路径", output_dir="output/api_regions")
print(result.count, result.boxes)

tracks = PaperDetector(preset="tracks")
result = tracks.detect("图片路径", rotate_ccw=True)

horizontal_tracks = PaperDetector(preset="tracks_preprocessed")
result = horizontal_tracks.detect("图片路径", rotate_ccw=True)
```

API 的旧预设名 `e3_3`、`paper_layers_v4` 仍可使用；原来的 API 默认值保持不变。
详见 [API说明](API_README.md)。

## 目录与检查

```text
main.py          单张图片入口
paper_counter/   两套算法及共用模块
configs/         两套入口配置、tracks增强配置
scripts/         统一批处理、打包、计数标注检查
tests/           保留算法的回归测试
docs/            算法结构、坐标与验证说明
annotations/     人工标注及其版本记录
output/          历史结果和新结果（不参与算法源码）
```

```powershell
python -m pytest tests -q
```

结构说明见 [算法说明](docs/ALGORITHMS.md)，坐标说明见 [坐标说明](docs/COORDINATES.md)，
清理前已有的问题和本轮核对见 [验证说明](docs/VALIDATION.md)。
