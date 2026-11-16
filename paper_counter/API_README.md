# 纸张检测 Python API

把 `paper_counter_api_0.1.0.zip` 整包发给使用者即可，包内有 wheel、源码、示例和本说明。无需原数据集、人工标注或历史结果，无需训练或下载模型。本接口是本地 Python 函数，不需要启动服务。

## 安装

Python 3.9 或更高版本。在解压目录执行：

```powershell
python -m pip install .\paper_layer_counter-0.1.0-py3-none-any.whl
```

pip 会安装 NumPy、OpenCV、SciPy、Matplotlib、PyYAML，需要能访问依赖源。已有这些依赖时可离线安装 wheel：`python -m pip install --no-deps .\paper_layer_counter-0.1.0-py3-none-any.whl`。也可从源码目录执行 `python -m pip install .`。安装后可以在任意目录导入。

本项目电脑使用 `conda activate pytorch`。若该环境启动时出现 `.pth` 编码错误，执行 Python 命令时增加 `-X utf8=0`，例如 `python -X utf8=0 example.py --image "纸张.bmp" --rotate-ccw`。

## 一次调用

```python
from paper_counter import detect_papers

result = detect_papers("纸张.bmp", rotate_ccw=True)
print(result.count)
print(result.status, result.warnings)
data = result.to_dict()  # 可直接 json.dumps(data, ensure_ascii=False)
```

图像须使纸层沿 y 轴上下排列。当前测试集和验证集的横向拍摄图需要 `rotate_ccw=True`，即逆时针旋转 90°。已经按上下排列的图像使用默认值 `False`；接口不会自动判断方向，也不会缩放检测图像。

## 批量与内存图像

```python
import cv2
from paper_counter import PaperDetector

detector = PaperDetector()  # 批量时复用配置
image = cv2.imread("sample.bmp")  # OpenCV BGR 顺序
result = detector.detect(image)
result2 = detector.detect("另一张.bmp", rotate_ccw=True, output_dir="result/sample2")
```

支持 `str` / `pathlib.Path` 路径（包括中文路径），或非空 NumPy 数组。数组支持 uint8 / uint16 灰度、BGR 或完全不透明 BGRA。Pillow/RGB 图像须先转 BGR；透明图须先合成背景。不会修改传入数组。调用默认不写文件；`output_dir` 指定后生成检测图、坐标 CSV/JSON 和精简诊断，同名文件会覆盖。每张图请使用单独输出目录。

## 两套算法

清理后只保留 `tracks` 与 `regions`。可直接选择：

```python
tracks = PaperDetector(preset="tracks")
corrugated = PaperDetector(recipe="corrugated_side_v1")  # 瓦楞纸框检测策略
regions = PaperDetector(preset="regions")
result = regions.detect("图片.bmp", output_dir="output/regions")
print(result.count, result.boxes)

preprocessed = PaperDetector(preset="tracks_preprocessed")
result = preprocessed.detect("图片.bmp", rotate_ccw=True)
```

`tracks` 是原默认命令行配置的别名；`tracks_preprocessed` 在 tracks 前端增加横向抑噪和暗缝增强，并保留原图作为 RAW 证据；自定义打开 `auto_geometry_enabled` 后可增加小角度自动旋正；`regions` 使用包内随附的225像素三条带矩形框配置，启用自动几何预处理。
上文“不自动缩放或判断方向”适用于原 tracks 默认预设；regions 的自动几何行为由其配置决定。
`e3_3`、`paper_layers_v4` 是 tracks 的配置版本，不是额外算法。原 API 默认预设没有改变。

## 配置版本

新 API 默认 `preset="paper_layers_v4"`，完整参数与本项目 `configs/paper_layers_v4.yaml` 一致，包含当前已修复的梯度、纸身证据与端部处理。现有命令行默认仍是 E3.3。需要复现它时：

```python
detector = PaperDetector(preset="e3_3")
# 或传入自己的完整配置；此时 preset 不参与配置合并
detector = PaperDetector(config="my_config.yaml")
```

`config` 也支持 `CounterConfig` 对象。注意 `CounterConfig()` 是底层基础参数，不等于上述两个发布预设。自定义配置调用时，结果的 `preset` 为 `custom`。内置 YAML 随安装包发布，不依赖启动目录。

## 返回值

`detect_papers()` 与 `PaperDetector.detect()` 都返回 `DetectionResult`：

| 字段 | 含义 |
|---|---|
| `count` | 最终纸张计数 |
| `status`, `warnings` | 原算法状态及复核提示；零计数不证明图中没有纸 |
| `confidence` | 未校准的一致性分数，不能当成正确概率 |
| `period` | 检测图中的估计层间距（像素），可能为 None |
| `tracks` | 最终计入轨迹，按检测图从上到下编号 |
| `original_shape`, `detection_shape` | 旋转前后数组尺寸，顺序为高、宽、可选通道 |
| `rotation_ccw_degrees` | 0 或 90 |
| `raw_result` | 完整底层结果，包含诊断和数组；不包含在 `to_dict()` 中 |

`tracks` 每项含 `number`、`track_id`、`center_detection_y`、`source`、`observations`。每个观测含零起始 `strip_index`、检测图坐标 `detection_x/y`、旋转前输入图坐标 `input_x/y`。坐标已包含 ROI 偏移；x 是聚合条带中点，不是逐像素边缘定位。中心 y 是轨迹汇总位置；仅 observations 是实测条带位置，不输出插值点。自定义 legacy 算法无轨迹时，`tracks` 为空，可查 `raw_result.peak_positions`。

图片不存在或不能读取会抛出 OSError/FileNotFoundError，解码失败、非法图像或配置会抛出 ValueError，传入类型错误会抛出 TypeError。调用方可捕获这些异常用于业务提示。

## 效果范围

封装复用原算法，不改变检测判断。最新 v4 在现有 83 张有人工真值图片中为 75 张计数完全正确；这些数据参与过开发，不是独立泛化保证。压实、弯曲纸边和部分灰色淋膜纸仍可能漏计或多计，应结合状态、警告和结果图复核。

## 修改源码后重新打包

在项目或交付包的 `source` 目录执行 `python scripts/build_api_release.py`（需安装 setuptools 和 wheel）。生成的 `dist/paper_counter_api_0.1.0.zip` 包含 wheel、源码、示例和 wheel 的 SHA256 校验值，不包含数据集和历史结果。本机使用 `conda run -n pytorch python -X utf8=0 scripts/build_api_release.py`。
