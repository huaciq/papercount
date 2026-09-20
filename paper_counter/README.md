# 离线纸张计数

当前版本为 **横向 multi-strip 轨迹提议 + 纵向序列优化**。默认 `algorithm: tracks`、`gradient_polarity: negative`、`num_strips: 12`；最终 Count 来自纵向 DP 选中的 `PaperEdgeTrack`。强轨迹门槛仍为 0.60，marginal 候选只在序列收益支持时计数。

本次目标函数、三类候选定义、候选优先恢复、自适应曲线窗口、文件清单及测试见 [纵向序列版本说明](docs/VERTICAL_SEQUENCE.md)。新增 `vertical_sequence.png` 和逐轨迹决策 JSON。`result_overlay.png` 中绿线为最终选中，黄线为序列恢复，红线为序列抑制；原图和历史输出保留。横向拍摄的 `数据/pictures` 图像须使用 `--rotate-ccw`。

本轮几何版本见 [Geometry-centric ordered-layer detection](docs/GEOMETRY_SEQUENCE.md)：RAW 灰度先生成 CORE/MARGIN/INVALID 三态 mask，默认关闭 n×period 解释，使用完整曲线 gap profile 与 non-crossing 约束；大间距只作异常提示，没有图像证据不会新增纸张。

当前主流程是 [Regular-first hybrid detector](docs/REGULAR_FIRST_HYBRID.md)：从 strong seed tracks 估计连续 `R(y)`，规律区增强 spacing/弱边搜索，异常区回到曲线几何和 gap profile。`regularity_map.png`、`hybrid_sequence.png`、`raw_stack_mask.png` 和 JSON 会记录局部权重、异常区、竞争纹理、恢复类型及未解决 gap。周期倍数只产生搜索提示，不能凭空增加计数；RAW INVALID 只来自边界连通的高置信黑背景，强轨迹可一次性扩展 mask。

新增局部 MAD strong/weak 候选、二阶动态规划关联、轨迹结构验证、局部周期、跨 strip 弱边恢复和有效纸堆 ROI。所有旧模块与测试保留，可通过 `algorithm: legacy` 使用旧流程。旧输出目录未覆盖。

完整修改清单、算法说明、限制和验证记录见 [轨迹版本说明](docs/TRACK_MIGRATION.md)。新增输出为 `strip_peak_candidates.png`、`edge_tracks.png`、`local_period.png`，原有输出继续保存。`result_overlay.png` 改为绘制实际轨迹折线；`profile.png` 标记的是轨迹中心，不应再把曲线上标记点当作全局峰检测结果。

当前 `prominence_k` 仅用于旧算法及初始方向比较；新主检测由 `strong_prominence_k` / `weak_prominence_k` 控制逐 strip 的局部阈值。`max_period` 默认仍为 50；`4.bmp` 应使用 80（沿用此前已确认的周期搜索范围）。

可通过 `--ground-truth ground_truth.json` 输出 count error、edge precision/recall/F1；没有真值时只报告检测变化，不把 Confidence 当作准确率。

运行入口和 Python 环境要求不变。以下旧样例数值和“算法与调试”部分描述首版 legacy 行为，当前版本以轨迹版本说明为准。

基于 OpenCV、NumPy、SciPy、Matplotlib 的单图计数程序，Python 3.9 及以上。仅使用传统图像处理和信号处理；检测、峰位置和周期均在原始分辨率计算。

## 运行

在 PowerShell 中：

```powershell
Set-Location 'F:\纸张\paper_counter'
conda activate pytorch
python main.py --image 'F:\纸张\数据\1.png' --config configs/default.yaml --output output
```

本机 `pytorch` 环境中有旧编码的 `.pth` 文件，若 Python 启动出现 `UnicodeDecodeError`，在当前终端先执行 `$env:PYTHONUTF8='0'`。也可直接运行已包含该兼容处理、且执行后恢复环境变量的一键脚本：

```powershell
& 'F:\纸张\paper_counter\run_sample.ps1'
```

缺少依赖时，在所选环境中运行 `python -m pip install -r requirements.txt`。程序本身不导入 PyTorch 或其他深度学习框架。

## 当前样例

`数据/1.png` 为 133 × 1219 的 uint8 BGRA 图，Alpha 全为 255。纹理集中在顶部，下方大部分像素恒定为 1。默认配置使用整图，未自动选择 ROI，也未针对目标张数调参。

初版默认配置运行结果：

- 自动选择正梯度，周期 28 px。
- 检出 6 条，补入 1 条，总计 7 条候选纸边。
- 原图 Y 坐标：3、19、47、79、106、134、149，其中 134 是补峰。
- 平均相邻间距 24.33 px，启发式 confidence 约 0.567。

这些是算法输出，尚未用人工标注核验真实纸张数。顶部截断边缘、纸层内纹理和黄色补峰需要核对。confidence 不代表“计数准确概率”；不能根据一次样例宣称工业精度。

## 配置和坐标

全部参数位于 `configs/default.yaml`，未知字段会报错，防止拼写错误悄悄失效。默认 `roi: null` 使用整图。手动 ROI 示例：

```yaml
roi: {x1: 0, x2: 133, y1: 0, y2: 170}
```

`x2`、`y2` 是不包含的右/下边界；越界会报错。`peak_positions` 和 JSON 内候选的 Y 均为原图坐标，内部 profile 的数组下标是 ROI 内行号；NPZ 的 `original_y` 提供对应关系。

主要参数：

| 参数 | 默认 | 含义 |
|---|---:|---|
| `illumination_sigma` | 15 | 光照背景 Gaussian sigma，原图像素 |
| `gradient_polarity` | auto | auto / positive / negative |
| `num_strips` | 8 | 横向等分的竖直条带数 |
| `horizontal_margin_ratio` | 0.05 | 左右各忽略比例 |
| `profile_smooth_sigma` | 1 | 一维平滑 sigma |
| `min_period`, `max_period` | 5, 50 | 自相关周期搜索范围，像素 |
| `min_acf_score` | 0.10 | 最低可靠自相关主峰强度 |
| `prominence_k` | 2 | 正常峰 prominence / profile MAD |
| `distance_ratio` | 0.55 | 正常候选最小间距 / 周期 |
| `vote_tolerance_ratio` | 0.20 | strip 投票容差 / 周期 |
| `min_vote_ratio` | 0.5 | 至少一半配置的 strip 支持 |
| `duplicate_ratio` | 0.50 | 去重距离 / 周期 |
| `max_peak_width_ratio` | 0.50 | 半 prominence 峰宽上限 / 周期 |
| `weak_prominence_ratio` | 0.5 | 补峰 prominence / 正常阈值 |
| `repair_tolerance_ratio` | 0.25 | 补峰搜索容差 / 周期 |
| `repair_min_vote_ratio` | 0.5 | 补峰 strip 支持下限 |
| `max_repair_gap_periods` | 4 | 最多检查跨多少个周期的间隙 |

额外阈值、置信度权重及显示大小也在 YAML。图像没有可靠周期时返回 `period=null`、`status=insufficient_evidence`，此时 `count=0` 表示没有通过验证的纸边，不能据此断言图中无纸。

## 算法与调试

按 ROI → 灰度 → 浮点除法光照归一化 → 有符号 Scharr Y → 正负响应分别投影 → strip median → 一维平滑 → 每 strip MAD 归一化 → 多 strip median 融合 → 自相关与间距规律性选方向 → 自适应检测 → strip 投票 → 去重 → 证据补峰 → confidence 执行。

完全平坦的 strip 不参与融合，但仍留在投票分母中，避免仅一个有效 strip 获得虚高支持率；少量异常 strip 由 median 抵抗。MAD 为零时使用非零绝对偏差的中位数作为备用尺度，并记录 `scale_method`。这适用于大量空白的稀疏图，避免用极小 epsilon 放大噪声。

自相关使用 FFT、减均值并除以零延迟能量，只在配置 lag 范围查找局部最大值。强度接近的谐波优先选择较短周期；若局部纹理本身存在强倍频，仍需通过调试曲线和标注验证周期范围。

补峰只检查相邻已通过验证的边缘之间、接近整数周期的间隙，必须同时有全局弱局部峰及多 strip 峰支持。找不到证据就记录 `no_image_evidence`，不按周期直接插值计数。较宽的光照归一化伪峰记录为 `peak_too_wide`。所有被投票拒绝或显式去重的正常候选都保留在 JSON。

`output` 包含：

| 文件 | 内容 |
|---|---|
| `result_overlay.png` | ROI 图；绿=检出、黄=补峰、红=拒绝；信息栏在图像外 |
| `profile.png` | 选定方向全局 profile 和各类候选 |
| `autocorrelation.png` | 双方向自相关、方向评分、选中周期 |
| `strip_profiles.png` | 两个方向所有 strip 的归一化曲线 |
| `gray.png`, `normalized.png` | 灰度、光照归一化预览 |
| `positive_gradient.png`, `negative_gradient.png` | 正负梯度分别显示 |
| `debug_info.json` | 配置、坐标、方向评分、候选 prominence/height/width/vote/source/reason、补峰尝试和置信度分项 |
| `profiles.npz` | 双方向原始/归一化 strip profile、融合 profile、自相关和原图行号 |
| `regularity_map.png` | 局部 spacing/MAD、ACF、track consistency 与连续 `R(y)` |
| `hybrid_sequence.png` | Regular/Mixed/Irregular 背景上的最终选择、恢复和抑制 |
| `raw_stack_mask.png` | RAW CORE/MARGIN/INVALID 许可及初始/修正边界 |
| `track_identity_profiles.png` | 最终轨迹的对齐 RAW/归一化/正负梯度剖面及本图稳健模板 |
| `render_metadata.json` | overlay 信息栏、缩放、ROI、旋转及坐标换算参数 |

显示预览可以缩放、归一化图可截断到 8 位显示；用于检测的数据从未缩放。gradient 响应按 strip 和行块计算，不同时保存两份全尺寸浮点响应；主要内存消耗仍随 ROI 像素数增长。PNG 是预览，NPZ 保留完整一维数值。

读取数值示例：

```python
import numpy as np
signals = np.load("output/profiles.npz")
y = signals["original_y"]
positive = signals["positive_global"]
```

## Python 接口

从项目目录执行：

```python
from paper_counter import CounterConfig, PaperCounter
from paper_counter.preprocessing import read_image
from paper_counter.visualization import save_debug

config = CounterConfig.from_yaml("configs/default.yaml")
result = PaperCounter(config).process(read_image(r"F:\纸张\数据\1.png"))
print(result.count, result.peak_positions, result.period, result.confidence)
save_debug(result, "output", config)
```

## 验证

```powershell
python -m unittest discover -s tests -v
```

当前共有 119 项回归测试，其中新增 20 项覆盖坐标映射、近似标注匹配、曲线对齐剖面、
稳健模板、保守终判、shadow/active 行为和终判后禁止恢复；合成图测试验证机制及回归，
不替代真实工业数据准确率评估。

## 最终轨迹边界身份核验

主检测、序列选择和全部恢复完成后，`Final Track Verifier` 沿每条弯曲轨迹采样固定长度的
RAW、归一化强度及正负梯度剖面，并与当前图像内高支持、规则区域的 provisional reference
轨迹建立的 median/MAD 模板比较。它同时记录纸体上方支持、背景连接、正梯度配对、暗区回弹、
条带一致性和短间距竞争。Regularity 只提供上下文，正梯度缺失和下方黑背景都不能单独删除轨迹。

`configs/default.yaml` 默认使用 shadow mode，只记录 `proposed_decision` 而不改变计数。
经过人工误检审计后，可用 `configs/final_verifier_active.yaml` 启用终判；终判位于最后一个恢复阶段之后，
被拒绝的轨迹不会再次被周期补回。`debug_info.json` 保存 `pre_verifier_sequence`、
`post_verifier_sequence`、完整身份特征、拒绝族和 before/after 计数。

人工 overlay 标注通过 `render_metadata.json` 映射到检测坐标。审计命令会生成坐标匹配、已知误检、
特征分布和可视化报告；标注仅用于验证，不参与生产决策：

```powershell
python scripts/audit_false_tracks.py --batch <output_batch> --annotations annotations/track_ground_truth.json
```
