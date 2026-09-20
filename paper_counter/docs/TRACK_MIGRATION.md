# Multi-strip-first 轨迹计数版本

本次目标是改进纸边轻微弯曲、局部低对比度、部分横向缺失和高度方向间距变化的处理能力。没有设置期望张数，没有按纸张序号或固定 Y 删除候选，没有降低原来的全局 prominence 来追求召回。

## 文件与接口

| 文件 | 作用 |
|---|---|
| `paper_counter/config.py`, `configs/default.yaml` | 新默认负梯度、12 strips、tracks；新增配置、类型/范围验证、注释 |
| `paper_counter/models.py`, `paper_counter/__init__.py` | 新增并导出 `StripPeakCandidate`、`PaperEdgeTrack`；结果增加轨迹列表和浮点 `edge_positions` |
| `paper_counter/local_detection.py` | `local_statistics()`、`detect_strip_peaks()`；逐 strip 精确滑窗 median/MAD 和 strong/weak 分级 |
| `paper_counter/tracking.py` | `associate_strip_peaks()` 二阶 DP、`make_track()`、`validate_tracks()`、`deduplicate_tracks()` |
| `paper_counter/local_period.py` | `LocalPeriod.at()`、`estimate_local_period()`、`missing_edge_hypotheses()` |
| `paper_counter/track_recovery.py` | `recover_weak_tracks()`、`positive_aux_score()`；基于曲线预测的横向弱证据恢复 |
| `paper_counter/active_roi.py` | `detect_active_stack_roi()`；原始梯度能量定位持续纹理区域 |
| `paper_counter/track_counter.py`, `paper_counter/counter.py` | 新流程编排、默认入口及显式旧算法选项 |
| `paper_counter/legacy_counter.py` | 修改前流程副本，用于可复现对照 |
| `paper_counter/confidence.py` | 新增 `compute_track_confidence()`；原函数保留 |
| `paper_counter/track_visualization.py`, `paper_counter/visualization.py` | 候选/轨迹/局部周期图、实际轨迹 polyline、完整数值导出 |
| `main.py` | 新摘要字段与逐条轨迹信息 |
| `tests/test_tracks.py` | 新结构和集成测试；原 `tests/test_counter.py` 未删除或改写 |
| `scripts/stage1_check.py`, `scripts/stage2_check.py` | 分阶段检查入口，输出目录带时间戳 |
| `scripts/compare_tracks.py` | 旧流程与新流程真实图对照，保存独立配置、日志、JSON 和图 |

## 算法变化

旧流程先融合 strip，再检测峰。各 strip 的真实边缘相差几像素时，median 会把同一纸边的窄峰摊宽、压低，甚至消除；全局阈值又容易受大块空白和上下对比度变化影响。

新流程先保留每个 strip 中实际观察到的局部峰。二阶 DP 的状态保留前两节点，代价包含每 strip 位移、斜率变化和跳过 strip 数量，因此允许平滑倾斜、弯曲及最多 2 个连续 strip 缺失。每条路径必须有 strong 节点，且普通轨迹默认至少 2 个 strong、3 个实际观察节点。弱节点可以补充路径，不能单独形成普通计数。

一条候选必须通过 coverage、continuity、smoothness 和 gradient 检查，再通过综合评分。Coverage 使用非平坦有效 strip 数作分母；continuity 使用总 strip 数作分母。缺失 strip 不插值成支持节点。DP 路径按得分进行节点互斥选择，不允许一个候选峰支持两条最终路径；这是二阶最优路径加贪心的多路径选择，不声称求得所有轨迹的联合全局最优解。

轨迹 center_y 为支持节点 Y 的中位数，可为半像素；`edge_positions` 保留浮点中心，兼容字段 `peak_positions` 使用四舍五入整数。所有公开坐标均在原图中，ROI 下边界不包含。

## 局部阈值与相对梯度评分

窗口默认是 `8 × initial_period`，取奇数长度。对每个窗口精确计算 median 和 MAD：

```text
effective_MAD(y) = max(local_MAD(y), global_robust_scale × mad_floor_ratio, epsilon)
strong_threshold(y) = 2.0 × effective_MAD(y)
weak_threshold(y)   = 0.8 × effective_MAD(y)
```

global_robust_scale 在全局 MAD 为零时沿用非零偏差中位数备用尺度。阈值只用于各 strip 的候选分级，没有全图唯一阈值。滑窗采用 float32 视图和分块向量化计算，不对像素逐点执行 Python 循环。

为避免“横向一致但极弱”的纹理靠几何评分通过，还对候选记录局部候选 prominence 的 75 分位参考强度。轨迹梯度分数取各 strip 的 `prominence / max(局部参考强度, strong_threshold)` 的中位数，再用 `r/(1+r)` 映射到 [0,1]。普通轨迹最低 0.20，恢复轨迹最低 0.10。该参考在高度方向随局部对比度变化，不使用人为固定灰度或固定全图峰值。MAD strong/weak 公式本身保持不变。

## 局部周期及恢复

全局 ACF 提供初始尺度；若融合曲线没有可靠周期，则允许多数 strip 的独立 ACF 周期一致时提供实测尺度，并在 `initial_period_source` 中记录。搜索始终遵守配置上下限，不因为找不到周期自动扩大。

高质量轨迹的相邻 gap 经范围筛选，排除明显小间距和约 2/3 倍周期的缺边 gap；用附近 7 个 gap 的 median/MAD 估计局部周期，再用 5 点 median filter 平滑。局部周期只指导去重和缺边搜索，不能直接生成纸张。

某个 gap 接近整数倍局部周期时，在邻接两条曲线之间插值生成每个 strip 的搜索位置。这些只是搜索假设；恢复必须在带内找到真实峰、重新关联为轨迹并满足支持率、连续性、平滑度、梯度及得分门槛。单 strip 噪声和只有 positive 证据均不能恢复。

Positive 仅在 negative 主分支的恢复阶段提供最多 0.05 的分数加成，不能绕过任何结构门槛。Recovered 也可能由原本 strong、但普通轨迹组合未过验证的节点组成；因此 `weak_support_count=0` 并不表示凭空补边。

重复轨迹必须同时满足中心距离和横向观测位置重叠条件。最终仍有间距异常时，写入 `spacing_warnings` 供人工核查；不按周期直接删除真实观测到的结构。

## 有效 ROI 与端部

先在手动 ROI 内用原始有符号梯度的幅度作为纹理能量（幅度只用于 ROI，绝不作为主计数响应），平滑后选取主要持续纹理区域，填合短内部间隙并向外留余量。首次 ROI 尚无实测周期时，使用配置周期上下限的中点作为平滑和 margin 的启动尺度，记录在诊断中；该尺度不作为实测周期输出。

Active ROI 不强制删除第一、最后纸边。局部检测允许端点峰以局部 baseline 作为外侧参考，从而保留部分可见边缘。整图无纹理时返回证据不足，并保存完整空结果调试图。

## 调试输出与 CLI

旧文件全部保留；新图为 `strip_peak_candidates.png`、`edge_tracks.png`、`local_period.png`。Overlay 画观测 strip 节点连接的折线：绿=直接检测，黄=恢复，红=拒绝。连接跳过的 strip 仅用于显示，JSON 的支持数只统计实际节点。

`profile.png` 中标记是轨迹中心在全局曲线的采样位置，不一定是全局曲线极大值。图标题明确说明使用局部阈值。`strip_peak_candidates.png` 的虚线显示 prominence 阈值，而非 height 阈值。

`profiles.npz` 额外保存 local baseline、MAD、effective MAD、strong/weak threshold 和 active ROI 能量；`original_y` 是实际计数 ROI 行号，`active_energy_original_y` 是 ROI 探测阶段对应原图行号。

JSON 包含所有 strip 候选、轨迹完整节点、拒绝原因、初始周期来源、局部周期样本、缺边假设、每次恢复结果、异常间距和置信度分项。CLI 每条最终轨迹输出中心、支持数、连续性、Y MAD、梯度/平滑/总分和 source。

`raw_track_count` 包含普通关联及恢复阶段生成的所有候选轨迹，等于最终轨迹数加拒绝轨迹数；`raw_seed_track_count`、`raw_recovery_track_count` 分别给出两个阶段的数量。拒绝轨迹多数是单个或少数 strip 的片段，并不代表这么多张真实纸被删除。

Confidence 是轨迹质量与一致性评分，由轨迹平均分、支持率、连续性、平滑度、局部间距一致性组成，按恢复比例、异常 gap 和小样本数量折减。新旧评分公式不同，不能直接把分数增减当作准确率增减。

## 开发和验证

按要求分阶段执行：阶段 1 保持旧 Count=88，只新增局部候选图；阶段 2/3 在保留旧入口的同时新增轨迹和结构评分；阶段 4 切换计数来源；阶段 5–8 接入局部周期、缺边假设、弱恢复和 positive 辅助；阶段 9 加入 active ROI；最后完成置信度、输出和测试。阶段目录保留。

原有 14 项测试保留；新增 19 项，包含指定的 12 类场景，并扩展单独 weak/positive 禁止计数、相对微弱纹理拒绝、配置类型验证、8/12/16 strips、整图弯曲/ROI 坐标和融合失效时的独立 strip 周期一致性。运行：

```powershell
& 'D:\Anc\anaconda3\envs\pytorch\python.exe' -X utf8=0 -m unittest discover -s tests -v
```

真实图对照入口：

```powershell
& 'D:\Anc\anaconda3\envs\pytorch\python.exe' -X utf8=0 scripts/compare_tracks.py
```

每次创建新目录。旧对照固定使用原先 auto + 8 strips；新算法使用统一的默认 negative + 12 strips，4.bmp 两分支都使用此前确认的 max_period=80。3.png 旧 auto 对照与之前手动 negative 的运行不是同一配置，不能混淆。

## 已知限制

最终验证：33 项测试全部通过（原有 14 项 + 新增 19 项）。真实图最终输出目录为 `output_tracks_20260914_100847_868173`，所有旧输出保留。

| 图像 | 旧算法对照 | 新 Count | detected | recovered | rejected | 初始 / 局部周期中位数 |
|---|---:|---:|---:|---:|---:|---|
| 2.bmp | 88 | 89 | 88 | 1 | 380 | 26 / 25.5 px |
| 4.bmp | 40 | 40 | 40 | 0 | 563 | 52 / 52 px |
| 3.png | 5（旧 auto） | 5 | 5 | 0 | 33 | 27 / 27 px |

2.bmp 自动有效 ROI 为原图 Y=1280～3710，confidence=0.6852。恢复轨迹中心 Y=1530，8/12 strips 支持；其他两次预测搜索没有形成合格轨迹，未计数。33 项测试及真实图对照均没有用真实图期望数量作断言；对照脚本额外验证 raw tracks = final tracks + rejected tracks 的记账关系。

4.bmp 原先 Y≈1841 的微弱误峰不再保留，但 Y≈1986 附近仍存在靠近上一轨迹的观测纹理；算法记录间距异常并保留供核对，没有按“应为 39”硬删。各图片新旧计数不能替代人工逐边标注。

- 无人工逐边标注，不能由 Count 增减证明召回率或准确率提高。
- 与真实纸边同样平滑且跨多个 strip 的周期内纹理仍可能误计；例如 4.bmp 底部另有近邻候选，需要核对。
- 当多数 strip 被遮挡、强反光破坏或曲率/相邻 strip 位移超限时仍可能漏检。
- 弱边远低于局部参考、或突然厚度变化超过局部周期筛选范围，可能不被恢复。
- 一个 ROI 存在两段分离纸堆时默认仅选主要持续区域；可关闭 auto_stack_roi 或指定 ROI 分次处理。
- DP 多路径冲突采用贪心选择；密集交叉纹理可能抢占节点，不能保证联合最优。
- 活动区域尺寸、初始周期搜索范围和每纸层单方向边缘假设仍需与成像尺度匹配。
