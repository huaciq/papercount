# 纵向序列优化版本说明

本次在原有 horizontal multi-strip tracking 后增加候选池与纵向序列决策。滤波、梯度、局部峰提取、横向关联和 `min_track_score: 0.60` 保持原有算法。最终 Count 是序列选中的**已有图像轨迹数量**，周期假设本身没有计数资格。

## 修改文件与入口

| 文件 | 新增或修改内容 |
|---|---|
| `paper_counter/vertical_sequence.py` | 新增 `SequenceResult`、`optimize_vertical_sequence()`、`unary_score()`、`transition_score()`、`period_mad_at()`、`huber()`；有向无环图 DP 与逐轨迹决策解释。 |
| `paper_counter/sequence_recovery.py` | 新增 `spacing_anomalies()`、`adaptive_search_radius()`、`recovery_hypotheses()`、`recover_sequence_candidates()`；先复用候选，再搜索真实弱峰。 |
| `paper_counter/tracking.py` | 新增 `classify_tracks()`、`deduplicate_obvious_tracks()`；修正单轨迹拒绝原因。原横向 DP 和旧去重函数保留。 |
| `paper_counter/models.py` | `PaperEdgeTrack` 增加稳定 `track_id`、`candidate_class`、`classification_reason`、`sequence_decision`。 |
| `paper_counter/track_counter.py` | 串联候选池、序列、恢复、再次序列；Count 切换为最终 selected；扩充 JSON。 |
| `paper_counter/config.py`、`configs/default.yaml` | 增加有注释、有校验的序列与自适应恢复参数。 |
| `paper_counter/visualization.py`、`track_visualization.py` | 新增 `vertical_sequence.png`；最终选中绿线、恢复黄线、序列抑制红线。全部垃圾碎片仍保留在 JSON 与 `edge_tracks.png`。 |
| `paper_counter/confidence.py` | 序列恢复也纳入原有恢复比例惩罚；仍是启发式一致性分数。 |
| `main.py` | 输出三类候选数量、selected/rescued/suppressed、序列总分、残留异常、两种恢复数量；记录旋转及坐标映射。 |
| `scripts/batch_detect.py` | 批量结果记录旋转、序列恢复/抑制数量；修复未旋转时仍显示“已旋转”的说明。 |
| `scripts/sequence_checkpoints.py` | 对历史横向证据做分阶段分类和 sequence-only 验证。 |
| `scripts/validate_sequence.py` | 从原图执行 2.bmp、4.bmp、3.png；新建目录，记录目标轨迹的完整决策，校验轨迹数量和 ID。 |
| `scripts/audit_sequence_outputs.py` | 校验 19 张结果的数量、几何、目标函数、旋转和产物，生成真实图报告及源码 SHA-256 清单。 |
| `tests/test_vertical_sequence.py` | 20 个新增测试，覆盖要求的十类情况及目标函数、边界和二阶 DP。 |
| `tests/test_tracks.py` | 仅更新两条预期 reason 字符串；原测试场景全部保留。 |
| `README.md` | 更新当前算法说明与文档入口。 |

`vertical_sequence_enabled: false` 可运行此前的横向轨迹接受/恢复流程作对照；`algorithm: legacy` 仍保留首版全局峰流程。

## strong / marginal / garbage

**Strong**：通过此前的独立门槛，默认至少 3 条带、support ratio ≥ 0.60、continuity ≥ 0.30、smoothness ≥ 0.45、至少 2 个 strong strip peaks、gradient score ≥ 0.20、total score ≥ 0.60。Strong 表示证据充分的候选，不等于一定计数。

**Marginal**：未通过 strong，但 total score ≥ `track_proposal_floor`（默认 0.45），且至少 3 条带、support ratio ≥ 0.30、continuity ≥ 0.15；原有梯度和几何底线继续约束它。这里只放宽候选提议阶段的覆盖/连续性要求，不降低最终计数门槛为一个新的单轨迹阈值。

**Garbage**：低于提议底线、支持条带过少、梯度太弱或严重几何失败。它不进入序列，不能仅凭周期位置被复活。弱图像恢复路径另需通过原有 recovery 门槛（默认 support ≥ 0.60、continuity ≥ 0.30、gradient ≥ 0.10、total ≥ 0.55），再作为 marginal 进入最终序列竞争。

`classification_reason` 保存最初分类失败项；`reason` 保存最终决策原因。因此 total 0.59 不再被误报成 `weak_gradient`，而是 `low_total_score`。

## DP 的具体目标函数

将 usable 轨迹按 `center_y` 排序，选择子序列 S，最大化：

```text
J(S) = sum U(t_i) + sum P(t_i, t_j) + sum C(previous_edge, next_edge)

strong:   U(t) = w_u * clip(total_score, 0, 1) - suppress_margin
marginal: U(t) = min(w_u * clip(total_score, 0, 1) + rescue_bonus,
                    w_u * (min_track_score - epsilon)) - suppress_margin

d     = local_period at midpoint of the two tracks
MAD   = interpolated local gap MAD at that midpoint
sigma = max(1 pixel, sequence_spacing_scale_ratio * d, 1.4826 * MAD)

P(i,j) = - min over n=1..max_sequence_gap_periods {
               w_spacing * Huber((gap - n*d) / (sqrt(n)*sigma))
               + missing_period_penalty * (n-1)
           }
         - short_gap_penalty * max(0, 1-gap/(min_spacing_ratio*d))

Huber(e) = 0.5*e^2                    if abs(e) <= 1
           abs(e)-0.5                 otherwise

r(edge) = gap / (n*d)
C(a,b)  = -w_change * Huber((r(b)-r(a))/spacing_scale_ratio)
```

候选 marginal 的 unary 始终低于 strong 下界；小 bonus 不能让弱候选压过合格 strong 的单节点证据。是否选中仍由全序列收益决定。每条边只允许 `0 < gap <= max_sequence_gap_periods*d`，默认最大 4 个周期；跨度更远的独立候选区间分别求解。每个被选中的连通区间至少包含一个 strong anchor，避免一整段纯弱噪声自我佐证。

一阶 DP 保留到当前节点的最优路径；开启二阶项时保留前一节点和是否存在 strong anchor，精确计算连续两条边的变化代价。当前 `sequence_spacing_change_weight=0`：已有目标点通过一阶项解决，尚无整图真值证明二阶项改善准确率。二阶实现已经过目标函数求和和小规模全子集穷举验证。

跨 n 个纸距的容差为 `sqrt(n)*sigma`。初次真实回归发现，固定单纸距容差会把短图末端“强证据但间距约 1.67d”的轨迹压掉；按跨度累积不确定性解决这一问题，没有增加图像特征，也没有添加坐标例外。末端仍可能因极端间距冲突被抑制，并非强制保留首尾。

序列 prior 从去掉明显重复后的 strong 候选估计；本次前后两次 DP 使用同一 prior，避免恢复改变周期后再为自身提供支持。对 selected 的每条实际间距输出误差与 pair score。对 suppressed 也输出把它插回相邻 selected 之间的两条边代价和局部插入净收益，方便核查。

## 为什么能同时抑制强轨迹、恢复 marginal

一个高分节点插入后，如果制造极短间距，其 unary 收益不足以抵消前后两条边的代价，DP 会跳过它。相反，一个 0.55～0.59 的轨迹若填补缺口并形成更合理的前后间距，加入它的全局收益可以为正。

没有候选的地方没有 DP 节点。`n=2`、`n=3` 仅表示为一个缺口支付代价，绝不据此新增一条或两条纸边。

## 去重与恢复

新前置去重只合并**更短轨迹至少 80% 的实测条带在 1 像素内重合**的路径。仅中心距离小于 0.5d 不再提前删除；几何独立的短间距轨迹保留给纵向 DP 竞争。旧 `deduplicate_tracks()` 保留给旧流程和测试。

恢复先检查 selected strong anchors 之间的异常间距；其中可能已经有第一轮序列选回的 marginal。候选中心位于缺口内部且距离预测中心不超过 `recovery_reuse_candidate_radius_ratio*d` 时，直接复用它的完整条带位置和支持数，在尝试记录中关联其最终 selected 状态。不会再把一条 10/12 的轨迹裁成窗口中的 7/12。

如果候选池中存在合理候选，即使它最后被 DP 抑制，也不针对同一缺口重复搜一次弱图像来绕过序列决定。只有无合理候选时才重搜未被已有轨迹占用的负梯度 strip peaks，仍调用原横向二阶关联。正梯度只能提供受限辅助加分，不能独立产生路径，也不能绕过结构门槛。

异常判据为：

```text
abs(gap-d) > max(1px, spacing_scale_ratio*d, spacing_anomaly_mad_k*1.4826*MAD)
```

因此 41px 对比 d=26px、低 MAD 时会被标记，不要求接近 2d。大于 d 的异常才尝试缺口恢复，短间距只保留复核记录。没有图像证据时保持缺口。

弱峰搜索半宽为：

```text
radius = clip(base_ratio*d + mad_weight*period_MAD
              + geometry_weight*mean(neighbor_y_MAD),
              min_search_ratio*d, max_search_ratio*d)
```

每条 strip 先按邻居的已观测位置插值出上、下曲线，再在缺口比例 k/n 处计算预测曲线：`predicted[s]=(1-k/n)*left[s]+(k/n)*right[s]`。搜索围绕每条带自身的预测位置展开，并限制在两条邻居曲线之间。补出的新路径仍需再过纵向序列，不能直接 append 到 Count。

## 输出与计数口径

JSON 包含 `track_candidate_pool`、`strong_tracks`、`marginal_tracks`、`garbage_tracks`、`vertical_sequence`（IDs、总分、transitions、track_decisions）、`sequence_suppressed_tracks`、`sequence_rescued_tracks`、`remaining_suspicious_gaps`、`candidate_reuse_recovery_attempts` 和 `weak_image_recovery_attempts`。

`Sequence selected` 包含 rescued，等于最终 Count；`Sequence rescued` 是其中 marginal 或弱图像恢复的子集；`Sequence suppressed` 是 usable 候选中未选中的数量，不等于全部垃圾碎片数量。`detected_peaks + repaired_peaks` 仍等于 Count。CLI 的 `Candidate-reuse recoveries` 是有尝试记录且最终选中的 marginal ID 数，可能少于全部 sequence rescued（有些 marginal 直接在第一轮序列中选回）。

恢复理由区分 `recovery_no_existing_candidate`、`recovery_reused_marginal_track`、`recovery_search_window_miss`、`recovery_no_image_evidence`、`recovery_low_support`。最终抑制区分 `sequence_suppressed_short_gap` 和 `sequence_spacing_conflict`；恢复标为 `sequence_rescued`。

overlay 只画实测 strip 曲线，绿/黄/红分别为 selected/rescued/suppressed。完整垃圾碎片见 `edge_tracks.png` 和 JSON。`vertical_sequence.png` 的上图是候选 unary 与决策，下图是 selected 间距和局部周期。

`--rotate-ccw` 在检测前逆时针旋转 90°，不改原文件。输出坐标对应旋转后的检测图：`original_x = original_width - 1 - detection_y`，`original_y = detection_x`。overlay 另有 116px 标题区和可能的预览缩放，不能直接把屏幕行号当作图像 Y。

## 测试与剩余限制

原有 33 项测试保留，新增 20 项，共 53 项通过。新增测试包括：强近邻竞争、0.58 marginal 恢复、噪声不全部保留、10/12 完整候选复用、空池启动弱搜、41/26 非整数异常但不造边、弯曲轨迹搜索、自适应宽度、正确拒绝原因、正梯度不能独立恢复、窗口漏搜理由、首尾不规则间距、marginal unary 上界、一阶和二阶 DP 穷举对照。

本次真实图完整结果见 [19 张图的验证报告](../output_vertical_sequence_20260914_143314_532167/validation_report.md)、[目标点决策](../output_vertical_sequence_20260914_143314_532167/regression.json) 和 [16 张旋转横图结果](../output_batch_20260914_143312_776459/summary.md)。2.bmp 为 90 条（恢复 8、抑制 9），4.bmp 为 39 条（恢复 0、抑制 6），3.png 为 5 条（恢复 0、抑制 0）。这里抑制数包含此前尚未计数的 marginal 候选，不能直接从旧 Count 减掉该数。

没有整图人工真值；Confidence 和剩余异常数均不能当作准确率。现有横向检测若把一条真实边拆成多个严重碎片，或使其低于垃圾底线，序列无法凭空重建。与真实纸边同样连贯且恰好具有合理间距的纹理也可能误入；厚度快速变化、先验错误、边界缺少上下文仍可能导致误抑制或漏恢复。剩余异常会输出供人工复核，而不是强行凑整周期。
