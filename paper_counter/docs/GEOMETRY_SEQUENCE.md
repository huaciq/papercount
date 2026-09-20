# Geometry-centric ordered-layer detection

本版本保留 negative gradient、逐条带峰检测、horizontal multi-strip tracking、候选池、Vertical Sequence DP、两级恢复、legacy 流程和 `vertical_sequence_enabled` 开关。改变的是纵向决策依据：局部 period 只作为 spacing scale，实际候选的完整 `y=f(x)` 几何、上下顺序和不交叉关系优先。

## RAW 三态权限

`detect_raw_stack_mask()` 在 illumination normalization 之前运行。它对原始 gray 的每行统计 median、P10、P90 和原始 Sobel-Y 纹理，用背景低分位数和 MAD 建立自适应亮度/纹理阈值，然后小范围闭合内部暗行，保留最大连续内容区。输出 `INVALID=0`、`MARGIN=1`、`CORE=2`。

INVALID 不允许峰、轨迹、复用或 recovery。MARGIN 允许有足够原始负梯度的 strong track；默认不允许 marginal 和 gap-triggered weak recovery。CORE 允许完整流程。黑背景不会因为 normalization 后出现微小梯度而获得计数资格。第一张和最后一张纸只要有真实 raw evidence，仍可在 MARGIN 保留。

## 曲线、gap profile 和 crossing

`fit_track_curve()` 对 `strip_positions` 做保形分段线性插值，只填少量内部 strip，不外推，也不修改原始支持 mask。`gap_profile(A,B)` 在共同观测或短缺口插值 strip 上计算 gap median/MAD/min/max、正间距比例、长期小间距比例、横向变化、共享几何置信度和 crossing ratio。lower 在任何有效 strip 上不应越过 upper；超过配置比例的 crossing transition 直接拒绝。

平滑倾斜、平滑弯曲和轻微起伏保留；连续 neighbor jump 合法即可。只有随机大跳或大量反转会降低 geometry evidence。track geometry 调试图中实点是真实 strip observations，虚线只是短距离插值。

## 新的 sequence objective

默认 `sequence_period_multiple_mode=false`，因此不会枚举 `n=1..4` 并把 `gap≈n*d` 当作缺纸证明。每一对实际轨迹只计算一次 geometry pair：

```text
J = Σ TrackUnary + Σ PairGeometry + very_weak_spacing_scale_prior

PairGeometry = shared_geometry * [non_crossing / gap_profile quality]
                - ambiguity
                - lambda_short * persistent_small_gap
                - lambda_long * weak_large_gap_penalty
                - weak_spacing_prior
```

短 gap 的惩罚强于大 gap。大 gap 只被标为 `large_gap_anomaly_not_missing`，不生成 predicted paper、不获得 missing bonus。若缺口内有真实 candidate，它凭自身 image unary 和 geometry 重新竞争；没有 negative evidence 就保持 gap。

旧的 n-period 逻辑保留为显式比较模式 `sequence_period_multiple_mode=true`，供历史测试/对照使用，默认生产配置关闭。

## Evidence-driven recovery

large/unusual gap 先生成一个“搜索走廊”，其上下界来自相邻完整曲线逐 strip 插值。走廊中心只用于搜索，不是预测纸层。候选池先复用完整 marginal/strong track；只有没有合理候选时才对 CORE 中的 negative weak strip peaks 重新做 horizontal association。MARGIN 默认禁止这一步。没有 image candidate 时不增加 Count。

## 调试输出

新增 `raw_stack_mask.png`、`track_geometry.png`，并扩充 `vertical_sequence.png` 的下图为 selected neighbor gap median、local spacing scale、short-gap warning 和 large-gap anomaly。JSON 保存 `raw_stack_mask`、INVALID 峰数、INVALID selected 数、margin selected、完整 gap profile、selected gap distribution、crossing、short suppression、large anomaly、evidence recovery 和 recovery reason。

Confidence 降低对 period 一致性的依赖，改由 raw validity、image support、non-crossing、shared geometry 和 sequence ambiguity 主导；大量 strong competition、suppression、rescue 或 crossing 会降低分数。Confidence 仍是质量/一致性启发式分数，不是计数准确率。

测试覆盖黑背景、边界 strong、MARGIN marginal、真实大 gap、厚纸 gap、secondary texture、真实近纸、倾斜/弯曲/起伏、随机跳跃、crossing、非均匀 gap、INVALID recovery、无证据大 gap、弱负梯度恢复、非对称 penalty、错误 spacing scale 和 gap profile 区分。
