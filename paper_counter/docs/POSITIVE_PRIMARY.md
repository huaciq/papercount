# 正梯度主检测

`configs/experiments/multiscale_positive.yaml` 使用正梯度（灰度随 Y 增大由暗到亮）作为主候选观测，计数线通常对应纸张上边缘。与 `multiscale_only.yaml` 保持相同尺度、12 条 strip 和后端阈值。周期锁定来自正梯度原生响应；候选和弱边缘恢复也来自正梯度。负梯度只给有限的辅助加分，不能独立补出纸张。

最终身份验证原先按纸张下边缘建立：负梯度居中、纸身在上方、正梯度配对在上方。正梯度模式在每条观测曲线周围反转局部采样方向，并交换主/辅助梯度通道，因此验证使用纸身在下方、配对负梯度在下方的原图证据。仅反转局部验证坐标，原图、ROI、轨迹和输出 Y 坐标不变。`auto` 极性会先解析成实际主极性再验证。

为兼容历史输出，身份特征中的 `negative` 表示主梯度、`positive` 表示相反梯度、`above` 表示纸身侧、`below` 表示间隙侧。`track_identity.primary_polarity`、`physical_paper_side`、`legacy_field_semantics` 明确记录解释；身份曲线图也标注主/辅助通道与局部坐标。其他 `positive_gradient.png` 和 `negative_gradient.png` 仍保留实际梯度符号。

`recovery_require_primary_evidence: true` 控制主流程恢复证据，旧的 `recovery_require_negative_evidence` 保留供历史恢复 API 直接调用。增强独有响应仍不能成为观测，恢复和计数都受原生支持、原图 mask、strip 支持及几何条件限制。两极计数不合并或平均。

```powershell
conda run -n pytorch python -m pytest tests -q
conda run -n pytorch python scripts/batch_detect.py --input "F:\纸张\数据\260921各纸种4面图集" --config configs/experiments/multiscale_positive.yaml --rotate-ccw --workers 2 --output-root "F:\纸张\paper_counter\output\multiscale_positive_260921"
conda run -n pytorch python scripts/compare_polarity_batches.py --negative NEGATIVE_BATCH --positive POSITIVE_BATCH
```

回退使用 `configs/experiments/multiscale_only.yaml`（负梯度多尺度）或 `configs/default.yaml`（原默认）。`--dual-polarity` 仍是旧诊断接口，正梯度审计分支保持关闭末端验证，其计数与新的正梯度主流程定义不同。

没有完整真值时，报告仅比较计数和诊断证据，不能报告准确率或把同数当作正确。正、负梯度代表不同边界相位，人工审核正梯度结果应看纸张上边缘，并依据新 PNG 的坐标重新记录。
