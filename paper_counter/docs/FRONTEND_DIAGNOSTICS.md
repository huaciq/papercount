# Front-end diagnostics and ablation

The normal configuration keeps the historical median-strip and native Scharr
front end. All new switches default to off, so `configs/default.yaml` remains
the one-step rollback path.

## Front-end variants

- `strip_aggregation_mode: median` keeps the full-width per-strip row median.
- `narrow_median` samples the centered fraction selected by
  `strip_width_ratio`; the experiment configuration also increases the strip
  count to keep spatial coverage measurable.
- `slope_aligned` divides each strip into column bins, estimates bounded integer
  row lags in fixed-height windows, aligns those bin profiles, and then takes
  their median. It does not use the estimated paper period to choose its window
  or maximum shift.
- `directional_smoothing_enabled` applies strong horizontal and weak vertical
  Gaussian smoothing before the signed Y-gradient.
- `multiscale_gradient_enabled` evaluates the fixed scales in
  `gradient_scales`. In the multiscale-only ablation the filters are isotropic;
  in the combined variant the directional base is preserved and only the
  vertical integration scale changes.

Positive and negative responses stay separate. A pointwise maximum merges
scales before peak detection, which removes duplicate scale proposals. Every
candidate records its winning scale and its support in the untouched native
gradient. An enhanced-only proposal is diagnostic evidence and cannot enter a
paper track as a real observation.

When either enhancement is enabled, `enhanced_period_from_native: true` also
estimates the back-end spacing scale from the untouched native gradient. This
prevents smoothing or scale fusion from changing an ACF harmonic and keeps the
back end comparable across front-end ablations. Set it to false only for an
explicit sensitivity experiment.

## Reproduction

Run a single fixed-backend variant:

```powershell
D:\Anc\anaconda3\envs\pytorch\python.exe scripts\batch_detect.py `
  --input "F:\纸张\数据\pictures" `
  --config configs\experiments\multiscale_only.yaml `
  --workers 2 --rotate-ccw `
  --output-root "F:\纸张\paper_counter\output\frontend_ablation"
```

Score count truth without treating it as edge truth:

```powershell
D:\Anc\anaconda3\envs\pytorch\python.exe scripts\evaluate_txt_ground_truth.py `
  --input "F:\纸张\数据\pictures" --batch <batch-directory>
```

Layer the partial coordinate annotations and save local evidence:

```powershell
D:\Anc\anaconda3\envs\pytorch\python.exe scripts\diagnose_error_layers.py `
  --batch <batch-directory> `
  --annotations annotations\track_ground_truth.json `
  --output "F:\纸张\paper_counter\output\error_layers"
```

The diagnostic categories are `原图不可辨`, `梯度弱`, `候选漏检`,
`关联失败`, `序列误选`, and `重复边界`. Image visibility is not inferred from
stability or bipolar agreement. When saved machine evidence cannot establish
visibility or when a nearby selected edge may belong to another sheet, the
case is marked `待人工核验`.

`scripts/compare_strip_aggregation.py` plots local normalized response curves
and reports the number of strips with a peak near each exact reviewer overlay
coordinate. `scripts/summarize_frontend_ablation.py` combines completed batch
truth reports and runtime into a single attribution table.
