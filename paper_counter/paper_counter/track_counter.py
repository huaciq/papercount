"""Multi-strip-first counting; global profile is only a scale prior/debug signal."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from .config import CounterConfig
from .models import PaperCountResult, PeakCandidate
from .preprocessing import (crop_roi, to_gray, normalize_illumination,
                            signed_gradient_levels, display_preview,
                            preprocess_tracks_signal)
from .profiles import (make_strip_profiles_diagnostic, normalize_and_fuse,
                       estimate_strip_orientation, aggregate_along_orientation)
from .period import score_polarity, estimate_period
from .local_detection import detect_strip_peaks
from .tracking import associate_strip_peaks, validate_tracks, deduplicate_tracks, classify_tracks, deduplicate_obvious_tracks
from .local_period import estimate_local_period, missing_edge_hypotheses
from .track_recovery import recover_weak_tracks
from .active_roi import detect_active_stack_roi
from .confidence import compute_track_confidence
from .vertical_sequence import optimize_vertical_sequence, SequenceResult, transition_score
from .sequence_recovery import recovery_hypotheses, recover_sequence_candidates, spacing_anomalies
from .raw_mask import detect_raw_stack_mask, filter_mask_peaks, track_mask_permission, StackState, refine_stack_mask_with_strong_tracks
from .track_geometry import estimate_local_spacing, geometry_pair_score
from .regularity import estimate_local_regularity, evaluate_middle_track_competition
from .track_identity import (analyze_track_identity, apply_final_track_verifier,
                             make_intensity_strip_profiles)
from .raw_final_guard import apply_raw_observation_guard
from .adaptive_structure import analyze_paper_structure
from .enhanced_edge_detection import enhanced_gradient_response


def process_tracks(image: np.ndarray, cfg: CounterConfig) -> PaperCountResult:
    """Detect at full resolution and export observed track centers."""
    # 1. 定位纸堆：图像为 [H, W]，条带剖面为 [S, H]；RAW 保持原始亮度。
    # bounds 使用传入图像坐标，内部信号使用裁剪后的 ROI 行号。
    roi, bounds = crop_roi(image, cfg.roi)
    requested_bounds = dict(bounds)
    # Keep RAW evidence on the untouched image.  The optional tracks frontend
    # receives a separate signal so enhancement cannot manufacture paper-body
    # evidence or change coordinate semantics.
    gray = to_gray(roi)
    signal_gray, preprocessing_info = preprocess_tracks_signal(gray, cfg)
    raw_preview = display_preview(gray, cfg.display_max_height, cfg.display_max_width)
    raw_mask = detect_raw_stack_mask(gray, cfg, offset=bounds["y1"])
    # The raw three-state mask decides whether individual candidates are
    # permitted; it is intentionally conservative and cannot locate the stack
    # when a scratch splits an otherwise empty black border.  Locate the main
    # sustained texture interval independently, then retain the raw mask for
    # candidate/track permission inside that interval.
    if cfg.stack_roi_mode == "foreground_envelope":
        top, bottom, active_info = detect_active_stack_roi(gray, None, cfg, original=roi)
    else:
        top, bottom, active_info = detect_active_stack_roi(gray, None, cfg)
    foreground_invalid_overrides = 0
    if cfg.stack_roi_mode == "foreground_envelope" and active_info.get("energy") is not None:
        # The historical relative-black mask can label a mostly saturated
        # image INVALID when its median background reference is also white.
        # RAW foreground is independent evidence that these are not dark rows.
        foreground = active_info["energy"] >= cfg.foreground_roi_min_width_ratio
        override = foreground & (raw_mask.states == StackState.INVALID)
        foreground_invalid_overrides = int(override.sum())
        raw_mask.states[override] = StackState.CORE
        raw_mask.evidence[override] = True
    roi = roi[top:bottom]
    gray = gray[top:bottom]
    signal_gray = signal_gray[top:bottom]
    # 二次裁剪必须累计 y1；候选检测时加此偏移才能回到整张检测图。
    bounds = {**bounds, "y1": bounds["y1"] + top, "y2": bounds["y1"] + bottom}
    normalized = normalize_illumination(signal_gray, cfg.illumination_sigma, cfg.epsilon)
    orientation_fields = (estimate_strip_orientation(gray, cfg) if
        cfg.strip_sequence_enabled or cfg.strip_aggregation_mode == "orientation_aligned" else None)
    gradient_levels, native_gy = signed_gradient_levels(normalized, cfg)
    invalid_rows = raw_mask.states[top:bottom] == StackState.INVALID
    native_gy[invalid_rows] = 0
    for _, gradient, _ in gradient_levels:
        gradient[invalid_rows] = 0
    preview = lambda a: display_preview(a, cfg.display_max_height, cfg.display_max_width)
    previews = {"gray": preview(gray), "normalized": preview(normalized)}
    preview_gradient = gradient_levels[-1][1]
    for name, sign in (("positive", 1.0), ("negative", -1.0)):
        rendered = np.empty(preview_gradient.shape, dtype=np.uint8)
        scale = max(float(np.max(preview_gradient) if sign > 0 else -np.min(preview_gradient)), cfg.epsilon)
        for start in range(0, len(preview_gradient), cfg.debug_chunk_rows):
            chunk = preview_gradient[start:start + cfg.debug_chunk_rows] * sign
            rendered[start:start + len(chunk)] = np.clip(chunk * (255 / scale), 0, 255).astype(np.uint8)
        previews[name + "_gradient"] = preview(rendered)
    del rendered
    data = {}
    # 2. 分开构建正/负梯度和多尺度条带信号，保留 native 分支约束增强候选。
    for polarity in ("positive", "negative"):
        level_profiles, level_details = [], []
        for scale_value, gradient, source in gradient_levels:
            raw_level, strips, aggregation = make_strip_profiles_diagnostic(gradient, polarity, cfg, orientation_fields)
            raw_level[:, invalid_rows] = 0
            level_profiles.append(raw_level)
            level_details.append({"scale": scale_value, "source": source,
                                  "aggregation": aggregation})
        native_raw, _, native_aggregation = make_strip_profiles_diagnostic(native_gy, polarity, cfg, orientation_fields)
        native_raw[:, invalid_rows] = 0
        stack = np.stack(level_profiles)
        winners = np.argmax(stack, axis=0)
        raw = np.take_along_axis(stack, winners[None, ...], axis=0)[0]
        scale_values = np.asarray([item[0] for item in gradient_levels], dtype=np.float32)
        scale_sources = scale_values[winners]
        profiles, fused, stats = normalize_and_fuse(raw, cfg.epsilon)
        native_profiles, native_fused, _ = normalize_and_fuse(native_raw, cfg.epsilon)
        data[polarity] = {"raw_profiles": raw, "profiles": profiles, "global": fused,
                          "native_raw_profiles": native_raw, "native_profiles": native_profiles,
                          "native_global": native_fused,
                          "scale_sources": scale_sources,
                          "strip_stats": stats, "assessment": score_polarity(fused, cfg),
                          "native_assessment": score_polarity(native_fused, cfg),
                          "gradient_levels": level_details,
                          "native_aggregation": native_aggregation}
        if cfg.strip_sequence_enabled:
            data[polarity]["independent_scale_profiles"] = [
                (float(detail["scale"]), normalize_and_fuse(level, cfg.epsilon)[0])
                for detail, level in zip(level_details, level_profiles)]
    raw_intensity_profiles = make_intensity_strip_profiles(gray, strips, cfg.debug_chunk_rows)
    body_profiles = None
    if cfg.evidence_model:
        from .body_evidence import narrow_intensity_profiles
        from .boundary_direction import trace_boundary_motion
        body_profiles = narrow_intensity_profiles(gray, strips)
        if cfg.oriented_association_enabled:
            orientation_fields = trace_boundary_motion(gray,strips,orientation_fields,cfg)
    normalized_intensity_profiles = make_intensity_strip_profiles(normalized, strips, cfg.debug_chunk_rows)
    raw_gradient_profiles = {}
    if cfg.strip_sequence_enabled:
        import cv2
        raw_gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        for polarity in ("positive", "negative"):
            raw_gradient_profiles[polarity], _, _ = make_strip_profiles_diagnostic(
                raw_gy, polarity, cfg, orientation_fields)
        del raw_gy
        if cfg.strip_aggregation_mode == "orientation_aligned":
            for s, (a, b) in enumerate(strips):
                raw_intensity_profiles[s] = aggregate_along_orientation(gray[:, a:b], orientation_fields[s])
                normalized_intensity_profiles[s] = aggregate_along_orientation(normalized[:, a:b], orientation_fields[s])
    del normalized, native_gy, gradient_levels, preview_gradient
    chosen = cfg.gradient_polarity
    if chosen == "auto":
        chosen = max(data, key=lambda p: (data[p]["assessment"]["score"], data[p]["assessment"]["acf_score"]))
    selected = data[chosen]
    opposite = "positive" if chosen == "negative" else "negative"
    # Resolve auto before the identity verifier interprets boundary direction.
    # Use to_dict() to avoid InitVar issues with replace()
    identity_cfg_dict = cfg.to_dict()
    identity_cfg_dict['gradient_polarity'] = chosen
    identity_cfg = CounterConfig(**identity_cfg_dict)
    use_native_period = cfg.enhanced_period_from_native and (
        cfg.directional_smoothing_enabled or cfg.multiscale_gradient_enabled)
    period_assessment = selected["native_assessment"] if use_native_period else selected["assessment"]
    period = period_assessment["period"]
    period_source = ("native_gradient_frontend_lock" if period is not None and use_native_period
                     else "global_profile" if period is not None else "unavailable")
    strip_periods = []
    # Large curvature can wash out the fused signal. Measured strip ACF consensus
    # supplies the scale in that case, without inferring any paper locations.
    if period is None:
        for signal in selected["profiles"]:
            measured, strength, _ = estimate_period(signal, cfg.min_period, cfg.max_period,
                cfg.min_acf_score, cfg.acf_harmonic_preference, cfg.epsilon)
            if measured is not None:
                strip_periods.append(measured)
        if len(strip_periods) >= np.ceil(cfg.min_track_support_ratio * cfg.num_strips):
            consensus = float(np.median(strip_periods))
            agreement = sum(abs(p - consensus) <= cfg.missing_gap_tolerance * consensus for p in strip_periods)
            if agreement >= np.ceil(cfg.min_track_support_ratio * cfg.num_strips):
                period = consensus
                period_source = "strip_acf_consensus"
    if period is None and not cfg.sequence_period_multiple_mode and np.any(raw_mask.evidence) and any(s["active"] for s in selected["strip_stats"]):
        period = (cfg.min_period + cfg.max_period) / 2
        period_source = "bootstrap_spacing_scale_not_periodicity"

    # ========== ADAPTIVE STRUCTURE ANALYSIS ==========
    # Analyze paper structure to enable adaptive algorithm behavior.
    # This replaces manual per-sample parameter tuning with automatic detection.
    structure_analysis = None
    adaptive_cfg = cfg
    structure_debug = {'enabled': True, 'analysis': None, 'adjustments_applied': {}}

    if period is not None:
        try:
            # Run structure analysis on the selected polarity gradient
            initial_gradient = selected["native_profiles"][cfg.num_strips // 2]  # Use middle strip as representative
            structure_analysis = analyze_paper_structure(
                gray, period, initial_gradient, cfg)

            structure_debug['analysis'] = {
                'structure_type': structure_analysis.structure_type,
                'confidence': structure_analysis.confidence,
                'has_narrow_gaps': structure_analysis.has_narrow_gaps,
                'gap_width_pixels': structure_analysis.gap_width_pixels,
                'tilt_angle_degrees': structure_analysis.tilt_angle_degrees,
                'tilt_confidence': structure_analysis.tilt_confidence,
                'local_period_variance': structure_analysis.local_period_variance,
                'edge_contrast_ratio': structure_analysis.edge_contrast_ratio,
                'bright_width_ratio': structure_analysis.bright_width_ratio,
                'dark_width_ratio': structure_analysis.dark_width_ratio
            }

            # Apply adaptive parameter adjustments
            adjustments = structure_analysis.recommended_adjustments
            if adjustments and structure_analysis.confidence > 0.5:
                # Create adapted configuration
                # Note: We can't use replace() with InitVar fields, so we use to_dict()
                adapted_params_dict = cfg.to_dict()

                # Map adjustments to config parameters
                if 'sequence_min_spacing_ratio' in adjustments:
                    adapted_params_dict['sequence_min_spacing_ratio'] = adjustments['sequence_min_spacing_ratio']
                if 'regularity_gap_min_ratio' in adjustments:
                    adapted_params_dict['regularity_gap_min_ratio'] = adjustments['regularity_gap_min_ratio']
                if 'sequence_short_gap_ratio' in adjustments:
                    adapted_params_dict['sequence_short_gap_ratio'] = adjustments['sequence_short_gap_ratio']
                if 'track_min_continuity_ratio' in adjustments:
                    adapted_params_dict['min_continuity_ratio'] = adjustments['track_min_continuity_ratio']
                if 'max_neighbor_shift_ratio' in adjustments:
                    adapted_params_dict['max_neighbor_shift_ratio'] = adjustments['max_neighbor_shift_ratio']
                if 'track_max_neighbor_jump_ratio' in adjustments:
                    adapted_params_dict['track_max_neighbor_jump_ratio'] = adjustments['track_max_neighbor_jump_ratio']
                if 'regularity_geometry_weight' in adjustments:
                    adapted_params_dict['regularity_geometry_weight'] = adjustments['regularity_geometry_weight']
                if 'regular_spacing_prior_weight' in adjustments:
                    adapted_params_dict['regular_spacing_prior_weight'] = adjustments['regular_spacing_prior_weight']
                if 'margin_min_gradient_score' in adjustments:
                    adapted_params_dict['margin_min_gradient_score'] = adjustments['margin_min_gradient_score']
                if 'min_recovery_gradient_score' in adjustments:
                    adapted_params_dict['min_recovery_gradient_score'] = adjustments['min_recovery_gradient_score']

                # Record what was changed
                structure_debug['adjustments_applied'] = {
                    k: v for k, v in adapted_params_dict.items()
                    if k in ['sequence_min_spacing_ratio', 'regularity_gap_min_ratio',
                            'sequence_short_gap_ratio', 'min_continuity_ratio',
                            'max_neighbor_shift_ratio', 'track_max_neighbor_jump_ratio',
                            'regularity_geometry_weight', 'regular_spacing_prior_weight',
                            'margin_min_gradient_score', 'min_recovery_gradient_score']
                    and adapted_params_dict[k] != cfg.to_dict().get(k)
                }

                if structure_debug['adjustments_applied']:
                    # Create new config from modified dict
                    adaptive_cfg = CounterConfig(**adapted_params_dict)

        except Exception as e:
            # Graceful degradation: log error but continue with original config
            structure_debug['error'] = str(e)
            structure_debug['enabled'] = False
    else:
        structure_debug['enabled'] = False
        structure_debug['reason'] = 'no_period_available'
    # ========== END ADAPTIVE STRUCTURE ANALYSIS ==========

    tracks, rejected, raw_tracks, candidates, statistics = [], [], [], [], []
    local = None
    hypotheses, attempts, recovered = [], [], []
    strong, marginal, garbage = [], [], []
    sequence = SequenceResult()
    reuse_attempts, weak_attempts, pool, remaining_gaps = [], [], [], []
    invalid_peak_count = 0
    # Evaluation-only snapshots, taken before later in-place decisions.
    evaluation_stages = {}
    strip_consensus_debug, strip_quality = {}, None
    strip_tracks, strip_rejected = [], []
    raw_period_debug = {'enabled': cfg.raw_period_consensus_enabled, 'changed': False}
    # 3. RAW 支持的独立条带序列：先按实测纸身配对/保序融合，得到补充候选。
    # 当前模式是 augment，后面的常规关联和纵向序列仍先建立主轨迹。
    if cfg.strip_sequence_enabled:
        from .strip_sequence import build_strip_consensus
        if period is None:
            period_source = 'bootstrap_spacing_scale_not_periodicity'
        period = period or (cfg.min_period+cfg.max_period)/2
        strip_tracks, strip_rejected, strip_local, strip_sequence, strip_candidates, strip_consensus_debug, strip_quality = build_strip_consensus(
            data, raw_intensity_profiles*(255/np.iinfo(gray.dtype).max), raw_gradient_profiles,
            gray, strips, orientation_fields, bounds["y1"], period, raw_mask, chosen, cfg)
        resolved = strip_consensus_debug['polarity_choice']['selected']
        if resolved != chosen:
            chosen = resolved
            selected = data[chosen]
            opposite = 'negative' if chosen == 'positive' else 'positive'
            identity_cfg_dict = cfg.to_dict()
            identity_cfg_dict['gradient_polarity'] = chosen
            identity_cfg = CounterConfig(**identity_cfg_dict)
            new_assessment = selected['native_assessment'] if use_native_period else selected['assessment']
            period = new_assessment['period'] or period
            period_source = 'raw_body_selected_polarity_native_prior'
        if cfg.strip_sequence_scale_consensus_enabled:
            from .strip_sequence import choose_strip_spacing
            period, scale_choice = choose_strip_spacing(strip_consensus_debug, chosen, period, cfg)
            strip_consensus_debug['scale_choice'] = scale_choice
            if scale_choice['changed']:
                period_source = 'native_strip_acf_with_unnormalized_raw_body_consensus'
        if cfg.strip_sequence_mode == "replace":
            tracks, rejected, local, sequence, candidates = strip_tracks, strip_rejected, strip_local, strip_sequence, strip_candidates
            raw_tracks = tracks+rejected
            strong = list(tracks)
            marginal = [t for t in rejected if t.reason == "local_evidence_pending"]
            pool = raw_tracks
            evaluation_stages['associated'] = [t.to_dict() for t in raw_tracks]
            remaining_gaps = spacing_anomalies(tracks, local, cfg, None)
    # Preserve independent RAW decoding and its existing spacing resolver.
    # The additional resolver only disambiguates the remaining main-association
    # prior; it cannot change the decoder's paper-body pairing window.
    if cfg.raw_period_consensus_enabled and period is not None:
        from .raw_period import raw_interface_period
        period, raw_period_debug = raw_interface_period(
            raw_intensity_profiles*(255/np.iinfo(gray.dtype).max), period, cfg)
        raw_period_debug['enabled'] = True
        if raw_period_debug['changed']:
            period_source = 'raw_interface_gap_and_acf_consensus'
    signal_model=None
    if period is not None and (not cfg.strip_sequence_enabled or cfg.strip_sequence_mode != "replace"):
        if cfg.evidence_model and getattr(orientation_fields,'motion',None):
            from .boundary_direction import direction_search_policy
            direction_policy=direction_search_policy(orientation_fields,period,cfg)
        if cfg.evidence_model:
            from .stripe_evidence import stripe_proposals
            from .signal_gaps import SignalGapModel
            signal_proposals,stripe_debug=stripe_proposals(body_profiles,strips,orientation_fields,
                bounds['y1'],period,chosen,cfg)
            if cfg.compressed_layers_enabled:
                from .compressed_layers import compressed_proposals
                compressed_tracks,compressed_debug=compressed_proposals(gray,strips,orientation_fields,
                    bounds['y1'],period,chosen,cfg)
            signal_model=SignalGapModel(body_profiles,stripe_debug,bounds['y1'],period,chosen,cfg,strips,orientation_fields,
                raw_intensity_profiles*(255/np.iinfo(gray.dtype).max))
        # 4. 局部峰 → 横向关联 → strong/marginal/garbage；strong 仍需序列终选。
        candidates, statistics = detect_strip_peaks(
            selected["profiles"], period, cfg, bounds["y1"],
            selected["native_profiles"], selected["scale_sources"])
        invalid_peak_count = sum(raw_mask.state_at(p.y) == StackState.INVALID for strip in candidates for p in strip)
        candidates = filter_mask_peaks(candidates, raw_mask)
        valid = sum(s["active"] for s in selected["strip_stats"])
        if cfg.oriented_association_enabled:
            from .strip_sequence import associate_oriented_candidates
            raw_tracks, ordered_association_debug = associate_oriented_candidates(
                candidates,strip_consensus_debug,chosen,strips,orientation_fields,
                strip_quality,bounds['y1'],period,cfg,valid)
            if cfg.evidence_model:
                ordered_association_debug['direction_policy']=locals().get('direction_policy',{})
        else:
            raw_tracks = associate_strip_peaks(candidates, period, adaptive_cfg, valid)
        if cfg.evidence_model:
            from .body_evidence import attach_body_evidence
            attach_body_evidence(raw_tracks,body_profiles,bounds['y1'],lambda y:period,chosen,cfg)
            if signal_model is not None:signal_model.protect_observed_transitions(raw_tracks)
        strong, marginal, garbage = classify_tracks(raw_tracks, adaptive_cfg)
        evaluation_stages['associated'] = [t.to_dict() for t in raw_tracks]
        mask_refinement = refine_stack_mask_with_strong_tracks(raw_mask, strong, cfg, period)
        for track in strong + marginal:
            if not track_mask_permission(track, raw_mask, cfg):
                garbage.append(track)
        strong = [t for t in strong if t.candidate_class == "strong"]
        marginal = [t for t in marginal if t.candidate_class == "marginal"]
        evaluation_stages['classified'] = [t.to_dict() for t in strong + marginal]
        if cfg.vertical_sequence_enabled:
            pool, duplicates = deduplicate_obvious_tracks(strong + marginal)
            estimate_spacing = estimate_local_period if cfg.sequence_period_multiple_mode else estimate_local_spacing
            local = estimate_spacing([t for t in pool if t.candidate_class == "strong"], period, adaptive_cfg)
            if signal_model is not None:signal_model.annotate(pool)
            regularity = estimate_local_regularity([t for t in pool if t.candidate_class == "strong"],
                                                   selected["global"], period, adaptive_cfg) if cfg.regularity_enabled else None
            # 5. 纵向选择真实轨迹，再检查异常空隙、复用候选或搜索弱图像证据。
            # 第二次优化只接收真实观测，不能在推测的周期位置插入虚拟纸层。
            if signal_model is not None:
                hypothesis_cfg_dict = adaptive_cfg.to_dict()
                hypothesis_cfg_dict['evidence_model'] = False
                hypothesis_cfg_dict['compressed_layers_enabled'] = False
                hypothesis_cfg_dict['composite_layers_enabled'] = False
                hypothesis_cfg = CounterConfig(**hypothesis_cfg_dict)
            else:
                hypothesis_cfg = adaptive_cfg
            sequence = optimize_vertical_sequence(pool, local, hypothesis_cfg, regularity)
            evaluation_stages['sequence_pass1'] = [t.to_dict() for t in sequence.selected]
            # Gaps between selected strong anchors may already contain a rescued
            # pool track. Log that reuse before allowing any weak-image search.
            anchors = [t for t in sequence.selected if t.candidate_class == "strong"]
            hypotheses = recovery_hypotheses(anchors, local, cfg, regularity)
            auxiliary = None
            if cfg.positive_aux_weight > 0:
                auxiliary, _ = detect_strip_peaks(
                    data[opposite]["profiles"], period, cfg, bounds["y1"],
                    data[opposite]["native_profiles"], data[opposite]["scale_sources"])
                auxiliary = filter_mask_peaks(auxiliary, raw_mask)
            added, failures, reuse_attempts, weak_attempts = recover_sequence_candidates(
                hypotheses, candidates, sequence.selected, pool, valid, cfg, auxiliary,
                raw_mask=raw_mask, negative_evidence=chosen == "negative", regularity=regularity,
                primary_evidence=True)
            if cfg.evidence_model:
                attach_body_evidence(added,body_profiles,bounds['y1'],local.at,chosen,cfg)
                if signal_model is not None:signal_model.protect_observed_transitions(added)
                absent=[t for t in added if t.local_evidence['paper_body']['state']=='absent']
                for t in absent:
                    t.reason='measured_non_paper_body';t.candidate_class='garbage'
                failures.extend(absent)
                added=[t for t in added if t.local_evidence['paper_body']['state']!='absent']
            pool, more_duplicates = deduplicate_obvious_tracks(pool + added)
            sequence = optimize_vertical_sequence(pool, local, hypothesis_cfg, regularity)
            tracks, recovered = sequence.selected, sequence.rescued
            rejected = garbage + duplicates + more_duplicates + failures + sequence.suppressed
            attempts = reuse_attempts + weak_attempts
            remaining_gaps = spacing_anomalies(tracks, local, cfg, regularity)
            for attempt in reuse_attempts:
                attempt["final_selected_track_ids"] = [t.track_id for t in tracks if t.track_id in attempt["candidate_track_ids"]]
            for attempt in weak_attempts:
                attempt["final_selected_track_ids"] = [t.track_id for t in tracks if t.track_id in attempt["accepted_candidate_ids"]]
        else:
            tracks, rejected = list(strong), list(garbage) + list(marginal)
            tracks, duplicates = deduplicate_tracks(tracks, lambda y: period, cfg)
            rejected.extend(duplicates)
            local = estimate_local_period(tracks, period, cfg)
            tracks, duplicates = deduplicate_tracks(tracks, local.at, cfg)
            rejected.extend(duplicates)
            local = estimate_local_period(tracks, period, cfg)
            hypotheses = missing_edge_hypotheses(tracks, local, cfg)
            auxiliary = None
            if cfg.positive_aux_weight > 0:
                auxiliary, _ = detect_strip_peaks(data[opposite]["profiles"], period, cfg, bounds["y1"],
                    data[opposite]["native_profiles"], data[opposite]["scale_sources"])
            recovery_candidates = [[p for p in strip if raw_mask.state_at(p.y) == StackState.CORE or
                (cfg.margin_allow_weak_recovery and raw_mask.state_at(p.y) == StackState.MARGIN)] for strip in candidates]
            recovered, failures, attempts = recover_weak_tracks(hypotheses, recovery_candidates, tracks, valid, cfg, auxiliary)
            rejected.extend(failures)
            tracks = sorted(tracks + recovered, key=lambda t: t.center_y)
            local = estimate_local_period(tracks, period, cfg)
    evaluation_stages['after_recovery'] = [t.to_dict() for t in tracks]
    # 6. 用独立 RAW 纸身补充主序列，再修复端部关联和孤立的实测纸身。
    if cfg.strip_sequence_enabled and cfg.strip_sequence_mode != "replace":
        from .strip_sequence import augment_observed_tracks
        if cfg.strip_sequence_mode == "augment":
            if cfg.evidence_model:
                from .body_evidence import attach_body_evidence
                from .stripe_evidence import stripe_proposals, fuse_body_proposals, select_body_sequence, deduplicate_body_identities
                proposals=signal_proposals if signal_model is not None else []
                if signal_model is None:stripe_debug={'reason':'no_measured_scale','proposals':0}
                attach_body_evidence(strip_tracks,body_profiles,bounds['y1'],lambda y:period,chosen,cfg)
                if signal_model is not None:
                    from .stripe_evidence import reconcile_signal_regions
                    tracks,body_competitors,augmentation=reconcile_signal_regions(tracks,proposals,signal_model,cfg,regularity,pool)
                    body_selection=augmentation
                    body_decisions=augmentation['decisions'];body_identity_decisions=augmentation['identity_duplicates']
                else:
                    body_competitors=[];body_selection={'selected_ids':[t.track_id for t in tracks],
                        'suppressed_ids':[],'reason':'no_new_body_identity_keeps_existing_sequence'}
                    body_decisions=[];body_identity_decisions=[]
                rejected.extend(body_competitors)
                evaluation_stages['body_sequence']=[t.to_dict() for t in tracks]
                augmentation={**locals().get('augmentation',{}),'mode':'augment','added':sum(t.track_id.startswith('body_') for t in tracks),
                              'decisions':body_decisions,'identity_duplicates':body_identity_decisions,
                              'stripe_evidence':stripe_debug,'joint_selection':body_selection}
            else:
                tracks, augmentation = augment_observed_tracks(tracks, strip_tracks, cfg, chosen)
            sequence.selected = list(tracks)
            sequence.rescued.extend(t for t in tracks if t.local_evidence.get('augmentation'))
            pool = pool + [t for t in tracks if t.local_evidence.get('augmentation')]
            remaining_gaps = spacing_anomalies(tracks, local, cfg, locals().get('regularity')) if local else []
            strip_consensus_debug['augmentation'] = augmentation
        else:
            strip_consensus_debug['augmentation'] = {'mode': 'shadow', 'added': 0}
        strip_consensus_debug['fusion_mode'] = cfg.strip_sequence_mode
    if cfg.composite_layers_enabled and period and orientation_fields is not None:
        from .composite_layers import CompositeLayerModel
        composite_model=CompositeLayerModel(gray,strips,orientation_fields,bounds['y1'],period,local,identity_cfg)
        tracks,composite_debug=composite_model.reconcile(tracks)
        evaluation_stages['composite_structure']=[t.to_dict() for t in tracks]
    if cfg.shared_raw_body_dedup_enabled and cfg.strip_sequence_enabled:
        from .strip_sequence import suppress_shared_raw_bodies
        tracks, body_rejected, body_dedup_debug = suppress_shared_raw_bodies(
            tracks,strip_consensus_debug,raw_gradient_profiles,bounds['y1'],chosen,cfg)
        rejected.extend(body_rejected)
        sequence.selected=list(tracks)
    outer_component_debug = {'enabled': cfg.isolated_raw_component_enabled}
    from .endpoint_association import repair_endpoint_association
    tracks, endpoint_replaced, endpoint_association_debug = repair_endpoint_association(
        tracks,candidates,raw_intensity_profiles*(255/np.iinfo(gray.dtype).max),bounds['y1'],chosen,period,cfg)
    rejected.extend(endpoint_replaced)
    if endpoint_association_debug['decisions']:
        evaluation_stages['after_endpoint_reassociation']=[t.to_dict() for t in tracks]
    if cfg.isolated_raw_component_enabled and period:
        from .isolated_component import recover_isolated_outer_components
        tracks, component_replaced, component_details = recover_isolated_outer_components(
            gray, strips, tracks, period, bounds['y1'], chosen, cfg)
        rejected.extend(component_replaced)
        outer_component_debug.update(component_details)
        evaluation_stages['after_outer_component'] = [t.to_dict() for t in tracks]
    positions = [int(round(t.center_y)) for t in tracks]
    def compatibility_peaks(items: list) -> list[PeakCandidate]:
        return [PeakCandidate(int(round(t.center_y)), float(np.median(list(t.strip_peak_strengths.values()))),
                float(np.median([p.height for p in t.peak_candidates])),
                float(np.median([p.width for p in t.peak_candidates])), t.support_ratio,
                "repaired" if t.source == "recovered" or t.sequence_decision == "rescued" else "detected", t.reason) for t in items]
    detected = compatibility_peaks([t for t in tracks if t.source == "detected" and t.sequence_decision != "rescued"])
    repaired = compatibility_peaks(recovered)
    rejected_peaks = compatibility_peaks(rejected)
    regularity = locals().get("regularity", None)
    confidence, confidence_parts = compute_track_confidence(tracks, local, cfg)
    if cfg.vertical_sequence_enabled and not cfg.sequence_period_multiple_mode:
        from .confidence import compute_geometry_confidence
        confidence, confidence_parts = compute_geometry_confidence(tracks, sequence, cfg, regularity)
    selected_pairs = [(transition_score(a, b, local, cfg, regularity)
                      or geometry_pair_score(a, b, local, cfg)) for a, b in zip(tracks, tracks[1:])] if local else []
    competitions = []
    if local and len(tracks) >= 3:
        for a, b, c in zip(tracks, tracks[1:], tracks[2:]):
            competitions.append(evaluate_middle_track_competition(
                a, b, c, local.at(b.center_y), cfg,
                regularity.at(b.center_y) if regularity is not None else 0.5))
    layer_gaps = [p["gap_median"] for p in selected_pairs]
    spacing_warnings = []
    if local is not None:
        for left, right in zip(tracks[:-1], tracks[1:]):
            expected = local.at((left.center_y + right.center_y) / 2)
            gap = right.center_y - left.center_y
            if abs(gap - expected) / expected > cfg.missing_gap_tolerance:
                spacing_warnings.append({"left": left.center_y, "right": right.center_y,
                    "gap": gap, "local_period": expected, "reason": "unexplained_spacing",
                    "action": "review_only_no_period_based_deletion"})
    if cfg.vertical_sequence_enabled:
        spacing_warnings = remaining_gaps
    if cfg.evidence_model:
        from .body_evidence import attach_body_evidence
        attach_body_evidence(tracks,body_profiles,bounds['y1'],local.at if local else lambda y:period,chosen,cfg)
        if signal_model is not None:
            signal_model.protect_observed_transitions(tracks)
            signal_model.annotate(tracks)
            from .stripe_evidence import deduplicate_body_identities
            tracks,signal_duplicates,_=deduplicate_body_identities(tracks,chosen,signal_model)
            rejected.extend(signal_duplicates)
            evaluation_stages['after_signal_identity_dedup']=[t.to_dict() for t in tracks]
    pre_verifier_tracks = list(tracks)
    evaluation_stages['pre_verifier'] = [t.to_dict() for t in tracks]
    identity_features = {}
    identity_bundle = {"debug": {"template": {"status": "disabled"}, "features": []},
                       "artifacts": {}}
    verifier_debug = {"enabled": False, "shadow_mode": cfg.final_track_verifier_shadow_mode,
                      "count_before": len(tracks), "count_after": len(tracks),
                      "proposed_reject_count": 0, "applied_reject_count": 0,
                      "proposed_reject_track_ids": [], "applied_reject_track_ids": [],
                      "track_decisions": [], "recovery_after_verifier": False}
    verifier_rejected = []
    if cfg.strip_sequence_enabled and cfg.strip_sequence_raw_body_safeguard_enabled:
        from .strip_sequence import attach_raw_body_confirmation
        attach_raw_body_confirmation(tracks, strip_consensus_debug, chosen, cfg)
    if cfg.final_track_verifier_enabled:
        # 7. 身份终判：按各轨迹实际极性对齐模板，结合纸身、暗缝和上下文裁决。
        if cfg.strip_sequence_dual_enabled and any(t.boundary_polarity and t.boundary_polarity != chosen for t in tracks):
            bundles = {}
            for polarity in (chosen, opposite):
                group = [t for t in tracks if (t.boundary_polarity or chosen) == polarity]
                if not group:
                    continue
                polarity_cfg_dict = identity_cfg.to_dict()
                polarity_cfg_dict['gradient_polarity'] = polarity
                polarity_cfg = CounterConfig(**polarity_cfg_dict)
                features, bundle = analyze_track_identity(
                    group, raw_intensity_profiles, normalized_intensity_profiles,
                    data["negative"]["profiles"], data["positive"]["profiles"],
                    bounds["y1"], local, period, regularity, raw_mask, competitions,
                    remaining_gaps, polarity_cfg)
                identity_features.update(features)
                bundles[polarity] = bundle
            identity_bundle = bundles[chosen] if chosen in bundles else bundles[opposite]
            identity_bundle["debug"] = {**identity_bundle["debug"],
                "per_polarity": {p: b["debug"] for p, b in bundles.items()},
                "features": [f for b in bundles.values() for f in b["debug"]["features"]],
                "mixed_boundary_polarities": True}
        else:
            identity_features, identity_bundle = analyze_track_identity(
                tracks, raw_intensity_profiles, normalized_intensity_profiles,
                data["negative"]["profiles"], data["positive"]["profiles"],
                bounds["y1"], local, period or (cfg.min_period + cfg.max_period) / 2,
                regularity, raw_mask, competitions, remaining_gaps, identity_cfg,
            )
        if cfg.compressed_layers_enabled:
            from .compressed_layers import prepare_compressed_candidates,recheck_compressed_identity
            additions,compressed_decisions=prepare_compressed_candidates(tracks,locals().get('compressed_tracks',[]),identity_cfg,
                strips,orientation_fields,bounds['y1'])
            compressed_rechecks=[recheck_compressed_identity(t,identity_features[t.track_id],tracks,identity_cfg)
                for t in tracks if t.local_evidence.get('compressed_layers',{}).get('verified')]
            if additions:
                # 原轨迹的身份特征冻结；新弱纹不能改变原模板或删除已选纸边。
                proposal_features,proposal_bundle=analyze_track_identity(
                    tracks+additions,raw_intensity_profiles,normalized_intensity_profiles,
                    data['negative']['profiles'],data['positive']['profiles'],bounds['y1'],local,
                    period,regularity,raw_mask,competitions,remaining_gaps,identity_cfg)
                identity_features.update({t.track_id:proposal_features[t.track_id] for t in additions})
                identity_bundle['debug']['features'].extend(proposal_features[t.track_id].to_dict() for t in additions)
                if 'profiles' in identity_bundle['artifacts']:
                    identity_bundle['artifacts']['profiles'].update({t.track_id:proposal_bundle['artifacts']['profiles'][t.track_id] for t in additions})
                tracks+=additions
                pre_verifier_tracks=list(tracks)
                evaluation_stages['compressed_candidates']=[t.to_dict() for t in additions]
                evaluation_stages['pre_verifier']=[t.to_dict() for t in tracks]
        tracks, verifier_rejected, verifier_debug = apply_final_track_verifier(
            tracks, identity_features, cfg)
        rejected.extend(verifier_rejected)
        tracks, guarded_rejected, _ = apply_raw_observation_guard(
            tracks, identity_features, bounds, cfg, verifier_debug)
        rejected.extend(guarded_rejected)
        from .background_region_guard import apply_background_region_guard
        tracks,background_rejected,_=apply_background_region_guard(tracks,identity_features,cfg,verifier_debug)
        rejected.extend(background_rejected)
    from .track_order_guard import apply_track_order_guard
    tracks, order_rejected, order_debug = apply_track_order_guard(tracks, chosen, cfg)
    rejected.extend(order_rejected)
    if order_rejected:
        verifier_debug['count_after']=len(tracks)
        ids={t.track_id for t in order_rejected}
        for key in ('proposed_reject_track_ids','applied_reject_track_ids'):
            verifier_debug.setdefault(key,[]).extend(sorted(ids-set(verifier_debug.get(key,[]))))
        verifier_debug['proposed_reject_count']=len(verifier_debug['proposed_reject_track_ids'])
        verifier_debug['applied_reject_count']=len(verifier_debug['applied_reject_track_ids'])
        for row in verifier_debug.get('track_decisions',[]):
            if row['track_id'] in ids:
                removed=next(t for t in order_rejected if t.track_id==row['track_id'])
                row['terminal_decision']=('REJECT_DUPLICATE_PHYSICAL_BODY' if removed.reason=='auxiliary_verified_pair_same_physical_body'
                                          else 'REJECT_CROSS_LAYER_AUXILIARY')
                row['applied']=True
    from .track_postprocessing import apply_horizontal_gap_postprocessing
    tracks, post_rejected, post_debug = apply_horizontal_gap_postprocessing(tracks, period, cfg)
    rejected.extend(post_rejected)
    if post_rejected:
        verifier_debug['count_after'] = len(tracks)
    # No recovery is called after identity verification.  The optional
    # horizontal duplicate cleanup below is terminal and only removes a
    # near-identical pair with shared real observations.
    evaluation_stages['final'] = [t.to_dict() for t in tracks]
    # 最终计数只由这份存活 tracks 决定；前面保存的阶段快照用于定位增删来源。
    positions = [int(round(t.center_y)) for t in tracks]
    recovered = [t for t in tracks if t.source == "recovered" or t.sequence_decision == "rescued"]
    detected = compatibility_peaks([t for t in tracks if t.source == "detected" and t.sequence_decision != "rescued"])
    repaired = compatibility_peaks(recovered)
    rejected_peaks = compatibility_peaks(rejected)
    if cfg.evidence_model:
        # Synchronize confidence with terminal survivors, not an earlier path.
        sequence.selected=list(tracks)
        sequence.rescued=list({t.track_id:t for t in recovered}.values())
        sequence.transitions=[edge for a,b in zip(tracks,tracks[1:])
                              if (edge:=transition_score(a,b,local,cfg,regularity)) is not None] if local else []
        for t in tracks:
            state=t.local_evidence.get('paper_body',{}).get('state','unknown')
            if t.local_evidence.get('isolated_raw_component',{}).get('verified'):state='confirmed'
            if cfg.composite_layers_enabled and t.local_evidence.get('composite_structure',{}).get('verified'):
                state='confirmed'
                t.local_evidence['identity_evidence_kind']='measured_composite_structure'
            t.local_evidence['identity_state']=state
    confidence, confidence_parts = compute_track_confidence(tracks, local, cfg)
    if cfg.vertical_sequence_enabled and not cfg.sequence_period_multiple_mode:
        from .confidence import compute_geometry_confidence
        confidence, confidence_parts = compute_geometry_confidence(tracks, sequence, cfg, regularity)
    debug = {"algorithm": "multi_strip_tracks", "config": cfg.to_dict(), "image_shape": list(image.shape),
        "image_dtype": str(image.dtype), "roi": bounds, "coordinate_system": "original image; x2/y2 exclusive",
        "requested_roi": requested_bounds, "active_stack_roi": bounds,
        "active_roi_diagnostics": {k: v for k, v in active_info.items() if k != "energy"},
        "raw_stack_mask": raw_mask.to_dict(), "mask_refinement": locals().get("mask_refinement", {}),
        "preprocessing": preprocessing_info,
        "foreground_invalid_overrides": foreground_invalid_overrides,
        "raw_mask_invalid_peaks_removed": invalid_peak_count,
        "invalid_selected_track_count": sum(any(raw_mask.state_at(y) == StackState.INVALID for y in t.strip_positions.values()) for t in tracks),
        "margin_selected_track_ids": [t.track_id for t in tracks if t.raw_mask_state == "MARGIN"],
        "adaptive_structure_analysis": structure_debug,
        "sequence_model": "independent_strip_ordered_fusion" if cfg.strip_sequence_enabled and cfg.strip_sequence_mode == "replace" else
                          "period_multiple_comparison" if cfg.sequence_period_multiple_mode else "geometry_ordered_layers",
        "strip_consensus": strip_consensus_debug,
        "outer_component": outer_component_debug,
        "endpoint_reassociation": endpoint_association_debug,
        "auxiliary_order_guard": order_debug,
        "horizontal_gap_postprocessing": post_debug,
        "raw_period_consensus": raw_period_debug,
        "ordered_association": locals().get('ordered_association_debug',{'enabled': False}),
        "shared_raw_body_dedup": locals().get('body_dedup_debug',{'enabled':False}),
        "hybrid_mode": "regular_first_geometry_exception" if cfg.regularity_enabled else "geometry_only",
        "regularity": regularity.to_dict() if regularity is not None else {},
        "initial_spacing_scale": period, "local_spacing_scale": local.to_samples() if local else [],
        "selected_neighbor_gap_profiles": selected_pairs,
        "secondary_texture_competitions": competitions,
        "track_identity": identity_bundle["debug"],
        "pre_verifier_sequence": {"count": len(pre_verifier_tracks),
                                  "track_ids": [t.track_id for t in pre_verifier_tracks]},
        "post_verifier_sequence": {"count": len(tracks),
                                   "track_ids": [t.track_id for t in tracks]},
        "final_track_verifier": verifier_debug,
        "evaluation_stages": evaluation_stages,
        "selected_neighbor_gap_distribution": {"median": float(np.median(layer_gaps)),
            "mad": float(np.median(np.abs(np.array(layer_gaps) - np.median(layer_gaps)))),
            "q10": float(np.percentile(layer_gaps, 10)), "q90": float(np.percentile(layer_gaps, 90))} if layer_gaps else {},
        "short_gap_suppression_count": sum(t.reason == "sequence_suppressed_short_gap" for t in sequence.suppressed),
        "large_gap_anomaly_count": sum(p["large_gap_anomaly"] for p in selected_pairs),
        "evidence_based_recovery_count": len(sequence.rescued),
        "gradient_polarity": chosen, "primary_evidence_polarity": chosen,
        "auxiliary_evidence_polarity": opposite,
        "counted_boundary": "per_track_actual_polarity" if cfg.strip_sequence_dual_enabled else
                            "paper_top" if chosen == "positive" else "paper_bottom",
        "recovery_evidence_policy": "unnormalized_raw_observation_required" if cfg.strip_sequence_enabled else "native_selected_polarity_required",
        "period": period, "initial_global_period": period,
        "initial_period_source": period_source, "strip_period_estimates": strip_periods,
        "median_local_period": float(np.median(local.values)) if local is not None and len(local.values) else None,
        "local_period_samples": local.to_samples() if local is not None else [],
        "missing_edge_hypotheses": hypotheses, "recovery_attempts": attempts,
        "exception_regions": hypotheses,
        "regular_missing_hypotheses": [h for h in hypotheses if h.get("regular_period_hint")],
        "regular_period_assisted_recoveries": [a for a in attempts if a.get("recovery_mode") == "period_assisted" or a.get("reason") == "recovery_observed_candidate" and a.get("regularity", 0) >= cfg.regularity_regular_threshold],
        "geometry_only_recoveries": [a for a in attempts if a.get("recovery_mode") == "geometry_only"],
        "unresolved_regular_gaps": ([g for g in remaining_gaps if g.get("regularity", 0) >= cfg.regularity_regular_threshold and not g.get("search", False)]
            + [a for a in attempts if a.get("regularity", 0) >= cfg.regularity_regular_threshold
               and a.get("reason") in {"recovery_no_image_evidence", "recovery_low_support", "recovery_search_window_miss"}]),
        "spacing_warnings": spacing_warnings,
        "composite_structure": locals().get('composite_debug',{'enabled':False}),
        "track_candidate_pool": [t.to_dict() for t in (pool if cfg.vertical_sequence_enabled else strong + marginal)],
        "strong_tracks": [t.to_dict() for t in strong],
        "marginal_tracks": [t.to_dict() for t in marginal],
        "garbage_tracks": [t.to_dict() for t in garbage],
        "vertical_sequence": sequence.to_dict(),
        "sequence_suppressed_tracks": [t.to_dict() for t in sequence.suppressed],
        "sequence_rescued_tracks": [t.to_dict() for t in sequence.rescued],
        "remaining_suspicious_gaps": remaining_gaps,
        "candidate_reuse_recovery_attempts": reuse_attempts,
        "weak_image_recovery_attempts": weak_attempts,
        "mean_peak_spacing": float(np.mean(np.diff(positions))) if len(positions) > 1 else None,
        "polarity_scores": {p: {k: v for k, v in d["assessment"].items() if k != "acf"} for p, d in data.items()},
        "strip_bounds_roi_x": strips, "strip_stats": {p: d["strip_stats"] for p, d in data.items()},
        "front_end": {"strip_aggregation_mode": cfg.strip_aggregation_mode,
            "strip_width_ratio": cfg.strip_width_ratio,
            "directional_smoothing_enabled": cfg.directional_smoothing_enabled,
            "directional_sigma_x": cfg.directional_sigma_x,
            "directional_sigma_y": cfg.directional_sigma_y,
            "multiscale_gradient_enabled": cfg.multiscale_gradient_enabled,
            "enhanced_period_from_native": cfg.enhanced_period_from_native,
            "period_evidence": period_source,
            "gradient_scales": cfg.gradient_scales,
            "scale_source": "fixed_config_not_period",
            "cross_scale_deduplication": "raw_peak_identity_after_independent_scale_detection" if cfg.strip_sequence_enabled else "pointwise_max_before_peak_detection",
            "enhanced_only_observations_count_as_tracks": False,
            "gradient_levels": {p: [{"scale": level["scale"], "source": level["source"],
                                      "aggregation": level["aggregation"]}
                                     for level in d["gradient_levels"]] for p, d in data.items()}},
        "prominence_threshold": None, "threshold_mode": "per_strip_local_adaptive",
        "strip_peak_candidates": [[p.to_dict() for p in strip] for strip in candidates],
        "strong_strip_candidate_count": sum(p.strength_type == "strong" and not p.reason for strip in candidates for p in strip),
        "weak_strip_candidate_count": sum(p.strength_type == "weak" and not p.reason for strip in candidates for p in strip),
        "enhanced_only_proposal_count": sum(p.reason == "enhanced_only_proposal" for strip in candidates for p in strip),
        "raw_seed_track_count": len(raw_tracks),
        "raw_recovery_track_count": sum(a["raw_track_count"] for a in attempts),
        "raw_track_count": len(raw_tracks) + sum(a["raw_track_count"] for a in attempts),
        "raw_candidate_count": len(raw_tracks) + sum(a["raw_track_count"] for a in attempts),
        "rejected_tracks": [t.to_dict() for t in rejected], "paper_edge_tracks": [t.to_dict() for t in tracks],
        "count": len(tracks), "peak_positions": positions, "edge_positions": [t.center_y for t in tracks],
        "detected_peaks": [p.to_dict() for p in detected], "repaired_peaks": [p.to_dict() for p in repaired],
        "rejected_peaks": [p.to_dict() for p in rejected_peaks],
        "confidence": confidence, "confidence_components": confidence_parts, "full_resolution_detection": True,
        "status": "review_required" if verifier_debug.get('raw_body_conflicts_retained') else
                  "ok" if len(tracks) >= 3 else "insufficient_evidence",
        "warnings": (["RAW body evidence conflicts with template rejection; retained tracks require review."]
                     if verifier_debug.get('raw_body_conflicts_retained') else [])
                    + ([] if period else ["No reliable period; no validated tracks. Count zero is not proof of no paper."])}
    if cfg.evidence_model:
        confirmed=sum(t.local_evidence.get('identity_state')=='confirmed' for t in tracks)
        uncertain=len(tracks)-confirmed
        debug['count_breakdown']={'confirmed':confirmed,'uncertain':uncertain,
            'period_assisted_observed':sum(t.local_evidence.get('period_assisted',False) for t in tracks),
            'estimated_total':len(tracks),'interpretation':'uncertain identities are explicitly included in the estimate'}
        if cfg.composite_layers_enabled:
            debug['count_breakdown']['composite_structure_supported']=sum(
                t.local_evidence.get('composite_structure',{}).get('verified',False) for t in tracks)
            debug['count_breakdown']['bright_band_supported']=sum(
                t.local_evidence.get('paper_body',{}).get('verified',False) for t in tracks)
            debug['count_breakdown']['interpretation']='measured local structure support is not count accuracy; bright-band and composite supports may overlap'
        if uncertain:
            debug['status']='review_required'
            debug['warnings'].append(f'{uncertain} counted tracks have unresolved paper-body identity.')
        debug['evidence_model']='paper_layers_v4'
        if cfg.compressed_layers_enabled:
            debug['compressed_layers']=locals().get('compressed_debug',{'enabled':True,'reason':'no_measured_scale'})
            debug['compressed_layers']['allocation']=locals().get('compressed_decisions',[])
            debug['compressed_layers']['existing_rechecks']=locals().get('compressed_rechecks',[])
            debug['compressed_layers']['counts']={
                'found':debug['compressed_layers'].get('proposals',0),
                'sent_to_terminal':len(locals().get('additions',[])),
                'retained':sum(t.track_id.startswith('compressed_') for t in tracks),
                'existing_rechecked':sum(r['used_local_identity'] for r in locals().get('compressed_rechecks',[]))}
        debug['signal_gap_model']=signal_model.to_dict() if signal_model is not None else {'reason':'no_measured_scale'}
        debug['signal_gap_checks']=signal_model.diagnostics(tracks) if signal_model is not None else []
        unresolved=[p for p in debug['signal_gap_checks'] if p['state'] in
            {'observed_bands_inside_gap','observed_transitions_inside_gap','unresolved_large_gap','crowded_unresolved','same_measured_body','unresolved_band_identity'}]
        debug['unresolved_signal_gap_count']=len(unresolved)
        if unresolved:
            debug['status']='review_required'
            debug['warnings'].append(f'{len(unresolved)} RAW signal gaps remain unresolved; count requires review.')
    return PaperCountResult(len(tracks), positions, detected, repaired, rejected_peaks, period, chosen, confidence,
        debug, {"roi": roi, "previews": previews, "polarities": data,
                "composite_model": locals().get('composite_model'),
        "raw_stack_mask": raw_mask, "raw_gray_preview": raw_preview,
                "track_identity": identity_bundle["artifacts"],
                "strip_candidates": candidates, "local_statistics": statistics, "local_period": local,
                "strip_quality": strip_quality,
                "strip_raw_intensity_profiles": raw_intensity_profiles*(255/np.iinfo(gray.dtype).max) if cfg.strip_sequence_enabled else None,
                "strip_raw_gradient_profiles": raw_gradient_profiles,
                "active_energy": active_info.get("energy"), "active_energy_y_offset": requested_bounds["y1"]}, tracks, rejected)
