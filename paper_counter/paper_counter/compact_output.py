"""Small production exports; full stage diagnostics remain explicitly available."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import json

from .visualization import save_overlay


def compact_track(track) -> dict:
    data = asdict(track) if is_dataclass(track) else dict(vars(track))
    fields = ('track_id', 'center_y', 'strip_positions', 'support_count', 'support_ratio',
              'continuity_ratio', 'total_score', 'source', 'sequence_decision',
              'boundary_polarity', 'reason')
    return {key: data[key] for key in fields if key in data}


def save_compact(result, output: str | Path, config) -> None:
    """Export the actual final tracks and coordinate mapping without repeated arrays."""
    from .exports import save_recipe_metadata
    save_recipe_metadata(result,output)
    if result.debug_info.get('algorithm')=='narrow_regions':
        from .region_output import save_regions
        return save_regions(result,output,config,full=False)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = save_overlay(result, directory, config)
    from .detection_coordinates import export_detection_coordinates
    export_detection_coordinates(result,directory,metadata)
    keys = ('algorithm', 'image_shape', 'image_dtype', 'roi', 'coordinate_system',
            'active_stack_roi', 'strip_bounds_roi_x', 'gradient_polarity', 'period',
            'median_local_period', 'count', 'detected_peaks', 'repaired_peaks', 'status',
              'warnings', 'confidence', 'count_breakdown', 'evidence_model', 'input_image', 'input_image_sha256',
            'input_original_shape', 'rotation_ccw_degrees', 'coordinate_mapping_to_original',
            'prediction_provenance', 'ground_truth', 'geometry_preprocessing', 'narrow_detection',
            'pipeline','resolved_recipe')
    data = {key: result.debug_info[key] for key in keys if key in result.debug_info}
    data.update(count=result.count, config=config.to_dict(), render_metadata=metadata,
                confidence_interpretation='uncalibrated consistency score; not a probability',
                export_mode='compact_final_result',
                paper_edge_tracks=[compact_track(t) for t in result.paper_edge_tracks],
                rejected_tracks=[compact_track(t) for t in result.rejected_tracks])
    if result.debug_info.get('compressed_layers'):
        data['compressed_layers']={k:v for k,v in result.debug_info['compressed_layers'].items()
            if k in {'enabled','counts','period_creates_positions','terminal_identity_exemption'}}
    # Retain measured endpoint proof without exporting full response arrays.
    if (config.isolated_raw_component_enabled or config.isolated_dark_track_guard_enabled
            or config.partial_endpoint_identity_enabled or config.auxiliary_order_guard_enabled):
        verifier=result.debug_info.get('final_track_verifier', {})
        data['endpoint_evidence'] = {
            'outer_component': result.debug_info.get('outer_component', {}),
            'raw_guard': verifier.get('raw_observation_guard', {}),
            'order_guard': result.debug_info.get('auxiliary_order_guard', {}),
            'endpoint_reassociation': result.debug_info.get('endpoint_reassociation', {}),
            'identity_overrides': [
                {'track_id':r['track_id'],'terminal_decision':r['terminal_decision'],
                 'partial_endpoint':r.get('features',{}).get('partial_endpoint')}
                for r in verifier.get('track_decisions',[])
                if r['terminal_decision'] in ['KEEP_MEASURED_PARTIAL_ENDPOINT','KEEP_MEASURED_FOREGROUND_ENDPOINT']]}
    (directory / 'debug_info.json').write_text(
        json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')),
        encoding='utf-8')
