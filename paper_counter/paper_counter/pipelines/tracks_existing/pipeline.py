"""Preserve the existing track algorithms behind an explicit pipeline."""
from dataclasses import replace
from ...contracts import PreparedFrame,PipelineExecution
from ...paper_geometry import prepare_geometry
from ...shared.preparation import foreground_envelope_bounds


class ExistingTracksPipeline:
    pipeline_id='tracks_existing_v1'

    def run(self,frame,recipe):
        from ...track_counter import process_tracks
        cfg=recipe.to_counter_config()
        if cfg.auto_geometry_enabled:
            image,geometry=prepare_geometry(frame.image,cfg.geometry_target_spacing,
                bounded_angle_analysis=False,auto_quarter_turn=False,center_width=None,
                preserve_extent=cfg.tracks_geometry_preserve_extent,stack_locator=foreground_envelope_bounds)
            prepared=PreparedFrame(image,geometry)
            result=process_tracks(image,replace(cfg,auto_geometry_enabled=False))
            result.debug_info['geometry_preprocessing']=geometry
            result.debug_info['input_original_shape']=list(frame.image.shape)
            result.debug_info['config']=cfg.to_dict()
            result.debug_info['coordinate_system']='prepared detection image; use geometry affine for input coordinates'
        else:
            prepared=PreparedFrame(frame.image)
            result=process_tracks(frame.image,cfg)
        return PipelineExecution(result,prepared)
