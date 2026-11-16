"""Geometry composition shared by strategies; callers choose the ROI policy."""
import numpy as np
from ..contracts import PreparedFrame
from ..paper_geometry import prepare_geometry,stack_bounds
from ..geometry_alignment import center_window


def bright_core_bounds(gray):return stack_bounds(gray,strict=True)


def foreground_envelope_bounds(gray):return stack_bounds(gray,strict=False)


def prepare_region_image(image,cfg,*,stack_locator):
    if cfg.auto_geometry_enabled:
        prepared,geometry=prepare_geometry(image,cfg.geometry_target_spacing,
            bounded_angle_analysis=True,auto_quarter_turn=True,center_width=cfg.narrow_width,
            preserve_extent=False,stack_locator=stack_locator)
    else:
        prepared,geometry=image,dict(enabled=False,applied=False,input_shape=list(image.shape),
            input_to_detection=np.eye(3).tolist(),detection_to_input=np.eye(3).tolist())
    if not geometry.get('center_window'):
        prepared,geometry=center_window(prepared,cfg.narrow_width,geometry)
    return PreparedFrame(prepared,geometry)


def attach_region_geometry(result,prepared,input_image,cfg):
    result.debug_info['geometry_preprocessing']=prepared.geometry
    result.debug_info['input_original_shape']=list(input_image.shape)
    result.debug_info['config']=cfg.to_dict()
    result.debug_info['coordinate_system']='prepared detection image; use geometry affine for input coordinates'
    if 'region_mapping' in result.artifacts:
        prepared.nonlinear_maps['aligned']=result.artifacts['region_mapping']
