"""Crop selection and local tangent alignment shared by regions geometry."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def center_window(image, width, geometry):
    """Crop about the prepared stack center and compose its original mapping.

    With adaptive strip selection enabled, finds the most horizontal region
    instead of using fixed center crop.
    """
    from .adaptive_strip_selection import find_best_horizontal_strip

    h, w = image.shape[:2]

    # Convert to grayscale if needed for analysis
    if len(image.shape) == 3:
        from .preprocessing import to_gray
        gray_for_analysis = to_gray(image)
    else:
        gray_for_analysis = image

    # Explicitly disabled geometry keeps the API mapping deterministic.  The
    # published regions preset enables geometry and may select a more
    # horizontal strip; custom configs with auto geometry off use the center
    # crop expected by the stable protocol.
    adaptive = bool(geometry.get('enabled', False))
    try:
        if not adaptive:
            raise RuntimeError('adaptive geometry disabled')
        (x_start, x_end), analysis = find_best_horizontal_strip(gray_for_analysis, width, sample_count=8)
        crop_info = {
            'adaptive_enabled': True,
            'method': 'adaptive_horizontal',
            'x_range': (int(x_start), int(x_end)),
            'original_shape': (h, w),
            'target_width': width,
            'analysis': analysis
        }
    except Exception as e:
        # Fallback to center crop
        import traceback
        actual = min(width, w)
        x_start = (w - actual) // 2
        x_end = x_start + actual
        crop_info = {
            'adaptive_enabled': False,
            'method': 'center_crop_fallback',
            'x_range': (int(x_start), int(x_end)),
            'error': str(e),
            'traceback': traceback.format_exc()
        }

    # Get actual crop coordinates
    x_start, x_end = crop_info['x_range']
    actual = x_end - x_start

    # Compose transformation matrix
    crop_matrix = np.array([[1., 0, -x_start], [0, 1., 0], [0, 0, 1.]])
    forward = crop_matrix @ np.asarray(geometry.get('input_to_detection', np.eye(3)))

    info = dict(geometry,
                input_to_detection=forward.tolist(),
                detection_to_input=np.linalg.inv(forward).tolist(),
                center_window=dict(
                    x1=x_start, x2=x_end, width=actual,
                    requested_width=width,
                    coordinate_system='prepared stack',
                    adaptive_selection=crop_info
                ),
                output_shape=list(image[:, x_start:x_end].shape))

    return image[:, x_start:x_end].copy(), info


def align_columns(gray):
    """Integrate a smooth local tangent from the center column, preserving order."""
    raw=gray.astype(np.float32)
    gx=cv2.Sobel(raw,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(raw,cv2.CV_32F,0,1,ksize=3)
    xx=gaussian_filter(gx*gx,(6,9)); yy=gaussian_filter(gy*gy,(6,9))
    xy=gaussian_filter(gx*gy,(6,9))
    coherence=np.sqrt((xx-yy)**2+4*xy*xy)/np.maximum(xx+yy,1e-6)
    slopes=np.clip(-xy/np.maximum(yy,1e-6),-.6,.6)
    slopes[coherence<.3]=0
    h,w=gray.shape; rows=np.arange(h,dtype=np.float32)
    mapping=np.empty((h,w),np.float32); center=w//2; mapping[:,center]=rows
    for direction,indices in ((1,range(center+1,w)),(-1,range(center-1,-1,-1))):
        current=rows.copy()
        for x in indices:
            field=slopes[:,x-direction]
            candidate=current+direction*np.interp(current,rows,field).astype(np.float32)
            # Never allow neighboring layer coordinates to fold or cross.
            if np.min(np.diff(candidate))<.1:
                candidate=current
            current=candidate; mapping[:,x]=current
    xs=np.broadcast_to(np.arange(w,dtype=np.float32),(h,w)).copy()
    valid=(mapping>=0)&(mapping<=h-1)
    return mapping,xs,valid

