"""Optional stack deskew and sampling normalization before signal extraction.

Geometry is estimated from the image, never from a requested paper count.
The recorded affine transform maps pixel centers back to the supplied image.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_closing
from scipy.signal import find_peaks


def weighted_median(values, weights):
    order = np.argsort(values)
    return float(np.asarray(values)[order][np.searchsorted(
        np.cumsum(np.asarray(weights)[order]), np.sum(weights) / 2)])


def stack_bounds(gray, strict=False):
    """Use sustained wide foreground, excluding small fixtures and black margins.

    Regions use a bright core plus nearby measured dim edges. The historical
    permissive envelope remains available to the tracks geometry path.
    """
    h, w = gray.shape
    ratio = min(1., 1200 / max(h, w))
    small = cv2.resize(gray.astype(np.float32), None, fx=ratio, fy=ratio,
                       interpolation=cv2.INTER_AREA) if ratio < 1 else gray.astype(np.float32)
    maximum = float(np.percentile(small, 99))
    if maximum <= 0:
        return (0, 0, w, h), False

    if strict:
        # A low fraction of the brightest pixel includes sensor glow, a lit
        # fixture, and isolated reflections. Locate the sustained paper mass
        # from the two intensity populations before expanding its envelope.
        values = np.clip(small * (255. / max(maximum, 1.)), 0, 255).astype(np.uint8)
        threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground = values > threshold
        occupancy = np.mean(foreground, axis=1)
        rows = occupancy > .15
        width = max(3, int(round(small.shape[1] * .04)) | 1)
        rows = binary_closing(rows, structure=np.ones(width), border_value=0)
        changes = np.diff(np.r_[False, rows, False].astype(np.int8))
        intervals = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        if not intervals:
            return (0, 0, w, h), False
        masses = [float(occupancy[a:b].sum()) for a, b in intervals]
        dominant = max(masses)
        # Retain multiple substantial paper groups, but not a small bright
        # fixture separated from the stack by hundreds of background rows.
        keep = [(a, b) for (a, b), mass in zip(intervals, masses) if mass >= .25 * dominant]
        top, bottom = keep[0][0], keep[-1][1]
        # A dim outer sheet can fall below the bright-population threshold.
        # Extend only to nearby coherent gray edges, rather than growing into
        # the entire low-threshold glow component that touches the stack.
        low = values[values <= threshold]
        high = values[values > threshold]
        background = float(np.median(low)) if len(low) else 0.
        contrast = (float(np.median(high)) if len(high) else maximum) - background
        row_signal = np.percentile(values, 75, axis=1)
        edge = abs(gaussian_filter1d(row_signal.astype(float), .65, order=1))
        weak_rows = (edge > max(4., .035 * contrast)) & (row_signal > background + .1 * contrast)
        margin = max(width, int(round(.25 * (bottom-top))))
        nearby = np.flatnonzero(weak_rows[max(0, top-margin):min(len(rows), bottom+margin)]) + max(0, top-margin)
        if len(nearby):
            top, bottom = min(top, int(nearby[0])), max(bottom, int(nearby[-1])+1)
        columns = np.flatnonzero(np.mean(foreground[top:bottom], axis=0) > .12)
        if len(columns) < .15 * small.shape[1]:
            return (0, 0, w, h), False
        sx, sy = w / small.shape[1], h / small.shape[0]
        return (int(columns[0] * sx), int(top * sy), min(w, int(np.ceil((columns[-1]+1)*sx))),
                min(h, int(np.ceil(bottom*sy)))), True
    foreground = small > .12 * maximum
    rows = np.mean(foreground, axis=1) > .2
    rows = binary_closing(rows, structure=np.ones(max(1, round(len(rows) * .006))), border_value=0)
    indices = np.flatnonzero(rows)
    if len(indices) < 12:
        return (0, 0, w, h), False

    # Take full envelope - no edge trimming
    # Let post-validation handle any background boxes instead of risking cutting real papers
    top, bottom = indices[0], indices[-1] + 1

    columns = np.flatnonzero(np.mean(foreground[top:bottom], axis=0) > .25)
    if len(columns) < .2 * small.shape[1]:
        return (0, 0, w, h), False
    sx, sy = w / small.shape[1], h / small.shape[0]
    return (int(columns[0]*sx), int(top*sy), min(w, int((columns[-1]+1)*sx)),
            min(h, int(bottom*sy))), True


def estimate_angle(gray, bounded_analysis=False):
    h, w = gray.shape
    scale = min(1., 1400 / (max(h,w) if bounded_analysis else w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    maximum = np.iinfo(small.dtype).max
    image = np.clip(small.astype(np.float32) * (255. / maximum), 0, 255).astype(np.uint8)
    edges = cv2.Canny(image, 35, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/1800, threshold=max(20, round(small.shape[1]*.08)),
                            minLineLength=max(20, round(small.shape[1]*.16)),
                            maxLineGap=max(3, round(small.shape[1]*.025)))
    angles, lengths = [], []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            dx,dy = float(x2-x1),float(y2-y1)
            if dx == 0: continue
            angle = np.degrees(np.arctan(dy/dx))
            if abs(angle) <= 15:
                angles.append(angle); lengths.append(float(np.hypot(dx,dy)))
    if len(angles) < 4 or sum(lengths) < 2*small.shape[1]:
        return 0., {'reliable':False,'lines':len(angles),'reason':'insufficient_long_layer_edges'}
    angle = weighted_median(angles,lengths)
    mad = weighted_median(np.abs(np.asarray(angles)-angle),lengths)
    return (angle if mad <= 2 else 0.), {'reliable':bool(mad<=2),'lines':len(angles),
        'angle_mad_degrees':mad,'measured_angle_degrees':angle}


def rotation_matrix(w,h,angle):
    matrix = cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),angle,1.)
    # Transform the full pixel footprint, so no corners are clipped.
    corners = np.array([[-.5,-.5,1],[w-.5,-.5,1],[-.5,h-.5,1],[w-.5,h-.5,1]])
    mapped = corners @ matrix.T
    low, high = mapped.min(axis=0), mapped.max(axis=0)
    matrix[:,2] += -.5-low
    return np.vstack([matrix,[0,0,1]]), tuple(np.ceil(high-low).astype(int))


def measure_spacing(gray):
    """Estimate intensity repetition using narrow independent column views.

    This deliberately searches below the detector's 5 px period floor. It is
    a sampling estimate, not a decision about paper identity or multiplicity.
    """
    h,w = gray.shape
    periods, strengths = [], []
    for fraction in np.linspace(.15,.85,11):
        x = int((w-1)*fraction)
        signal = np.median(gray[:,max(0,x-2):min(w,x+3)].astype(np.float32),axis=1)
        signal -= gaussian_filter1d(signal,max(4.,h/120))
        energy = float(signal@signal)
        if energy <= 1e-6: continue
        length = min(100,h//5)
        size = 1 << (2*h-1).bit_length()
        spectrum = np.fft.rfft(signal,size)
        acf = np.fft.irfft(spectrum*np.conj(spectrum),size)[:length+1]/energy
        peaks,_ = find_peaks(acf,prominence=.06)
        peaks = peaks[(peaks>=2)&(acf[peaks]>=.12)]
        if not len(peaks): continue
        best = float(np.max(acf[peaks]))
        # Prefer a visible short repetition over its stronger double-period.
        # This sets sampling only; it never inserts a counted line.
        chosen = int(peaks[acf[peaks]>=.25*best][0])
        periods.append(chosen); strengths.append(float(acf[chosen]))
    if len(periods)<5:
        return None, {'reliable':False,'view_periods':periods}
    period = weighted_median(periods,strengths)
    agreement = sum(abs(p-period)<=max(1.,.2*period) for p in periods)
    reliable = agreement>=max(5,int(np.ceil(.6*len(periods))))
    return (period if reliable else None), {'reliable':bool(reliable),'view_periods':periods,
        'agreeing_views':agreement,'measured_spacing_px':period}


def estimate_quarter_turn(gray):
    """Distinguish vertical layer edges from horizontal ones using image content."""
    h,w=gray.shape;scale=min(1.,1000/max(h,w))
    sample=cv2.resize(gray.astype(np.float32),None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else gray.astype(np.float32)
    sample=cv2.GaussianBlur(sample,(0,0),.6)
    gx=cv2.Sobel(sample,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(sample,cv2.CV_32F,0,1,ksize=3)
    energy=gx*gx+gy*gy
    # Clip very strong fixtures/frame edges so many repeated layer edges win.
    cap=float(np.percentile(energy,95))
    weights=np.minimum(energy,cap)
    total=float(weights.sum())
    fraction=float(np.sum(weights*gx*gx/np.maximum(energy,1e-6))/total) if total>1e-6 else .5
    # Choose the dominant normal axis. The earlier 70% hard gate wrongly
    # treated strong paper texture as proof that vertical edges were absent.
    rotation=90 if fraction>.5 else 0
    reliable=bool(total>1e-6 and abs(fraction-.5)>.04)
    detail=dict(normal_x_energy_fraction=fraction,reliable=reliable,
                chosen_ccw_degrees=rotation,uses_image_aspect_ratio=False,method='gradient_normal_energy')
    return rotation,detail


def prepare_geometry(image, target_spacing=14., bounded_angle_analysis=False, auto_quarter_turn=False,
                     center_width=None, preserve_extent=False, stack_locator=None):
    from .preprocessing import to_gray
    gray = to_gray(image)
    input_shape=list(image.shape)
    coarse=np.eye(3);quarter=0;axis_info={'enabled':False}
    if auto_quarter_turn:
        quarter,axis_info=estimate_quarter_turn(gray)
        axis_info['enabled']=True
        if quarter:
            coarse=np.array([[0.,1.,0.],[-1.,0.,gray.shape[1]-1.],[0.,0.,1.]])
            image=np.rot90(image,1).copy();gray=np.rot90(gray,1).copy()
    h,w = gray.shape
    bounds, located = (stack_locator(gray) if stack_locator is not None
                       else stack_bounds(gray, strict=center_width is not None))
    x1,y1,x2,y2 = bounds
    if not located:
        return image, {'enabled':True,'applied':bool(quarter),'reason':'stack_not_located',
            'input_to_detection':coarse.tolist(),'detection_to_input':np.linalg.inv(coarse).tolist(),
            'input_shape':input_shape,'coarse_rotation_ccw_degrees':quarter,'axis_estimation':axis_info}
    # Estimate the layer angle on the detected envelope, but let tracks keep
    # the full image footprint.  The later active-stack ROI can then discard
    # background after rotation without clipping real outer sheets here.
    analysis_bounds = (x1, y1, x2, y2)
    angle, angle_info = estimate_angle(gray[y1:y2,x1:x2],bounded_angle_analysis)
    angle_info['bounded_analysis']=bounded_angle_analysis
    if center_width is not None:
        # Estimate the direction where detection will actually sample. A wider
        # analysis guard keeps tilted long sheets visible; it is not rendered.
        middle=(x1+x2)//2
        guard=min(x2-x1,max(2*center_width,int((y2-y1)*np.tan(np.deg2rad(4)))+center_width))
        left=max(x1,middle-guard//2);right=min(x2,left+guard)
        local_angle,local_info=estimate_angle(gray[y1:y2,left:right],True)
        angle_info=dict(global_estimate=angle_info,center_estimate=local_info,
            analysis_xyxy=[left,y1,right,y2],reliable=local_info['reliable'] or angle_info['reliable'],
            selected_source='center_guard' if local_info['reliable'] else 'global_fallback')
        if local_info['reliable']:angle=local_angle
    if abs(angle)<.25:
        angle = 0.  # Avoid resampling an already straight image for tiny jitter.
    # Keep real outer edges and enough surrounding background for the verifier.
    if preserve_extent:
        x1,y1,x2,y2 = 0, 0, w, h
    else:
        px,py = max(12,round((x2-x1)*.025)), max(12,round((y2-y1)*.04))
        x1,y1,x2,y2 = max(0,x1-px),max(0,y1-py),min(w,x2+px),min(h,y2+py)
    rotation, (rw,rh) = rotation_matrix(x2-x1,y2-y1,angle)
    rotated = cv2.warpAffine(gray[y1:y2,x1:x2],rotation[:2],(rw,rh),flags=cv2.INTER_LINEAR)
    spacing, spacing_info = measure_spacing(rotated)
    # Upsampling changes numerical sampling, not the optical information.
    scale = min(4., max(1.,target_spacing/spacing)) if spacing else 1.
    # Bound memory without silently shrinking a previously valid original.
    cap = max(1., np.sqrt(32_000_000 / (rw*rh)))
    scale = min(scale,cap)
    ow,oh = max(1,round(rw*scale)),max(1,round(rh*scale))
    sx,sy = ow/rw,oh/rh
    resize = np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    crop = np.array([[1.,0,-x1],[0,1.,-y1],[0,0,1.]])
    local_forward = resize @ rotation @ crop

    # First, create full prepared image without center crop
    forward_temp = local_forward @ coarse
    prepared_full = cv2.warpAffine(image,local_forward[:2],(ow,oh),flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT,borderValue=0)

    center_info=None
    if center_width is not None:
        # Apply adaptive strip selection on the prepared image
        from .adaptive_strip_selection import find_best_horizontal_strip
        from .preprocessing import to_gray

        gray_prepared = to_gray(prepared_full) if len(prepared_full.shape) == 3 else prepared_full

        try:
            # Find most horizontal strip
            (x_start, x_end), analysis = find_best_horizontal_strip(
                gray_prepared,
                strip_width=int(center_width),
                sample_count=8
            )

            center_info=dict(
                x1=x_start, x2=x_end,
                width=x_end-x_start,
                requested_width=int(center_width),
                coordinate_system='prepared stack with adaptive horizontal selection',
                adaptive_selection={
                    'enabled': True,
                    'method': 'adaptive_horizontal',
                    'x_range': (int(x_start), int(x_end)),
                    'analysis': analysis
                }
            )

            # Crop the prepared image
            prepared = prepared_full[:, x_start:x_end].copy()
            actual = x_end - x_start
            left = x_start

        except Exception as e:
            # Fallback to center crop
            import traceback
            actual=min(int(center_width),ow)
            left=(ow-actual)//2
            x_start = left
            x_end = left + actual

            center_info=dict(
                x1=left, x2=left+actual,
                width=actual,
                requested_width=int(center_width),
                coordinate_system='prepared stack center crop (adaptive fallback)',
                adaptive_selection={
                    'enabled': False,
                    'method': 'center_crop_fallback',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
            )
            prepared = prepared_full[:, left:left+actual].copy()

        # Update forward transform to include the horizontal crop
        local_forward=np.array([[1.,0.,-left],[0.,1.,0.],[0.,0.,1.]])@local_forward
        ow=actual
        # Use the full prepared image only to select the analysis window.
        # Sample the detector image directly with its final composed affine;
        # a cropped preview can differ at interpolation/border rounding.
        prepared=cv2.warpAffine(image,local_forward[:2],(ow,oh),flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    else:
        prepared = prepared_full

    forward=local_forward @ coarse
    info={'enabled':True,'applied':True,'input_shape':input_shape,
        'coarse_rotation_ccw_degrees':quarter,'axis_estimation':axis_info,
        'output_shape':list(prepared.shape),'crop_xyxy':[x1,y1,x2,y2],
        'rotation_degrees':float(angle),'angle_estimation':angle_info,
        'spacing_estimation':spacing_info,'scale_x':sx,'scale_y':sy,
        'target_spacing_px':target_spacing,'input_to_detection':forward.tolist(),
        'detection_to_input':np.linalg.inv(forward).tolist(),
        'count_semantics':'repeating observed lines; no automatic divide by two',
        'preserve_extent': bool(preserve_extent),
        'angle_analysis_crop_xyxy': list(analysis_bounds)}
    if center_info is not None:
        info.update(center_window=center_info,detector_resampling_passes=1,
                    geometry_method='center_direction_guard_then_single_affine_sampling')
    return prepared,info
