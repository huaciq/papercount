"""RAW 边界方向积分：在小步长上跟随转弯，而不是用整条带平均斜率外推。"""
from __future__ import annotations
import cv2
import numpy as np


class DirectionFields(list):
    """保留旧的逐条带方向接口，同时携带相邻条带间的实测预测位移。"""
    def __init__(self, fields, motion, reliability=None):
        super().__init__(fields)
        self.motion = motion
        self.reliability = reliability or {}


def trace_boundary_motion(gray, strips, fields, cfg):
    image = gray.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    # 尺度只用于局部切线估计，不移动图像、不生成候选峰。
    xx = cv2.GaussianBlur(gx*gx, (0, 0), 3., sigmaY=1.2)
    yy = cv2.GaussianBlur(gy*gy, (0, 0), 3., sigmaY=1.2)
    xy = cv2.GaussianBlur(gx*gy, (0, 0), 3., sigmaY=1.2)
    coherence = np.sqrt((xx-yy)**2+4*xy*xy)/np.maximum(xx+yy, cfg.epsilon)
    tangent = -xy/np.maximum(yy, cfg.epsilon)
    tangent[(coherence < cfg.orientation_min_coherence) | (np.abs(tangent) > cfg.orientation_max_slope)] = 0
    centers = [int((a+b-1)/2) for a,b in strips]
    rows = np.arange(len(gray), dtype=float)
    motion = {}; reliability = {}
    for source, start in enumerate(centers):
        for target in range(max(0, source-cfg.max_skip_strips-1), min(len(strips),source+cfg.max_skip_strips+2)):
            if target == source:
                continue
            end = centers[target]
            steps = np.linspace(start, end, max(2, int(abs(end-start)/2)+1))
            y = rows.copy()
            good = np.zeros(len(rows)); same_sign = np.zeros(len(rows))
            initial = np.sign(gy[:,start])
            for left,right in zip(steps,steps[1:]):
                x = int(round((left+right)/2))
                good += np.interp(y, rows, coherence[:,x]) >= cfg.orientation_min_coherence
                same_sign += np.sign(np.interp(y, rows, gy[:,x])) == initial
                dy = np.interp(y, rows, tangent[:,x])*(right-left)
                y += np.interp(y+dy*.5, rows, tangent[:,x])*(right-left)
            # Rasterized RAW edges have two-pixel Sobel plateaus. Refine the
            # integration endpoint only to a nearby actual edge of the same
            # sign and comparable strength; this is bounded by 2.5 pixels,
            # not a periodic row-lag alignment or a generated observation.
            best=.5*np.abs(gy[:,start])
            refined=y.copy();nearest=np.full(len(rows),np.inf)
            for delta in range(-2,3):
                target_rows=np.clip(np.rint(y).astype(int)+delta,0,len(rows)-1)
                strength=initial*gy[target_rows,end]
                distance=np.abs(target_rows-y)
                better=(strength>best+cfg.epsilon)|((np.abs(strength-best)<=cfg.epsilon)&(distance<nearest))
                better&=(strength>cfg.epsilon)&(distance<=2.5)
                refined[better]=target_rows[better];nearest[better]=distance[better];best[better]=strength[better]
            shifts=refined-rows
            usable=(good/(len(steps)-1)>=.8)&(same_sign/(len(steps)-1)>=.95)
            # 用邻近多条 RAW 边界的共同形变压制纸面颗粒的随机切线。
            # 它们提供方向场，不共享纸张身份，不把周期对齐当作位移。
            knots=[];values=[];valid=[]
            half=max(3,cfg.orientation_window_rows//2)
            for row in range(0,len(rows),max(1,half//2)):
                lo=max(0,row-half);hi=min(len(rows),row+half+1)
                strength=np.abs(gy[lo:hi,start])
                threshold=max(cfg.epsilon,.25*float(np.percentile(strength,95)))
                boundary_rows=usable[lo:hi]&(strength>=threshold)
                measured=shifts[lo:hi][boundary_rows]
                weights=strength[boundary_rows]**2
                knots.append(row);valid.append(len(measured)>=3)
                if len(measured)>=3:
                    order=np.argsort(measured)
                    middle=np.searchsorted(np.cumsum(weights[order]),.5*weights.sum())
                    values.append(float(measured[order[min(middle,len(order)-1)]]))
                else:values.append(0.)
            motion[source,target]=np.interp(rows,knots,values).astype(np.float32)
            reliability[source,target]=np.interp(rows,knots,valid)>=.5
    return DirectionFields(fields, motion, reliability)


def predicted_shift(fields, centers, source, target, y, offset=0):
    if not getattr(fields,'association_active',True):
        return 0.
    row = float(np.clip(y-offset, 0, len(fields[source])-1))
    motion = getattr(fields, 'motion', {})
    reliable = getattr(fields,'reliability',{}).get((source,target))
    if (source,target) in motion and (reliable is None or reliable[int(round(row))]):
        a = int(row); b = min(a+1, len(fields[source])-1)
        return float(motion[source,target][a]*(b-row)+motion[source,target][b]*(row-a)) if a != b else float(motion[source,target][a])
    if (source,target) in motion:
        return 0.
    at = int(round(row))
    return float((fields[source][at]+fields[target][at])*.5*(centers[target]-centers[source]))


def direction_search_policy(fields, period, cfg):
    """水平纸沿用位置邻域；可测形变超出邻域时才需要方向扩展。"""
    shifts=[]
    for (a,b),motion in getattr(fields,'motion',{}).items():
        if b != a+1:
            continue
        usable=fields.reliability[a,b]
        if np.any(usable):shifts.append(float(np.max(np.abs(motion[usable]))))
    maximum=max(shifts,default=0.)
    gate=cfg.max_neighbor_shift_ratio*period
    fields.association_active=maximum>=gate
    return {'enabled':fields.association_active,'raw_motion_max':maximum,
            'position_neighborhood_px':gate,'rule':'measured_deformation_exceeds_position_neighborhood'}
