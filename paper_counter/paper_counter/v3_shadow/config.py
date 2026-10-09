from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class ShadowConfig:
    version: str = 'V3-P1-1'
    stage: str = 'A4'
    rho: float = 0.3
    eta: float = 0.8
    gamma: float = 1.0
    scales: int = 5
    tail_sigma: float = 5.0
    min_sigma_px: float = 0.5
    width_measure_smoothing_px: float = 0.7
    reference_end_fraction: float = 0.05
    region_area_ratio: float = 0.10
    region_peak_ratio: float = 0.10
    absolute_peak_floor: float = 1e-7
    max_saturated_fraction: float = 0.50

    def __post_init__(self):
        if self.stage not in ('A1', 'A2', 'A3', 'A4'):
            raise ValueError('Only standalone P1 stages A1--A4 are implemented')
        if not (0 < self.rho <= 1 and self.eta >= 0 and self.scales >= 1 and self.tail_sigma >= 3):
            raise ValueError('Invalid filter parameters')
        if not (0 < self.reference_end_fraction <= .5 and self.min_sigma_px > 0):
            raise ValueError('Invalid sampling/reference settings')
        if min(self.region_area_ratio, self.region_peak_ratio, self.absolute_peak_floor) < 0:
            raise ValueError('Verification thresholds must be nonnegative')
        if any(isinstance(v, float) and not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Nonfinite configuration')

    def to_dict(self):
        return asdict(self)
