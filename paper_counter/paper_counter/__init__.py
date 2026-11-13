from __future__ import annotations

"""Traditional image and signal processing for paper edge counting."""

from .config import CounterConfig
from .counter import PaperCounter
from .models import PaperCountResult, PaperEdgeTrack, StripPeakCandidate
from .api import DetectionResult, PaperDetector, detect_papers

__all__ = ["CounterConfig", "PaperCounter", "PaperCountResult", "PaperEdgeTrack", "StripPeakCandidate",
           "DetectionResult", "PaperDetector", "detect_papers"]

