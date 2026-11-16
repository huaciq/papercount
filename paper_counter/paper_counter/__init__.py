from __future__ import annotations

"""Traditional image and signal processing for paper edge counting."""

from .config import CounterConfig
from .counter import PaperCounter
from .models import PaperCountResult, PaperEdgeTrack, StripPeakCandidate
from .api import DetectionResult, PaperDetector, detect_papers
from .recipe import Recipe,CaptureProfile
from .engine import DetectionEngine
from .registry import PipelineSpec,PipelineRegistry,get_registry
from .contracts import Evidence,PaperInstance,InputFrame,PreparedFrame,PipelineExecution

__all__ = ["CounterConfig", "PaperCounter", "PaperCountResult", "PaperEdgeTrack", "StripPeakCandidate",
           "DetectionResult", "PaperDetector", "detect_papers", "Recipe", "CaptureProfile", "DetectionEngine",
           "PipelineSpec", "PipelineRegistry", "get_registry", "Evidence", "PaperInstance", "InputFrame",
           "PreparedFrame", "PipelineExecution"]

