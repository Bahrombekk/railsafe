"""
Type definitions va dataclass'lar
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime
import numpy as np

@dataclass
class CameraConfig:
    """Kamera konfiguratsiyasi"""
    id: int
    name: str
    source: str
    polygon_file: str
    enabled: bool = True

@dataclass
class ModelConfig:
    """Model konfiguratsiyasi"""
    path: str
    target_classes: List[int]
    class_names: dict
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640

@dataclass
class ThresholdsConfig:
    """Vaqt chegaralari konfiguratsiyasi"""
    warning: float
    violation: float

@dataclass
class ProcessingConfig:
    """Ishlash konfiguratsiyasi"""
    adaptive_mode: bool = True
    frame_skip_idle: int = 3
    frame_skip_active: int = 2
    timeout_seconds: float = 3.0
    empty_threshold: int = 3

@dataclass
class VehicleTrackData:
    """Avtomobil tracking ma'lumotlari"""
    class_id: int
    start_time: Optional[float] = None
    in_polygon: bool = False
    total_time: float = 0.0
    entered_polygon: bool = False
    last_seen_time: float = 0.0
    violation_saved: bool = False
    exit_saved: bool = False

@dataclass
class FrameEvent:
    """Frame hodisasi ma'lumotlari"""
    frame: np.ndarray
    camera_id: int
    camera_name: str
    track_id: int
    event_type: str  # 'enter', 'exit', 'violation'
    timestamp: datetime
    box_coords: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    time_in_polygon: float = 0.0
    class_id: int = 0

@dataclass
class DetectionResult:
    """Deteksiya natijasi"""
    boxes: np.ndarray  # N x 4 (x1, y1, x2, y2)
    track_ids: np.ndarray  # N
    class_ids: np.ndarray  # N
    confidences: np.ndarray  # N

@dataclass
class PolygonState:
    """Polygon holati"""
    state: str  # 'empty', 'detected', 'violation'
    max_time: float = 0.0
    objects_count: int = 0