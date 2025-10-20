"""
RailCore - RailSafe tizimining asosiy moduli
"""
from railcore.system import MultiCameraSystem
from railcore.camera import PolygonCamera
from railcore.saver import ImageSaver
from railcore.logging_setup import setup_logger

__version__ = "1.0.0"

__all__ = [
    'MultiCameraSystem',
    'PolygonCamera',
    'ImageSaver',
    'setup_logger'
]