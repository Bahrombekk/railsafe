"""
YOLO detector wrapper (Ultralytics)
"""
import numpy as np
import torch
from ultralytics import YOLO
from typing import Optional
from railcore.types import ModelConfig, DetectionResult
from railcore.logging_setup import setup_logger

logger = setup_logger(__name__)

class YOLODetector:
    """YOLO model wrapper"""
    
    def __init__(self, config: ModelConfig, camera_id: int):
        """
        Args:
            config: Model konfiguratsiyasi
            camera_id: Kamera ID (logging uchun)
        """
        self.config = config
        self.camera_id = camera_id
        
        logger.info(f"Kamera {camera_id} uchun YOLO model yuklanmoqda: {config.path}")
        
        # Model yuklash
        self.model = YOLO(config.path)
        
        # CUDA optimizatsiya
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.set_float32_matmul_precision("high")
        
        # Model fuse
        self.model.fuse()
        
        logger.info(f"Kamera {camera_id} uchun YOLO model yuklandi")
    
    def detect(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """
        Frame'da object detection va tracking
        
        Args:
            frame: Input frame
        
        Returns:
            DetectionResult yoki None
        """
        try:
            # YOLO track
            results = self.model.track(
                frame,
                persist=True,
                classes=self.config.target_classes,
                conf=self.config.conf,
                iou=self.config.iou,
                tracker="bytetrack.yaml",
                device=0,
                verbose=False,
                half=True
            )
            
            # Natijalarni parse qilish
            if results[0].boxes is None or len(results[0].boxes) == 0:
                return None
            
            boxes = results[0].boxes
            
            # Box koordinatalari
            xyxy = boxes.xyxy.detach().cpu().numpy()
            
            # Track IDs
            if boxes.id is not None:
                track_ids = boxes.id.detach().cpu().numpy().astype(int)
            else:
                return None
            
            # Class IDs
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)
            
            # Confidences
            confidences = boxes.conf.detach().cpu().numpy()
            
            return DetectionResult(
                boxes=xyxy,
                track_ids=track_ids,
                class_ids=class_ids,
                confidences=confidences
            )
            
        except Exception as e:
            logger.error(f"Kamera {self.camera_id} detection xato: {e}")
            return None
    
    def get_class_name(self, class_id: int) -> str:
        """
        Class ID bo'yicha nom olish
        
        Args:
            class_id: Class ID
        
        Returns:
            str: Class nomi
        """
        return self.config.class_names.get(str(class_id), f"class_{class_id}")