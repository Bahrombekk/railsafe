"""
MultiCameraSystem - ko'p kamerali tizim
"""
import yaml
import torch
import threading
import time
import cv2
from typing import List
from railcore.camera import PolygonCamera
from railcore.saver import ImageSaver
from railcore.types import CameraConfig, ModelConfig, ThresholdsConfig, ProcessingConfig
from railcore.logging_setup import setup_logger

logger = setup_logger(__name__)

class MultiCameraSystem:
    """Ko'p kamerali monitoring tizimi"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Args:
            config_path: Config fayl yo'li
        """
        logger.info(f"Config yuklanmoqda: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Image saver yaratish (bitta umumiy)
        self.image_saver = ImageSaver(save_dir='saved_images')
        
        # Model config
        self.model_config = ModelConfig(
            path=self.config['model']['path'],
            target_classes=self.config['model']['target_classes'],
            class_names=self.config['model']['class_names'],
            conf=self.config['model'].get('conf', 0.35),
            iou=self.config['model'].get('iou', 0.5),
            imgsz=self.config['model'].get('imgsz', 640)
        )
        
        # Thresholds config
        self.thresholds_config = ThresholdsConfig(
            warning=self.config['thresholds']['warning'],
            violation=self.config['thresholds']['violation']
        )
        
        # Processing config
        self.processing_config = ProcessingConfig(
            adaptive_mode=self.config['processing'].get('adaptive_mode', True),
            frame_skip_idle=self.config['processing'].get('frame_skip_idle', 3),
            frame_skip_active=self.config['processing'].get('frame_skip_active', 2),
            timeout_seconds=self.config['processing'].get('timeout_seconds', 3.0),
            empty_threshold=self.config['processing'].get('empty_threshold', 3)
        )
        
        # CUDA optimizatsiyasi
        if torch.cuda.is_available():
            logger.info(f"CUDA mavjud: {torch.cuda.get_device_name(0)}")
            torch.multiprocessing.set_start_method('spawn', force=True)
        else:
            logger.warning("CUDA mavjud emas. CPU ishlatiladi")
        
        # Kameralarni yaratish
        self.cameras: List[PolygonCamera] = []
        self.threads: List[threading.Thread] = []
        
        for cam_config_dict in self.config['cameras']:
            if cam_config_dict.get('enabled', True):
                try:
                    cam_config = CameraConfig(
                        id=cam_config_dict['id'],
                        name=cam_config_dict['name'],
                        source=cam_config_dict['source'],
                        polygon_file=cam_config_dict['polygon_file'],
                        enabled=cam_config_dict.get('enabled', True)
                    )
                    
                    camera = PolygonCamera(
                        cam_config,
                        self.model_config,
                        self.thresholds_config,
                        self.processing_config,
                        self.image_saver
                    )
                    
                    self.cameras.append(camera)
                    logger.info(f"Kamera {cam_config.id} - {cam_config.name} qo'shildi")
                    
                except Exception as e:
                    logger.error(f"Kamera {cam_config_dict['id']} xato: {e}")
    
    def start(self):
        """Tizimni ishga tushirish"""
        if not self.cameras:
            logger.error("Hech qanday faol kamera topilmadi!")
            return
        
        logger.info(f"{len(self.cameras)} ta kamera ishga tushirilmoqda...")
        
        # Har bir kamera uchun thread yaratish
        for camera in self.cameras:
            thread = threading.Thread(target=camera.run, daemon=True)
            thread.start()
            self.threads.append(thread)
            time.sleep(0.05)  # Threadlar orasida kichik pauza
        
        logger.info("Barcha kameralar ishga tushdi. To'xtatish uchun Ctrl+C bosing...")
        
        # Barcha threadlarni kutish
        try:
            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            logger.info("Dastur to'xtatilmoqda...")
            self._stop_all()
    
    def _stop_all(self):
        """Barcha kameralarni to'xtatish"""
        for camera in self.cameras:
            camera.stop()
        
        self.image_saver.stop()
        cv2.destroyAllWindows()
        logger.info("Barcha kameralar to'xtatildi")