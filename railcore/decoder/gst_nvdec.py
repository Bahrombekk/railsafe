"""
GStreamer NVDEC decoder
"""
import cv2
from typing import Tuple
import numpy as np
from railcore.decoder.base import VideoDecoder
from railcore.logging_setup import setup_logger

logger = setup_logger(__name__)

class GStreamerNVDECDecoder(VideoDecoder):
    """GStreamer NVDEC hardware decoder"""
    
    def __init__(self, source: str):
        """
        Args:
            source: Video manba (RTSP URL yoki fayl)
        """
        self.source = source
        self.cap = None
        self._open()
    
    def _open(self) -> bool:
        """Decoderni ochish"""
        pipeline = (
            f"rtspsrc location={self.source} latency=100 protocols=tcp ! "
            "rtph264depay ! h264parse ! nvh264dec ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
        )
        
        try:
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info(f"GStreamer NVDEC decoder ochildi: {self.source}")
                return True
            else:
                logger.warning(f"GStreamer NVDEC decoder ochilmadi: {self.source}")
                return False
        except Exception as e:
            logger.error(f"GStreamer NVDEC xato: {e}")
            return False
    
    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Frame o'qish
        
        Returns:
            Tuple[bool, np.ndarray]: (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        return self.cap.read()
    
    def reopen(self) -> bool:
        """
        Decoderni qayta ochish
        
        Returns:
            bool: Muvaffaqiyatli ochilsa True
        """
        logger.info(f"Decoder qayta ochilmoqda: {self.source}")
        self.release()
        return self._open()
    
    def release(self):
        """Decoderni yopish"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_properties(self) -> dict:
        """
        Decoder xususiyatlarini olish
        
        Returns:
            dict: width, height, fps
        """
        if self.cap is None or not self.cap.isOpened():
            return {'width': 0, 'height': 0, 'fps': 0}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS)
        }
    
    def is_opened(self) -> bool:
        """
        Decoder ochiq ekanligini tekshirish
        
        Returns:
            bool: Ochiq bo'lsa True
        """
        return self.cap is not None and self.cap.isOpened()