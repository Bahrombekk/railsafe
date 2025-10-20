"""
ImageSaver - Alohida thread'da rasmlarni saqlash
"""
import cv2
import time
import threading
from queue import Queue
from pathlib import Path
from datetime import datetime
from railcore.types import FrameEvent
from railcore.logging_setup import setup_logger

logger = setup_logger(__name__)

class ImageSaver:
    """Alohida thread'da rasmlarni saqlash uchun"""
    
    def __init__(self, save_dir: str = 'saved_images'):
        """
        Args:
            save_dir: Rasmlarni saqlash papkasi
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.queue = Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info(f"ImageSaver ishga tushdi: {self.save_dir}")
    
    def _worker(self):
        """Queue'dan rasmlarni olish va saqlash"""
        while self.running:
            try:
                if not self.queue.empty():
                    event = self.queue.get(timeout=1)
                    self._save_image(event)
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"ImageSaver xato: {e}")
    
    def _save_image(self, event: FrameEvent):
        """
        Rasmni saqlash
        
        Args:
            event: FrameEvent ma'lumotlari
        """
        frame = event.frame
        camera_id = event.camera_id
        camera_name = event.camera_name
        track_id = event.track_id
        event_type = event.event_type
        timestamp = event.timestamp
        box_coords = event.box_coords
        time_in_polygon = event.time_in_polygon
        class_id = event.class_id
        
        # Frame nusxasini olish
        img = frame.copy()
        
        # Box koordinatalari
        x1, y1, x2, y2 = box_coords
        
        # Rang va matn tanlash
        if event_type == 'enter':
            color = (0, 255, 0)  # Yashil
            event_text = "KIRISH"
        elif event_type == 'exit':
            color = (255, 0, 0)  # Ko'k
            event_text = "CHIQISH"
        elif event_type == 'violation':
            color = (0, 0, 255)  # Qizil
            event_text = "QOIDABUZARLIK"
        else:
            color = (255, 255, 255)
            event_text = event_type.upper()
        
        # Box chizish
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Ma'lumotlarni yozish
        cv2.putText(img, f"ID: {track_id}", (x1, y1 - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, event_text, (x1, y1 - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if time_in_polygon > 0:
            cv2.putText(img, f"Vaqt: {time_in_polygon:.1f}s", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Fayl nomini yaratish
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"cam{camera_id}_{event_type}_id{track_id}_{timestamp_str}.jpg"
        
        # Papkalar yaratish
        camera_dir = self.save_dir / f"camera_{camera_id}"
        camera_dir.mkdir(exist_ok=True)
        
        event_dir = camera_dir / event_type
        event_dir.mkdir(exist_ok=True)
        
        filepath = event_dir / filename
        
        # Rasm saqlash
        cv2.imwrite(str(filepath), img)
        
        # TXT fayl saqlash (YOLO format)
        h, w = frame.shape[:2]
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        
        txt_filename = filepath.with_suffix('.txt').name
        txt_path = event_dir / txt_filename
        
        with open(txt_path, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        logger.debug(f"Saqlandi: {camera_name} - {event_text} - ID:{track_id} -> {filepath}")
    
    def add_to_queue(self, event: FrameEvent):
        """
        Queue'ga rasm qo'shish
        
        Args:
            event: FrameEvent ma'lumotlari
        """
        self.queue.put(event)
    
    def stop(self):
        """Image saver'ni to'xtatish"""
        logger.info("ImageSaver to'xtatilmoqda...")
        self.running = False
        self.thread.join()
        logger.info("ImageSaver to'xtatildi")