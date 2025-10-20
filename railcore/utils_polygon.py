"""
Polygon mask, point-in-polygon va chizish funksiyalari
"""
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple

class PolygonUtils:
    """Polygon bilan ishlash uchun utility class"""
    
    def __init__(self, polygon_file: str, frame_width: int, frame_height: int):
        """
        Args:
            polygon_file: Polygon JSON fayl yo'li
            frame_width: Frame kengligi
            frame_height: Frame balandligi
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.polygon_points = self._load_polygon(polygon_file)
        self.polygon_mask = self._create_mask()
    
    def _load_polygon(self, polygon_file: str) -> np.ndarray:
        """
        Polygon nuqtalarini yuklash
        
        Args:
            polygon_file: JSON fayl yo'li
        
        Returns:
            np.ndarray: Polygon nuqtalari (N, 2)
        """
        with open(polygon_file, 'r') as f:
            polygon_data = json.load(f)
        
        points = np.array(
            polygon_data['annotations'][0]['segmentation'][0]
        ).reshape(-1, 2).astype(np.int32)
        
        return points
    
    def _create_mask(self) -> np.ndarray:
        """
        Polygon mask yaratish
        
        Returns:
            np.ndarray: Binary mask (H, W)
        """
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.polygon_points], 255)
        return mask
    
    def point_in_polygon(self, x: float, y: float) -> bool:
        """
        Nuqta polygon ichida ekanligini tekshirish (mask orqali)
        
        Args:
            x: X koordinata
            y: Y koordinata
        
        Returns:
            bool: Nuqta ichida bo'lsa True
        """
        ix, iy = int(x), int(y)
        if 0 <= iy < self.frame_height and 0 <= ix < self.frame_width:
            return self.polygon_mask[iy, ix] > 0
        return False
    
    def draw_polygon(self, frame: np.ndarray, state: str, max_time: float = 0.0) -> np.ndarray:
        """
        Polygon va holatini chizish
        
        Args:
            frame: Frame
            state: Polygon holati ('empty', 'detected', 'violation')
            max_time: Maksimal vaqt
        
        Returns:
            np.ndarray: Chizilgan frame
        """
        # Rang tanlash
        if state == "empty":
            color = (0, 255, 0)  # Yashil
        elif state == "detected":
            color = (0, 255, 255)  # Sariq
        else:  # violation
            color = (0, 0, 255)  # Qizil
        
        # Polygon chizish
        cv2.polylines(frame, [self.polygon_points], True, color, 3)
        
        # Holat matni
        text = f"Polygon: {state} ({max_time:.1f}s)"
        cv2.putText(frame, text, 
                   (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def draw_box(self, frame: np.ndarray, 
                 box: Tuple[int, int, int, int],
                 track_id: int,
                 time_in_polygon: float,
                 is_inside: bool,
                 threshold_warning: float,
                 threshold_violation: float) -> np.ndarray:
        """
        Box va ma'lumotlarni chizish
        
        Args:
            frame: Frame
            box: Box koordinatalari (x1, y1, x2, y2)
            track_id: Track ID
            time_in_polygon: Polygon ichida vaqt
            is_inside: Polygon ichida ekanligini
            threshold_warning: Ogohlantirish chegarasi
            threshold_violation: Qoidabuzarlik chegarasi
        
        Returns:
            np.ndarray: Chizilgan frame
        """
        x1, y1, x2, y2 = box
        
        # Rang tanlash
        if not is_inside:
            color = (0, 255, 0)  # Tashqarida - yashil
            time_text = "Tashqarida"
        elif time_in_polygon < threshold_warning:
            color = (255, 0, 0)  # Xavfsiz - ko'k
            time_text = f"{time_in_polygon:.1f}s"
        elif time_in_polygon < threshold_violation:
            color = (0, 255, 255)  # Ogohlantirish - sariq
            time_text = f"{time_in_polygon:.1f}s"
        else:
            color = (0, 0, 255)  # Qoidabuzarlik - qizil
            time_text = f"{time_in_polygon:.1f}s"
        
        # Box chizish
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Ma'lumotlar
        cv2.putText(frame, f"ID:{track_id}  -  {time_text}", 
                   (x1, y1 - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame