"""
Tracking holati va enter/exit/violation mantiqi
"""
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from railcore.types import VehicleTrackData, FrameEvent, ThresholdsConfig
from railcore.utils_polygon import PolygonUtils
from railcore.logging_setup import setup_logger

logger = setup_logger(__name__)

class VehicleTracker:
    """Avtomobil tracking va hodisalarni boshqarish"""
    
    def __init__(self, 
                 camera_id: int,
                 camera_name: str,
                 polygon_utils: PolygonUtils,
                 thresholds: ThresholdsConfig,
                 timeout_seconds: float = 3.0):
        """
        Args:
            camera_id: Kamera ID
            camera_name: Kamera nomi
            polygon_utils: Polygon utilities
            thresholds: Vaqt chegaralari
            timeout_seconds: Timeout vaqti
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.polygon_utils = polygon_utils
        self.thresholds = thresholds
        self.timeout_seconds = timeout_seconds
        
        # Tracking ma'lumotlari
        self.vehicles: Dict[int, VehicleTrackData] = {}
        
        # Counters
        self.entered_count = 0
        self.passed_count = 0
    
    def update(self, 
               track_id: int,
               class_id: int,
               box: Tuple[int, int, int, int],
               current_time: float,
               frame) -> List[FrameEvent]:
        """
        Track'ni yangilash va hodisalarni qaytarish
        
        Args:
            track_id: Track ID
            class_id: Class ID
            box: Box koordinatalari (x1, y1, x2, y2)
            current_time: Hozirgi vaqt (sekund)
            frame: Current frame
        
        Returns:
            List[FrameEvent]: Hodisalar ro'yxati
        """
        events = []
        
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Polygon ichida ekanligini tekshirish
        is_inside = self.polygon_utils.point_in_polygon(center_x, center_y)
        
        # Yangi track
        if track_id not in self.vehicles:
            self.vehicles[track_id] = VehicleTrackData(
                class_id=class_id,
                last_seen_time=current_time
            )
        
        vehicle = self.vehicles[track_id]
        
        # Polygon ichida
        if is_inside:
            # KIRISH hodisasi
            if not vehicle.in_polygon:
                vehicle.start_time = current_time
                vehicle.in_polygon = True
                vehicle.entered_polygon = True
                vehicle.violation_saved = False
                vehicle.exit_saved = False
                self.entered_count += 1
                self.passed_count += 1
                
                # KIRISH event
                events.append(FrameEvent(
                    frame=frame.copy(),
                    camera_id=self.camera_id,
                    camera_name=self.camera_name,
                    track_id=track_id,
                    event_type='enter',
                    timestamp=datetime.now(),
                    box_coords=box,
                    time_in_polygon=0.0,
                    class_id=class_id
                ))
            
            # Vaqtni hisoblash
            time_in_polygon = current_time - vehicle.start_time
            vehicle.total_time = time_in_polygon
            vehicle.last_seen_time = current_time
            
            # QOIDABUZARLIK hodisasi (faqat 1 marta)
            if (time_in_polygon >= self.thresholds.violation and 
                not vehicle.violation_saved):
                events.append(FrameEvent(
                    frame=frame.copy(),
                    camera_id=self.camera_id,
                    camera_name=self.camera_name,
                    track_id=track_id,
                    event_type='violation',
                    timestamp=datetime.now(),
                    box_coords=box,
                    time_in_polygon=time_in_polygon,
                    class_id=class_id
                ))
                vehicle.violation_saved = True
        
        else:  # Tashqarida
            # CHIQISH hodisasi
            if vehicle.in_polygon and not vehicle.exit_saved:
                time_in_polygon = vehicle.total_time
                events.append(FrameEvent(
                    frame=frame.copy(),
                    camera_id=self.camera_id,
                    camera_name=self.camera_name,
                    track_id=track_id,
                    event_type='exit',
                    timestamp=datetime.now(),
                    box_coords=box,
                    time_in_polygon=time_in_polygon,
                    class_id=class_id
                ))
                vehicle.in_polygon = False
                vehicle.exit_saved = True
            
            vehicle.last_seen_time = current_time
        
        return events
    
    def cleanup_expired(self, current_time: float):
        """
        Eski tracklarni tozalash
        
        Args:
            current_time: Hozirgi vaqt
        """
        expired_ids = [
            tid for tid, data in self.vehicles.items()
            if current_time - data.last_seen_time > self.timeout_seconds
        ]
        
        for tid in expired_ids:
            del self.vehicles[tid]
    
    def get_polygon_state(self) -> Tuple[str, float, int]:
        """
        Polygon holatini olish
        
        Returns:
            Tuple[str, float, int]: (state, max_time, objects_count)
        """
        vehicles_inside = 0
        max_time = 0.0
        
        for vehicle in self.vehicles.values():
            if vehicle.in_polygon:
                vehicles_inside += 1
                if vehicle.total_time > max_time:
                    max_time = vehicle.total_time
        
        if vehicles_inside == 0:
            state = "empty"
            max_time = 0.0
        elif max_time >= self.thresholds.violation:
            state = "violation"
        else:
            state = "detected"
        
        return state, max_time, vehicles_inside
    
    def get_vehicle_data(self, track_id: int) -> Optional[VehicleTrackData]:
        """
        Track ID bo'yicha vehicle ma'lumotlarini olish
        
        Args:
            track_id: Track ID
        
        Returns:
            VehicleTrackData yoki None
        """
        return self.vehicles.get(track_id)