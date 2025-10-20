"""
PolygonCamera - barcha komponentlarni birlashtiradi
"""
import cv2
import time
import numpy as np
from railcore.types import CameraConfig, ModelConfig, ThresholdsConfig, ProcessingConfig
from railcore.decoder import create_decoder
from railcore.utils_polygon import PolygonUtils
from railcore.vision import YOLODetector, VehicleTracker
from railcore.saver import ImageSaver
from railcore.logging_setup import setup_logger

logger = setup_logger(__name__)

class PolygonCamera:
    """Polygon monitoring camera"""
    
    def __init__(self,
                 camera_config: CameraConfig,
                 model_config: ModelConfig,
                 thresholds_config: ThresholdsConfig,
                 processing_config: ProcessingConfig,
                 image_saver: ImageSaver):
        """
        Args:
            camera_config: Kamera konfiguratsiyasi
            model_config: Model konfiguratsiyasi
            thresholds_config: Vaqt chegaralari
            processing_config: Ishlash sozlamalari
            image_saver: Rasm saqlash
        """
        self.camera_id = camera_config.id
        self.camera_name = camera_config.name
        self.image_saver = image_saver
        
        # Decoder yaratish
        logger.info(f"Kamera {self.camera_id} uchun decoder ochilmoqda...")
        self.decoder = create_decoder(camera_config.source)
        
        if not self.decoder.is_opened():
            raise ValueError(f"Kamera ochilmadi: {camera_config.source}")
        
        # Video xususiyatlari
        props = self.decoder.get_properties()
        self.frame_width = props['width']
        self.frame_height = props['height']
        self.video_fps = props['fps'] if props['fps'] > 0 else 25.0
        
        logger.info(f"Kamera {self.camera_id}: {self.frame_width}x{self.frame_height} @ {self.video_fps} FPS")
        
        # Polygon utils
        self.polygon_utils = PolygonUtils(
            camera_config.polygon_file,
            self.frame_width,
            self.frame_height
        )
        
        # YOLO detector
        self.detector = YOLODetector(model_config, self.camera_id)
        
        # Vehicle tracker
        self.tracker = VehicleTracker(
            self.camera_id,
            self.camera_name,
            self.polygon_utils,
            thresholds_config,
            processing_config.timeout_seconds
        )
        
        # Processing config
        self.adaptive_mode = processing_config.adaptive_mode
        self.frame_skip_idle = processing_config.frame_skip_idle
        self.frame_skip_active = processing_config.frame_skip_active
        self.empty_threshold = processing_config.empty_threshold
        
        # FPS
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # Counters
        self.frame_count = 0
        self.process_count = 0
        self.running = True
        
        # Adaptive processing
        self.current_frame_skip = self.frame_skip_idle
        self.frame_counter = 0
        self.consecutive_empty_frames = 0
        
        # Thresholds
        self.threshold_warning = thresholds_config.warning
        self.threshold_violation = thresholds_config.violation
        
        logger.info(f"Kamera {self.camera_id} - {self.camera_name} tayyor")
    
    def _update_fps(self):
        """FPS hisoblash"""
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        #if elapsed >= 1.0:
        self.current_fps = self.fps_frame_count / elapsed
        self.fps_frame_count = 0
        self.fps_start_time = current_time
    
    def run(self):
        """Asosiy loop"""
        logger.info(f"Kamera {self.camera_id} - {self.camera_name} boshlandi")
        
        while self.running:
            # Frame o'qish
            success, frame = self.decoder.read()
            
            if not success:
                logger.warning(f"Kamera {self.camera_id} frame o'qiy olmadi. Qayta ulanish...")
                time.sleep(1)
                if not self.decoder.reopen():
                    logger.error(f"Kamera {self.camera_id} qayta ulanmadi")
                    time.sleep(5)
                continue
            
            self.frame_count += 1
            current_time = self.frame_count / self.video_fps
            self.frame_counter += 1
            self._update_fps()
            
            # Frame qayta ishlash kerakmi?
            process_this_frame = self.frame_counter % self.current_frame_skip == 0
            
            if process_this_frame:
                self.process_count += 1
                
                # Detection
                detection_result = self.detector.detect(frame)
                
                if detection_result is not None:
                    detected_count = len(detection_result.boxes)
                    
                    # Adaptive mode
                    if self.adaptive_mode:
                        if detected_count == 0:
                            self.consecutive_empty_frames += 1
                            if self.consecutive_empty_frames >= self.empty_threshold:
                                self.current_frame_skip = self.frame_skip_idle
                        else:
                            self.consecutive_empty_frames = 0
                            self.current_frame_skip = self.frame_skip_active
                    
                    # Tracking va event handling
                    for i in range(len(detection_result.boxes)):
                        track_id = detection_result.track_ids[i]
                        class_id = detection_result.class_ids[i]
                        box = tuple(detection_result.boxes[i].astype(int))
                        
                        # Update tracker va hodisalarni olish
                        events = self.tracker.update(track_id, class_id, box, current_time, frame)
                        
                        # Hodisalarni saqlash
                        for event in events:
                            self.image_saver.add_to_queue(event)
                else:
                    # Bo'sh frame
                    if self.adaptive_mode:
                        self.consecutive_empty_frames += 1
                        if self.consecutive_empty_frames >= self.empty_threshold:
                            self.current_frame_skip = self.frame_skip_idle
                
                # Eski tracklarni tozalash
                self.tracker.cleanup_expired(current_time)
            
            # Vizualizatsiya
            self._draw_visualization(frame, detection_result if process_this_frame else None)
            
            # Display
            H, W = frame.shape[:2]
            frame_resized = cv2.resize(frame, (W // 2, H // 2))
            
            window_name = f"Camera {self.camera_id} - {self.camera_name}"
            cv2.imshow(window_name, frame_resized)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break
        
        # Cleanup
        self.decoder.release()
        cv2.destroyWindow(f"Camera {self.camera_id} - {self.camera_name}")
        logger.info(f"Kamera {self.camera_id} to'xtatildi")
    
    def _draw_visualization(self, frame, detection_result):
        """Vizualizatsiya chizish"""
        # Polygon holati
        state, max_time, objects_count = self.tracker.get_polygon_state()
        self.polygon_utils.draw_polygon(frame, state, max_time)
        
        # Detections
        if detection_result is not None:
            for i in range(len(detection_result.boxes)):
                track_id = detection_result.track_ids[i]
                box = tuple(detection_result.boxes[i].astype(int))
                
                vehicle_data = self.tracker.get_vehicle_data(track_id)
                if vehicle_data:
                    self.polygon_utils.draw_box(
                        frame, box, track_id,
                        vehicle_data.total_time,
                        vehicle_data.in_polygon,
                        self.threshold_warning,
                        self.threshold_violation
                    )
        
        # Info text
        cv2.putText(frame, f"{self.camera_name} | FPS: {self.current_fps:.1f} | Frame: {self.frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (227, 30, 206), 2)
        
        cv2.putText(frame, f"Count: {self.tracker.passed_count}  | Inside: {objects_count}", 
                   (10, self.frame_height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if self.adaptive_mode:
            mode_text = f"{'ACTIVE' if self.current_frame_skip <= 2 else 'IDLE'} (1/{self.current_frame_skip})"
            cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def stop(self):
        """Kamerani to'xtatish"""
        self.running = False