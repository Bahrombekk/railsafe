import cv2
import numpy as np
import json
from ultralytics import YOLO
import yaml
import time
import threading
from queue import Queue
import torch
from pathlib import Path
from datetime import datetime

class ImageSaver:
    """Alohida thread'da rasmlarni saqlash uchun"""
    def __init__(self, save_dir='saved_images'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.queue = Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        """Queue'dan rasmlarni olish va saqlash"""
        while self.running:
            try:
                if not self.queue.empty():
                    data = self.queue.get(timeout=1)
                    self._save_image(data)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] ImageSaver: {e}")
    
    def _save_image(self, data):
        """Rasmni saqlash"""
        frame = data['frame']
        camera_id = data['camera_id']
        camera_name = data['camera_name']
        track_id = data['track_id']
        event_type = data['event_type']
        timestamp = data['timestamp']
        box_coords = data['box_coords']
        time_in_polygon = data.get('time_in_polygon', 0)
        class_id = data.get('class_id', 0)  # Default class_id if not provided
        
        # Frame nusxasini olish (faqat bu object uchun clean copy)
        img = frame.copy()
        
        # Faqat tegishli box'ni chizish
        x1, y1, x2, y2 = box_coords
        
        # Rang tanlash
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
        cv2.putText(img, f"ID: {track_id}", (x1, y1-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, event_text, (x1, y1-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if time_in_polygon > 0:
            cv2.putText(img, f"Vaqt: {time_in_polygon:.1f}s", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Fayl nomini yaratish
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"cam{camera_id}_{event_type}_id{track_id}_{timestamp_str}.jpg"
        
        # Kamera bo'yicha papka yaratish
        camera_dir = self.save_dir / f"camera_{camera_id}"
        camera_dir.mkdir(exist_ok=True)
        
        # Hodisa bo'yicha papka yaratish
        event_dir = camera_dir / event_type
        event_dir.mkdir(exist_ok=True)
        
        filepath = event_dir / filename
        
        # Rasm saqlash
        cv2.imwrite(str(filepath), img)
        
        # Box TXT faylini saqlash (YOLO format: class_id x_center y_center width height, normalized)
        h, w = frame.shape[:2]
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        
        txt_filename = filepath.with_suffix('.txt').name
        txt_path = event_dir / txt_filename
        
        with open(txt_path, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        #print(f"[SAVED] {camera_name} - {event_text} - ID:{track_id} -> {filepath} (TXT: {txt_path})")
    
    def add_to_queue(self, frame, camera_id, camera_name, track_id, event_type, 
                     box_coords, time_in_polygon=0, class_id=0):
        """Queue'ga rasm qo'shish"""
        data = {
            'frame': frame,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'track_id': track_id,
            'event_type': event_type,
            'timestamp': datetime.now(),
            'box_coords': box_coords,
            'time_in_polygon': time_in_polygon,
            'class_id': class_id
        }
        self.queue.put(data)
    
    def stop(self):
        """Image saver'ni to'xtatish"""
        self.running = False
        self.thread.join()


class PolygonCamera:
    def __init__(self, camera_config, model_path, model_config_data, thresholds_config, 
                 processing_config, image_saver):
        self.camera_id = camera_config['id']
        self.camera_name = camera_config['name']
        self.source = camera_config['source']
        self.polygon_file = camera_config['polygon_file']
        self.image_saver = image_saver
        self.passed_count = 0  # Poligondan o'tgan avtomobillar soni

        
        # HAR BIR KAMERA O'Z MODELIGA EGA (CUDA threading muammosini hal qilish)
        print(f"[INFO] Kamera {self.camera_id} uchun model yuklanmoqda...")
        self.model = YOLO(model_path)
        
        # Har bir model o'z device'ida
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.set_float32_matmul_precision("high")

        
        self.model.fuse()
        self.target_classes = model_config_data['target_classes']
        self.class_names = model_config_data['class_names']
        
        # Thresholds
        self.threshold_warning = thresholds_config['warning']
        self.threshold_violation = thresholds_config['violation']
        
        # Processing config
        self.adaptive_mode = processing_config.get('adaptive_mode', True)
        self.frame_skip_idle = processing_config.get('frame_skip_idle', 3)
        self.frame_skip_active = processing_config.get('frame_skip_active', 2)
        
        # Kamera ochish
        pipeline = (
            f"rtspsrc location={self.source} latency=100 protocols=tcp ! "
                "rtph264depay ! h264parse ! nvh264dec ! "
                "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
            )
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                raise ValueError(f"Kamera ochilmadi: {self.source}")
            print(f"[INFO] Kamera {self.camera_id}: CPU/FFmpeg dekoder ishlatilyapti.")
        else:
             print(f"[INFO] Kamera {self.camera_id}: GStreamer NVDEC (nvh264dec).")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Polygon yuklash
        with open(self.polygon_file, 'r') as f:
            polygon_data = json.load(f)
        self.polygon_points = np.array(polygon_data['annotations'][0]['segmentation'][0]).reshape(-1, 2).astype(np.int32)
        
        # Polygon mask
        self.polygon_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        cv2.fillPoly(self.polygon_mask, [self.polygon_points], 255)
        
        # FPS
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        
        # Counters
        self.frame_count = 0
        self.process_count = 0
        self.detected_count = 0
        self.entered_count = 0
        self.running = True
        
        # Adaptive processing
        self.current_frame_skip = self.frame_skip_idle
        self.frame_counter = 0
        self.consecutive_empty_frames = 0
        self.empty_threshold = 3
        
        # Tracking
        self.vehicle_tracking = {}
        self.current_time = 0
        self.timeout_seconds = 3
        self.polygon_state = "empty"
        self.max_time_in_polygon = 0
        
        # Cache
        self.last_detection_result = None
        self.last_process_frame = 0
        
        # Colors
        self.color_safe = (255, 0, 0)
        self.color_warning = (0, 255, 255)
        self.color_violation = (0, 0, 255)
        self.color_outside = (0, 255, 0)
        self.color_empty = (0, 255, 0)
        self.color_detected = (0, 255, 255)
    
    def _update_fps(self):
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        self.current_fps = self.fps_frame_count / elapsed
        self.fps_frame_count = 0
        self.fps_start_time = current_time
    
    def _point_in_polygon_fast(self, x, y):
        if 0 <= int(y) < self.frame_height and 0 <= int(x) < self.frame_width:
            return self.polygon_mask[int(y), int(x)] > 0
        return False
    
    def _update_polygon_state(self):
        vehicles_inside = 0
        max_time = 0
        
        for data in self.vehicle_tracking.values():
            if data['in_polygon']:
                vehicles_inside += 1
                if data['total_time'] > max_time:
                    max_time = data['total_time']
        
        if vehicles_inside == 0:
            self.max_time_in_polygon = 0
            self.polygon_state = "empty"
        elif max_time >= self.threshold_violation:
            self.max_time_in_polygon = max_time
            self.polygon_state = "violation"
        else:
            self.max_time_in_polygon = max_time
            self.polygon_state = "detected"
    
    def _draw_simple_polygon(self, frame):
        color = self.color_empty if self.polygon_state == "empty" else self.color_detected if self.polygon_state == "detected" else self.color_violation
        cv2.polylines(frame, [self.polygon_points], True, color, 3)
        cv2.putText(frame, f"Polygon: {self.polygon_state} ({self.max_time_in_polygon:.1f}s)", 
                   (10, self.frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _get_box_color(self, time_in_polygon):
        if time_in_polygon < self.threshold_warning:
            return self.color_safe
        elif time_in_polygon < self.threshold_violation:
            return self.color_warning
        else:
            return self.color_violation
    
    def _save_event_image(self, frame, track_id, event_type, box_coords, time_in_polygon=0, class_id=0):
        """Hodisa rasmini saqlash uchun"""
        self.image_saver.add_to_queue(
            frame=frame,
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            track_id=track_id,
            event_type=event_type,
            box_coords=box_coords,
            time_in_polygon=time_in_polygon,
            class_id=class_id
        )
    
    def run(self):
        print(f"[INFO] Kamera {self.camera_id} - {self.camera_name} boshlandi...")
        
        while self.running:
            success, frame = self.cap.read()
            if not success:
                print(f"[WARNING] Kamera {self.camera_id} frame o'qiy olmadi. Qayta ulanish...")
                time.sleep(1)
                pipeline = (
                    f"rtspsrc location={self.source} latency=100 ! "
                        "rtph264depay ! h264parse ! nvh264dec ! "
                        "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
                    )   
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                continue
            
            self.frame_count += 1
            self.current_time = self.frame_count / self.video_fps
            self.frame_counter += 1
            self._update_fps()
            
            process_this_frame = self.frame_counter % self.current_frame_skip == 0
            
            if process_this_frame:
                self.process_count += 1
                self.last_process_frame = self.frame_count
                
                # HAR BIR KAMERA O'Z MODELIDAN FOYDALANADI
                results = self.model.track(frame, persist=True, classes=self.target_classes, 
                                          conf=0.35,iou=0.5,tracker="bytetrack.yaml",device=0,verbose=False,half=True)
                #results = self.model.track(frame, persist=True, classes=self.target_classes, 
                #                          conf=0.35, imgsz=640, verbose=False, half=True)
                
                self.last_detection_result = results
                
                detected_objects = len(results[0].boxes) if results[0].boxes is not None else 0
                self.detected_count += detected_objects
                
                if self.adaptive_mode:
                    if detected_objects == 0:
                        self.consecutive_empty_frames += 1
                        if self.consecutive_empty_frames >= self.empty_threshold:
                            self.current_frame_skip = self.frame_skip_idle
                    else:
                        self.consecutive_empty_frames = 0
                        self.current_frame_skip = self.frame_skip_active
                
                # Eski tracklarni tozalash
                expired_ids = [tid for tid, data in self.vehicle_tracking.items() 
                              if self.current_time - data.get('last_seen_time', self.current_time) > self.timeout_seconds]
                for tid in expired_ids:
                    del self.vehicle_tracking[tid]
                
                if results[0].boxes is not None:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        ids  = (boxes.id.detach().cpu().numpy().astype(int)
                            if boxes.id is not None else np.array([], dtype=int))
                        clss = boxes.cls.detach().cpu().numpy().astype(int)
                        for i in range(len(xyxy)):
                            if boxes.id is None: continue
                            track_id = ids[i]
                            class_id = clss[i]
                            x1, y1, x2, y2 = xyxy[i].astype(int)
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            box_coords = (int(x1), int(y1), int(x2), int(y2))
                            
                            is_inside = self._point_in_polygon_fast(center_x, center_y)
                            
                            if track_id not in self.vehicle_tracking:
                                self.vehicle_tracking[track_id] = {
                                    'class_id': class_id,
                                    'start_time': None,
                                    'in_polygon': False,
                                    'total_time': 0,
                                    'entered_polygon': False,
                                    'last_seen_time': self.current_time,
                                    'violation_saved': False,  # Qoidabuzarlik rasmi saqlangani
                                    'exit_saved': False  # Chiqish rasmi saqlangani
                                }
                            
                            if is_inside:
                                # KIRISH hodisasi
                                if not self.vehicle_tracking[track_id]['in_polygon']:
                                    self.vehicle_tracking[track_id]['start_time'] = self.current_time
                                    self.vehicle_tracking[track_id]['in_polygon'] = True
                                    self.vehicle_tracking[track_id]['entered_polygon'] = True
                                    self.vehicle_tracking[track_id]['violation_saved'] = False
                                    self.vehicle_tracking[track_id]['exit_saved'] = False
                                    self.entered_count += 1
                                    self.passed_count += 1
                                    
                                    # KIRISH rasmini saqlash
                                    self._save_event_image(frame, track_id, 'enter', box_coords, class_id=class_id)
                                
                                time_in_polygon = self.current_time - self.vehicle_tracking[track_id]['start_time']
                                self.vehicle_tracking[track_id]['total_time'] = time_in_polygon
                                self.vehicle_tracking[track_id]['last_seen_time'] = self.current_time
                                
                                # QOIDABUZARLIK hodisasi (faqat 1 marta)
                                if (time_in_polygon >= self.threshold_violation and 
                                    not self.vehicle_tracking[track_id]['violation_saved']):
                                    self._save_event_image(frame, track_id, 'violation', box_coords, time_in_polygon, class_id=class_id)
                                    self.vehicle_tracking[track_id]['violation_saved'] = True
                            else:
                                # CHIQISH hodisasi
                                if (self.vehicle_tracking[track_id]['in_polygon'] and 
                                    not self.vehicle_tracking[track_id]['exit_saved']):
                                    time_in_polygon = self.vehicle_tracking[track_id]['total_time']
                                    self._save_event_image(frame, track_id, 'exit', box_coords, time_in_polygon, class_id=class_id)
                                    self.vehicle_tracking[track_id]['in_polygon'] = False
                                    self.vehicle_tracking[track_id]['exit_saved'] = True
                                
                                self.vehicle_tracking[track_id]['last_seen_time'] = self.current_time
            
            self._update_polygon_state()
            self._draw_simple_polygon(frame)
            objects_in_polygon = sum(1 for v in self.vehicle_tracking.values() if v['in_polygon'])
            #cv2.putText(frame, f"Inside: {objects_in_polygon}", 
            #(10, self.frame_height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Tracklarni chizish
            if self.last_detection_result and self.last_detection_result[0].boxes is not None:
                boxes = self.last_detection_result[0].boxes
                for box in boxes:
                    if box.id is not None:
                        track_id = int(box.id[0])
                        if track_id in self.vehicle_tracking:
                            data = self.vehicle_tracking[track_id]
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            class_id = data['class_id']
                            
                            box_color = self._get_box_color(data['total_time']) if data['in_polygon'] else self.color_outside
                            time_text = f"{data['total_time']:.1f}s" if data['in_polygon'] else "Tashqarida"
                            
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(frame, f"ID:{track_id}  -  {time_text}", (x1, y1-40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            #cv2.putText(frame, time_text, (x1, y1-20), 
                            #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Ma'lumotlar
            cv2.putText(frame, f"{self.camera_name} | FPS: {self.current_fps:.1f} | Frame: {self.frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (227, 30, 206), 2)
            cv2.putText(frame, f"Count: {self.passed_count}  | Inside: {objects_in_polygon}", 
            (10, self.frame_height-90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if self.adaptive_mode:
                mode_text = f"{'ACTIVE' if self.current_frame_skip <= 2 else 'IDLE'} (1/{self.current_frame_skip})"
                cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            H, W = frame.shape[:2]
            frame = cv2.resize(frame, (W//2, H//2))
            
            # Har bir kamera uchun alohida window
            window_name = f"Camera {self.camera_id} - {self.camera_name}"
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break
        
        self.cap.release()
        cv2.destroyWindow(window_name)
        print(f"[INFO] Kamera {self.camera_id} to'xtatildi.")
    
    def stop(self):
        self.running = False


class MultiCameraSystem:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Image saver yaratish (bitta umumiy)
        self.image_saver = ImageSaver(save_dir='saved_images')
        
        # Model path va config
        self.model_path = self.config['model']['path']
        self.model_config_data = {
            'target_classes': self.config['model']['target_classes'],
            'class_names': self.config['model']['class_names']
        }
        
        self.thresholds_config = self.config['thresholds']
        self.processing_config = self.config['processing']
        
        # CUDA optimizatsiyasi
        if torch.cuda.is_available():
            print(f"[INFO] CUDA mavjud: {torch.cuda.get_device_name(0)}")
            torch.multiprocessing.set_start_method('spawn', force=True)
        
        # Enabled kameralarni topish
        self.cameras = []
        self.threads = []
        
        for cam_config in self.config['cameras']:
            if cam_config.get('enabled', True):
                try:
                    camera = PolygonCamera(
                        cam_config, 
                        self.model_path,
                        self.model_config_data, 
                        self.thresholds_config, 
                        self.processing_config,
                        self.image_saver
                    )
                    self.cameras.append(camera)
                    print(f"[SUCCESS] Kamera {cam_config['id']} - {cam_config['name']} tayyor")
                except Exception as e:
                    print(f"[ERROR] Kamera {cam_config['id']} xato: {e}")
    
    def start(self):
        if not self.cameras:
            print("[ERROR] Hech qanday faol kamera topilmadi!")
            return
        
        print(f"\n[INFO] {len(self.cameras)} ta kamera ishga tushirilmoqda...\n")
        
        # Har bir kamera uchun thread yaratish
        for camera in self.cameras:
            thread = threading.Thread(target=camera.run, daemon=True)
            thread.start()
            self.threads.append(thread)
            time.sleep(0.05)
        
        # Barcha threadlarni kutish
        try:
            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            print("\n[INFO] Dastur to'xtatilmoqda...")
            for camera in self.cameras:
                camera.stop()
            self.image_saver.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        system = MultiCameraSystem('config.yaml')
        system.start()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()