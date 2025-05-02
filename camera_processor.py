import threading
import time
import logging
import math
import os
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

from notifications import NotificationHandler

logger = logging.getLogger(__name__)

class CameraProcessor(threading.Thread):
    def __init__(
        self,
        camera_config: Dict[str, Any],
        yolo_model: YOLO,
        notifier: NotificationHandler,
        stop_event: threading.Event,
        display: bool = False
    ):
        super().__init__(daemon=True)
        self.cam_name = camera_config['name']
        self.cam_url = camera_config['url']
        self.notifier = notifier
        self.stop_event = stop_event
        self.display = display

        self.conf_threshold: float = camera_config.get('confidence_threshold', 0.5)
        self.target_classes: List[str] = camera_config.get('classes_to_detect', ['person'])

        md = camera_config.get('motion_detection', {})
        self.motion_enabled: bool = md.get('enabled', False)
        self.skip_frames: int = md.get('skip_frames', 5)
        self.min_area: int = md.get('min_area', 5000)
        self.motion_thresh: int = md.get('threshold', 25)
        self.blur_kernel: tuple = tuple(md.get('blur_kernel', (21, 21)))

        ld = camera_config.get('linger_detection', {})
        self.linger_enabled: bool = ld.get('enabled', False)
        self.roi: Optional[tuple] = tuple(ld.get('roi', [])) if self.linger_enabled else None
        self.linger_time: float = ld.get('linger_time_seconds', 5)
        self.tracking_dist: float = ld.get('tracking_distance_threshold', 75)

        self.cooldown: float = camera_config.get('alert_cooldown_seconds', 60)
        self._last_general_alert: float = 0.0

        self.yolo_model = yolo_model
        self.class_indices: Optional[List[int]] = self._map_class_indices(self.target_classes)

        self.cap: Optional[cv2.VideoCapture] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_count: int = 0
        self.detections: List[Dict[str, Any]] = []

        self.tracker: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 0

        self.detections_dir = camera_config.get('save_directory', './detections')
        os.makedirs(self.detections_dir, exist_ok=True)

        logger.info(f"{self.cam_name}: Iniciado (motion={self.motion_enabled}, linger={self.linger_enabled})")

    def _map_class_indices(self, names: List[str]) -> Optional[List[int]]:
        if not names:
            return None
        mapping = []
        rev = {v: k for k, v in self.yolo_model.names.items()}
        for n in names:
            idx = rev.get(n)
            if idx is None:
                logger.warning(f"{self.cam_name}: Clase '{n}' no encontrada en modelo; se ignora filtro.")
            else:
                mapping.append(idx)
        return mapping if mapping else None

    def _connect(self) -> bool:
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.cam_url)
        if not self.cap.isOpened():
            logger.error(f"{self.cam_name}: No se pudo abrir stream.")
            return False
        logger.info(f"{self.cam_name}: Stream conectado.")
        return True

    def run(self):
        while not self.stop_event.is_set() and not self._connect():
            time.sleep(5)

        while not self.stop_event.is_set():
            ret, frame = self.cap.read() if self.cap else (False, None)
            if not ret or frame is None:
                logger.error(f"{self.cam_name}: Lectura de frame fallida; reconectando...")
                if not self._connect():
                    time.sleep(5)
                continue
            try:
                self._process(frame)
            except Exception:
                logger.exception(f"{self.cam_name}: Error en _process().")
        self._cleanup()

    def _process(self, frame: np.ndarray):
        now = time.monotonic()
        self.frame_count += 1

        if self.motion_enabled and (self.frame_count % self.skip_frames != 0):
            return

        moved = True
        if self.motion_enabled:
            moved = self._detect_motion(frame)

        if moved:
            self.detections = self._detect_objects(frame)

        if self.linger_enabled and self.roi:
            self._draw_roi(frame)
            if moved:
                self._detect_linger(self.detections, frame, now)
            else:
                self._update_linger_timing(frame, now)

        self._draw_boxes(frame, self.detections)
        self._maybe_general_alert(self.detections, frame, now)

        if self.display:
            cv2.imshow(self.cam_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()

    def _detect_motion(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
        delta = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(delta, self.motion_thresh, 255, cv2.THRESH_BINARY)
        dil = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.prev_gray = gray
        return any(cv2.contourArea(c) > self.min_area for c in cnts)

    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        try:
            results = self.yolo_model.predict(frame, conf=self.conf_threshold, classes=self.class_indices, verbose=False)
        except Exception as e:
            logger.error(f"{self.cam_name}: YOLO fallo: {e}")
            return []
        dets: List[Dict[str, Any]] = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                name = self.yolo_model.names.get(cls, str(cls))
                dets.append({'name': name, 'confidence': conf, 'box': (x1, y1, x2, y2)})
        return dets

    def _draw_boxes(self, frame: np.ndarray, dets: List[Dict[str, Any]]):
        for d in dets:
            x1, y1, x2, y2 = d['box']
            lbl = f"{d['name']} {d['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    def _draw_roi(self, frame: np.ndarray):
        x1,y1,x2,y2 = self.roi
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 1)

    def _save_frame(self, frame: np.ndarray, alert_type: str):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.detections_dir}/{self.cam_name}_{alert_type}_{timestamp}.jpg"
        try:
            cv2.imwrite(filename, frame)
            logger.info(f"{self.cam_name}: Frame guardado en {filename}")
        except Exception as e:
            logger.error(f"{self.cam_name}: Error al guardar frame: {e}")

    def _detect_linger(self, dets: List[Dict[str, Any]], frame: np.ndarray, now: float):
        pts = []
        for d in dets:
            if d['name'] == 'person':
                x1,y1,x2,y2 = d['box']
                cx, cy = (x1+x2)//2, (y1+y2)//2
                if self.roi[0] <= cx <= self.roi[2] and self.roi[1] <= cy <= self.roi[3]:
                    pts.append((cx,cy,d))
        active = set()
        for cx,cy,d in pts:
            bid, mind = None, self.tracking_dist
            for tid,data in self.tracker.items():
                px,py = data['centroid']
                dist = math.hypot(cx-px, cy-py)
                if dist < mind:
                    mind, bid = dist, tid
            if bid is not None:
                self.tracker[bid].update({'centroid':(cx,cy),'last_ts':now,'box':d})
                active.add(bid)
            else:
                self.tracker[self._next_id] = {
                    'centroid':(cx,cy),
                    'first_ts':now,
                    'last_ts':now,
                    'alerted':False,
                    'box':d
                }
                active.add(self._next_id)
                self._next_id += 1
        for tid in list(self.tracker):
            if tid not in active:
                del self.tracker[tid]
        for tid,data in self.tracker.items():
            dur = data['last_ts'] - data['first_ts']
            cx,cy = data['centroid']
            cv2.putText(frame, f"ID{tid} {dur:.1f}s", (cx-20, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)
            if dur>self.linger_time and not data['alerted']:
                logger.info(f"{self.cam_name}: LINGER alerta ID{tid} ({dur:.1f}s)")
                try:
                    self.notifier.send_alert(self.cam_name, "[LINGER]", [data['box']], frame)
                    self._save_frame(frame, "LINGER")
                except Exception as e:
                    logger.error(f"{self.cam_name}: fallo alerta linger: {e}")
                data['alerted']=True

    def _update_linger_timing(self, frame: np.ndarray, now: float):
        for tid, data in self.tracker.items():
            data['last_ts'] = now
            dur = data['last_ts'] - data['first_ts']
            cx, cy = data['centroid']
            cv2.putText(frame, f"ID{tid} {dur:.1f}s", (cx-20, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)
            if dur > self.linger_time and not data['alerted']:
                logger.info(f"{self.cam_name}: LINGER alerta ID{tid} ({dur:.1f}s)")
                try:
                    # TODO: corregir codigo repetido. Se repite en detect_linger y aqui
                    self.notifier.send_alert(self.cam_name, "[LINGER]", [data['box']], frame)
                    self._save_frame(frame, "LINGER")
                except Exception as e:
                    logger.error(f"{self.cam_name}: fallo alerta linger: {e}")
                data['alerted'] = True

    def _maybe_general_alert(self, dets: List[Dict[str, Any]], frame: np.ndarray, now: float):
        to_alert = [d for d in dets if not (self.linger_enabled and d['name'] == 'person')]
        if to_alert and (now - self._last_general_alert) > self.cooldown:
            logger.info(f"{self.cam_name}: GENERAL alerta, objetos: {[d['name'] for d in to_alert]}")
            try:
                self.notifier.send_alert(self.cam_name, "[GENERAL]", to_alert, frame)
                self._save_frame(frame, "GENERAL")
            except Exception as e:
                logger.error(f"{self.cam_name}: fallo alerta general: {e}")
            self._last_general_alert = now

    def _cleanup(self):
        logger.info(f"{self.cam_name}: Limpiando recursos.")
        if self.cap:
            self.cap.release()
        if self.display:
            cv2.destroyAllWindows()
