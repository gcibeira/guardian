import threading
import time
import logging
from pathlib import Path
import cv2

from camera_manager import CameraManager
from motion_detector import MotionDetector
from object_detector import ObjectDetector
from tracker import Tracker
from linger_detector import LingerDetector
from overlay_renderer import OverlayRenderer
from notifications import NotificationManager

logger = logging.getLogger(__name__)

class CameraProcessor(threading.Thread):
    """
    Subclase de Thread que instancia todos los componentes por cámara
    y ejecuta el bucle principal de procesamiento de frames.
    """
    def __init__(
        self,
        cam_cfg,
        detection_cfg,
        model_path: str,
        notifier: NotificationManager,
        stop_event: threading.Event,
        display: bool,
    ):
        super().__init__(daemon=True, name=cam_cfg.name)
        self.cam_cfg = cam_cfg
        self.detection_cfg = detection_cfg
        self.model_path = model_path
        self.notifier = notifier
        self.stop_event = stop_event
        self.display_flag = display
        self.skip_frames = detection_cfg.skip_frames
        self.force_interval = detection_cfg.force_interval
        self.frame_count = 0
        self.tracked = []
        self.linger_events = []

        # Instanciar módulos
        self.camera = CameraManager(
            cam_cfg.url,
            reconnect_interval=5.0,
            window_name=cam_cfg.name if display else None,
        )
        self.motion = MotionDetector(
            threshold=cam_cfg.motion_detection.threshold,
            blur_kernel=tuple(cam_cfg.motion_detection.blur_kernel),
            min_area=cam_cfg.motion_detection.min_area,
        )
        self.detector = ObjectDetector(
            model_path=self.model_path,
            classes=cam_cfg.classes_to_detect,
            confidence_threshold=cam_cfg.confidence_threshold,
        )
        self.tracker = Tracker(
            max_missing_frames=cam_cfg.linger_detection.max_missing_frames,
            dist_threshold=cam_cfg.linger_detection.tracking_distance_threshold,
        )
        self.linger = LingerDetector(
            roi=tuple(cam_cfg.linger_detection.roi),
            linger_time=cam_cfg.linger_detection.linger_time_seconds,
        )
        self.renderer = OverlayRenderer(roi=tuple(cam_cfg.linger_detection.roi))
        self.detections = []

    def run(self):
        while not self.stop_event.is_set():
            frame = self.camera.read_frame()
            # Si no hay frame disponible, esperar un poco y reintentar
            if frame is None:
                continue
            
            self.frame_count += 1
            if not self.frame_count % self.skip_frames:
                now = time.monotonic()
                moved = self.motion.detect(frame)
                if moved or self.frame_count % self.force_interval == 0:
                    self.detections = self.detector.detect(frame)

                self.tracked = self.tracker.update(self.detections)
                self.linger_events = self.linger.update(self.tracked, now)
                
                # 7) Guardar snapshots y notificar
                for linger_event in self.linger_events:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"{self.name}_{linger_event.label}_{timestamp}.jpg"
                    out_dir = Path(self.cam_cfg.save_directory)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    path = out_dir / fname
                    cv2.imwrite(str(path), frame)
                    # Enviar alerta
                    self.notifier.send(
                        camera_name=self.name,
                        subject_detail=linger_event.label,
                        linger_event=linger_event,
                        frame_img=frame
                    )
                

            # 8) Mostrar en ventana
            annotated = self.renderer.render(frame, self.tracked, self.linger_events)
            if self.display_flag and not self.camera.display(annotated):
                self.stop_event.set()
                break

        # Limpieza al terminar
        self.camera.cleanup()