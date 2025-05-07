import threading
import time
import logging
from camera_manager import CameraManager
from motion_detector import MotionDetector
from object_detector import ObjectDetector
from tracker import Tracker
from linger_detector import LingerDetector
from overlay_renderer import OverlayRenderer
from alert_manager import AlertManager
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

        # Instanciar módulos
        self.camera = CameraManager(
            cam_cfg.url,
            reconnect_interval=5.0,
            window_name=cam_cfg.name if display else None,
        )
        self.motion = MotionDetector(
            skip_frames=cam_cfg.motion_detection.skip_frames,
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
            max_missing=5,
            dist_threshold=cam_cfg.linger_detection.tracking_distance_threshold,
        )
        self.linger = LingerDetector(
            roi=tuple(cam_cfg.linger_detection.roi),
            linger_time=cam_cfg.linger_detection.linger_time_seconds,
        )
        self.renderer = OverlayRenderer(roi=tuple(cam_cfg.linger_detection.roi))
        self.alert_mgr = AlertManager(general_cooldown=cam_cfg.alert_cooldown_seconds)

        self.detections = []

    def run(self):
        while not self.stop_event.is_set():
            frame = self.camera.read_frame()
            if frame is None:
                # Si no hay frame disponible, esperar un poco y reintentar
                time.sleep(0.1)
                continue

            now = time.monotonic()

            # 1) Detección de movimiento
            moved = self.motion.detect(frame)

            # 2) Detecciones de objetos
            if moved:
                self.detections = self.detector.detect(frame)
                if self.display_flag and not self.camera.display(frame):
                    self.stop_event.set()
                    break

            # 3) Tracking robusto
            tracked = self.tracker.update(self.detections)

            annotated = self.renderer.render(frame, tracked)
            if self.display_flag and not self.camera.display(annotated):
                self.stop_event.set()
                break

            '''
            # 4) Detección de permanencia en ROI
            linger_events = self.linger.update(tracked, now)

            # 5) Evaluar alertas
            # Ignorar detecciones de "person" si hay linger activo
            general = [
                d for d in self.detections
                if not (d.label == "person" and linger_events)
            ]
            alerts = self.alert_mgr.evaluate(
                general=general,
                linger=linger_events,
                timestamp=now
            )

            # 6) Renderizar overlays
            annotated = self.renderer.render(frame, tracked, linger_events)

            # 7) Guardar snapshots y notificar
            for alert in alerts:
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = f"{self.name}_{alert.type}_{ts}.jpg"
                out_dir = Path(self.cam_cfg.save_directory)
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / fname
                cv2.imwrite(str(path), annotated)
                # Enviar alerta
                self.notifier.send(
                    camera_name=self.name,
                    subject_detail=alert.type,
                    detected_objects=alert.objects,
                    frame_img=annotated
                )

            # 8) Mostrar en ventana
            if self.display_flag and not self.camera.display(annotated):
                self.stop_event.set()
                break
            '''

        # Limpieza al terminar
        self.camera.cleanup()