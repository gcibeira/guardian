# main_monitor.py

import sys
import time
import threading
import signal
import logging
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from config_loader import load_config, ConfigError
from camera_manager import CameraManager
from motion_detector import MotionDetector
from object_detector import ObjectDetector
from tracker import Tracker
from linger_detector import LingerDetector
from overlay_renderer import OverlayRenderer
from alert_manager import AlertManager
from notifications import NotificationManager


class CameraThread(threading.Thread):
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

        self.last_detections = []

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
                detections = self.detector.detect(frame)
                self.last_detections = detections
            else:
                # Persistir detecciones previas para mantener overlays
                detections = self.last_detections

            # 3) Tracking robusto
            tracked = self.tracker.update(detections)

            # 4) Detección de permanencia en ROI
            linger_events = self.linger.update(tracked, now)

            # 5) Evaluar alertas
            # Ignorar detecciones de "person" si hay linger activo
            general = [
                d for d in detections
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

        # Limpieza al terminar
        self.camera.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Sistema de Monitoreo de Cámaras")
    parser.add_argument(
        "-c", "--config", default="config.yaml",
        help="Ruta al archivo de configuración YAML"
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Mostrar ventana de video por cámara"
    )
    parser.add_argument(
        "--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging"
    )
    return parser.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )


def main():
    args = parse_args()
    setup_logging(args.log)
    logger = logging.getLogger("main_monitor")

    # Cargar configuración
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        logger.critical(f"Error en configuración: {e}")
        sys.exit(1)

    # Instanciar manager de notificaciones
    notifier = NotificationManager(
        email_cfg=cfg.alerting.email,
        google_cfg=cfg.alerting.google_home
    )

    # Cargar modelo YOLO
    logger.info(f"Cargando modelo: {cfg.detection.model}")
    try:
        _ = YOLO(cfg.detection.model)  # precarga
    except Exception as e:
        logger.critical(f"Error cargando modelo YOLO: {e}")
        sys.exit(1)

    # Preparar y arrancar hilos por cámara
    stop_evt = threading.Event()
    threads = []

    for cam in cfg.cameras:
        t = CameraThread(
            cam_cfg=cam,
            detection_cfg=cfg.detection,
            model_path=cfg.detection.model,
            notifier=notifier,
            stop_event=stop_evt,
            display=args.display,
        )
        threads.append(t)
        t.start()
        time.sleep(0.2)  # escalonar conexiones

    # Señales de apagado limpio
    def shutdown(sig, frame):
        logger.info("Se recibió señal de terminación. Apagando...")
        stop_evt.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Esperar finalización
    while not stop_evt.is_set():
        time.sleep(1)
    for t in threads:
        t.join(timeout=5)
    logger.info("Sistema detenido.")


if __name__ == "__main__":
    main()
