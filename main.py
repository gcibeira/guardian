import sys
import time
import threading
import signal
import logging
import argparse
from ultralytics import YOLO
from config_loader import load_config, ConfigError
from notifications import NotificationManager
from camera_processor import CameraProcessor

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
    logger = logging.getLogger(__name__)

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
        t = CameraProcessor(
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
