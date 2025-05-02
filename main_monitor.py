import sys
import time
import threading
import signal
import logging
from ultralytics import YOLO

# Módulos propios
from config_loader import load_config
from camera_processor import CameraProcessor
from notifications import GoogleHomeNotificationHandler # Importar handlers necesarios

# --- Configuración Global ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        #logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
CONFIG_FILE = 'config.yaml'

def main():
    logger.info("Iniciando Sistema de Monitoreo de Cámaras")
    logger.info(f"Usando Python {sys.version}")
    logger.info(f"Hora actual: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}") # Incluye zona horaria si está configurada

    # --- Cargar Configuración ---
    config = load_config(CONFIG_FILE)
    if not config:
        logger.critical("Error crítico: No se pudo cargar la configuración. Saliendo.")
        sys.exit(1)

    detection_config = config.get('detection', {})
    alerting_config = config.get('alerting', {})

    # --- Cargar Modelo YOLO (una sola vez) ---
    model_name = detection_config.get('model', 'yolov8n.pt')
    logger.info(f"Cargando modelo YOLO global: {model_name}...")
    try:
        # Podrías añadir device='cuda' o device='cpu' si es necesario
        yolo_model = YOLO(model_name)
        # Ejecutar una predicción dummy para 'calentar' el modelo si es necesario
        _ = yolo_model.predict(source=[(255*__import__('numpy').random.rand(480, 640, 3)).astype(__import__('numpy').uint8)], verbose=False)
        logger.info("Modelo YOLO cargado y listo.")
    except Exception as e:
        logger.critical(f"Error crítico al cargar el modelo YOLO '{model_name}': {e}")
        sys.exit(1)

    # --- Instanciar Manejador de Notificaciones ---
    google_home_config = alerting_config.get('google_home', {})
    notifier = GoogleHomeNotificationHandler(
        google_home_config.get('device_name', 'Google Home'),
        google_home_config.get('sound_server_url', 'http://localhost:8000')
    )

    # --- Preparar Hilos ---
    threads = []
    stop_event = threading.Event() # Evento para señalar la detención

    # --- Crear e Iniciar un Hilo por Cámara ---
    global_detection_params = {
         'confidence_threshold': detection_config.get('confidence_threshold', 0.5),
         'classes_to_detect': detection_config.get('classes_to_detect', ['person'])
    }
    global_alert_cooldown = alerting_config.get('cooldown_seconds', 60)

    logger.info(f"Configurando {len(config['cameras'])} cámaras...")
    for camera_cfg in config['cameras']:
        # Pasar parámetros globales si no están definidos específicamente en la cámara
        camera_cfg.setdefault('confidence_threshold', global_detection_params['confidence_threshold'])
        camera_cfg.setdefault('classes_to_detect', global_detection_params['classes_to_detect'])
        camera_cfg.setdefault('alert_cooldown_seconds', global_alert_cooldown)
        camera_cfg.setdefault('motion_detection', detection_config.get('motion_detection', {}))

        processor = CameraProcessor(camera_cfg, yolo_model, notifier, stop_event, display=False)
        threads.append(processor)
        processor.start()
        time.sleep(0.5) # Pequeña pausa entre inicios de hilo para evitar sobrecarga inicial

    logger.info("Monitoreo Activo")
    logger.info("Presiona Ctrl+C para detener.")

    # --- Manejo de Señal de Interrupción (Ctrl+C) ---
    def signal_handler(sig, frame):
        logger.info("¡Ctrl+C detectado! Iniciando apagado ordenado...")
        stop_event.set() # Señalar a todos los hilos que se detengan

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler) # Manejar también señal TERM

    # --- Esperar a que los hilos terminen (o el evento de parada se active) ---
    try:
        # Mantener el hilo principal vivo mientras los otros hilos corren
        while not stop_event.is_set():
            # Podríamos comprobar aquí si los hilos siguen vivos (is_alive())
            # y relanzarlos si fallan, pero eso añade complejidad.
            time.sleep(1) # Esperar revisando el evento de parada

    except Exception as e:
         logger.error(f"Error inesperado en el hilo principal: {e}")
         stop_event.set() # Asegurarse de detener todo si hay un error aquí

    finally:
        # --- Limpieza Final ---
        logger.info("Esperando a que todos los hilos finalicen...")
        if not stop_event.is_set():
             stop_event.set() # Asegurarse de que el evento esté activo

        for t in threads:
             # Esperar un tiempo razonable a que cada hilo termine
             t.join(timeout=10) # Esperar hasta 10 segundos por hilo
             if t.is_alive():
                 logger.warning(f"Advertencia: El hilo para {t.cam_name} no finalizó a tiempo.")

        logger.info("Todos los hilos han sido procesados.")
        logger.info("Sistema de Monitoreo Finalizado")

if __name__ == "__main__":
    main()