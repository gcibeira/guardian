# notifications.py
import abc
import time
import logging
import pychromecast

logger = logging.getLogger(__name__)

# TODO: Implementar array de notificadores para enviar a múltiples destinos (ej. Email, Telegram, Google Home, etc.)
class NotificationHandler(abc.ABC):
    """Clase base abstracta para manejadores de notificaciones."""

    @abc.abstractmethod
    def send_alert(self, camera_name, subject_detail, detected_objects, frame_img):
        """
        Envía una alerta.

        Args:
            camera_name (str): Nombre de la cámara que generó la alerta.
            subject_detail (str): Detalle para el asunto (ej. "[LINGER]", "[GENERAL]").
            detected_objects (list): Lista de diccionarios con la info de los objetos detectados.
            frame_img (numpy.ndarray): Imagen (frame) con la detección (puede ser None).
        """
        pass

class EmailNotificationHandler(NotificationHandler):
    """Implementación (Placeholder) para notificaciones por Email."""
    def __init__(self, email_config):
        self.config = email_config
        if not self.config or not self.config.get('enabled', False):
            logger.info("Email Notifier: Deshabilitado por configuración.")
            self.enabled = False
        else:
            # Validar configuración esencial
            required_keys = ['smtp_server', 'smtp_port', 'sender_email', 'sender_password', 'recipient_email']
            if not all(key in self.config for key in required_keys):
                logger.warning("Email Notifier: Configuración incompleta. Deshabilitado.")
                self.enabled = False
            else:
                logger.info("Email Notifier: Inicializado y habilitado.")
                self.enabled = True

    def send_alert(self, camera_name, subject_detail, detected_objects, frame_img):
        if not self.enabled:
            return

        # --- IMPLEMENTACIÓN FUTURA DE SMTP ---
        logger.info("--- NOTIFICACIÓN EMAIL (Placeholder) ---")
        logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cámara: {camera_name}")
        logger.info(f"Tipo: {subject_detail}")
        obj_desc = [f"{obj['name']} ({obj['confidence']:.2f})" for obj in detected_objects]
        logger.info(f"Objetos: {', '.join(obj_desc)}")
        logger.info(f"Imagen Adjunta: {'Sí' if frame_img is not None else 'No'}")
        logger.info(f"Destinatario: {self.config.get('recipient_email')}")
        logger.info("------------------------------------")
        # Simular un pequeño delay de envío
        time.sleep(0.1)

class NoOpNotificationHandler(NotificationHandler):
    """Implementación que no hace nada (útil para pruebas o desactivar alertas)."""
    def __init__(self, *args, **kwargs):
        logger.info("NoOp Notifier: Inicializado. No se enviarán alertas.")
        pass

    def send_alert(self, camera_name, subject_detail, detected_objects, frame_img):
        # No hace nada
        pass

class GoogleHomeNotificationHandler(NotificationHandler):
    """Implementación para notificaciones a través de Google Home."""
    def __init__(self, device_name, sound_server_url):
        self.device_name = device_name
        self.sound_server_url = sound_server_url.rstrip('/')  # Asegurar que no termine con '/'
        self.cast = None
        self.browser = None

        try:
            # Buscar dispositivos Chromecast
            chromecasts, self.browser = pychromecast.get_chromecasts()
            self.cast = next((cc for cc in chromecasts if cc.cast_info.friendly_name == self.device_name), None)
            if self.cast:
                logger.info(f"Google Home Notifier: Conectando a '{self.device_name}'...")
                self.cast.wait()
                logger.info(f"Google Home Notifier: Dispositivo '{self.device_name}' conectado y listo.")
            else:
                logger.error(f"Google Home Notifier: Dispositivo '{self.device_name}' no encontrado.")
        except Exception as e:
            logger.error(f"Google Home Notifier: Error al inicializar - {e}")
        finally:
            if self.browser:
                self.browser.stop_discovery()

    def send_alert(self, camera_name, subject_detail, detected_objects, frame_img):
        if not self.cast:
            logger.warning("Google Home Notifier: No se puede enviar la notificación, dispositivo no inicializado.")
            return

        try:
            if detected_objects:
                for obj in detected_objects:
                    sound_file = f"{obj['name'].lower()}.mp3"
                    sound_url = f"{self.sound_server_url}/{sound_file}"
                    logger.info(f"Google Home Notifier: Enviando sonido para objeto '{obj['name']}' desde {sound_url}")
                    mc = self.cast.media_controller
                    mc.play_media(sound_url, "audio/mpeg")
                    mc.block_until_active()
            else:
                logger.info("Google Home Notifier: No se detectaron objetos, no se enviará sonido.")
        except Exception as e:
            logger.error(f"Google Home Notifier: Error al enviar la notificación - {e}")

