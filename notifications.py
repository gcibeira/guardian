# notifications.py

import logging
import smtplib
from email.message import EmailMessage
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import pychromecast


logger = logging.getLogger(__name__)


class NotificationHandler(ABC):
    """Interfaz para handlers de notificación."""

    @abstractmethod
    def send_alert(
        self,
        camera_name: str,
        subject_detail: str,
        detected_objects: List[Dict[str, Any]],
        frame_img,
    ) -> None:
        """Envía la alerta al canal correspondiente."""
        pass


class EmailNotificationHandler(NotificationHandler):
    """Envía alertas por email usando SMTP."""

    def __init__(self, config):
        self.enabled = config.enabled
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def send_alert(
        self,
        camera_name: str,
        subject_detail: str,
        detected_objects: List[Dict[str, Any]],
        frame_img,
    ) -> None:
        if not self.enabled:
            return

        msg = EmailMessage()
        msg["Subject"] = f"{camera_name} {subject_detail}"
        msg["From"] = self.config.sender_email
        msg["To"] = self.config.recipient_email

        body = [
            f"Cámara: {camera_name}",
            f"Alerta: {subject_detail}",
            "Objetos detectados: "
            + ", ".join(f"{o['label']} ({o['confidence']:.2f})" for o in detected_objects),
        ]
        msg.set_content("\n".join(body))

        # Adjuntar imagen si existe
        if frame_img is not None:
            import cv2

            ok, buf = cv2.imencode(".jpg", frame_img)
            if ok:
                msg.add_attachment(
                    buf.tobytes(),
                    maintype="image",
                    subtype="jpeg",
                    filename=f"{camera_name}_{subject_detail}.jpg",
                )

        try:
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as smtp:
                smtp.starttls()
                smtp.login(self.config.sender_email, self.config.sender_password)
                smtp.send_message(msg)
                self.logger.info("Email enviado correctamente.")
        except Exception as e:
            self.logger.error(f"Error enviando email: {e}")


class GoogleHomeNotificationHandler(NotificationHandler):
    """Reproduce audio de alerta en Google Home vía Chromecast."""

    def __init__(self, config):
        self.enabled = config.enabled
        self.device_name = config.device_name
        self.server_url = config.sound_server_url.rstrip("/")
        self.cast = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_cast()

    def _init_cast(self):
        if not self.enabled:
            return
        try:
            chromecasts, browser = pychromecast.get_chromecasts()
            self.cast = next(
                (cc for cc in chromecasts if cc.cast_info.friendly_name == self.device_name),
                None,
            )
            if self.cast:
                self.cast.wait()
                self.logger.info(f"Conectado a Google Home '{self.device_name}'")
            else:
                self.logger.error(f"Dispositivo '{self.device_name}' no encontrado")
            browser.stop_discovery()
        except Exception as e:
            self.logger.error(f"Error inicializando Google Home: {e}")

    def send_alert(
        self,
        camera_name: str,
        subject_detail: str,
        detected_objects: List[Dict[str, Any]],
        frame_img,
    ) -> None:
        if not self.enabled or not self.cast or not detected_objects:
            return

        mc = self.cast.media_controller
        for obj in detected_objects:
            url = f"{self.server_url}/{obj['label'].lower()}.mp3"
            try:
                mc.play_media(url, "audio/mpeg")
                mc.block_until_active()
            except Exception as e:
                self.logger.error(f"Error reproduciendo en Google Home: {e}")


class NoOpNotificationHandler(NotificationHandler):
    """Handler que no hace nada (fallback)."""

    def send_alert(
        self,
        camera_name: str,
        subject_detail: str,
        detected_objects: List[Dict[str, Any]],
        frame_img,
    ) -> None:
        return


class NotificationManager:
    """
    Orquesta múltiples NotificationHandler según configuración.
    Si no hay ninguno habilitado, usa el NoOpNotificationHandler.
    """

    def __init__(self, email_cfg, google_cfg):
        self.handlers: List[NotificationHandler] = []
        if email_cfg.enabled:
            self.handlers.append(EmailNotificationHandler(email_cfg))
        if google_cfg.enabled:
            self.handlers.append(GoogleHomeNotificationHandler(google_cfg))
        if not self.handlers:
            self.handlers.append(NoOpNotificationHandler())

    def send(
        self,
        camera_name: str,
        subject_detail: str,
        detected_objects: List[Dict[str, Any]],
        frame_img,
    ) -> None:
        for handler in self.handlers:
            handler.send_alert(camera_name, subject_detail, detected_objects, frame_img)
