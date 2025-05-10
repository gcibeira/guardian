import cv2
import time
import logging
import numpy as np

class CameraManager:
    """
    Gestiona la conexión al stream, reconexión inteligente y lectura de un solo frame.
    También expone un método para mostrar frames con cv2.imshow().
    """
    def __init__(
        self,
        url: str,
        reconnect_interval: float = 5.0,
        window_name: str = None,
    ):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.window_name = window_name
        self.cap = None
        self.last_attempt = 0.0
        self.logger = logging.getLogger(self.__class__.__name__)
        self._connect()

    def _connect(self):
        """Intenta abrir la conexión al stream."""
        self.last_attempt = time.time()
        if self.cap:
            self.cap.release()
        self.logger.info(f"Conectando a {self.url}…")
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            self.logger.error(f"No se pudo abrir stream {self.url}")
            self.cap = None

    def read_frame(self) -> np.ndarray:
        """
        Devuelve un único frame BGR.
        Si falla la lectura, la conexión se cierra y retorna None.
        Tras el intervalo de reconexión, intenta reconectar automáticamente.
        """
        # Si no hay conexión activa, intentar reconectar tras el intervalo
        if self.cap is None:
            if time.time() - self.last_attempt >= self.reconnect_interval:
                self._connect()
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.logger.warning("Lectura fallida; reconectando…")
            self.cap.release()
            self.cap = None
            return None

        return frame

    def display(self, frame) -> bool:
        """
        Muestra el frame en ventana si window_name está definido.
        Devuelve False si el usuario presionó 'q' para salir.
        """
        if self.window_name:
            cv2.imshow(self.window_name, cv2.resize(frame, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    def cleanup(self):
        """Libera recursos al finalizar."""
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.window_name:
            cv2.destroyWindow(self.window_name)
