import cv2
import logging
from typing import Tuple

class MotionDetector:
    """
    Detecta movimiento significativo comparando el frame actual con el previo,
    respetando un skip_frames para aligerar carga de CPU.
    """
    def __init__(
        self,
        threshold: int = 25,
        blur_kernel: Tuple[int, int] = (21, 21),
        min_area: int = 5000,
    ):
        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self.min_area = min_area

        self.prev_gray = None
        self.frame_count = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect(self, frame) -> bool:
        """
        Retorna True si hay movimiento significativo en este frame.
        Internamente actualiza self.prev_gray y usa skip_frames.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        self.prev_gray = gray

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            if cv2.contourArea(c) >= self.min_area:
                return True
        return False
