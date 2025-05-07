import cv2
import logging
from typing import List, Tuple

from tracker import TrackedObject
from linger_detector import LingerEvent

class OverlayRenderer:
    """
    Dibuja bounding boxes, IDs y timers de linger sobre cada frame.
    """
    def __init__(self, roi: Tuple[int,int,int,int] = None):
        self.roi = roi
        self.logger = logging.getLogger(self.__class__.__name__)

    def render(
        self,
        frame,
        tracked: List[TrackedObject] = [],
        linger_events: List[LingerEvent] = []
    ):
        # Dibujar ROI si existe
        if self.roi:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Dibujar objetos trackeados
        for obj in tracked:
            x1, y1, x2, y2 = obj.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID{obj.id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # Dibujar eventos de linger
        for ev in linger_events:
            x1, y1, x2, y2 = ev.box
            cv2.putText(
                frame,
                f"Linger {ev.duration:.1f}s",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        return frame
