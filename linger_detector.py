from dataclasses import dataclass
from typing import List, Tuple, Dict
import logging
import time

@dataclass
class LingerEvent:
    id: int
    duration: float
    box: tuple
    label: str

class LingerDetector:
    """
    Emite un event cuando un objeto trackeado permanece dentro de la ROI
    más tiempo del umbral configurado.
    """
    def __init__(self, roi: Tuple[int,int,int,int], linger_time: float):
        self.roi = roi
        self.linger_time = linger_time
        self.enter_times: Dict[int, float] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def update(self, tracked: List, now: float = None) -> List[LingerEvent]:
        now = now or time.monotonic()
        events: List[LingerEvent] = []
        x1, y1, x2, y2 = self.roi

        for obj in tracked:
            cx = (obj.box[0] + obj.box[2]) // 2
            cy = (obj.box[1] + obj.box[3]) // 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if obj.id not in self.enter_times:
                    self.enter_times[obj.id] = now
                else:
                    dur = now - self.enter_times[obj.id]
                    if dur >= self.linger_time:
                        events.append(LingerEvent(
                            id=obj.id, duration=dur, box=obj.box, label=obj.label
                        ))
            else:
                # salió de la ROI: reiniciamos timer
                self.enter_times.pop(obj.id, None)

        return events
