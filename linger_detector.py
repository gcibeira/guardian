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
    def __init__(self, roi: Tuple[int, int, int, int], linger_time: float):
        self.roi = roi
        self.linger_time = linger_time
        self.objects_in_roi: Dict[int, Dict[str, float]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def update(self, tracked: List, now: float = None) -> List[LingerEvent]:
        now = now or time.monotonic()
        events: List[LingerEvent] = []
        x1, y1, x2, y2 = self.roi

        for obj in tracked:
            cx = (obj.box[0] + obj.box[2]) // 2
            cy = (obj.box[1] + obj.box[3]) // 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                # El objeto está dentro de la ROI
                if obj.id not in self.objects_in_roi:
                    self.objects_in_roi[obj.id] = {"enter_time": now, "alert_emitted": False}
                    self.logger.debug(f"Objeto {obj.id} entró en la ROI.")
                else:
                    linger_duration = now - self.objects_in_roi[obj.id]["enter_time"]
                    if linger_duration > self.linger_time and not self.objects_in_roi[obj.id]["alert_emitted"]:
                        # Emitir evento si supera linger_time y no se ha emitido alerta
                        events.append(LingerEvent(
                            id=obj.id,
                            duration=linger_duration,
                            box=obj.box,
                            label=obj.label
                        ))
                        self.logger.info(f"Evento emitido para objeto {obj.id}.")
                        self.objects_in_roi[obj.id]["alert_emitted"] = True
            else:
                # El objeto está fuera de la ROI
                if obj.id in self.objects_in_roi:
                    self.logger.debug(f"Objeto {obj.id} salió de la ROI.")
                    del self.objects_in_roi[obj.id]  # Limpiar registro al salir

        return events
