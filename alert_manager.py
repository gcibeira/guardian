import time
import logging
from dataclasses import dataclass
from typing import List

@dataclass
class Alert:
    type: str                # 'general' o 'linger'
    objects: List            # lista de Detection o LingerEvent

class AlertManager:
    """
    Decide cuándo lanzar alertas generales o de linger
    aplicando cooldowns y reglas de filtrado.
    """
    def __init__(self, general_cooldown: float = 60.0):
        self.general_cooldown = general_cooldown
        self.last_general = 0.0
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(
        self,
        general: List,
        linger: List,
        timestamp: float = None
    ) -> List[Alert]:
        timestamp = timestamp or time.monotonic()
        alerts: List[Alert] = []

        # Primero linger (si los hay)
        for ev in linger:
            self.logger.info(f"Linger alert ID{ev.id}")
            alerts.append(Alert(type="linger", objects=[ev]))

        # Luego alerta general (si venció cooldown)
        if general and (timestamp - self.last_general >= self.general_cooldown):
            self.logger.info(f"General alert: {len(general)} objetos")
            alerts.append(Alert(type="general", objects=general))
            self.last_general = timestamp

        return alerts
