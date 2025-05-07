from ultralytics import YOLO
from typing import List, Tuple
from dataclasses import dataclass
import logging

@dataclass
class Detection:
    box: Tuple[int, int, int, int]
    label: str
    confidence: float

class ObjectDetector:
    """
    Wrapper de YOLO que filtra por clases y umbral de confianza.
    """
    def __init__(
        self,
        model_path: str,
        classes: List[str] = None,
        confidence_threshold: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.classes = classes or []
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect(self, frame) -> List[Detection]:
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception as e:
            self.logger.error(f"ObjectDetector error: {e}")
            return []

        dets: List[Detection] = []
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                label = self.model.names[cls]
                if self.classes and label not in self.classes:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                dets.append(Detection(box=(x1, y1, x2, y2), label=label, confidence=conf))
        return dets
