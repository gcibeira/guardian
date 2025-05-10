import math
import time
import logging
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class TrackedObject:
    id: int
    box: tuple
    label: str
    confidence: float
    last_seen: float

class Tracker:
    """
    Asocia detecciones a IDs persistentes usando distancia de centroides.
    Permite reaparecer objetos tras breves pérdidas.
    """
    def __init__(self, max_missing_frames: int = 5, dist_threshold: float = 50.0):
        self.max_missing_frames = max_missing_frames
        self.dist_threshold = dist_threshold
        self.next_id = 0
        self.tracks: Dict[int, Dict] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def update(self, detections: List[TrackedObject]) -> List[TrackedObject]:
        updated = {}
        used_ids = set()

        # Asociar detecciones a tracks existentes
        for det in detections:
            x1, y1, x2, y2 = det.box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            best_id, best_dist = None, self.dist_threshold

            for tid, tr in self.tracks.items():
                tx, ty = tr['centroid']
                d = math.hypot(cx - tx, cy - ty)
                if d < best_dist:
                    best_id, best_dist = tid, d

            if best_id is None:
                tid = self.next_id
                self.next_id += 1
            else:
                tid = best_id

            updated[tid] = {
                'centroid': (cx, cy),
                'box': det.box,
                'label': det.label,
                'confidence': det.confidence,
                'missing': 0
            }
            used_ids.add(tid)

        # Incrementar missing para tracks no actualizados
        for tid, tr in self.tracks.items():
            if tid not in used_ids:
                tr['missing'] += 1
                if tr['missing'] <= self.max_missing_frames:
                    updated[tid] = tr

        self.tracks = updated

        # Construir lista de salida
        now = time.monotonic()
        out = []
        for tid, tr in self.tracks.items():
            out.append(TrackedObject(
                id=tid,
                box=tr['box'],
                label=tr['label'],
                confidence=tr['confidence'],
                last_seen=now
            ))
        return out
