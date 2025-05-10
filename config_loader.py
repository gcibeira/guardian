# config_loader.py

import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


class ConfigError(Exception):
    """Error al cargar o validar la configuración."""


@dataclass
class MotionConfig:
    enabled: bool = False
    min_area: int = 5000
    threshold: int = 25
    blur_kernel: List[int] = field(default_factory=lambda: [21, 21])


@dataclass
class LingerConfig:
    enabled: bool = False
    roi: Optional[List[int]] = None
    linger_time_seconds: float = 5.0
    tracking_distance_threshold: float = 75.0
    max_missing_frames: int = 5


@dataclass
class CameraConfig:
    name: str
    url: str
    confidence_threshold: float
    classes_to_detect: List[str]
    motion_detection: MotionConfig
    linger_detection: LingerConfig
    alert_cooldown_seconds: float
    save_directory: Path


@dataclass
class DetectionConfig:
    model: str = "yolov8n.pt"
    classes_to_detect: List[str] = field(default_factory=lambda: ["person"])
    confidence_threshold: float = 0.5
    motion_detection: MotionConfig = field(default_factory=MotionConfig)
    skip_frames: int = 5
    force_interval: int = 25


@dataclass
class GoogleHomeConfig:
    enabled: bool = False
    device_name: str = ""
    sound_server_url: str = ""


@dataclass
class EmailConfig:
    enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 0
    sender_email: str = ""
    sender_password: str = ""
    recipient_email: str = ""


@dataclass
class AlertingConfig:
    cooldown_seconds: float = 60.0
    save_directory: Path = Path("./detections")
    google_home: GoogleHomeConfig = field(default_factory=GoogleHomeConfig)
    email: EmailConfig = field(default_factory=EmailConfig)


@dataclass
class AppConfig:
    cameras: List[CameraConfig]
    detection: DetectionConfig
    alerting: AlertingConfig


def load_config(path: str = "config.yaml") -> AppConfig:
    """
    Carga y valida la configuración desde YAML, devolviendo instancias
    tipadas de AppConfig o lanza ConfigError.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Archivo de configuración no encontrado: {path!r}")

    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parseando YAML: {e}") from e

    # --- Detection ---
    d_conf = raw.get("detection", {}) or {}
    default_classes = ["person"]
    det_classes = d_conf.get("classes_to_detect") or default_classes
    det_motion = MotionConfig(**(d_conf.get("motion_detection") or {}))
    detection = DetectionConfig(
        model=d_conf.get("model", DetectionConfig.model),
        classes_to_detect=list(det_classes),
        confidence_threshold=d_conf.get("confidence_threshold", DetectionConfig.confidence_threshold),
        motion_detection=det_motion,
        skip_frames=d_conf.get("skip_frames", DetectionConfig.skip_frames),
        force_interval=d_conf.get("force_interval", DetectionConfig.force_interval)
    )

    # --- Alerting ---
    a_conf = raw.get("alerting", {}) or {}
    gh_conf = a_conf.get("google_home", {}) or {}
    em_conf = a_conf.get("email", {}) or {}

    google_home = GoogleHomeConfig(
        enabled=gh_conf.get("enabled", False),
        device_name=gh_conf.get("device_name", ""),
        sound_server_url=gh_conf.get("sound_server_url", ""),
    )
    email = EmailConfig(
        enabled=em_conf.get("enabled", False),
        smtp_server=em_conf.get("smtp_server", ""),
        smtp_port=em_conf.get("smtp_port", 0),
        sender_email=em_conf.get("sender_email", ""),
        sender_password=em_conf.get("sender_password", ""),
        recipient_email=em_conf.get("recipient_email", ""),
    )
    alerting = AlertingConfig(
        cooldown_seconds=a_conf.get("cooldown_seconds", AlertingConfig.cooldown_seconds),
        save_directory=Path(a_conf.get("save_directory") or AlertingConfig.save_directory),
        google_home=google_home,
        email=email,
    )

    # --- Cameras ---
    cams_raw = raw.get("cameras", []) or []
    cameras: List[CameraConfig] = []
    for cam in cams_raw:
        name = cam.get("name")
        url = cam.get("url")
        if not name or not url:
            continue

        cam_classes = cam.get("classes_to_detect") or detection.classes_to_detect
        mc = MotionConfig(**(cam.get("motion_detection") or det_motion.__dict__))
        lc = LingerConfig(**(cam.get("linger_detection") or LingerConfig().__dict__))

        cameras.append(
            CameraConfig(
                name=name,
                url=url,
                confidence_threshold=cam.get("confidence_threshold", detection.confidence_threshold),
                classes_to_detect=list(cam_classes),
                motion_detection=mc,
                linger_detection=lc,
                alert_cooldown_seconds=cam.get("alert_cooldown_seconds", alerting.cooldown_seconds),
                save_directory=Path(cam.get("save_directory") or alerting.save_directory),
            )
        )

    if not cameras:
        raise ConfigError("No se encontraron cámaras válidas en la configuración.")

    return AppConfig(cameras=cameras, detection=detection, alerting=alerting)
