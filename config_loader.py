# config_loader.py
import yaml
import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_LINGER_CONFIG = {'enabled': False}
DEFAULT_ALERTING_CONFIG = {'cooldown_seconds': 60}

def load_config(config_path='config.yaml'):
    """Carga y valida la configuración desde un archivo YAML."""
    if not os.path.exists(config_path):
        logger.error(f"Archivo de configuración '{config_path}' no encontrado.")
        return None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuración cargada exitosamente.")

        # --- Validación y Valores por Defecto ---
        if 'cameras' not in config or not isinstance(config['cameras'], list):
            logger.error("La sección 'cameras' es inválida o no existe en config.yaml.")
            return None

        if 'detection' not in config:
            logger.warning("Sección 'detection' no encontrada. Usando valores por defecto si es posible.")
            config['detection'] = {} # Permitir continuar si no es esencial para todo

        if 'alerting' not in config:
            logger.warning("Sección 'alerting' no encontrada. Usando valores por defecto.")
            config['alerting'] = DEFAULT_ALERTING_CONFIG
        else:
            config['alerting'].setdefault('cooldown_seconds', 60)

        # Validar cada cámara
        valid_cameras = []
        for i, cam_info in enumerate(config['cameras']):
            if not isinstance(cam_info, dict) or 'name' not in cam_info or 'url' not in cam_info:
                logger.warning(f"Entrada de cámara inválida en el índice {i}. Omitiendo.")
                continue

            # Configuración de Linger por defecto/validación
            linger_cfg = cam_info.get('linger_detection', DEFAULT_LINGER_CONFIG)
            linger_cfg.setdefault('enabled', False)
            if linger_cfg['enabled']:
                # Validar ROI si está habilitado
                roi = linger_cfg.get('roi')
                if not roi or not isinstance(roi, list) or len(roi) != 4:
                    logger.warning(f"ROI inválida o faltante para {cam_info['name']}. Deshabilitando Linger Detection.")
                    linger_cfg['enabled'] = False
                linger_cfg.setdefault('linger_time_seconds', 5)
                linger_cfg.setdefault('tracking_distance_threshold', 75)
            cam_info['linger_detection'] = linger_cfg
            valid_cameras.append(cam_info)

        config['cameras'] = valid_cameras
        if not config['cameras']:
            logger.error("No se encontraron configuraciones de cámara válidas.")
            return None

        logger.info("Configuración validada.")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error al parsear el archivo de configuración YAML: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado al cargar la configuración: {e}")
        return None