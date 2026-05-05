"""
camera.py
=========
Módulo mejorado para captura de imágenes mediante la webcam.
Diseñado para integrarse de forma segura con interfaces gráficas (Tkinter/PyQt)
sin causar conflictos con los hilos del sistema operativo.
"""

import typing
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def is_camera_available(camera_index: int = 0) -> bool:
    """Verifica si la cámara especificada está disponible y lista para usarse."""
    if not CV2_AVAILABLE:
        return False
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False
    
    cap.release()
    return True


class CameraStream:
    """
    Clase para manejar el flujo de la cámara de forma continua.
    Ideal para incrustar la vista previa directamente dentro de un panel de Tkinter.
    """
    
    def __init__(self, camera_index: int = 0):
        if not CV2_AVAILABLE:
            raise RuntimeError(
                "OpenCV no está instalado. Instálalo ejecutando:\n"
                "pip install opencv-python"
            )
        self.camera_index = camera_index
        self._cap = None

    def start(self) -> bool:
        """Inicia el hardware de la cámara."""
        self._cap = cv2.VideoCapture(self.camera_index)
        return self._cap.isOpened()

    def read_frame(self) -> typing.Optional[Image.Image]:
        """
        Lee el frame actual de la cámara en tiempo real.
        
        Retorna:
            Objeto PIL.Image en formato RGB, o None si hay un error de lectura.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        # OpenCV utiliza formato BGR por defecto; lo convertimos a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def release(self):
        """Apaga la cámara y libera los recursos del sistema de manera segura."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def capture_single_frame(camera_index: int = 0, stabilize_frames: int = 5) -> Image.Image:
    """
    Función de utilidad para capturar una sola imagen rápidamente (One-Shot).
    Enciende la cámara, toma la foto y la apaga inmediatamente.

    Parámetros:
        camera_index     : Índice de la cámara (0 = predeterminada).
        stabilize_frames : Cantidad de frames a descartar al inicio para que el 
                           sensor ajuste el autoenfoque y la exposición a la luz.

    Retorna:
        Objeto PIL.Image en modo 'RGB'.

    Lanza:
        RuntimeError si falla la inicialización o la lectura.
    """
    cam = CameraStream(camera_index)
    
    if not cam.start():
        raise RuntimeError(
            f"No se pudo abrir la cámara (índice {camera_index}).\n"
            "Verifica que esté conectada y no esté siendo usada por otra app."
        )

    try:
        # Descartar los primeros frames (suelen verse oscuros o borrosos)
        for _ in range(stabilize_frames):
            cam._cap.read()

        # Tomar la fotografía definitiva
        img = cam.read_frame()
        if img is None:
            raise RuntimeError("Se inicializó la cámara, pero falló la lectura del frame.")
        
        return img
    finally:
        # Asegurarse de que la cámara se apague incluso si ocurre un error
        cam.release()


def capture_with_preview(camera_index: int = 0, stabilize_frames: int = 5) -> Image.Image:
    """
    Alias para compatibilidad con la interfaz principal.
    Actualmente realiza una captura de un solo frame tras estabilizar la cámara.
    """
    return capture_single_frame(camera_index=camera_index, stabilize_frames=stabilize_frames)