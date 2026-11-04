"""Clase CamaraInspeccion.
Captura la imagen desde MuJoCo, la procesa con OpenCV y detecta la forma
geométrica del objeto. Cuando detecta una pieza, actualiza la variable
booleana `objeto_detectado` y comunica a la cinta asociada que debe
detenerse."""

from __future__ import annotations
from typing import Optional
import cv2
import numpy as np

from .cinta_transportadora import CintaTransportadora


class CamaraInspeccion:
    """Cámara de inspección asociada a una cinta concreta."""

    def __init__(self, nombre_camara: str, etiqueta: str,
                 cinta: CintaTransportadora):
        self.nombre_camara = nombre_camara       # nombre en el MJCF
        self.etiqueta = etiqueta                 # texto para el mosaico
        self.cinta = cinta                       # COMPOSICIÓN
        self.objeto_detectado: bool = False
        self.forma_detectada: Optional[str] = None

    # ---------- captura ----------
    def capturar(self, renderer, data) -> np.ndarray:
        renderer.update_scene(data, camera=self.nombre_camara)
        rgb = renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ---------- detección de forma ----------
    def detectar_forma(self, imagen: np.ndarray) -> Optional[str]:
        """Devuelve 'circulo', 'cuadrado', 'rectangulo' o None."""
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Eliminar la banda negra inferior (suelo) si aparece
        th[: th.shape[0] // 3, :] = 0

        contornos, _ = cv2.findContours(
            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mejor = None
        mejor_area = 0
        for c in contornos:
            area = cv2.contourArea(c)
            if area < 400:            # filtrar ruido
                continue
            if area > mejor_area:
                mejor_area = area
                mejor = c

        if mejor is None:
            return None

        # Aproximación poligonal
        perimetro = cv2.contourPerimeter(mejor)
        approx = cv2.approxPolyDP(mejor, 0.04 * perimetro, True)
        n_vertices = len(approx)

        # Circularidad
        circularidad = (4 * np.pi * mejor_area) / (perimetro * perimetro + 1e-6)

        if circularidad > 0.78:
            return "circulo"
        if n_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / (h + 1e-6)
            if 0.85 <= aspect <= 1.15:
                return "cuadrado"
            return "rectangulo"
        # Otros polígonos: los tratamos como rectángulo genérico
        return "rectangulo"

    # ---------- actualización por paso ----------
    def actualizar(self, renderer, data) -> np.ndarray:
        """Captura, detecta y devuelve la imagen etiquetada."""
        imagen = self.capturar(renderer, data)
        forma = self.detectar_forma(imagen)

        self.forma_detectada = forma
        self.objeto_detectado = forma is not None

        # Comunicación con la cinta (composición)
        if self.objeto_detectado:
            self.cinta.detener()

        # Etiquetado visual
        color = (0, 220, 0) if self.objeto_detectado else (200, 200, 200)
        texto = f"{self.etiqueta}  |  {forma or '---'}"
        cv2.rectangle(imagen, (0, 0), (imagen.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(imagen, texto, (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return imagen

    def __repr__(self) -> str:
        return (f"CamaraInspeccion({self.nombre_camara}, "
                f"cinta={self.cinta.nombre}, "
                f"detectado={self.objeto_detectado}, forma={self.forma_detectada})")