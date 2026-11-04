# -*- coding: utf-8 -*-
"""Detección de piezas y clasificación de su forma con OpenCV."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Deteccion:
    forma: str            # circulo | cuadrado | rectangulo | triangulo | poligono
    area: float
    centroide: tuple      # (cx, cy) en píxeles
    caja: tuple           # (x, y, w, h)
    contorno: np.ndarray
    circularidad: float
    vertices: int
    extension: float      # área / área del rectángulo mínimo
    color: str
    en_zona: bool         # el centroide está en la franja de inspección

    @property
    def categoria(self):
        """Clase de objeto industrial equivalente."""
        if self.forma == "circulo":
            return "cilindro"
        if self.forma in ("cuadrado", "rectangulo"):
            return "caja"
        return None


class DetectorFormas:
    """Pipeline: HSV -> segmentación -> morfología -> contornos -> forma.

    La banda y el suelo son grises (baja saturación); las piezas son colores
    saturados, de modo que umbralizar la saturación las separa del fondo.
    """

    def __init__(self, sat_min=120, val_min=50, area_min=600, zona=(0.38, 0.62),
                 extension_rect=0.90, circularidad_min=0.80, ratio_cuadrado=1.20):
        self.sat_min = sat_min
        self.val_min = val_min
        self.area_min = area_min
        self.zona = zona
        self.extension_rect = extension_rect
        self.circularidad_min = circularidad_min
        self.ratio_cuadrado = ratio_cuadrado

    def segmentar(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, (0, self.sat_min, self.val_min), (179, 255, 255))
        k = np.ones((5, 5), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k)
        return mascara

    def clasificar(self, contorno):
        """Devuelve (forma, circularidad, vertices, extension)."""
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.04 * perimetro, True)
        n = len(aprox)
        circ = 4.0 * np.pi * area / (perimetro * perimetro) if perimetro > 0 else 0.0
        (_, _), (w, h), _ = cv2.minAreaRect(contorno)
        ext = area / (w * h) if w * h > 0 else 0.0

        if n == 3 and ext < 0.70:
            forma = "triangulo"
        elif ext >= self.extension_rect:
            ratio = max(w, h) / max(min(w, h), 1e-6)
            forma = "cuadrado" if ratio <= self.ratio_cuadrado else "rectangulo"
        elif circ >= self.circularidad_min:
            forma = "circulo"
        else:
            forma = "poligono"
        return forma, float(circ), int(n), float(ext)

    @staticmethod
    def nombre_color(bgr, contorno):
        mascara = np.zeros(bgr.shape[:2], np.uint8)
        cv2.drawContours(mascara, [contorno], -1, 255, -1)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = float(cv2.mean(hsv[:, :, 0], mask=mascara)[0])
        if h < 10 or h >= 170:
            return "rojo"
        if h < 40:
            return "amarillo"
        if h < 90:
            return "verde"
        if h < 135:
            return "azul"
        return "morado"

    def detectar(self, bgr):
        """Lista de Deteccion (mayor área primero)."""
        alto, ancho = bgr.shape[:2]
        mascara = self.segmentar(bgr)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        resultado = []
        for c in contornos:
            area = cv2.contourArea(c)
            if area < self.area_min:
                continue
            forma, circ, n, ext = self.clasificar(c)
            m = cv2.moments(c)
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            en_zona = self.zona[0] * ancho <= cx <= self.zona[1] * ancho
            resultado.append(Deteccion(
                forma=forma, area=float(area), centroide=(cx, cy),
                caja=cv2.boundingRect(c), contorno=c, circularidad=circ,
                vertices=n, extension=ext, color=self.nombre_color(bgr, c),
                en_zona=en_zona))
        resultado.sort(key=lambda d: d.area, reverse=True)
        return resultado

    def dibujar(self, bgr, detecciones, resaltar=False):
        """Devuelve una copia con la franja de inspección y las detecciones."""
        img = bgr.copy()
        alto, ancho = img.shape[:2]
        x0, x1 = int(self.zona[0] * ancho), int(self.zona[1] * ancho)
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, 0), (x1, alto), (255, 200, 0), -1)
        img = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
        cv2.line(img, (x0, 0), (x0, alto), (255, 200, 0), 1)
        cv2.line(img, (x1, 0), (x1, alto), (255, 200, 0), 1)
        for d in detecciones:
            color = (0, 255, 0) if d.en_zona else (170, 170, 170)
            cv2.drawContours(img, [d.contorno], -1, color, 2)
            x, y, w, h = d.caja
            cv2.putText(img, d.forma.upper(), (x, max(y - 6, 52)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return img
