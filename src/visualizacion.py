# -*- coding: utf-8 -*-
"""Ventana OpenCV con el mosaico de cámaras de la planta."""

import time
from pathlib import Path

import cv2
import numpy as np


class VisorCamaras:
    NOMBRE_VENTANA = "Camaras de la planta"
    AYUDA = "Q/ESC salir | P pausa | 1-5 cintas | R reset | S captura | +/- horno | M modo robot | A auto-reanudar"

    def __init__(self, ancho, alto, columnas=3, directorio_capturas=None):
        self.ancho = ancho
        self.alto = alto
        self.columnas = columnas
        self.directorio = Path(directorio_capturas) if directorio_capturas else None

    def abrir(self, filas=2):
        cv2.namedWindow(self.NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.NOMBRE_VENTANA, self.ancho * self.columnas,
                         self.alto * filas + 56)

    @staticmethod
    def _etiquetar(imagen, titulo, estado="", alarma=False):
        img = imagen.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(img, titulo, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65 if len(titulo) < 26 else 0.55, (245, 245, 245), 2, cv2.LINE_AA)
        if estado:
            cv2.putText(img, estado, (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (245, 245, 245), 1, cv2.LINE_AA)
        if alarma:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 3)
        return img

    def componer(self, camaras, texto_global):
        teselas = [self._etiquetar(c.imagen, c.titulo, c.texto_estado(), c.hay_alarma())
                   for c in camaras]
        while len(teselas) % self.columnas:
            teselas.append(np.zeros_like(teselas[0]))
        filas = [np.hstack(teselas[i:i + self.columnas])
                 for i in range(0, len(teselas), self.columnas)]
        mosaico = np.vstack(filas)
        pie = np.full((56, mosaico.shape[1], 3), 25, np.uint8)
        cv2.putText(pie, texto_global, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pie, self.AYUDA, (10, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (190, 190, 190), 1, cv2.LINE_AA)
        return np.vstack((mosaico, pie))

    def mostrar(self, mosaico):
        cv2.imshow(self.NOMBRE_VENTANA, mosaico)

    @staticmethod
    def leer_tecla():
        return cv2.waitKey(1) & 0xFF

    def guardar(self, mosaico):
        self.directorio.mkdir(parents=True, exist_ok=True)
        ruta = self.directorio / f"mosaico_{int(time.time())}.png"
        cv2.imwrite(str(ruta), mosaico)
        return ruta

    def cerrar(self):
        cv2.destroyAllWindows()
