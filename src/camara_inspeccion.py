# -*- coding: utf-8 -*-
"""Cámaras de la planta: captura, visión y control de la cinta asociada."""

import cv2

from .vision import DetectorFormas


class Camara:
    """Cámara MuJoCo genérica: sabe capturar una imagen (BGR para OpenCV)."""

    def __init__(self, nombre, titulo=None):
        self.nombre = nombre
        self.titulo = titulo or nombre
        self.imagen_cruda = None
        self.imagen = None

    def capturar(self, renderer, data):
        renderer.update_scene(data, camera=self.nombre)
        rgb = renderer.render()
        self.imagen_cruda = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.imagen = self.imagen_cruda
        return self.imagen_cruda

    def refrescar(self, renderer, data):
        """Captura sin ejecutar lógica (p. ej. en pausa)."""
        self.capturar(renderer, data)
        return self.imagen

    def texto_estado(self):
        return ""

    def hay_alarma(self):
        return False


class CamaraInspeccion(Camara):
    """Cámara de inspección asociada a una cinta.

    Cada vez que se actualiza: captura -> detecta forma -> actualiza
    `objeto_detectado` -> si hay pieza en la zona de inspección, detiene su
    cinta. La reanudación es una decisión controlada (ver `_decidir`):
      LIBRE          esperando una pieza
      INSPECCIONANDO cinta parada por esta cámara
      DESPEJANDO     pieza ya inspeccionada: no se vuelve a parar hasta que
                     la zona quede libre (evita parar dos veces por la misma pieza)
    """

    LIBRE, INSPECCIONANDO, DESPEJANDO = "LIBRE", "INSPECCIONANDO", "DESPEJANDO"

    def __init__(self, nombre, cinta, titulo=None, vision=None,
                 tiempo_inspeccion=2.0, reanudacion_automatica=True):
        super().__init__(nombre, titulo or cinta.nombre)
        self.cinta = cinta
        self.vision = vision or DetectorFormas()
        self.tiempo_inspeccion = tiempo_inspeccion
        self.reanudacion_automatica = reanudacion_automatica

        self.objeto_detectado = False     # pieza en la zona de inspección
        self.pieza_visible = False        # pieza en cualquier parte del campo de visión
        self.forma_detectada = None
        self.deteccion = None
        self.detecciones = []
        self.estado = self.LIBRE
        self._t_parada = 0.0

    # ----------------------------------------------------------------- visión
    def detectar_forma(self, imagen):
        """Procesa la imagen BGR; devuelve la forma de la pieza en zona o None."""
        self.detecciones = self.vision.detectar(imagen)
        self.pieza_visible = len(self.detecciones) > 0
        en_zona = [d for d in self.detecciones if d.en_zona]
        self.deteccion = en_zona[0] if en_zona else None
        return self.deteccion.forma if self.deteccion else None

    # ---------------------------------------------------------------- control
    def _decidir(self, ahora):
        if self.estado == self.LIBRE:
            if self.objeto_detectado:
                if self.cinta.esta_activa():
                    self.cinta.detener(motivo=f"camara:{self.nombre}")
                    self.estado = self.INSPECCIONANDO
                    self._t_parada = ahora
                    print(f"[{self.nombre}] pieza '{self.forma_detectada}' detectada "
                          f"-> {self.cinta.nombre} DETENIDA")
                else:
                    self.estado = self.DESPEJANDO   # ya estaba parada por el operario
        elif self.estado == self.INSPECCIONANDO:
            if self.cinta.esta_activa():            # el operario la ha rearrancado
                self.estado = self.DESPEJANDO
            elif (self.reanudacion_automatica
                  and ahora - self._t_parada >= self.tiempo_inspeccion):
                self.cinta.arrancar()
                self.estado = self.DESPEJANDO
                print(f"[{self.nombre}] inspección terminada -> {self.cinta.nombre} reanudada")
        elif self.estado == self.DESPEJANDO:
            if not self.objeto_detectado:
                self.estado = self.LIBRE

    def procesar(self, imagen, tiempo):
        self.forma_detectada = self.detectar_forma(imagen)
        self.objeto_detectado = self.forma_detectada is not None
        self._decidir(tiempo)
        self.imagen = self.vision.dibujar(imagen, self.detecciones)

    def actualizar(self, renderer, data):
        """Ciclo completo. Devuelve la imagen anotada (BGR)."""
        imagen = self.capturar(renderer, data)
        self.procesar(imagen, float(data.time))
        return self.imagen

    def refrescar(self, renderer, data):
        imagen = self.capturar(renderer, data)
        self.imagen = self.vision.dibujar(imagen, self.detecciones)
        return self.imagen

    def reiniciar(self):
        self.objeto_detectado = False
        self.pieza_visible = False
        self.forma_detectada = None
        self.deteccion = None
        self.detecciones = []
        self.estado = self.LIBRE

    def texto_estado(self):
        forma = self.forma_detectada or "-"
        cinta = "ON" if self.cinta.esta_activa() else "OFF"
        return f"{self.estado[:5]} | forma: {forma} | cinta {cinta}"

    def hay_alarma(self):
        return self.objeto_detectado and not self.cinta.esta_activa()
