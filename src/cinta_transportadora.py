# -*- coding: utf-8 -*-
"""Clase CintaTransportadora: estado, configuración y fuerza de transporte."""

import mujoco
import numpy as np

from .config import (BELT_ROLLER_SPEED, CENTER_KP, CORNER_SWITCH, MAX_FORCE,
                     TURN_START, VEL_KP)


class CintaTransportadora:
    """Una cinta con su propio estado de marcha/parada.

    Sustituye a las banderas globales BANDERA_CINTA_n y a las funciones
    cinta_activa(), conmutar_cinta() y transport_force() del código procedural.
    """

    def __init__(self, identificador, nombre, velocidad, direccion, centro,
                 direccion_siguiente=None, indice_siguiente=None, actuadores=(),
                 signo_rodillos=1.0, velocidad_rodillos=BELT_ROLLER_SPEED,
                 kp_velocidad=VEL_KP, kp_centrado=CENTER_KP,
                 fuerza_maxima=MAX_FORCE, inicio_giro=TURN_START,
                 cambio_esquina=CORNER_SWITCH):
        self.identificador = identificador
        self.nombre = nombre
        self.velocidad = float(velocidad)
        self.direccion = np.asarray(direccion, dtype=float)
        self.centro = np.asarray(centro, dtype=float)
        self.direccion_siguiente = (None if direccion_siguiente is None
                                    else np.asarray(direccion_siguiente, dtype=float))
        self.indice_siguiente = indice_siguiente
        self.nombres_actuadores = tuple(actuadores)
        self.signo_rodillos = float(signo_rodillos)
        self.velocidad_rodillos = float(velocidad_rodillos)
        self.kp_velocidad = kp_velocidad
        self.kp_centrado = kp_centrado
        self.fuerza_maxima = fuerza_maxima
        self.inicio_giro = inicio_giro
        self.cambio_esquina = cambio_esquina

        # Eje transversal (el que no es de avance) para centrar las piezas.
        self._mascara_transversal = 1.0 - np.abs(self.direccion)
        self._eje_transversal = int(np.argmax(self._mascara_transversal))

        self._activa = True
        self.motivo_parada = None      # None | "manual" | "camara:<nombre>"
        self._ids_actuadores = []

    # ------------------------------------------------------------------ estado
    @property
    def activa(self):
        return self._activa

    def esta_activa(self):
        return self._activa

    def arrancar(self):
        self._activa = True
        self.motivo_parada = None

    def detener(self, motivo="manual"):
        self._activa = False
        self.motivo_parada = motivo

    def conmutar(self):
        """Cambia marcha/parada (acción del operario: motivo 'manual')."""
        if self._activa:
            self.detener("manual")
        else:
            self.arrancar()
        return self._activa

    def detenida_por_camara(self):
        return (not self._activa and self.motivo_parada is not None
                and self.motivo_parada.startswith("camara"))

    # ------------------------------------------------------------- geometría
    @property
    def tiene_giro(self):
        return self.direccion_siguiente is not None

    def progreso(self, pos):
        """Coordenada de la pieza a lo largo del sentido de avance."""
        return float(np.dot(np.asarray(pos[:2], dtype=float), self.direccion))

    def _alpha(self, pos):
        if not self.tiene_giro:
            return 0.0
        rango = max(self.cambio_esquina - self.inicio_giro, 1e-6)
        return float(np.clip((self.progreso(pos) - self.inicio_giro) / rango, 0.0, 1.0))

    def debe_transferir(self, pos):
        """True si la pieza ha llegado a la placa de esquina y pasa a la siguiente."""
        return (self.indice_siguiente is not None
                and self.progreso(pos) >= self.cambio_esquina)

    def coordenada_centrado(self, pos):
        """(eje, valor) a fijar para mantener una pieza sobre la línea central.

        Solo en tramo recto (antes de empezar a girar). None si no procede.
        """
        if self.tiene_giro and self.progreso(pos) >= self.inicio_giro:
            return None
        return self._eje_transversal, float(self.centro[self._eje_transversal])

    # --------------------------------------------------------------- física
    def fuerza_transporte(self, pos, vel):
        """Fuerza (fx, fy) sobre una pieza situada sobre esta cinta."""
        p = np.asarray(pos[:2], dtype=float)
        v = np.asarray(vel[:2], dtype=float)

        if not self._activa:
            # Frenado viscoso: la física sigue activa, solo cambia el objetivo.
            f = self.kp_velocidad * (0.0 - v)
        else:
            alpha = self._alpha(p)
            if self.tiene_giro:
                v_des = self.velocidad * ((1.0 - 0.45 * alpha) * self.direccion
                                          + 0.85 * alpha * self.direccion_siguiente)
            else:
                v_des = self.velocidad * self.direccion
            f = (self.kp_velocidad * (v_des - v)
                 + (1.0 - alpha) * self.kp_centrado
                 * (self.centro - p) * self._mascara_transversal)

        f = np.clip(f, -self.fuerza_maxima, self.fuerza_maxima)
        return float(f[0]), float(f[1])

    # ------------------------------------------------------------- MuJoCo
    def vincular(self, model):
        """Resuelve los ids de los actuadores de rodillo."""
        self._ids_actuadores = []
        for nombre in self.nombres_actuadores:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nombre)
            if aid < 0:
                raise RuntimeError(f"No se encontró el actuador de rodillo '{nombre}'.")
            self._ids_actuadores.append(aid)

    def aplicar_rodillos(self, data):
        """Los rodillos giran solo si la cinta está en marcha."""
        valor = self.signo_rodillos * self.velocidad_rodillos if self._activa else 0.0
        for aid in self._ids_actuadores:
            data.ctrl[aid] = valor

    def texto_estado(self):
        return f"C{self.identificador}:{'ON' if self._activa else 'OFF'}"

    def __repr__(self):
        return f"CintaTransportadora({self.nombre!r}, activa={self._activa})"
