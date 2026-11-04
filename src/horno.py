# -*- coding: utf-8 -*-
"""Clase Horno: registro de piezas, temperatura y color en MuJoCo."""

import mujoco
import numpy as np

_AZUL = np.array([0.10, 0.30, 1.00, 1.0])
_NARANJA = np.array([1.00, 0.60, 0.10, 1.0])
_ROJO = np.array([1.00, 0.10, 0.10, 1.0])


class Horno:
    """Final de la línea. Gestiona piezas recibidas, temperatura y color.

    Toda la lógica (temperatura, límites, color) está encapsulada aquí; no hay
    variables globales.
    """

    NOMBRE_GEOM = "oven_glow"

    def __init__(self, centro_x, entrada_x, temperatura_inicial=20.0,
                 temperatura_minima=20.0, temperatura_maxima=200.0,
                 temperatura_media=100.0, incremento_pieza=30.0,
                 enfriamiento=0.5, geom_id=-1):
        self.centro_x = centro_x
        self.entrada_x = entrada_x
        self.temperatura_minima = float(temperatura_minima)
        self.temperatura_maxima = float(temperatura_maxima)
        self.temperatura_media = float(temperatura_media)
        self.temperatura_inicial = float(temperatura_inicial)
        self.incremento_pieza = float(incremento_pieza)
        self.enfriamiento = float(enfriamiento)   # °C/s
        self.geom_id = geom_id

        self.numero_piezas = 0
        self.piezas_por_objeto = {}
        self.temperatura = self._limitar(self.temperatura_inicial)

        self._tiempo = 0.0
        self._t_ultima_entrada = None
        self._color_aplicado = None

    # ----------------------------------------------------------- temperatura
    def _limitar(self, t):
        return float(min(max(t, self.temperatura_minima), self.temperatura_maxima))

    def aumentar_temperatura(self, incremento):
        self.temperatura = self._limitar(self.temperatura + incremento)

    def disminuir_temperatura(self, decremento):
        self.temperatura = self._limitar(self.temperatura - decremento)

    def establecer_temperatura(self, valor):
        self.temperatura = self._limitar(valor)

    def obtener_temperatura(self):
        return self.temperatura

    @property
    def estado(self):
        if self.temperatura >= self.temperatura_maxima:
            return "SOBRECALENTADO"
        if self.temperatura >= self.temperatura_media:
            return "CALIENTE"
        if self.temperatura >= 50.0:
            return "TEMPLADO"
        return "FRIO"

    # ----------------------------------------------------------------- piezas
    def pieza_entra(self, x):
        """True si una pieza de la cinta del horno ha cruzado la boca."""
        return x <= self.entrada_x

    def registrar_pieza(self, id_objeto=None):
        self.numero_piezas += 1
        if id_objeto is not None:
            self.piezas_por_objeto[id_objeto] = self.piezas_por_objeto.get(id_objeto, 0) + 1
        self._t_ultima_entrada = self._tiempo
        self.aumentar_temperatura(self.incremento_pieza)

    @property
    def pieza_entrante(self):
        """True durante 1 s simulado tras la entrada de una pieza."""
        return (self._t_ultima_entrada is not None
                and self._tiempo - self._t_ultima_entrada < 1.0)

    # ------------------------------------------------------------------ color
    def color_actual(self):
        """Azul (frío) -> naranja (medio) -> rojo (caliente), interpolado."""
        t, tmin, tmed, tmax = (self.temperatura, self.temperatura_minima,
                               self.temperatura_media, self.temperatura_maxima)
        if t <= tmin:
            return _AZUL.copy()
        if t < tmed:
            f = (t - tmin) / max(tmed - tmin, 1e-9)
            return (1 - f) * _AZUL + f * _NARANJA
        f = min((t - tmed) / max(tmax - tmed, 1e-9), 1.0)
        return (1 - f) * _NARANJA + f * _ROJO

    def actualizar_color(self, model):
        """Escribe el color en model.geom_rgba. Devuelve True si cambió."""
        color = self.color_actual()
        if (self._color_aplicado is not None
                and np.allclose(color, self._color_aplicado, atol=1e-3)):
            return False
        model.geom_rgba[self.geom_id] = color
        self._color_aplicado = color
        return True

    # ------------------------------------------------------------ simulación
    def actualizar(self, model, dt):
        """Enfriamiento pasivo + actualización del color. Una vez por paso."""
        self._tiempo += dt
        if self.temperatura > self.temperatura_minima:
            self.disminuir_temperatura(self.enfriamiento * dt)
        self.actualizar_color(model)

    def reiniciar(self, model):
        self.numero_piezas = 0
        self.piezas_por_objeto.clear()
        self.temperatura = self._limitar(self.temperatura_inicial)
        self._t_ultima_entrada = None
        self._color_aplicado = None
        self.actualizar_color(model)

    def vincular(self, model):
        self.geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.NOMBRE_GEOM)
        if self.geom_id < 0:
            raise RuntimeError(f"No se encontró el geom '{self.NOMBRE_GEOM}' del horno.")
        self._color_aplicado = None
        self.actualizar_color(model)

    def a_mjcf(self):
        c = " ".join(f"{v:.3f}" for v in self.color_actual())
        return f"""
        <body name="oven" pos="{self.centro_x} 0 0">
            <!-- base y fondo -->
            <geom type="box" pos="0 0 0.12" size="0.55 0.62 0.10" material="oven_mat"/>
            <geom type="box" pos="-0.50 0 0.68" size="0.08 0.62 0.58" material="oven_mat"/>
            <!-- paredes laterales, dejando el frontal abierto hacia +X -->
            <geom type="box" pos="0 -0.56 0.68" size="0.55 0.06 0.58" material="oven_mat"/>
            <geom type="box" pos="0  0.56 0.68" size="0.55 0.06 0.58" material="oven_mat"/>
            <geom type="box" pos="0 0 1.23" size="0.55 0.62 0.07" material="oven_mat"/>
            <!-- marco frontal -->
            <geom type="box" pos="0.50 -0.48 0.68" size="0.05 0.08 0.58" material="oven_dark_mat"/>
            <geom type="box" pos="0.50  0.48 0.68" size="0.05 0.08 0.58" material="oven_dark_mat"/>
            <geom type="box" pos="0.50 0 1.16" size="0.05 0.40 0.10" material="oven_dark_mat"/>
            <!-- interior: su color (geom_rgba) refleja la temperatura. Sin material
                 para que geom_rgba controle el color en tiempo de ejecución. -->
            <geom name="{self.NOMBRE_GEOM}" type="box" pos="-0.42 0 0.58" size="0.025 0.39 0.31"
                  rgba="{c}" contype="0" conaffinity="0"/>
            <light pos="-0.18 0 0.82" dir="1 0 -0.15" diffuse="1.0 0.25 0.05"/>
        </body>"""
