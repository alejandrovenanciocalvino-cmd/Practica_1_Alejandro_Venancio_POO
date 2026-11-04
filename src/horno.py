"""Clase Horno.
Encapsula la temperatura, el contador de piezas recibidas y el cambio
de color del geom del horno en MuJoCo en función de la temperatura."""

from __future__ import annotations
import mujoco

from . import config as cfg


class Horno:
    """Horno industrial al final de la Cinta 5."""

    # Paleta de color según temperatura (RGBA)
    COLOR_FRIO   = (0.10, 0.30, 1.00, 1.0)   # azul
    COLOR_TIBIO  = (1.00, 0.55, 0.10, 1.0)   # naranja
    COLOR_CALIDO = (1.00, 0.25, 0.05, 1.0)   # rojo
    COLOR_BASE   = (0.22, 0.24, 0.26, 1.0)   # gris oscuro (apagado)

    def __init__(self, model, temperatura_inicial: float = 20.0):
        self.temperatura = float(temperatura_inicial)
        self.numero_piezas = 0
        self.geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, cfg.OVEN_GEOM_NAME)
        if self.geom_id < 0:
            raise RuntimeError(
                f"No se encontró el geom '{cfg.OVEN_GEOM_NAME}' en el modelo. "
                "Asegúrate de nombrar así el geom interior del horno en build_mjcf().")

    # ---------- gestión de piezas ----------
    def registrar_pieza(self) -> None:
        self.numero_piezas += 1

    # ---------- gestión de temperatura ----------
    def aumentar_temperatura(self, incremento: float) -> None:
        self.temperatura = min(self.temperatura + incremento, cfg.OVEN_TEMP_MAX)

    def disminuir_temperatura(self, decremento: float) -> None:
        self.temperatura = max(self.temperatura - decremento, cfg.OVEN_TEMP_MIN)

    def obtener_temperatura(self) -> float:
        return self.temperatura

    # ---------- representación visual ----------
    def actualizar_color(self, model) -> None:
        """Cambia el color del geom del horno según la temperatura."""
        t = self.temperatura
        if t < 50.0:
            color = self.COLOR_FRIO
        elif t < 120.0:
            # interpolación lineal entre azul y naranja
            alpha = (t - 50.0) / 70.0
            color = tuple(a + alpha * (b - a)
                          for a, b in zip(self.COLOR_FRIO, self.COLOR_TIBIO))
        else:
            # interpolación entre naranja y rojo
            alpha = min((t - 120.0) / 80.0, 1.0)
            color = tuple(a + alpha * (b - a)
                          for a, b in zip(self.COLOR_TIBIO, self.COLOR_CALIDO))
        model.geom_rgba[self.geom_id] = color

    def __repr__(self) -> str:
        return (f"Horno(T={self.temperatura:.1f}°C, "
                f"piezas={self.numero_piezas})")