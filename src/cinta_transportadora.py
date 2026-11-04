"""Clase CintaTransportadora.
Encapsula el estado de marcha/parada, la velocidad y la lógica de fuerza
de transporte que antes estaba repartida en funciones procedurales."""

from __future__ import annotations
import numpy as np

from . import config as cfg


class CintaTransportadora:
    """Una cinta con su propio estado, velocidad y dirección de transporte."""

    # Dirección de flujo según el número de cinta (1 a 5)
    DIRECCIONES = {
        1: (+1,  0),   # inferior  +X
        1: ( 0, +1),   # derecha   +Y
        2: (-1,  0),   # superior  -X
        3: ( 0, -1),   # izquierda -Y
        4: (-1,  0),   # rama horno -X
    }

    def __init__(self, numero: int, nombre: str, velocidad: float,
                 activa: bool = True):
        if numero not in self.DIRECCIONES:
            raise ValueError(f"Número de cinta inválido: {numero}")
        self.numero = numero
        self.nombre = nombre
        self.velocidad = velocidad
        self.activa = bool(activa)
        self.dir_x, self.dir_y = self.DIRECCIONES[numero]

    # ---------- control de marcha/parada ----------
    def arrancar(self) -> None:
        self.activa = True

    def detener(self) -> None:
        self.activa = False

    def conmutar(self) -> bool:
        self.activa = not self.activa
        return self.activa

    def esta_activa(self) -> bool:
        return self.activa

    # ---------- física de transporte ----------
    def calcular_fuerza(self, pos, vel) -> tuple[float, float]:
        """Devuelve (fx, fy) que la cinta aplica al objeto."""
        x, y = float(pos[0]), float(pos[1])
        vx, vy = float(vel[0]), float(vel[1])

        if not self.activa:
            fx = cfg.VEL_KP * (0.0 - vx)
            fy = cfg.VEL_KP * (0.0 - vy)
            return (float(np.clip(fx, -cfg.MAX_FORCE, cfg.MAX_FORCE)),
                    float(np.clip(fy, -cfg.MAX_FORCE, cfg.MAX_FORCE)))

        seg = self.numero
        if seg == 1:  # inferior: +X -> +Y
            alpha = np.clip((x - cfg.TURN_START) /
                            max(cfg.CORNER_SWITCH - cfg.TURN_START, 1e-6), 0.0, 1.0)
            desired_vx = self.velocidad * (1.0 - 0.45 * alpha)
            desired_vy = self.velocidad * (0.85 * alpha)
            fx = cfg.VEL_KP * (desired_vx - vx)
            fy = cfg.VEL_KP * (desired_vy - vy) + (1.0 - alpha) * cfg.CENTER_KP * (-cfg.LANE - y)

        elif seg == 2:  # derecha: +Y -> -X
            alpha = np.clip((y - cfg.TURN_START) /
                            max(cfg.CORNER_SWITCH - cfg.TURN_START, 1e-6), 0.0, 1.0)
            desired_vx = -self.velocidad * (0.85 * alpha)
            desired_vy = self.velocidad * (1.0 - 0.45 * alpha)
            fx = cfg.VEL_KP * (desired_vx - vx) + (1.0 - alpha) * cfg.CENTER_KP * (cfg.LANE - x)
            fy = cfg.VEL_KP * (desired_vy - vy)

        elif seg == 3:  # superior: -X -> -Y
            alpha = np.clip((-x - cfg.TURN_START) /
                            max(cfg.CORNER_SWITCH - cfg.TURN_START, 1e-6), 0.0, 1.0)
            desired_vx = -self.velocidad * (1.0 - 0.45 * alpha)
            desired_vy = -self.velocidad * (0.85 * alpha)
            fx = cfg.VEL_KP * (desired_vx - vx)
            fy = cfg.VEL_KP * (desired_vy - vy) + (1.0 - alpha) * cfg.CENTER_KP * (cfg.LANE - y)

        elif seg == 4:  # izquierda: -Y -> +X en esquina inferior
            alpha = np.clip((-y - cfg.TURN_START) /
                            max(cfg.CORNER_SWITCH - cfg.TURN_START, 1e-6), 0.0, 1.0)
            desired_vx = self.velocidad * (0.85 * alpha)
            desired_vy = -self.velocidad * (1.0 - 0.45 * alpha)
            fx = cfg.VEL_KP * (desired_vx - vx) + (1.0 - alpha) * cfg.CENTER_KP * (-cfg.LANE - x)
            fy = cfg.VEL_KP * (desired_vy - vy)

        else:  # seg == 5 -> rama al horno
            desired_vx = -cfg.BRANCH_SPEED
            desired_vy = 0.0
            fx = cfg.VEL_KP * (desired_vx - vx)
            fy = cfg.VEL_KP * (desired_vy - vy) + cfg.CENTER_KP * (cfg.BRANCH_Y - y)

        return (float(np.clip(fx, -cfg.MAX_FORCE, cfg.MAX_FORCE)),
                float(np.clip(fy, -cfg.MAX_FORCE, cfg.MAX_FORCE)))

    # ---------- rodillos visuales ----------
    def velocidad_rodillos(self) -> float:
        """Velocidad angular de los actuadores de rodillos de esta cinta."""
        if not self.activa:
            return 0.0
        # Signos coherentes con el flujo de cada cinta (1 a 5)
        signos = {1: +5.5, 2: -5.5, 3: -5.5, 4: +5.5, 5: -5.5}
        return signos[self.numero]

    def __repr__(self) -> str:
        estado = "ON" if self.activa else "OFF"
        return f"CintaTransportadora({self.nombre}, v={self.velocidad}, {estado})"