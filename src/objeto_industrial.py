"""Clase ObjetoIndustrial.
Encapsula el identificador, tipo/forma, segmento actual y los IDs
de MuJoCo necesarios para acceder al cuerpo y al freejoint."""

from __future__ import annotations
import mujoco

from . import config as cfg


class ObjetoIndustrial:
    """Una pieza que circula por la planta."""

    # Tipos de forma según el índice del objeto (coincide con build_mjcf)
    TIPOS = {
        0: "caja",        # cubo rojo
        1: "cilindro",    # cilindro azul
        2: "rectangulo",  # caja amarilla
        3: "cilindro",    # cilindro verde
        4: "rectangulo",  # caja violeta
    }

    def __init__(self, indice: int, model):
        self.indice = indice
        self.tipo = self.TIPOS.get(indice, "desconocido")
        self.segmento = 0  # se fija en reset
        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"obj{indice}")
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{indice}_free")
        self.joint_id = jid
        self.qpos_addr = model.jnt_qposadr[jid]
        self.dof_addr = model.jnt_dofadr[jid]

    # ---------- máquina de estados del circuito ----------
    def actualizar_segmento(self, pos) -> None:
        """Cambia de cinta al alcanzar la esquina correspondiente."""
        x, y = float(pos[0]), float(pos[1])
        s = self.segmento
        if s == 0 and x >= cfg.CORNER_SWITCH:
            self.segmento = 1
        elif s == 1 and y >= cfg.CORNER_SWITCH:
            self.segmento = 2
        elif s == 2 and x <= -cfg.CORNER_SWITCH:
            self.segmento = 3
        elif s == 3 and y <= -cfg.CORNER_SWITCH:
            self.segmento = 0
        # El segmento 4 (horno) no vuelve al circuito: lo gestiona el Horno.

    def ha_llegado_al_horno(self, pos) -> bool:
        return self.segmento == 4 and float(pos[0]) <= cfg.OVEN_ENTRY_X

    def __repr__(self) -> str:
        return (f"ObjetoIndustrial(idx={self.indice}, tipo={self.tipo}, "
                f"seg={self.segmento})")