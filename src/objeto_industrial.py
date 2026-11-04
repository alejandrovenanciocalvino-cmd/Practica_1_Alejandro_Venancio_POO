# -*- coding: utf-8 -*-
"""Objetos industriales que circulan por la planta (jerarquía Caja / Cilindro)."""

import mujoco

from .config import (BELT_TOP_Z, LANE, LOCK_CYLINDERS_TO_CENTERLINE,
                     LOCK_OBJECT_ROTATION, OBJECT_CLEARANCE_Z)

_QUAT_IDENTIDAD = [1.0, 0.0, 0.0, 0.0]


class ObjetoIndustrial:
    """Pieza con cuerpo libre (freejoint) en MuJoCo.

    Atributos: identificador, forma, segmento (índice de la cinta actual),
    ids de MuJoCo (body, qpos, dof) y posición/segmento inicial.
    """

    forma = "objeto"

    def __init__(self, identificador, masa, rgba, inicio, segmento):
        self.identificador = identificador
        self.masa = masa
        self.rgba = tuple(rgba)
        self.posicion_inicial = (float(inicio[0]), float(inicio[1]))
        self.segmento_inicial = segmento
        self.segmento = segmento
        self.body_id = -1
        self.qpos_adr = -1
        self.dof_adr = -1

    # ---- a implementar por las subclases
    @property
    def semialtura(self):
        raise NotImplementedError

    def a_mjcf(self):
        raise NotImplementedError

    # ---- comunes
    @property
    def nombre(self):
        return f"obj{self.identificador}"

    @property
    def z_reposo(self):
        return BELT_TOP_Z + self.semialtura + OBJECT_CLEARANCE_Z

    def _rgba_txt(self):
        return " ".join(f"{v}" for v in self.rgba)

    def vincular(self, model):
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.nombre)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.nombre}_free")
        if self.body_id < 0 or jid < 0:
            raise RuntimeError(f"No se encontró el cuerpo/joint de {self.nombre}.")
        self.qpos_adr = int(model.jnt_qposadr[jid])
        self.dof_adr = int(model.jnt_dofadr[jid])

    def posicion(self, data):
        return data.xpos[self.body_id]

    def velocidad(self, data):
        return data.qvel[self.dof_adr:self.dof_adr + 3]

    def aplicar_fuerza(self, data, fx, fy):
        data.xfrc_applied[self.body_id, 0] = fx
        data.xfrc_applied[self.body_id, 1] = fy

    def colocar(self, data, x, y, segmento, vx=0.0):
        """Teletransporta la pieza (reposo sobre la banda) y fija su segmento."""
        q = self.qpos_adr
        data.qpos[q:q + 3] = [x, y, self.z_reposo]
        data.qpos[q + 3:q + 7] = _QUAT_IDENTIDAD
        data.qvel[self.dof_adr:self.dof_adr + 6] = [vx, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.segmento = segmento

    def reiniciar(self, data):
        self.colocar(data, *self.posicion_inicial, self.segmento_inicial)

    def reaparecer_en_cinta1(self, data, x):
        self.colocar(data, x, -LANE, 0)

    def fuera_de_limites(self, pos):
        return pos[2] < -0.1 or abs(pos[0]) > 5.0 or abs(pos[1]) > 2.4

    def bloquear_rotacion(self, data):
        if LOCK_OBJECT_ROTATION:
            q = self.qpos_adr
            data.qpos[q + 3:q + 7] = _QUAT_IDENTIDAD
            data.qvel[self.dof_adr + 3:self.dof_adr + 6] = 0.0

    def estabilizar(self, data, cinta):
        """Estabilización posterior al paso físico (por defecto, solo rotación)."""
        self.bloquear_rotacion(data)

    def __repr__(self):
        return f"{type(self).__name__}(id={self.identificador}, segmento={self.segmento})"


class Caja(ObjetoIndustrial):
    forma = "caja"

    def __init__(self, identificador, medias, masa, rgba, inicio, segmento):
        super().__init__(identificador, masa, rgba, inicio, segmento)
        self.medias = tuple(medias)

    @property
    def semialtura(self):
        return self.medias[2]

    def a_mjcf(self):
        x, y = self.posicion_inicial
        hx, hy, hz = self.medias
        return f"""
        <body name="{self.nombre}" pos="{x} {y} {self.z_reposo}">
            <freejoint name="{self.nombre}_free"/>
            <geom name="{self.nombre}_geom" type="box" size="{hx} {hy} {hz}"
                  mass="{self.masa}" rgba="{self._rgba_txt()}"/>
        </body>"""


class Cilindro(ObjetoIndustrial):
    forma = "cilindro"

    def __init__(self, identificador, radio, semialtura, masa, rgba, inicio, segmento):
        super().__init__(identificador, masa, rgba, inicio, segmento)
        self.radio = radio
        self._semialtura = semialtura

    @property
    def semialtura(self):
        return self._semialtura

    def a_mjcf(self):
        x, y = self.posicion_inicial
        return f"""
        <body name="{self.nombre}" pos="{x} {y} {self.z_reposo}">
            <freejoint name="{self.nombre}_free"/>
            <geom name="{self.nombre}_geom" type="cylinder" size="{self.radio} {self._semialtura}"
                  mass="{self.masa}" rgba="{self._rgba_txt()}"/>
        </body>"""

    def estabilizar(self, data, cinta):
        """Además de fijar la rotación, mantiene el cilindro en la línea central."""
        super().estabilizar(data, cinta)
        if not LOCK_CYLINDERS_TO_CENTERLINE:
            return
        q = self.qpos_adr
        res = cinta.coordenada_centrado(data.qpos[q:q + 2])
        if res is not None:
            eje, valor = res
            data.qpos[q + eje] = valor
            data.qvel[self.dof_adr + eje] = 0.0


def crear_objeto(cfg):
    """Fábrica: construye Caja o Cilindro a partir de una entrada de OBJETOS_CFG."""
    comunes = dict(identificador=cfg["identificador"], masa=cfg["masa"], rgba=cfg["rgba"],
                   inicio=cfg["inicio"], segmento=cfg["segmento"])
    if cfg["tipo"] == "caja":
        return Caja(medias=cfg["medias"], **comunes)
    if cfg["tipo"] == "cilindro":
        return Cilindro(radio=cfg["radio"], semialtura=cfg["semialtura"], **comunes)
    raise ValueError(f"Tipo de objeto desconocido: {cfg['tipo']}")
