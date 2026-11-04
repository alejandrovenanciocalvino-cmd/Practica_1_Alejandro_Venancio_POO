# -*- coding: utf-8 -*-
"""Clase EstacionRobotizada: interfaz orientada a objetos del Panda.

Envuelve el módulo robot.py (IK proporcionada, usada como caja negra). La IK
NO se reimplementa: aquí solo se encapsula el estado del robot y se adaptan
las llamadas a los objetos de la planta.
"""

import mujoco
import numpy as np

from . import robot


class EstacionRobotizada:
    def __init__(self, config, ik_manual=True, objetivo_posicion=(0, 0, 0),
                 objetivo_rpy_deg=(0, 0, 0), marco_orientacion="home",
                 gripper_abierto=True):
        self.config = config
        self.ik_manual = ik_manual
        self.objetivo_posicion = np.asarray(objetivo_posicion, dtype=float)
        self.objetivo_rpy_deg = np.asarray(objetivo_rpy_deg, dtype=float)
        self.marco_orientacion = marco_orientacion
        self.gripper_abierto = gripper_abierto
        self._ids = None
        self._estado = robot.new_robot_state()

    # ------------------------------------------------------------------ modelo
    def construir_modelo(self, xml):
        """Compone el MJCF de la planta con el Panda de Menagerie."""
        return robot.build_model_with_franka(xml, self.config)

    def vincular(self, model):
        self._ids = robot.get_robot_ids(model)

    # ------------------------------------------------------------------ estado
    @property
    def estado(self):
        return self._estado["state"]

    @property
    def ciclos(self):
        return self._estado["cycles"]

    @property
    def indice_objeto_agarrado(self):
        return self._estado["grasped_obj"]

    @property
    def indice_objeto_reservado(self):
        return self._estado["target_obj"]

    def tiene_agarrado(self, objeto):
        return objeto.identificador == self._estado["grasped_obj"]

    def retiene(self, objeto):
        """True si el robot ha reservado la pieza y la cinta no debe moverla."""
        return (objeto.identificador == self._estado["target_obj"]
                and self._estado["state"] != "idle")

    def fuerza_retencion(self, pos, vel, fuerza_maxima):
        return robot.robot_hold_force(pos, vel, self.config, fuerza_maxima)

    def texto_objetivo(self):
        return robot.ik_target_text(self._estado)

    # ------------------------------------------------------------ inicialización
    def reiniciar(self, model, data):
        self._estado = robot.new_robot_state()
        robot.initialize_robot(model, data, self._ids, self._estado)
        if self.ik_manual:
            self.ir_a_objetivo(model, data)

    def ir_a_objetivo(self, model, data):
        robot.set_ik_target_rpy(model, self._ids, self._estado,
                                self.objetivo_posicion, self.objetivo_rpy_deg,
                                orientation_frame=self.marco_orientacion)
        mujoco.mj_forward(model, data)

    def ir_a_home(self, model, data):
        robot.set_ik_target_home(model, self._ids, self._estado)
        mujoco.mj_forward(model, data)

    def conmutar_modo(self, model, data):
        """Alterna entre IK manual y pick-and-place automático."""
        self.ik_manual = not self.ik_manual
        self.reiniciar(model, data)
        return self.ik_manual

    # ----------------------------------------------------------------- control
    def actualizar_estado(self, model, data, objetos):
        """Máquina de estados pick-and-place (solo en modo automático).

        robot.py trabaja con listas de índices; se adaptan y se devuelven los
        segmentos modificados (p. ej. pieza depositada en la Cinta 5).
        """
        if self.ik_manual:
            return
        segmentos = [o.segmento for o in objetos]
        robot.update_robot_state_machine(
            model, data, [o.body_id for o in objetos], self._ids, self._estado,
            segmentos, [o.z_reposo for o in objetos], self.config)
        for o, s in zip(objetos, segmentos):
            o.segmento = s

    def aplicar_control(self, model, data):
        if self.ik_manual:
            robot.control_manual_ik(model, data, self._ids, self._estado,
                                    gripper_open=self.gripper_abierto)
        else:
            robot.control_robot_ik(model, data, self._ids, self._estado, self.config)
            robot.follow_tcp_with_grasped_object(model, data, self._ids, self._estado)

    def seguir_pieza_agarrada(self, model, data):
        if not self.ik_manual:
            robot.follow_tcp_with_grasped_object(model, data, self._ids, self._estado)
