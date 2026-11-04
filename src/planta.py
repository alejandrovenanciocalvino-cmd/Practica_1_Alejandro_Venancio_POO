# -*- coding: utf-8 -*-
"""Clase PlantaIndustrial: coordina cintas, cámaras, objetos, horno y robot."""

import time

import mujoco
import mujoco.viewer

from .camara_inspeccion import Camara, CamaraInspeccion
from .cinta_transportadora import CintaTransportadora
from .config import (CAM_FPS, CAM_H, CAM_W, CAMARA_GENERAL, CAMARAS_CFG, CINTAS_CFG,
                     MAX_FORCE, OBJETOS_CFG, OUTPUT_DIR, OVEN_CENTER_X,
                     OVEN_ENFRIAMIENTO, OVEN_ENTRY_X, OVEN_INCREMENTO_PIEZA,
                     OVEN_PASO_TECLA, OVEN_TEMP_INICIAL, OVEN_TEMP_MAX, OVEN_TEMP_MEDIA,
                     OVEN_TEMP_MIN, ROBOT_CONFIG, ROBOT_IK_GRIPPER_OPEN,
                     ROBOT_IK_ORIENTATION_FRAME, ROBOT_IK_TARGET_POSITION,
                     ROBOT_IK_TARGET_RPY_DEG, ROBOT_IK_TEST_MODE, SHOW_CAMERA_MOSAIC,
                     SHOW_MUJOCO_VIEWER, TIEMPO_INSPECCION, VISION_AREA_MIN,
                     VISION_SAT_MIN, VISION_VAL_MIN, VISION_ZONA)
from .escena_mjcf import construir_mjcf
from .estacion_robotizada import EstacionRobotizada
from .horno import Horno
from .objeto_industrial import crear_objeto
from .vision import DetectorFormas
from .visualizacion import VisorCamaras


class PlantaIndustrial:
    """Elemento coordinador del sistema (sustituye a run() y a las funciones globales)."""

    def __init__(self, mostrar_visor=SHOW_MUJOCO_VIEWER, mostrar_camaras=SHOW_CAMERA_MOSAIC,
                 robot_ik_manual=ROBOT_IK_TEST_MODE):
        self.mostrar_visor = mostrar_visor
        self.mostrar_camaras = mostrar_camaras

        # --- composición: la planta posee a todos sus elementos
        self.cintas = [CintaTransportadora(**c) for c in CINTAS_CFG]
        self.cinta_horno = self.cintas[-1]
        self.objetos = [crear_objeto(c) for c in OBJETOS_CFG]
        self.horno = Horno(
            centro_x=OVEN_CENTER_X, entrada_x=OVEN_ENTRY_X,
            temperatura_inicial=OVEN_TEMP_INICIAL, temperatura_minima=OVEN_TEMP_MIN,
            temperatura_maxima=OVEN_TEMP_MAX, temperatura_media=OVEN_TEMP_MEDIA,
            incremento_pieza=OVEN_INCREMENTO_PIEZA, enfriamiento=OVEN_ENFRIAMIENTO)
        self.robot = EstacionRobotizada(
            ROBOT_CONFIG, ik_manual=robot_ik_manual,
            objetivo_posicion=ROBOT_IK_TARGET_POSITION,
            objetivo_rpy_deg=ROBOT_IK_TARGET_RPY_DEG,
            marco_orientacion=ROBOT_IK_ORIENTATION_FRAME,
            gripper_abierto=ROBOT_IK_GRIPPER_OPEN)
        self.camaras = [
            CamaraInspeccion(
                nombre, cinta,
                vision=DetectorFormas(VISION_SAT_MIN, VISION_VAL_MIN, VISION_AREA_MIN,
                                      VISION_ZONA),
                tiempo_inspeccion=TIEMPO_INSPECCION)
            for nombre, cinta in zip(CAMARAS_CFG, self.cintas)]
        self.camara_general = Camara(CAMARA_GENERAL, "VISTA GENERAL")
        self.visor = VisorCamaras(CAM_W, CAM_H, directorio_capturas=OUTPUT_DIR) \
            if mostrar_camaras else None

        # --- recursos de MuJoCo (se crean en inicializar())
        self.model = None
        self.data = None
        self.renderer = None
        self.viewer = None
        self.mosaico = None

        self.pausada = False
        self.ejecutando = False
        self._t_camaras = -1.0
        self._t_render_pausa = 0.0

    # ---------------------------------------------------------- inicialización
    def inicializar(self):
        xml = construir_mjcf(self.objetos, self.horno)
        self.model = self.robot.construir_modelo(xml)
        self.data = mujoco.MjData(self.model)

        for o in self.objetos:
            o.vincular(self.model)
        for c in self.cintas:
            c.vincular(self.model)
        self.horno.vincular(self.model)
        self.robot.vincular(self.model)

        self.reiniciar()

        if self.mostrar_camaras:
            self.renderer = mujoco.Renderer(self.model, height=CAM_H, width=CAM_W)
            self.visor.abrir()
        if self.mostrar_visor:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMARA_GENERAL)
        self._imprimir_banner()

    def reiniciar(self):
        """Reinicia piezas, robot, horno y cámaras (tecla R)."""
        for o in self.objetos:
            o.reiniciar(self.data)
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.robot.reiniciar(self.model, self.data)
        self.horno.reiniciar(self.model)
        for cam in self.camaras:
            cam.reiniciar()
        for cinta in self.cintas:
            if cinta.detenida_por_camara():
                cinta.arrancar()
        self._aplicar_rodillos()

    def _imprimir_banner(self):
        print("\n=== PLANTA ROBOTIZADA: 4 CINTAS + PANDA + HORNO (POO) ===")
        print("Circuito: inferior -> derecha -> superior -> izquierda -> inferior")
        print("Estación robot: Cinta 4 -> Franka Panda -> Cinta 5 -> horno")
        print("Q/ESC salir | P pausa | 1..5 cintas | R reset | S captura | "
              "+/- temperatura horno | M modo robot | A auto-reanudar cámaras")
        if self.robot.ik_manual:
            print("Modo IK TEST: T=pose objetivo | H=HOME | Target:", self.robot.texto_objetivo())
        else:
            print("Modo robot: pick-and-place automático")
        print("Cintas:", " ".join(c.texto_estado() for c in self.cintas), "\n")

    # ------------------------------------------------------------- simulación
    def _aplicar_rodillos(self):
        for cinta in self.cintas:
            cinta.aplicar_rodillos(self.data)

    def _aplicar_transporte(self):
        """Fuerza de cada cinta sobre las piezas, transferencias, horno y rescate."""
        data = self.data
        data.xfrc_applied[:] = 0.0

        for obj in self.objetos:
            if self.robot.tiene_agarrado(obj):
                continue                      # lo lleva el robot: la cinta no actúa
            pos = obj.posicion(data)
            vel = obj.velocidad(data)

            if self.robot.retiene(obj):       # pieza reservada: parada en zona de pick
                fx, fy = self.robot.fuerza_retencion(pos, vel, MAX_FORCE)
            else:
                cinta = self.cintas[obj.segmento]
                if cinta.debe_transferir(pos):
                    obj.segmento = cinta.indice_siguiente
                    cinta = self.cintas[obj.segmento]
                fx, fy = cinta.fuerza_transporte(pos, vel)
            obj.aplicar_fuerza(data, fx, fy)

            if (self.cintas[obj.segmento] is self.cinta_horno
                    and self.horno.pieza_entra(pos[0])):
                self.horno.registrar_pieza(obj.identificador)
                obj.reaparecer_en_cinta1(data, -1.05 + 0.18 * (obj.identificador % 3))
                continue

            if obj.fuera_de_limites(pos):
                obj.reaparecer_en_cinta1(data, -0.8 + 0.25 * obj.identificador)

    def actualizar(self):
        """Un paso de simulación (física + control + estabilización)."""
        self.robot.actualizar_estado(self.model, self.data, self.objetos)
        self._aplicar_rodillos()
        self._aplicar_transporte()
        self.robot.aplicar_control(self.model, self.data)

        mujoco.mj_step(self.model, self.data)

        for obj in self.objetos:
            obj.estabilizar(self.data, self.cintas[obj.segmento])
        self.robot.seguir_pieza_agarrada(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.horno.actualizar(self.model, self.model.opt.timestep)

    # ----------------------------------------------------------------- visión
    def texto_estado_global(self):
        flags = " ".join(c.texto_estado() for c in self.cintas)
        return (f"t={self.data.time:6.1f}s | {'PAUSA' if self.pausada else 'EN MARCHA'} | "
                f"{flags} | Panda={self.robot.estado} ciclos={self.robot.ciclos} | "
                f"horno: {self.horno.numero_piezas} pzs, "
                f"{self.horno.obtener_temperatura():.0f}C {self.horno.estado}")

    def actualizar_camaras(self):
        """Captura, detecta, muestra el mosaico OpenCV y lee el teclado."""
        ahora = self.data.time
        if self.pausada:
            ahora_real = time.perf_counter()
            toca = ahora_real - self._t_render_pausa >= 1.0 / CAM_FPS
            if toca:
                self._t_render_pausa = ahora_real
        else:
            toca = self._t_camaras < 0.0 or ahora - self._t_camaras >= 1.0 / CAM_FPS

        if toca:
            for cam in self.camaras:
                if self.pausada:
                    cam.refrescar(self.renderer, self.data)
                else:
                    cam.actualizar(self.renderer, self.data)
            self.camara_general.capturar(self.renderer, self.data)
            self.mosaico = self.visor.componer(self.camaras + [self.camara_general],
                                               self.texto_estado_global())
            self.visor.mostrar(self.mosaico)
            self._t_camaras = ahora

        self.procesar_teclado(self.visor.leer_tecla())

    # ---------------------------------------------------------------- teclado
    def procesar_teclado(self, tecla):
        if tecla in (ord('q'), 27):
            self.ejecutando = False
        elif tecla == ord('p'):
            self.pausada = not self.pausada
            print("PAUSA GLOBAL" if self.pausada else "SIMULACIÓN REANUDADA")
        elif tecla in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            n = tecla - ord('0')
            activa = self.cintas[n - 1].conmutar()
            self._aplicar_rodillos()
            print(f"Cinta {n}: {'ON' if activa else 'OFF'}")
        elif tecla == ord('r'):
            self.reiniciar()
            print("Planta, Panda y horno reiniciados")
        elif tecla == ord('t') and self.robot.ik_manual:
            self.robot.ir_a_objetivo(self.model, self.data)
            print("[IK] Target configurado:", self.robot.texto_objetivo())
        elif tecla == ord('h') and self.robot.ik_manual:
            self.robot.ir_a_home(self.model, self.data)
            print("[IK] Target HOME cartesiano activado")
        elif tecla == ord('m'):
            manual = self.robot.conmutar_modo(self.model, self.data)
            print("Modo robot:", "IK manual" if manual else "pick-and-place automático")
        elif tecla in (ord('+'), ord('=')):
            self.horno.aumentar_temperatura(OVEN_PASO_TECLA)
            print(f"Horno: {self.horno.obtener_temperatura():.0f} °C ({self.horno.estado})")
        elif tecla in (ord('-'), ord('_')):
            self.horno.disminuir_temperatura(OVEN_PASO_TECLA)
            print(f"Horno: {self.horno.obtener_temperatura():.0f} °C ({self.horno.estado})")
        elif tecla == ord('a'):
            nuevo = not self.camaras[0].reanudacion_automatica
            for cam in self.camaras:
                cam.reanudacion_automatica = nuevo
            print("Reanudación automática tras inspección:", "SÍ" if nuevo else "NO (manual)")
        elif tecla == ord('s') and self.mosaico is not None:
            print(f"Captura guardada: {self.visor.guardar(self.mosaico)}")

    # ------------------------------------------------------------ bucle principal
    def ejecutar(self):
        self.inicializar()
        self.ejecutando = True
        try:
            while self.ejecutando:
                if self.viewer is not None and not self.viewer.is_running():
                    break
                inicio = time.perf_counter()

                if not self.pausada:
                    self.actualizar()
                if self.viewer is not None:
                    self.viewer.sync()
                if self.visor is not None:
                    self.actualizar_camaras()

                restante = self.model.opt.timestep - (time.perf_counter() - inicio)
                if restante > 0:
                    time.sleep(restante)
        finally:
            self.cerrar()

    def cerrar(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        if self.visor is not None:
            self.visor.cerrar()
