"""Clase PlantaIndustrial.
Coordinador global: crea y conecta cintas, cámaras, objetos, horno y robot.
Sustituye a la función procedural `run()`."""

from __future__ import annotations
import time
from pathlib import Path
from typing import List

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from . import config as cfg
from .cinta_transportadora import CintaTransportadora
from .camara_inspeccion import CamaraInspeccion
from .objeto_industrial import ObjetoIndustrial
from .horno import Horno
from . import robot


class PlantaIndustrial:
    """Coordinador de toda la planta."""

    def __init__(self):
        self.cintas: List[CintaTransportadora] = []
        self.camaras: List[CamaraInspeccion] = []
        self.objetos: List[ObjetoIndustrial] = []
        self.horno: Horno | None = None

        self.model = None
        self.data = None
        self.renderer = None
        self.viewer_ctx = None
        self.robot_ids = None
        self.robot_state = None
        self.robot_config = None

        self.paused = False
        self.running = True
        self.last_cam_time = -1.0
        self.output_dir = Path(__file__).resolve().parent.parent / "capturas"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # INICIALIZACIÓN
    # ==================================================================
    def inicializar(self) -> None:
        self.robot_config = robot.make_config(
            lane=cfg.LANE, branch_y=cfg.BRANCH_Y, branch_speed=cfg.BRANCH_SPEED)

        # 1) Modelo MuJoCo
        xml = self._build_mjcf()
        self.model = robot.build_model_with_franka(xml, self.robot_config)
        self.data = mujoco.MjData(self.model)

        # 2) Cintas
        nombres = [
            (1, "CINTA 1 · INFERIOR · +X", cfg.BELT_SPEED),
            (2, "CINTA 2 · DERECHA  · +Y", cfg.BELT_SPEED),
            (3, "CINTA 3 · SUPERIOR · -X", cfg.BELT_SPEED),
            (4, "CINTA 4 · IZQUIERDA· -Y", cfg.BELT_SPEED),
            (5, "CINTA 5 · HORNO    · -X", cfg.BRANCH_SPEED),
        ]
        self.cintas = [CintaTransportadora(n, nom, v) for n, nom, v in nombres]

        # 3) Objetos
        self.objetos = [ObjetoIndustrial(i, self.model) for i in range(5)]

        # 4) Horno
        self.horno = Horno(self.model, temperatura_inicial=cfg.OVEN_TEMP_MIN)

        # 5) Cámaras (una por cinta, composición cámara→cinta)
        camaras_cfg = [
            ("cam_bottom", "CINTA 1", self.cintas[0]),
            ("cam_right",  "CINTA 2", self.cintas[1]),
            ("cam_top",    "CINTA 3", self.cintas[2]),
            ("cam_left",   "CINTA 4", self.cintas[3]),
            ("cam_oven",   "CINTA 5", self.cintas[4]),
        ]
        self.camaras = [CamaraInspeccion(n, e, c) for n, e, c in camaras_cfg]

        # 6) Robot
        self.robot_ids = robot.get_robot_ids(self.model)
        self.robot_state = robot.new_robot_state()
        self._reset_objetos()
        robot.initialize_robot(self.model, self.data,
                               self.robot_ids, self.robot_state)

        if cfg.ROBOT_IK_TEST_MODE:
            robot.set_ik_target_rpy(
                self.model, self.robot_ids, self.robot_state,
                cfg.ROBOT_IK_TARGET_POSITION,
                cfg.ROBOT_IK_TARGET_RPY_DEG,
                orientation_frame=cfg.ROBOT_IK_ORIENTATION_FRAME,
            )

        mujoco.mj_forward(self.model, self.data)
        self._actualizar_rodillos()

        # 7) Renderer y viewer
        if cfg.SHOW_CAMERA_MOSAIC:
            self.renderer = mujoco.Renderer(self.model,
                                            height=cfg.CAM_H, width=cfg.CAM_W)
            cv2.namedWindow("Camaras de la planta", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camaras de la planta", cfg.CAM_W * 3, cfg.CAM_H * 2)

        if cfg.SHOW_MUJOCO_VIEWER:
            self.viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data)
            overview_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
            self.viewer_ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer_ctx.cam.fixedcamid = overview_id

        self._imprimir_info()

    # ==================================================================
    # BUCLE PRINCIPAL
    # ==================================================================
    def ejecutar(self) -> None:
        try:
            while self.running:
                if self.viewer_ctx is not None and not self.viewer_ctx.is_running():
                    break
                step_start = time.perf_counter()
                if not self.paused:
                    self._paso_simulacion()
                mujoco.mj_step(self.model, self.data)
                self._estabilizar_objetos()
                mujoco.mj_forward(self.model, self.data)
                if self.viewer_ctx is not None:
                    self.viewer_ctx.sync()
                self._refrescar_camaras()
                self._procesar_teclado()
                elapsed = time.perf_counter() - step_start
                sleep_time = self.model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self._cerrar()

    # ==================================================================
    # PASO DE SIMULACIÓN
    # ==================================================================
    def _paso_simulacion(self) -> None:
        self.data.xfrc_applied[:] = 0.0

        # Transporte de cada objeto
        for obj in self.objetos:
            # Si el robot porta la pieza, la cinta no actúa
            if obj.indice == self.robot_state.get("grasped_obj", -1):
                continue
            # Cubo reservado por el robot
            if (obj.indice == self.robot_state.get("target_obj", -1)
                    and self.robot_state.get("state") != "idle"):
                pos = self.data.xpos[obj.body_id]
                vel = self.data.qvel[obj.dof_addr:obj.dof_addr + 3]
                fx, fy = robot.robot_hold_force(
                    pos, vel, self.robot_config, cfg.MAX_FORCE)
            else:
                pos = self.data.xpos[obj.body_id]
                vel = self.data.qvel[obj.dof_addr:obj.dof_addr + 3]
                obj.actualizar_segmento(pos)
                cinta = self.cintas[obj.segmento]
                fx, fy = cinta.calcular_fuerza(pos, vel)

                # Entrada al horno
                if obj.ha_llegado_al_horno(pos):
                    self.horno.registrar_pieza()
                    self._reintegrar_objeto(obj)
                    continue

            self.data.xfrc_applied[obj.body_id, 0] = fx
            self.data.xfrc_applied[obj.body_id, 1] = fy

        # Robot
        if cfg.ROBOT_IK_TEST_MODE:
            robot.control_manual_ik(
                self.model, self.data, self.robot_ids, self.robot_state,
                gripper_open=cfg.ROBOT_IK_GRIPPER_OPEN,
            )
        else:
            robot.update_robot_state_machine(
                self.model, self.data,
                [o.body_id for o in self.objetos],
                self.robot_ids, self.robot_state,
                [o.segmento for o in self.objetos],
                cfg.OBJECT_REST_Z, self.robot_config,
            )
            robot.control_robot_ik(
                self.model, self.data, self.robot_ids,
                self.robot_state, self.robot_config)
            robot.follow_tcp_with_grasped_object(
                self.model, self.data, self.robot_ids, self.robot_state)

        # Horno: subir temperatura mientras hay piezas y refrescar color
        if self.horno.numero_piezas > 0:
            self.horno.aumentar_temperatura(0.5)
        else:
            self.horno.disminuir_temperatura(0.2)
        self.horno.actualizar_color(self.model)

    def _estabilizar_objetos(self) -> None:
        if not cfg.LOCK_OBJECT_ROTATION:
            return
        for obj in self.objetos:
            self.data.qpos[obj.qpos_addr + 3:obj.qpos_addr + 7] = [1, 0, 0, 0]
            self.data.qvel[obj.dof_addr + 3:obj.dof_addr + 6] = 0.0
        if cfg.LOCK_CYLINDERS_TO_CENTERLINE:
            for obj in self.objetos:
                if obj.indice not in cfg.CYLINDER_OBJECT_IDS:
                    continue
                x = float(self.data.qpos[obj.qpos_addr])
                y = float(self.data.qpos[obj.qpos_addr + 1])
                s = obj.segmento
                if s == 0 and x < cfg.TURN_START:
                    self.data.qpos[obj.qpos_addr + 1] = -cfg.LANE
                    self.data.qvel[obj.dof_addr + 1] = 0.0
                elif s == 1 and y < cfg.TURN_START:
                    self.data.qpos[obj.qpos_addr] = cfg.LANE
                    self.data.qvel[obj.dof_addr] = 0.0
                elif s == 2 and x > -cfg.TURN_START:
                    self.data.qpos[obj.qpos_addr + 1] = cfg.LANE
                    self.data.qvel[obj.dof_addr + 1] = 0.0
                elif s == 3 and y > -cfg.TURN_START:
                    self.data.qpos[obj.qpos_addr] = -cfg.LANE
                    self.data.qvel[obj.dof_addr] = 0.0
                elif s == 4:
                    self.data.qpos[obj.qpos_addr + 1] = cfg.BRANCH_Y
                    self.data.qvel[obj.dof_addr + 1] = 0.0

    def _reintegrar_objeto(self, obj: ObjetoIndustrial) -> None:
        """Tras entrar al horno, reaparece en Cinta 1."""
        x0 = -1.05 + 0.18 * (obj.indice % 3)
        self.data.qpos[obj.qpos_addr:obj.qpos_addr + 3] = [
            x0, -cfg.LANE, cfg.OBJECT_REST_Z[obj.indice]]
        self.data.qpos[obj.qpos_addr + 3:obj.qpos_addr + 7] = [1, 0, 0, 0]
        self.data.qvel[obj.dof_addr:obj.dof_addr + 6] = 0.0
        obj.segmento = 0

    def _reset_objetos(self) -> None:
        starts = [
            (-cfg.LANE,  0.95, cfg.OBJECT_REST_Z[0], 3),
            ( cfg.LANE, -0.72, cfg.OBJECT_REST_Z[1], 1),
            ( 0.58,  cfg.LANE, cfg.OBJECT_REST_Z[2], 2),
            (-0.62, -cfg.LANE, cfg.OBJECT_REST_Z[3], 0),
            (-0.45,  cfg.LANE, cfg.OBJECT_REST_Z[4], 2),
        ]
        for obj, (x, y, z, seg) in zip(self.objetos, starts):
            self.data.qpos[obj.qpos_addr:obj.qpos_addr + 3] = [x, y, z]
            self.data.qpos[obj.qpos_addr + 3:obj.qpos_addr + 7] = [1, 0, 0, 0]
            self.data.qvel[obj.dof_addr:obj.dof_addr + 6] = 0.0
            obj.segmento = seg
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ==================================================================
    # RODILLOS
    # ==================================================================
    def _actualizar_rodillos(self) -> None:
        actuadores = {
            1: ["belt_act_b1", "belt_act_b2"],
            2: ["belt_act_r1", "belt_act_r2"],
            3: ["belt_act_t1", "belt_act_t2"],
            4: ["belt_act_l1", "belt_act_l2"],
            5: ["belt_act_o1", "belt_act_o2"],
        }
        for cinta in self.cintas:
            v = cinta.velocidad_rodillos()
            for nombre in actuadores[cinta.numero]:
                aid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nombre)
                if aid >= 0:
                    self.data.ctrl[aid] = v

    # ==================================================================
    # CÁMARAS / MOSAICO
    # ==================================================================
    def _refrescar_camaras(self) -> None:
        if not cfg.SHOW_CAMERA_MOSAIC or self.renderer is None:
            return
        if (self.last_cam_time >= 0.0
                and (self.data.time - self.last_cam_time) < (1.0 / cfg.CAM_FPS)
                and not self.paused):
            return
        frames = [cam.actualizar(self.renderer, self.data) for cam in self.camaras]
        # Vista general
        self.renderer.update_scene(self.data, camera="overview")
        rgb = self.renderer.render()
        general = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frames.append(self._etiquetar(general, "VISTA GENERAL"))

        top = np.hstack(frames[:3])
        bottom = np.hstack(frames[3:])
        mosaic = np.vstack((top, bottom))
        flags_txt = " ".join(
            f"C{c.numero}:{'ON' if c.esta_activa() else 'OFF'}"
            for c in self.cintas)
        status = (f"t={self.data.time:6.1f}s · "
                  f"{'PAUSA' if self.paused else 'MARCHA'} · "
                  f"{flags_txt} · "
                  f"Panda={self.robot_state['state']} · "
                  f"ciclos={self.robot_state['cycles']} · "
                  f"horno={self.horno.numero_piezas} "
                  f"T={self.horno.obtener_temperatura():.0f}°C")
        cv2.putText(mosaic, status, (12, mosaic.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Camaras de la planta", mosaic)
        self.last_cam_time = self.data.time

    @staticmethod
    def _etiquetar(img, texto):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(out, texto, (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
        return out

    # ==================================================================
    # TECLADO
    # ==================================================================
    def _procesar_teclado(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return
        if key in (ord('q'), 27):
            self.running = False
        elif key == ord('p'):
            self.paused = not self.paused
            print("PAUSA GLOBAL" if self.paused else "SIMULACIÓN REANUDADA")
        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            n = key - ord('0')
            cinta = self.cintas[n - 1]
            activa = cinta.conmutar()
            self._actualizar_rodillos()
            print(f"Cinta {n}: {'ON' if activa else 'OFF'}")
        elif key == ord('r'):
            self._reset_objetos()
            self.horno.numero_piezas = 0
            self.robot_state.clear()
            self.robot_state.update(robot.new_robot_state())
            robot.initialize_robot(self.model, self.data,
                                   self.robot_ids, self.robot_state)
            if cfg.ROBOT_IK_TEST_MODE:
                robot.set_ik_target_rpy(
                    self.model, self.robot_ids, self.robot_state,
                    cfg.ROBOT_IK_TARGET_POSITION, cfg.ROBOT_IK_TARGET_RPY_DEG,
                    orientation_frame=cfg.ROBOT_IK_ORIENTATION_FRAME)
            mujoco.mj_forward(self.model, self.data)
            self._actualizar_rodillos()
            print("Planta y Panda reiniciados")
        elif key == ord('t') and cfg.ROBOT_IK_TEST_MODE:
            robot.set_ik_target_rpy(
                self.model, self.robot_ids, self.robot_state,
                cfg.ROBOT_IK_TARGET_POSITION, cfg.ROBOT_IK_TARGET_RPY_DEG,
                orientation_frame=cfg.ROBOT_IK_ORIENTATION_FRAME)
            mujoco.mj_forward(self.model, self.data)
            print("[IK] Target:", robot.ik_target_text(self.robot_state))
        elif key == ord('h') and cfg.ROBOT_IK_TEST_MODE:
            robot.set_ik_target_home(self.model, self.robot_ids, self.robot_state)
            mujoco.mj_forward(self.model, self.data)
            print("[IK] Target HOME activado")
        elif key == ord('s'):
            self._guardar_captura()
        elif key == ord('+'):
            self.horno.aumentar_temperatura(10.0)
            self.horno.actualizar_color(self.model)
            print(f"Horno T={self.horno.obtener_temperatura():.1f}°C")
        elif key == ord('-'):
            self.horno.disminuir_temperatura(10.0)
            self.horno.actualizar_color(self.model)
            print(f"Horno T={self.horno.obtener_temperatura():.1f}°C")

    def _guardar_captura(self) -> None:
        if not cfg.SHOW_CAMERA_MOSAIC or self.renderer is None:
            return
        frames = [cam.actualizar(self.renderer, self.data) for cam in self.camaras]
        self.renderer.update_scene(self.data, camera="overview")
        rgb = self.renderer.render()
        frames.append(self._etiquetar(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                      "VISTA GENERAL"))
        top = np.hstack(frames[:3])
        bottom = np.hstack(frames[3:])
        mosaic = np.vstack((top, bottom))
        filename = self.output_dir / f"mosaico_{int(time.time())}.png"
        cv2.imwrite(str(filename), mosaic)
        print(f"Captura guardada: {filename}")

    # ==================================================================
    # UTILIDADES
    # ==================================================================
    def _imprimir_info(self) -> None:
        print("\n=== PLANTA ROBOTIZADA (POO) ===")
        print("Circuito: inferior -> derecha -> superior -> izquierda -> inferior")
        print("Estación robot: Cinta 4 -> Panda -> Cinta 5 -> horno")
        print(f"Cintas: {len(self.cintas)}")
        print(f"Cámaras: {len(self.camaras)}")
        print(f"Objetos: {len(self.objetos)}")
        print(f"Horno: {self.horno}")
        flags = " ".join(
            f"C{c.numero}={'ON' if c.esta_activa() else 'OFF'}" for c in self.cintas)
        print(f"Banderas: {flags}")
        print("Teclas: Q/ESC salir | P pausa | 1..5 cintas | R reset | "
              "S captura | +/- horno")

    def _cerrar(self) -> None:
        if self.viewer_ctx is not None:
            self.viewer_ctx.close()
        if self.renderer is not None:
            self.renderer.close()
        cv2.destroyAllWindows()

    # ==================================================================
    # MJCF (idéntico al procedural, pero como método de la clase)
    # ==================================================================
    def _build_mjcf(self) -> str:
        """Construye el MJCF. Mismo XML que el original, pero con el
        geom del horno llamado 'oven_interior' para poder cambiar su color."""
        cam_bottom = (0.0, -(cfg.LANE + cfg.CAM_LENS_SIDE_OFFSET), cfg.CAM_HEIGHT)
        cam_right  = (cfg.LANE + cfg.CAM_LENS_SIDE_OFFSET, 0.0, cfg.CAM_HEIGHT)
        cam_top    = (0.0, cfg.LANE + cfg.CAM_LENS_SIDE_OFFSET, cfg.CAM_HEIGHT)
        cam_left = (-(cfg.LANE + cfg.BELT_HALF_WIDTH + cfg.CAM_LEFT_HEAD_CLEARANCE),
                    cfg.CAM_LEFT_INSPECTION_Y, cfg.CAM_HEIGHT)
        cam_oven = (cfg.BRANCH_CENTER_X,
                    cfg.BRANCH_Y - cfg.CAM_OVEN_SIDE_OFFSET, cfg.CAM_OVEN_HEIGHT)

        target_bottom = (0.0, -cfg.LANE, cfg.BELT_TOP_Z + 0.015)
        target_right  = (cfg.LANE, 0.0, cfg.BELT_TOP_Z + 0.015)
        target_top    = (0.0, cfg.LANE, cfg.BELT_TOP_Z + 0.015)
        target_left   = (-cfg.LANE, cfg.CAM_LEFT_INSPECTION_Y, cfg.BELT_TOP_Z + 0.015)
        target_oven   = (cfg.BRANCH_CENTER_X, cfg.BRANCH_Y, cfg.BELT_TOP_Z + 0.015)

        pole_bottom_y = -(cfg.LANE + cfg.CAM_POLE_SIDE_OFFSET)
        pole_right_x  =  (cfg.LANE + cfg.CAM_POLE_SIDE_OFFSET)
        pole_top_y    =  (cfg.LANE + cfg.CAM_POLE_SIDE_OFFSET)
        pole_left_x   = -(cfg.LANE + cfg.BELT_HALF_WIDTH + cfg.CAM_LEFT_POLE_CLEARANCE)
        pole_left_y   = cfg.CAM_LEFT_INSPECTION_Y
        pole_oven_y   = cfg.BRANCH_Y - (cfg.BRANCH_HALF_WIDTH + 0.52)

        def xyaxes(cam_pos, target):
            cam = np.asarray(cam_pos, dtype=float)
            tgt = np.asarray(target, dtype=float)
            z_axis = cam - tgt
            z_axis /= np.linalg.norm(z_axis)
            up = np.array([0.0, 0.0, 1.0])
            x_axis = np.cross(up, z_axis)
            if np.linalg.norm(x_axis) < 1e-8:
                up = np.array([0.0, 1.0, 0.0])
                x_axis = np.cross(up, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            vals = np.concatenate([x_axis, y_axis])
            return " ".join(f"{v:.6f}" for v in vals)

        xy_bottom = xyaxes(cam_bottom, target_bottom)
        xy_right  = xyaxes(cam_right,  target_right)
        xy_top    = xyaxes(cam_top,    target_top)
        xy_left   = xyaxes(cam_left,   target_left)
        xy_oven   = xyaxes(cam_oven,   target_oven)

        xml = f"""
<mujoco model="square_conveyor_factory">
<compiler angle="degree" autolimits="true"/>
<option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
<visual>
  <global offwidth="1280" offheight="960"/>
  <quality shadowsize="2048"/>
  <headlight ambient="0.35 0.35 0.35" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
</visual>
<default>
  <geom contype="1" conaffinity="1" friction="1.2 0.02 0.002" condim="4"/>
  <joint damping="0.15"/>
</default>
<asset>
  <texture name="floor_tex" type="2d" builtin="checker"
           rgb1="0.18 0.20 0.22" rgb2="0.23 0.25 0.27" width="512" height="512"/>
  <material name="floor_mat" texture="floor_tex" texrepeat="8 8" reflectance="0.10"/>
  <material name="belt_mat" rgba="0.08 0.10 0.12 1"/>
  <material name="belt_edge_mat" rgba="0.28 0.32 0.36 1"/>
  <material name="transfer_mat" rgba="0.18 0.20 0.23 1"/>
  <material name="camera_mat" rgba="0.07 0.07 0.08 1"/>
  <material name="pole_mat" rgba="0.30 0.33 0.36 1"/>
  <material name="roller_mat" rgba="0.13 0.15 0.17 1"/>
  <material name="island_mat" rgba="0.24 0.28 0.31 1"/>
  <material name="island_panel_mat" rgba="0.10 0.38 0.52 1"/>
  <material name="oven_mat" rgba="0.22 0.24 0.26 1"/>
  <material name="oven_dark_mat" rgba="0.06 0.06 0.07 1"/>
</asset>
<worldbody>
  <light pos="0 0 6" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
  <light pos="3 -3 4" dir="-0.5 0.5 -1" diffuse="0.45 0.45 0.45"/>
  <geom name="floor" type="plane" size="6 6 0.1" material="floor_mat"/>

  <geom name="belt_bottom" type="box" pos="0 -{cfg.LANE} {cfg.BELT_Z}"
        size="{cfg.BELT_HALF_LENGTH} {cfg.BELT_HALF_WIDTH} 0.06" material="belt_mat"/>
  <geom type="box" pos="0 -{cfg.LANE-cfg.BELT_HALF_WIDTH-cfg.RAIL_THICKNESS} {cfg.RAIL_CENTER_Z}"
        size="{cfg.RAIL_HALF_LENGTH} {cfg.RAIL_THICKNESS} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
  <geom type="box" pos="0 -{cfg.LANE+cfg.BELT_HALF_WIDTH+cfg.RAIL_THICKNESS} {cfg.RAIL_CENTER_Z}"
        size="{cfg.RAIL_HALF_LENGTH} {cfg.RAIL_THICKNESS} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

  <geom name="belt_right" type="box" pos="{cfg.LANE} 0 {cfg.BELT_Z}"
        size="{cfg.BELT_HALF_WIDTH} {cfg.BELT_HALF_LENGTH} 0.06" material="belt_mat"/>
  <geom type="box" pos="{cfg.LANE-cfg.BELT_HALF_WIDTH-cfg.RAIL_THICKNESS} 0 {cfg.RAIL_CENTER_Z}"
        size="{cfg.RAIL_THICKNESS} {cfg.RAIL_HALF_LENGTH} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
  <geom type="box" pos="{cfg.LANE+cfg.BELT_HALF_WIDTH+cfg.RAIL_THICKNESS} 0 {cfg.RAIL_CENTER_Z}"
        size="{cfg.RAIL_THICKNESS} {cfg.RAIL_HALF_LENGTH} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

  <geom name="belt_top" type="box" pos="0 {cfg.LANE} {cfg.BELT_Z}"
        size="{cfg.BELT_HALF_LENGTH} {cfg.BELT_HALF_WIDTH} 0.06" material="belt_mat"/>
  <geom type="box" pos="0 {cfg.LANE-cfg.BELT_HALF_WIDTH-cfg.RAIL_THICKNESS} {cfg.RAIL_CENTER_Z}"
        size="{cfg.RAIL_HALF_LENGTH} {cfg.RAIL_THICKNESS} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
  <geom type="box" pos="0 {cfg.LANE+cfg.BELT_HALF_WIDTH+cfg.RAIL_THICKNESS} {cfg.RAIL_CENTER_Z}"
        size="{cfg.RAIL_HALF_LENGTH} {cfg.RAIL_THICKNESS} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

  <geom name="belt_left" type="box" pos="-{cfg.LANE} 0 {cfg.BELT_Z}"
        size="{cfg.BELT_HALF_WIDTH} {cfg.BELT_HALF_LENGTH} 0.06" material="belt_mat"/>
  <geom type="box" pos="-{cfg.LANE-cfg.BELT_HALF_WIDTH-cfg.RAIL_THICKNESS} {cfg.LEFT_RAIL_UPPER_CENTER_Y} {cfg.RAIL_CENTER_Z}"
        size="{cfg.RAIL_THICKNESS} {cfg.LEFT_RAIL_UPPER_HALF} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
  <geom type="box" pos="-{cfg.LANE-cfg.BELT_HALF_WIDTH-cfg.RAIL_THICKNESS} {cfg.LEFT_RAIL_LOWER_CENTER_Y} {cfg.RAIL_CENTER_Z}"
        size="{cfg.RAIL_THICKNESS} {cfg.LEFT_RAIL_LOWER_HALF} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
  <geom type="box" pos="-{cfg.LANE+cfg.BELT_HALF_WIDTH+cfg.RAIL_THICKNESS} {cfg.BRANCH_RAIL_CENTER_Y} {cfg.RAIL_CENTER_Z}"
        size="{cfg.RAIL_THICKNESS} {cfg.BRANCH_RAIL_HALF} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
  <geom type="box" pos="-{cfg.LANE+cfg.BELT_HALF_WIDTH+cfg.RAIL_THICKNESS} -{cfg.BRANCH_RAIL_CENTER_Y} {cfg.RAIL_CENTER_Z}"
        size="{cfg.RAIL_THICKNESS} {cfg.BRANCH_RAIL_HALF} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

  <geom name="belt_oven" type="box" pos="{cfg.BRANCH_CENTER_X} {cfg.BRANCH_Y} {cfg.BELT_Z}"
        size="{cfg.BRANCH_HALF_LENGTH} {cfg.BRANCH_HALF_WIDTH} 0.06" material="belt_mat"/>
  <geom name="branch_transfer" type="box" pos="-{cfg.LANE} {cfg.BRANCH_Y} {cfg.BELT_Z}"
        size="{cfg.BELT_HALF_WIDTH+0.06} {cfg.BRANCH_HALF_WIDTH} 0.06" material="transfer_mat"/>
  <geom type="box" pos="{cfg.BRANCH_RAIL_CENTER_X} {cfg.BRANCH_Y-cfg.BRANCH_HALF_WIDTH-cfg.RAIL_THICKNESS} {cfg.RAIL_CENTER_Z}"
        size="{cfg.BRANCH_RAIL_HALF_LENGTH} {cfg.RAIL_THICKNESS} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
  <geom type="box" pos="{cfg.BRANCH_RAIL_CENTER_X} {cfg.BRANCH_Y+cfg.BRANCH_HALF_WIDTH+cfg.RAIL_THICKNESS} {cfg.RAIL_CENTER_Z}"
        size="{cfg.BRANCH_RAIL_HALF_LENGTH} {cfg.RAIL_THICKNESS} {cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

  <body name="process_island" pos="{self.robot_config['base_x']} {self.robot_config['base_y']} 0">
    <geom name="island_base" type="box" pos="0 0 {self.robot_config['base_z']/2}"
          size="{0.18} {0.38} {self.robot_config['base_z']/2}"
          material="island_mat" contype="0" conaffinity="0"/>
    <geom name="island_panda_plate" type="cylinder" pos="0 0 {self.robot_config['base_z']+0.025}"
          size="0.16 0.025" material="belt_edge_mat" contype="0" conaffinity="0"/>
  </body>

  <body name="oven" pos="{cfg.OVEN_CENTER_X} 0 0">
    <geom type="box" pos="0 0 0.12" size="0.55 0.62 0.10" material="oven_mat"/>
    <geom type="box" pos="-0.50 0 0.68" size="0.08 0.62 0.58" material="oven_mat"/>
    <geom type="box" pos="0 -0.56 0.68" size="0.55 0.06 0.58" material="oven_mat"/>
    <geom type="box" pos="0  0.56 0.68" size="0.55 0.06 0.58" material="oven_mat"/>
    <geom type="box" pos="0 0 1.23" size="0.55 0.62 0.07" material="oven_mat"/>
    <geom type="box" pos="0.50 -0.48 0.68" size="0.05 0.08 0.58" material="oven_dark_mat"/>
    <geom type="box" pos="0.50  0.48 0.68" size="0.05 0.08 0.58" material="oven_dark_mat"/>
    <geom type="box" pos="0.50 0 1.16" size="0.05 0.40 0.10" material="oven_dark_mat"/>
    <!-- GEOM DEL HORNO QUE CAMBIA DE COLOR -->
    <geom name="oven_interior" type="box" pos="-0.42 0 0.58"
          size="0.025 0.39 0.31" rgba="0.22 0.24 0.26 1"
          contype="0" conaffinity="0"/>
    <light pos="-0.18 0 0.82" dir="1 0 -0.15" diffuse="1.0 0.25 0.05"/>
  </body>

  <geom name="corner_br" type="box" pos=" {cfg.LANE} -{cfg.LANE} {cfg.BELT_Z}"
        size="{cfg.BELT_HALF_WIDTH} {cfg.BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
  <geom name="corner_tr" type="box" pos=" {cfg.LANE}  {cfg.LANE} {cfg.BELT_Z}"
        size="{cfg.BELT_HALF_WIDTH} {cfg.BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
  <geom name="corner_tl" type="box" pos="-{cfg.LANE}  {cfg.LANE} {cfg.BELT_Z}"
        size="{cfg.BELT_HALF_WIDTH} {cfg.BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
  <geom name="corner_bl" type="box" pos="-{cfg.LANE} -{cfg.LANE} {cfg.BELT_Z}"
        size="{cfg.BELT_HALF_WIDTH} {cfg.BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>

  <body name="roller_b1" pos="-1.15 -{cfg.LANE} 0.20">
    <joint name="jr_b1" type="hinge" axis="0 1 0"/>
    <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
  </body>
  <body name="roller_b2" pos="1.15 -{cfg.LANE} 0.20">
    <joint name="jr_b2" type="hinge" axis="0 1 0"/>
    <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
  </body>
  <body name="roller_r1" pos="{cfg.LANE} -1.15 0.20">
    <joint name="jr_r1" type="hinge" axis="1 0 0"/>
    <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
  </body>
  <body name="roller_r2" pos="{cfg.LANE} 1.15 0.20">
    <joint name="jr_r2" type="hinge" axis="1 0 0"/>
    <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
  </body>
  <body name="roller_t1" pos="1.15 {cfg.LANE} 0.20">
    <joint name="jr_t1" type="hinge" axis="0 1 0"/>
    <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
  </body>
  <body name="roller_t2" pos="-1.15 {cfg.LANE} 0.20">
    <joint name="jr_t2" type="hinge" axis="0 1 0"/>
    <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
  </body>
  <body name="roller_l1" pos="-{cfg.LANE} 1.15 0.20">
    <joint name="jr_l1" type="hinge" axis="1 0 0"/>
    <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
  </body>
  <body name="roller_l2" pos="-{cfg.LANE} -1.15 0.20">
    <joint name="jr_l2" type="hinge" axis="1 0 0"/>
    <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
  </body>
  <body name="roller_o1" pos="-2.18 {cfg.BRANCH_Y} 0.20">
    <joint name="jr_o1" type="hinge" axis="0 1 0"/>
    <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
  </body>
  <body name="roller_o2" pos="-3.25 {cfg.BRANCH_Y} 0.20">
    <joint name="jr_o2" type="hinge" axis="0 1 0"/>
    <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
  </body>

  <body name="cam_station_bottom" pos="0 0 0">
    <geom type="cylinder" pos="0 {pole_bottom_y} {cfg.CAM_HEIGHT/2}"
          size="0.025 {cfg.CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
    <geom type="box" pos="0 {(pole_bottom_y + cam_bottom[1])/2} {cfg.CAM_HEIGHT}"
          size="0.025 {abs(cam_bottom[1]-pole_bottom_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
    <geom type="box" pos="{cam_bottom[0]} {cam_bottom[1]} {cam_bottom[2]}"
          size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
    <camera name="cam_bottom" pos="{cam_bottom[0]} {cam_bottom[1]} {cam_bottom[2]}"
            xyaxes="{xy_bottom}" fovy="{cfg.CAM_FOVY}"/>
  </body>
  <body name="cam_station_right" pos="0 0 0">
    <geom type="cylinder" pos="{pole_right_x} 0 {cfg.CAM_HEIGHT/2}"
          size="0.025 {cfg.CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
    <geom type="box" pos="{(pole_right_x + cam_right[0])/2} 0 {cfg.CAM_HEIGHT}"
          size="{abs(cam_right[0]-pole_right_x)/2} 0.025 0.022" material="pole_mat" contype="0" conaffinity="0"/>
    <geom type="box" pos="{cam_right[0]} {cam_right[1]} {cam_right[2]}"
          size="0.055 0.075 0.055" material="camera_mat" contype="0" conaffinity="0"/>
    <camera name="cam_right" pos="{cam_right[0]} {cam_right[1]} {cam_right[2]}"
            xyaxes="{xy_right}" fovy="{cfg.CAM_FOVY}"/>
  </body>
  <body name="cam_station_top" pos="0 0 0">
    <geom type="cylinder" pos="0 {pole_top_y} {cfg.CAM_HEIGHT/2}"
          size="0.025 {cfg.CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
    <geom type="box" pos="0 {(pole_top_y + cam_top[1])/2} {cfg.CAM_HEIGHT}"
          size="0.025 {abs(cam_top[1]-pole_top_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
    <geom type="box" pos="{cam_top[0]} {cam_top[1]} {cam_top[2]}"
          size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
    <camera name="cam_top" pos="{cam_top[0]} {cam_top[1]} {cam_top[2]}"
            xyaxes="{xy_top}" fovy="{cfg.CAM_FOVY}"/>
  </body>
  <body name="cam_station_left" pos="0 0 0">
    <geom type="cylinder" pos="{pole_left_x} {pole_left_y} {cfg.CAM_HEIGHT/2}"
          size="0.025 {cfg.CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
    <geom type="box" pos="{(pole_left_x + cam_left[0])/2} {pole_left_y} {cfg.CAM_HEIGHT}"
          size="{abs(cam_left[0]-pole_left_x)/2} 0.025 0.022" material="pole_mat" contype="0" conaffinity="0"/>
    <geom type="box" pos="{cam_left[0]} {cam_left[1]} {cam_left[2]}"
          size="0.055 0.075 0.055" material="camera_mat" contype="0" conaffinity="0"/>
    <camera name="cam_left" pos="{cam_left[0]} {cam_left[1]} {cam_left[2]}"
            xyaxes="{xy_left}" fovy="{cfg.CAM_FOVY}"/>
  </body>
  <body name="cam_station_oven" pos="0 0 0">
    <geom type="cylinder" pos="{cfg.BRANCH_CENTER_X} {pole_oven_y} {cfg.CAM_OVEN_HEIGHT/2}"
          size="0.025 {cfg.CAM_OVEN_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
    <geom type="box" pos="{cfg.BRANCH_CENTER_X} {(pole_oven_y + cam_oven[1])/2} {cfg.CAM_OVEN_HEIGHT}"
          size="0.025 {abs(cam_oven[1]-pole_oven_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
    <geom type="box" pos="{cam_oven[0]} {cam_oven[1]} {cam_oven[2]}"
          size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
    <camera name="cam_oven" pos="{cam_oven[0]} {cam_oven[1]} {cam_oven[2]}"
            xyaxes="{xy_oven}" fovy="{cfg.CAM_OVEN_FOVY}"/>
  </body>

  <camera name="overview" pos="4.6 -4.6 5.2" xyaxes="0.707 0.707 0 -0.49 0.49 0.72" fovy="50"/>

  <body name="obj0" pos="-0.95 -{cfg.LANE} {cfg.OBJECT_REST_Z[0]}">
    <freejoint name="obj0_free"/>
    <geom name="obj0_geom" type="box" size="{cfg.OBJECT_HALF} {cfg.OBJECT_HALF} {cfg.OBJECT_HALF}"
          mass="0.18" rgba="0.90 0.16 0.12 1"/>
  </body>
  <body name="obj1" pos="{cfg.LANE} -0.70 {cfg.OBJECT_REST_Z[1]}">
    <freejoint name="obj1_free"/>
    <geom name="obj1_geom" type="cylinder" size="{cfg.OBJ1_RADIUS} {cfg.OBJ1_HALF_HEIGHT}"
          mass="0.08" rgba="0.12 0.45 0.95 1"/>
  </body>
  <body name="obj2" pos="0.55 {cfg.LANE} {cfg.OBJECT_REST_Z[2]}">
    <freejoint name="obj2_free"/>
    <geom name="obj2_geom" type="box" size="0.10 0.07 0.07"
          mass="0.15" rgba="0.95 0.76 0.10 1"/>
  </body>
  <body name="obj3" pos="-{cfg.LANE} 0.45 {cfg.OBJECT_REST_Z[3]}">
    <freejoint name="obj3_free"/>
    <geom name="obj3_geom" type="cylinder" size="{cfg.OBJ3_RADIUS} {cfg.OBJ3_HALF_HEIGHT}"
          mass="0.085" rgba="0.18 0.75 0.30 1"/>
  </body>
  <body name="obj4" pos="0.15 -{cfg.LANE} {cfg.OBJECT_REST_Z[4]}">
    <freejoint name="obj4_free"/>
    <geom name="obj4_geom" type="box" size="0.07 0.11 0.065"
          mass="0.14" rgba="0.72 0.28 0.90 1"/>
  </body>
</worldbody>
<actuator>
  <velocity name="belt_act_b1" joint="jr_b1" kv="3" ctrlrange="-9 9"/>
  <velocity name="belt_act_b2" joint="jr_b2" kv="3" ctrlrange="-9 9"/>
  <velocity name="belt_act_r1" joint="jr_r1" kv="3" ctrlrange="-9 9"/>
  <velocity name="belt_act_r2" joint="jr_r2" kv="3" ctrlrange="-9 9"/>
  <velocity name="belt_act_t1" joint="jr_t1" kv="3" ctrlrange="-9 9"/>
  <velocity name="belt_act_t2" joint="jr_t2" kv="3" ctrlrange="-9 9"/>
  <velocity name="belt_act_l1" joint="jr_l1" kv="3" ctrlrange="-9 9"/>
  <velocity name="belt_act_l2" joint="jr_l2" kv="3" ctrlrange="-9 9"/>
  <velocity name="belt_act_o1" joint="jr_o1" kv="3" ctrlrange="-9 9"/>
  <velocity name="belt_act_o2" joint="jr_o2" kv="3" ctrlrange="-9 9"/>
</actuator>
</mujoco>
"""
        return xml