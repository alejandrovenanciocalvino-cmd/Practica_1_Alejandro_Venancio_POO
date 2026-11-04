# -*- coding: utf-8 -*-
"""Generación del modelo MJCF de la planta (cintas, cámaras, isla, esquinas).

El horno y los objetos generan su propio XML mediante sus métodos a_mjcf().
"""

import numpy as np

from .config import (
    BELT_HALF_LENGTH,
    BELT_HALF_WIDTH,
    BELT_TOP_Z,
    BELT_Z,
    BRANCH_CENTER_X,
    BRANCH_HALF_LENGTH,
    BRANCH_HALF_WIDTH,
    BRANCH_RAIL_CENTER_X,
    BRANCH_RAIL_CENTER_Y,
    BRANCH_RAIL_HALF,
    BRANCH_RAIL_HALF_LENGTH,
    BRANCH_Y,
    CAM_FOVY,
    CAM_HEIGHT,
    CAM_LEFT_HEAD_CLEARANCE,
    CAM_LEFT_INSPECTION_Y,
    CAM_LEFT_POLE_CLEARANCE,
    CAM_LENS_SIDE_OFFSET,
    CAM_OVEN_FOVY,
    CAM_OVEN_HEIGHT,
    CAM_OVEN_SIDE_OFFSET,
    CAM_POLE_SIDE_OFFSET,
    ISLAND_HALF_X,
    ISLAND_HALF_Y,
    ISLAND_X,
    ISLAND_Y,
    ISLAND_Z,
    LANE,
    LEFT_RAIL_LOWER_CENTER_Y,
    LEFT_RAIL_LOWER_HALF,
    LEFT_RAIL_UPPER_CENTER_Y,
    LEFT_RAIL_UPPER_HALF,
    RAIL_CENTER_Z,
    RAIL_HALF_HEIGHT,
    RAIL_HALF_LENGTH,
    RAIL_THICKNESS,
)

def camera_xyaxes(cam_pos, target):
    """Devuelve los ejes X/Y de una cámara MuJoCo que mira hacia `target`."""
    cam = np.asarray(cam_pos, dtype=float)
    tgt = np.asarray(target, dtype=float)

    # La cámara de MuJoCo mira a lo largo de su eje local -Z.
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


def construir_mjcf(objetos, horno):
    """Construye el MJCF de la planta. Horno y objetos aportan su propio XML."""

    # Cámaras de inspección: el soporte/poste está a un lado de cada cinta,
    # mientras que el objetivo óptico queda cerca del borde de la banda. Así
    # la imagen contiene esencialmente banda + pieza y resulta adecuada para
    # reconocimiento posterior de color y forma.
    cam_bottom = (0.0, -(LANE + CAM_LENS_SIDE_OFFSET), CAM_HEIGHT)
    cam_right  = (LANE + CAM_LENS_SIDE_OFFSET, 0.0, CAM_HEIGHT)
    cam_top    = (0.0, LANE + CAM_LENS_SIDE_OFFSET, CAM_HEIGHT)

    # La cámara 4 NO queda encima de la banda. Su x está más allá del borde
    # exterior de la Cinta 4 y su y se aleja de la intersección con Cinta 5.
    cam_left = (
        -(LANE + BELT_HALF_WIDTH + CAM_LEFT_HEAD_CLEARANCE),
        CAM_LEFT_INSPECTION_Y,
        CAM_HEIGHT,
    )
    cam_oven   = (BRANCH_CENTER_X, BRANCH_Y - CAM_OVEN_SIDE_OFFSET, CAM_OVEN_HEIGHT)

    # Se apunta al centro de una zona local de inspección de la banda. El
    # pequeño offset hace que la cámara siga estando físicamente a un lado,
    # pero la dirección de visión sea casi cenital.
    target_bottom = (0.0, -LANE, BELT_TOP_Z + 0.015)
    target_right  = (LANE, 0.0, BELT_TOP_Z + 0.015)
    target_top    = (0.0, LANE, BELT_TOP_Z + 0.015)
    target_left   = (-LANE, CAM_LEFT_INSPECTION_Y, BELT_TOP_Z + 0.015)
    target_oven   = (BRANCH_CENTER_X, BRANCH_Y, BELT_TOP_Z + 0.015)

    pole_bottom_y = -(LANE + CAM_POLE_SIDE_OFFSET)
    pole_right_x  =  (LANE + CAM_POLE_SIDE_OFFSET)
    pole_top_y    =  (LANE + CAM_POLE_SIDE_OFFSET)
    pole_left_x   = -(LANE + BELT_HALF_WIDTH + CAM_LEFT_POLE_CLEARANCE)
    pole_left_y   = CAM_LEFT_INSPECTION_Y
    pole_oven_y   = BRANCH_Y - (BRANCH_HALF_WIDTH + 0.52)

    xy_bottom = camera_xyaxes(cam_bottom, target_bottom)
    xy_right  = camera_xyaxes(cam_right, target_right)
    xy_top    = camera_xyaxes(cam_top, target_top)
    xy_left   = camera_xyaxes(cam_left, target_left)
    xy_oven   = camera_xyaxes(cam_oven, target_oven)

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
        <material name="oven_hot_mat" rgba="0.95 0.22 0.05 1" emission="0.8"/>
    </asset>

    <worldbody>
        <light pos="0 0 6" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
        <light pos="3 -3 4" dir="-0.5 0.5 -1" diffuse="0.45 0.45 0.45"/>
        <geom name="floor" type="plane" size="6 6 0.1" material="floor_mat"/>

        <!-- CINTA 1: INFERIOR, flujo +X -->
        <geom name="belt_bottom" type="box" pos="0 -{LANE} {BELT_Z}"
              size="{BELT_HALF_LENGTH} {BELT_HALF_WIDTH} 0.06" material="belt_mat"/>
        <geom type="box" pos="0 -{LANE-BELT_HALF_WIDTH-RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="0 -{LANE+BELT_HALF_WIDTH+RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <!-- CINTA 2: DERECHA, flujo +Y -->
        <geom name="belt_right" type="box" pos="{LANE} 0 {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_LENGTH} 0.06" material="belt_mat"/>
        <geom type="box" pos="{LANE-BELT_HALF_WIDTH-RAIL_THICKNESS} 0 {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {RAIL_HALF_LENGTH} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="{LANE+BELT_HALF_WIDTH+RAIL_THICKNESS} 0 {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {RAIL_HALF_LENGTH} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <!-- CINTA 3: SUPERIOR, flujo -X -->
        <geom name="belt_top" type="box" pos="0 {LANE} {BELT_Z}"
              size="{BELT_HALF_LENGTH} {BELT_HALF_WIDTH} 0.06" material="belt_mat"/>
        <geom type="box" pos="0 {LANE-BELT_HALF_WIDTH-RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="0 {LANE+BELT_HALF_WIDTH+RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <!-- CINTA 4: IZQUIERDA, flujo -Y -->
        <geom name="belt_left" type="box" pos="-{LANE} 0 {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_LENGTH} 0.06" material="belt_mat"/>

        <!-- Barandilla interior de Cinta 4: abierta delante de la isla robotizada. -->
        <geom type="box" pos="-{LANE-BELT_HALF_WIDTH-RAIL_THICKNESS} {LEFT_RAIL_UPPER_CENTER_Y} {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {LEFT_RAIL_UPPER_HALF} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="-{LANE-BELT_HALF_WIDTH-RAIL_THICKNESS} {LEFT_RAIL_LOWER_CENTER_Y} {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {LEFT_RAIL_LOWER_HALF} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <!-- La barandilla exterior queda abierta en y=0: esa abertura ya no
             hace el desvío; sirve solo como paso hacia la Cinta 5. -->
        <geom type="box" pos="-{LANE+BELT_HALF_WIDTH+RAIL_THICKNESS} {BRANCH_RAIL_CENTER_Y} {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {BRANCH_RAIL_HALF} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="-{LANE+BELT_HALF_WIDTH+RAIL_THICKNESS} -{BRANCH_RAIL_CENTER_Y} {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {BRANCH_RAIL_HALF} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <!-- CINTA 5: RAMA LATERAL HACIA EL HORNO, flujo -X. -->
        <geom name="belt_oven" type="box" pos="{BRANCH_CENTER_X} {BRANCH_Y} {BELT_Z}"
              size="{BRANCH_HALF_LENGTH} {BRANCH_HALF_WIDTH} 0.06" material="belt_mat"/>
        <geom name="branch_transfer" type="box" pos="-{LANE} {BRANCH_Y} {BELT_Z}"
              size="{BELT_HALF_WIDTH+0.06} {BRANCH_HALF_WIDTH} 0.06" material="transfer_mat"/>

        <!-- Las guías de Cinta 5 empiezan después de la zona de descarga del
             robot. Así el cubo no cae contra una pared al ser depositado. -->
        <geom type="box" pos="{BRANCH_RAIL_CENTER_X} {BRANCH_Y-BRANCH_HALF_WIDTH-RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{BRANCH_RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="{BRANCH_RAIL_CENTER_X} {BRANCH_Y+BRANCH_HALF_WIDTH+RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{BRANCH_RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <!-- ISLA ROBOTIZADA. La plataforma es baja y queda alineada con la
             intersección Cinta 4 / Cinta 5. El Panda se adjunta después con MjSpec. -->
        <body name="process_island" pos="{ISLAND_X} {ISLAND_Y} 0">
            <geom name="island_base" type="box" pos="0 0 {ISLAND_Z/2}"
                  size="{ISLAND_HALF_X} {ISLAND_HALF_Y} {ISLAND_Z/2}"
                  material="island_mat" contype="0" conaffinity="0"/>
            <geom name="island_panda_plate" type="cylinder" pos="0 0 {ISLAND_Z+0.025}"
                  size="0.16 0.025" material="belt_edge_mat" contype="0" conaffinity="0"/>
        </body>

        <!-- HORNO INDUSTRIAL AL FINAL DE LA RAMA (generado por Horno) -->
        {horno.a_mjcf()}

        <!-- TRANSFERENCIAS EN LAS CUATRO ESQUINAS -->
        <geom name="corner_br" type="box" pos=" {LANE} -{LANE} {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
        <geom name="corner_tr" type="box" pos=" {LANE}  {LANE} {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
        <geom name="corner_tl" type="box" pos="-{LANE}  {LANE} {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
        <geom name="corner_bl" type="box" pos="-{LANE} -{LANE} {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>

        <!-- RODILLOS VISUALES: 2 POR CINTA -->
        <body name="roller_b1" pos="-1.15 -{LANE} 0.20">
            <joint name="jr_b1" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_b2" pos="1.15 -{LANE} 0.20">
            <joint name="jr_b2" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>

        <body name="roller_r1" pos="{LANE} -1.15 0.20">
            <joint name="jr_r1" type="hinge" axis="1 0 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_r2" pos="{LANE} 1.15 0.20">
            <joint name="jr_r2" type="hinge" axis="1 0 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>

        <body name="roller_t1" pos="1.15 {LANE} 0.20">
            <joint name="jr_t1" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_t2" pos="-1.15 {LANE} 0.20">
            <joint name="jr_t2" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>

        <body name="roller_l1" pos="-{LANE} 1.15 0.20">
            <joint name="jr_l1" type="hinge" axis="1 0 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_l2" pos="-{LANE} -1.15 0.20">
            <joint name="jr_l2" type="hinge" axis="1 0 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>

        <!-- Rodillos de la cinta hacia el horno. -->
        <body name="roller_o1" pos="-2.18 {BRANCH_Y} 0.20">
            <joint name="jr_o1" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_o2" pos="-3.25 {BRANCH_Y} 0.20">
            <joint name="jr_o2" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>

        <!-- CÁMARAS DE INSPECCIÓN: SOPORTE LATERAL + VISIÓN SOBRE LA BANDA -->
        <!-- El poste permanece fuera de la cinta; un brazo corto sitúa el cabezal
             óptico junto al borde. El FOV estrecho muestra casi exclusivamente
             la superficie de la banda para visión artificial. -->
        <body name="cam_station_bottom" pos="0 0 0">
            <geom type="cylinder" pos="0 {pole_bottom_y} {CAM_HEIGHT/2}"
                  size="0.025 {CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="0 {(pole_bottom_y + cam_bottom[1])/2} {CAM_HEIGHT}"
                  size="0.025 {abs(cam_bottom[1]-pole_bottom_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{cam_bottom[0]} {cam_bottom[1]} {cam_bottom[2]}"
                  size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
            <camera name="cam_bottom" pos="{cam_bottom[0]} {cam_bottom[1]} {cam_bottom[2]}"
                    xyaxes="{xy_bottom}" fovy="{CAM_FOVY}"/>
        </body>

        <body name="cam_station_right" pos="0 0 0">
            <geom type="cylinder" pos="{pole_right_x} 0 {CAM_HEIGHT/2}"
                  size="0.025 {CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{(pole_right_x + cam_right[0])/2} 0 {CAM_HEIGHT}"
                  size="{abs(cam_right[0]-pole_right_x)/2} 0.025 0.022" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{cam_right[0]} {cam_right[1]} {cam_right[2]}"
                  size="0.055 0.075 0.055" material="camera_mat" contype="0" conaffinity="0"/>
            <camera name="cam_right" pos="{cam_right[0]} {cam_right[1]} {cam_right[2]}"
                    xyaxes="{xy_right}" fovy="{CAM_FOVY}"/>
        </body>

        <body name="cam_station_top" pos="0 0 0">
            <geom type="cylinder" pos="0 {pole_top_y} {CAM_HEIGHT/2}"
                  size="0.025 {CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="0 {(pole_top_y + cam_top[1])/2} {CAM_HEIGHT}"
                  size="0.025 {abs(cam_top[1]-pole_top_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{cam_top[0]} {cam_top[1]} {cam_top[2]}"
                  size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
            <camera name="cam_top" pos="{cam_top[0]} {cam_top[1]} {cam_top[2]}"
                    xyaxes="{xy_top}" fovy="{CAM_FOVY}"/>
        </body>

        <body name="cam_station_left" pos="0 0 0">
            <geom type="cylinder" pos="{pole_left_x} {pole_left_y} {CAM_HEIGHT/2}"
                  size="0.025 {CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{(pole_left_x + cam_left[0])/2} {pole_left_y} {CAM_HEIGHT}"
                  size="{abs(cam_left[0]-pole_left_x)/2} 0.025 0.022" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{cam_left[0]} {cam_left[1]} {cam_left[2]}"
                  size="0.055 0.075 0.055" material="camera_mat" contype="0" conaffinity="0"/>
            <camera name="cam_left" pos="{cam_left[0]} {cam_left[1]} {cam_left[2]}"
                    xyaxes="{xy_left}" fovy="{CAM_FOVY}"/>
        </body>

        <!-- Cámara de inspección de la cinta 5 hacia el horno. -->
        <body name="cam_station_oven" pos="0 0 0">
            <geom type="cylinder" pos="{BRANCH_CENTER_X} {pole_oven_y} {CAM_OVEN_HEIGHT/2}"
                  size="0.025 {CAM_OVEN_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{BRANCH_CENTER_X} {(pole_oven_y + cam_oven[1])/2} {CAM_OVEN_HEIGHT}"
                  size="0.025 {abs(cam_oven[1]-pole_oven_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{cam_oven[0]} {cam_oven[1]} {cam_oven[2]}"
                  size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
            <camera name="cam_oven" pos="{cam_oven[0]} {cam_oven[1]} {cam_oven[2]}"
                    xyaxes="{xy_oven}" fovy="{CAM_OVEN_FOVY}"/>
        </body>

        <!-- CÁMARA GENERAL DE LA PLANTA -->
        <camera name="overview" pos="4.6 -4.6 5.2" xyaxes="0.707 0.707 0 -0.49 0.49 0.72" fovy="50"/>

        <!-- OBJETOS (generados por cada ObjetoIndustrial) -->
        {"".join(o.a_mjcf() for o in objetos)}
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
