# -*- coding: utf-8 -*-
"""
Configuración de la planta industrial.

Este módulo contiene únicamente CONSTANTES de solo lectura (parámetros
geométricos y de control) y tablas de descripción de los elementos de la
planta. No hay estado mutable: el estado de cada elemento (por ejemplo, la
marcha/parada de una cinta) vive dentro de su objeto.
"""

from pathlib import Path

from . import robot

# -----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------------------------------------------
SHOW_MUJOCO_VIEWER = True
SHOW_CAMERA_MOSAIC = True

CAM_W = 360
CAM_H = 260
CAM_FPS = 20.0

BELT_SPEED = 0.55          # m/s, velocidad objetivo de los objetos
BELT_ROLLER_SPEED = 5.5    # rad/s, velocidad de los rodillos visuales

VEL_KP = 7.0               # control de velocidad
CENTER_KP = 12.0           # mantiene el objeto centrado en la cinta
TURN_KP = 16.0             # ayuda a realizar el giro en las esquinas
MAX_FORCE = 12.0           # N por eje

BELT_HALF_LENGTH = 1.45
BELT_HALF_WIDTH = 0.28
BELT_Z = 0.12
BELT_TOP_Z = 0.18
LANE = 1.45                 # coordenada de la línea central de cada cinta
TURN_START = 1.08           # comienza a orientar el objeto hacia la siguiente cinta
CORNER_SWITCH = 1.27        # cambio efectivo de cinta dentro de la placa de esquina
RAIL_END_GAP = 0.06
RAIL_HALF_LENGTH = LANE - BELT_HALF_WIDTH - RAIL_END_GAP
RAIL_THICKNESS = 0.035
RAIL_HALF_HEIGHT = 0.10
RAIL_CENTER_Z = 0.22

# Cámaras de inspección
CAM_HEIGHT = 0.92
CAM_FOVY = 32.0
CAM_LENS_SIDE_OFFSET = 0.10
CAM_POLE_SIDE_OFFSET = BELT_HALF_WIDTH + 0.45

CAM_OVEN_HEIGHT = 0.78
CAM_OVEN_SIDE_OFFSET = 0.46
CAM_OVEN_FOVY = 34.0

CAM_LEFT_INSPECTION_Y = 0.82
CAM_LEFT_HEAD_CLEARANCE = 0.20
CAM_LEFT_POLE_CLEARANCE = 0.52

# Estabilización de las piezas
LOCK_OBJECT_ROTATION = True
LOCK_CYLINDERS_TO_CENTERLINE = True

OBJECT_CLEARANCE_Z = 0.01

# Rama lateral hacia el horno (Cinta 5)
BRANCH_SPEED = 0.48
BRANCH_Y = 0.0
BRANCH_HALF_LENGTH = 1.15
BRANCH_HALF_WIDTH = 0.32
BRANCH_CENTER_X = -(LANE + BRANCH_HALF_LENGTH - 0.05)
BRANCH_OPEN_HALF = 0.38
BRANCH_RAIL_HALF = (RAIL_HALF_LENGTH - BRANCH_OPEN_HALF) / 2.0
BRANCH_RAIL_CENTER_Y = (RAIL_HALF_LENGTH + BRANCH_OPEN_HALF) / 2.0
BRANCH_RAIL_START_X = -2.00
BRANCH_RAIL_END_X = -3.48
BRANCH_RAIL_CENTER_X = 0.5 * (BRANCH_RAIL_START_X + BRANCH_RAIL_END_X)
BRANCH_RAIL_HALF_LENGTH = 0.5 * abs(BRANCH_RAIL_END_X - BRANCH_RAIL_START_X)

PICK_RAIL_OPEN_Y_MIN = -0.10
PICK_RAIL_OPEN_Y_MAX = 0.72
LEFT_RAIL_UPPER_CENTER_Y = 0.5 * (RAIL_HALF_LENGTH + PICK_RAIL_OPEN_Y_MAX)
LEFT_RAIL_UPPER_HALF = 0.5 * (RAIL_HALF_LENGTH - PICK_RAIL_OPEN_Y_MAX)
LEFT_RAIL_LOWER_CENTER_Y = 0.5 * (-RAIL_HALF_LENGTH + PICK_RAIL_OPEN_Y_MIN)
LEFT_RAIL_LOWER_HALF = 0.5 * (PICK_RAIL_OPEN_Y_MIN + RAIL_HALF_LENGTH)

# -----------------------------------------------------------------------------
# ROBOT (la IK proporcionada vive en src/robot.py y NO se modifica)
# -----------------------------------------------------------------------------
ROBOT_CONFIG = robot.make_config(
    lane=LANE,
    branch_y=BRANCH_Y,
    branch_speed=BRANCH_SPEED,
)

# True: modo de prueba IK (el TCP va a una pose 6D). False: pick-and-place.
ROBOT_IK_TEST_MODE = True
ROBOT_IK_TARGET_POSITION = (-1.45, 0.05, 0.62)
ROBOT_IK_TARGET_RPY_DEG = (0.0, 0.0, 20.0)
ROBOT_IK_ORIENTATION_FRAME = "home"   # "home" o "world"
ROBOT_IK_GRIPPER_OPEN = True

ISLAND_X = ROBOT_CONFIG["base_x"]
ISLAND_Y = ROBOT_CONFIG["base_y"]
ISLAND_Z = ROBOT_CONFIG["base_z"]
ISLAND_HALF_X = 0.18
ISLAND_HALF_Y = 0.38

# -----------------------------------------------------------------------------
# HORNO
# -----------------------------------------------------------------------------
OVEN_CENTER_X = -4.05
OVEN_ENTRY_X = -3.58

OVEN_TEMP_INICIAL = 20.0      # °C
OVEN_TEMP_MIN = 20.0          # °C
OVEN_TEMP_MAX = 200.0         # °C
OVEN_TEMP_MEDIA = 100.0       # °C: color naranja
OVEN_INCREMENTO_PIEZA = 30.0  # °C que sube al recibir una pieza
OVEN_ENFRIAMIENTO = 0.5       # °C/s de enfriamiento pasivo (tiempo simulado)
OVEN_PASO_TECLA = 10.0        # °C por pulsación de las teclas + / -

# -----------------------------------------------------------------------------
# VISIÓN
# -----------------------------------------------------------------------------
VISION_SAT_MIN = 120          # saturación HSV mínima (la banda gris es ~85)
VISION_VAL_MIN = 50
VISION_AREA_MIN = 600         # px^2
VISION_ZONA = (0.38, 0.62)    # franja central de inspección (fracción del ancho)
TIEMPO_INSPECCION = 2.0       # s simulados que la cinta permanece parada

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "capturas"

# -----------------------------------------------------------------------------
# DESCRIPCIÓN DE LAS CINTAS (índice = segmento 0..4)
# direccion: sentido de avance; direccion_siguiente: hacia dónde gira en la
# esquina; centro: punto de la línea central de la banda.
# -----------------------------------------------------------------------------
CINTAS_CFG = [
    dict(identificador=1, nombre="CINTA 1 · INFERIOR · +X", velocidad=BELT_SPEED,
         direccion=(1.0, 0.0), direccion_siguiente=(0.0, 1.0), indice_siguiente=1,
         centro=(0.0, -LANE), actuadores=("belt_act_b1", "belt_act_b2"),
         signo_rodillos=+1.0),
    dict(identificador=2, nombre="CINTA 2 · DERECHA · +Y", velocidad=BELT_SPEED,
         direccion=(0.0, 1.0), direccion_siguiente=(-1.0, 0.0), indice_siguiente=2,
         centro=(LANE, 0.0), actuadores=("belt_act_r1", "belt_act_r2"),
         signo_rodillos=-1.0),
    dict(identificador=3, nombre="CINTA 3 · SUPERIOR · -X", velocidad=BELT_SPEED,
         direccion=(-1.0, 0.0), direccion_siguiente=(0.0, -1.0), indice_siguiente=3,
         centro=(0.0, LANE), actuadores=("belt_act_t1", "belt_act_t2"),
         signo_rodillos=-1.0),
    dict(identificador=4, nombre="CINTA 4 · IZQUIERDA · -Y", velocidad=BELT_SPEED,
         direccion=(0.0, -1.0), direccion_siguiente=(1.0, 0.0), indice_siguiente=0,
         centro=(-LANE, 0.0), actuadores=("belt_act_l1", "belt_act_l2"),
         signo_rodillos=+1.0),
    dict(identificador=5, nombre="CINTA 5 · HORNO · -X", velocidad=BRANCH_SPEED,
         direccion=(-1.0, 0.0), direccion_siguiente=None, indice_siguiente=None,
         centro=(0.0, BRANCH_Y), actuadores=("belt_act_o1", "belt_act_o2"),
         signo_rodillos=-1.0),
]

# Cámara MuJoCo asociada a cada cinta (mismo orden que CINTAS_CFG).
CAMARAS_CFG = ["cam_bottom", "cam_right", "cam_top", "cam_left", "cam_oven"]
CAMARA_GENERAL = "overview"

# -----------------------------------------------------------------------------
# DESCRIPCIÓN DE LOS OBJETOS
# -----------------------------------------------------------------------------
OBJETOS_CFG = [
    dict(identificador=0, tipo="caja", medias=(0.085, 0.085, 0.085), masa=0.18,
         rgba=(0.90, 0.16, 0.12, 1.0), inicio=(-LANE, 0.95), segmento=3),
    dict(identificador=1, tipo="cilindro", radio=0.075, semialtura=0.045, masa=0.08,
         rgba=(0.12, 0.45, 0.95, 1.0), inicio=(LANE, -0.72), segmento=1),
    dict(identificador=2, tipo="caja", medias=(0.10, 0.07, 0.07), masa=0.15,
         rgba=(0.95, 0.76, 0.10, 1.0), inicio=(0.58, LANE), segmento=2),
    dict(identificador=3, tipo="cilindro", radio=0.085, semialtura=0.0375, masa=0.085,
         rgba=(0.18, 0.75, 0.30, 1.0), inicio=(-0.62, -LANE), segmento=0),
    dict(identificador=4, tipo="caja", medias=(0.07, 0.11, 0.065), masa=0.14,
         rgba=(0.72, 0.28, 0.90, 1.0), inicio=(-0.45, LANE), segmento=2),
]
