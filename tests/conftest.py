"""Permite ejecutar las pruebas unitarias aunque MuJoCo no esté instalado.

Las clases probadas (cinta, horno, visión, cámara) solo necesitan MuJoCo para
resolver ids; aquí se sustituye por un módulo vacío si no está disponible.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import mujoco  # noqa: F401
except ImportError:
    stub = types.ModuleType("mujoco")
    stub.mjtObj = types.SimpleNamespace(mjOBJ_ACTUATOR=0, mjOBJ_BODY=1, mjOBJ_JOINT=2,
                                        mjOBJ_GEOM=3, mjOBJ_SITE=4, mjOBJ_CAMERA=5)
    stub.viewer = types.ModuleType("mujoco.viewer")
    stub.mj_name2id = lambda *a, **k: 0
    sys.modules["mujoco"] = stub
    sys.modules["mujoco.viewer"] = stub.viewer
