import numpy as np
from src.horno import Horno


class ModeloFalso:
    def __init__(self):
        self.geom_rgba = np.zeros((4, 4))


def crear():
    return Horno(-4.05, -3.58, geom_id=2)


def test_temperatura_y_limites():
    h = crear()
    h.aumentar_temperatura(50)
    assert h.obtener_temperatura() == 70
    h.aumentar_temperatura(1000)
    assert h.obtener_temperatura() == h.temperatura_maxima
    h.disminuir_temperatura(1000)
    assert h.obtener_temperatura() == h.temperatura_minima


def test_registro_de_piezas():
    h = crear()
    t0 = h.temperatura
    assert h.pieza_entra(-3.60) and not h.pieza_entra(-3.0)
    h.registrar_pieza(0)
    assert h.numero_piezas == 1 and h.pieza_entrante and h.temperatura > t0


def test_color_cambia_en_mujoco():
    m, h = ModeloFalso(), crear()
    h.actualizar_color(m)
    frio = m.geom_rgba[2].copy()
    h.establecer_temperatura(100); h.actualizar_color(m)
    medio = m.geom_rgba[2].copy()
    h.establecer_temperatura(200); h.actualizar_color(m)
    caliente = m.geom_rgba[2].copy()
    assert frio[2] > frio[0]                  # azul
    assert medio[0] > 0.9 and 0.4 < medio[1] < 0.8   # naranja
    assert caliente[0] > 0.9 and caliente[1] < 0.2   # rojo


def test_enfriamiento_pasivo():
    m, h = ModeloFalso(), crear()
    h.establecer_temperatura(100)
    h.actualizar(m, 10.0)
    assert h.temperatura < 100
