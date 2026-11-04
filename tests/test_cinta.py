import numpy as np
from src.cinta_transportadora import CintaTransportadora
from src.config import CINTAS_CFG, LANE


def crear(i):
    return CintaTransportadora(**CINTAS_CFG[i])


def test_estado_independiente():
    cintas = [crear(i) for i in range(5)]
    cintas[2].detener()
    assert [c.esta_activa() for c in cintas] == [True, True, False, True, True]
    cintas[2].conmutar()
    assert cintas[2].esta_activa()


def test_parada_frena_la_pieza():
    c = crear(0)
    f_on = c.fuerza_transporte((0, -LANE, 0.2), (0, 0, 0))
    assert f_on[0] > 0                       # empuja en +X
    c.detener()
    f_off = c.fuerza_transporte((0, -LANE, 0.2), (0.5, 0, 0))
    assert f_off[0] < 0                      # frena


def test_transferencia_entre_cintas():
    c = crear(0)
    assert not c.debe_transferir((1.0, -LANE))
    assert c.debe_transferir((1.3, -LANE)) and c.indice_siguiente == 1
    assert crear(4).indice_siguiente is None  # cinta del horno no transfiere


def test_motivo_parada():
    c = crear(1)
    c.detener("camara:cam_right")
    assert c.detenida_por_camara()
    c.arrancar()
    assert not c.detenida_por_camara()
