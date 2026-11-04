import cv2
import numpy as np
import pytest
from src.camara_inspeccion import CamaraInspeccion
from src.cinta_transportadora import CintaTransportadora
from src.config import CINTAS_CFG
from src.vision import DetectorFormas

W, H = 360, 260
FONDO = (30, 26, 22)   # BGR gris oscuro, poco saturado


def escena(forma, cx=180, color=(30, 40, 230)):
    img = np.full((H, W, 3), FONDO, np.uint8)
    if forma == "circulo":
        cv2.circle(img, (cx, 130), 40, color, -1)
    elif forma == "cuadrado":
        cv2.rectangle(img, (cx - 40, 90), (cx + 40, 170), color, -1)
    elif forma == "rectangulo":
        cv2.rectangle(img, (cx - 55, 100), (cx + 55, 160), color, -1)
    elif forma == "triangulo":
        cv2.fillPoly(img, [np.array([[cx, 80], [cx - 50, 170], [cx + 50, 170]])], color)
    return img


@pytest.mark.parametrize("forma", ["circulo", "cuadrado", "rectangulo", "triangulo"])
def test_clasifica_formas(forma):
    dets = DetectorFormas().detectar(escena(forma))
    assert len(dets) == 1 and dets[0].forma == forma and dets[0].en_zona


def test_cilindro_visto_en_perspectiva_es_circulo():
    img = np.full((H, W, 3), FONDO, np.uint8)
    cv2.ellipse(img, (180, 130), (42, 35), 0, 0, 360, (240, 110, 30), -1)
    cv2.rectangle(img, (138, 130), (222, 150), (240, 110, 30), -1)   # lateral
    cv2.ellipse(img, (180, 150), (42, 35), 0, 0, 180, (240, 110, 30), -1)
    assert DetectorFormas().detectar(img)[0].forma == "circulo"


def test_fondo_gris_no_se_detecta():
    img = np.full((H, W, 3), FONDO, np.uint8)
    cv2.rectangle(img, (0, 0), (W, 40), (92, 82, 72), -1)   # carril gris claro
    assert DetectorFormas().detectar(img) == []


def test_zona_de_inspeccion():
    d = DetectorFormas()
    assert d.detectar(escena("cuadrado", cx=180))[0].en_zona
    assert not d.detectar(escena("cuadrado", cx=70))[0].en_zona


def test_colores():
    d = DetectorFormas()
    assert d.detectar(escena("cuadrado", color=(30, 40, 230)))[0].color == "rojo"
    assert d.detectar(escena("cuadrado", color=(240, 110, 30)))[0].color == "azul"


# ---- cámara + cinta: variable booleana y parada --------------------------------
def camara():
    cinta = CintaTransportadora(**CINTAS_CFG[0])
    return CamaraInspeccion("cam_bottom", cinta, tiempo_inspeccion=2.0), cinta


def test_deteccion_activa_booleana_y_para_cinta():
    cam, cinta = camara()
    cam.procesar(escena("cuadrado", cx=70), 0.0)       # visible pero fuera de zona
    assert cam.pieza_visible and not cam.objeto_detectado and cinta.esta_activa()
    cam.procesar(escena("cuadrado", cx=180), 0.05)     # en zona
    assert cam.objeto_detectado and cam.forma_detectada == "cuadrado"
    assert not cinta.esta_activa()


def test_sin_pieza_booleana_false():
    cam, cinta = camara()
    cam.procesar(np.full((H, W, 3), FONDO, np.uint8), 0.0)
    assert not cam.objeto_detectado and cinta.esta_activa()


def test_reanudacion_controlada():
    cam, cinta = camara()
    img = escena("circulo")
    cam.procesar(img, 0.0)
    assert not cinta.esta_activa()
    cam.procesar(img, 1.0)                              # aún inspeccionando
    assert not cinta.esta_activa()
    cam.procesar(img, 2.1)                              # tiempo cumplido -> reanuda
    assert cinta.esta_activa()
    cam.procesar(img, 2.2)                              # misma pieza: NO vuelve a parar
    assert cinta.esta_activa()
    cam.procesar(np.full((H, W, 3), FONDO, np.uint8), 3.0)   # zona libre -> rearme
    cam.procesar(img, 3.1)                              # nueva pieza -> para
    assert not cinta.esta_activa()


def test_reanudacion_manual():
    cam, cinta = camara()
    cam.reanudacion_automatica = False
    img = escena("cuadrado")
    cam.procesar(img, 0.0)
    cam.procesar(img, 10.0)
    assert not cinta.esta_activa()                      # sigue parada
    cinta.conmutar()                                    # operario rearranca
    cam.procesar(img, 10.1)
    assert cinta.esta_activa()                          # no se vuelve a parar
