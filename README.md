# Planta industrial MuJoCo — Práctica I (POO)

Refactorización orientada a objetos de la planta procedural (`planta_4_cintas.py`):
4 cintas en circuito + cinta al horno, 5 cámaras de inspección con OpenCV,
horno con temperatura y color, y un Franka Emika Panda con IK proporcionada.

## Instalación y ejecución

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                 # prueba IK (como el original)
python main.py --robot auto    # pick-and-place: las piezas llegan al horno
python main.py --sin-visor     # solo mosaico OpenCV
```

El modelo del Panda se busca en `MUJOCO_MENAGERIE_PATH`, `./vendor/mujoco_menagerie`,
`./mujoco_menagerie`, `../mujoco_menagerie` o `~/mujoco_menagerie`; si no está se
descarga con git (sparse) en `vendor/`.

## Controles (ventana de cámaras)

| Tecla | Acción |
|---|---|
| `Q` / `ESC` | salir |
| `P` | pausa |
| `1`…`5` | marcha/parada de cada cinta |
| `R` | reiniciar piezas, robot, horno y cámaras |
| `S` | guardar captura del mosaico (`capturas/`) |
| `+` / `-` | subir / bajar 10 °C la temperatura del horno |
| `M` | alternar robot: prueba IK ↔ pick-and-place |
| `T` / `H` | (modo IK) pose objetivo / pose HOME |
| `A` | alternar reanudación automática de las cintas tras la inspección |

## Arquitectura

```
main.py                      punto de entrada
src/
  planta.py                  PlantaIndustrial   (coordinador, bucle, teclado)
  cinta_transportadora.py    CintaTransportadora (estado marcha/parada, fuerza)
  camara_inspeccion.py       Camara, CamaraInspeccion (captura + visión + control)
  vision.py                  DetectorFormas, Deteccion (OpenCV)
  objeto_industrial.py       ObjetoIndustrial -> Caja, Cilindro
  horno.py                   Horno (piezas, temperatura, color en MuJoCo)
  estacion_robotizada.py     EstacionRobotizada (envuelve la IK)
  robot.py                   IK proporcionada (caja negra, sin modificar el algoritmo)
  visualizacion.py           VisorCamaras (mosaico OpenCV)
  escena_mjcf.py             generación del MJCF
  config.py                  constantes de solo lectura y tablas de descripción
tests/                       pruebas unitarias (pytest)
informe/informe.tex          borrador del informe
```

Flujo de detección: `CamaraInspeccion.actualizar()` → captura → HSV + umbral →
contornos → forma (`circulo`/`cuadrado`/`rectangulo`/`triangulo`) →
`objeto_detectado` → `cinta.detener()`. La reanudación es controlada
(`tiempo_inspeccion`, `reanudacion_automatica`, rearme al despejarse la zona).

## Pruebas

```bash
pip install pytest && pytest -q
```
