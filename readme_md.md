# Planta Industrial Robotizada (POO)

Simulación interactiva de una planta industrial automatizada desarrollada en **Python** utilizando **MuJoCo** para la simulación física 3D, **OpenCV** para la adquisición y procesamiento de visión artificial, y **NumPy** para operaciones matriciales y cinemática.

Este proyecto corresponde a la **Práctica I** de la asignatura *Programación Modular y Orientada a Objetos* (PMOO) del Grado en Ingeniería Informática de la **Universidad de Burgos (UBU)**, Curso 2026-2027.

---

## 📌 Descripción General

El proyecto refactoriza una implementación procedural original transformándola en una arquitectura limpia y modular basada en **Programación Orientada a Objetos (POO)**. 

La planta consta de:
- **Circuito cerrado de transporte:** 4 cintas transportadoras dispuestas en formato cuadrado (Cintas 1 a 4) más una cinta de desvío hacia el proceso térmico (Cinta 5).
- **Sistema de visión artificial:** Cámaras de inspección dedicadas sobre cada cinta que capturan frames y detectan piezas mediante contornos y geometría con OpenCV.
- **Estación de tratamiento térmico (Horno):** Registro de piezas procesadas y control de temperatura en tiempo real, reflejado visualmente mediante cambios de color en el modelo MuJoCo.
- **Estación robotizada:** Manipulador industrial **Franka Emika Panda** integrado con un solucionador de cinemática inversa (IK) cartesiana para la manipulación de objetos.

---

## 🛠️ Estructura del Proyecto

El código está organizado manteniendo la separación entre el punto de entrada y los módulos de lógica interna:

```text
planta_industrial/
├── main.py                     # Punto de entrada principal a la aplicación
├── requirements.txt            # Dependencias de Python necesarias
├── README.md                   # Documentación del repositorio
└── src/                        # Código fuente modularizado
    ├── __init__.py
    ├── config.py               # Constantes numéricas e hiperparámetros del sistema
    ├── cinta_transportadora.py # Clase CintaTransportadora (Cintas 1 a 5)
    ├── camara_inspeccion.py    # Clase CamaraInspeccion y visión con OpenCV
    ├── objeto_industrial.py    # Clase ObjetoIndustrial (geometría y tracking)
    ├── horno.py                # Clase Horno (temperatura y cambio de color RGBA)
    ├── planta.py               # Clase PlantaIndustrial (Coordinador global)
    └── robot.py                # Control e integración del robot Franka Panda
```

---

## ⚙️ Requisitos e Instalación

### Prerrequisitos
- Python 3.9 o superior.
- Entorno de ejecución compatible con OpenGL / renderizado para MuJoCo.

### Instalación de dependencias

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/usuario/planta_industrial_ubu.git
   cd planta_industrial_ubu
   ```

2. **Crear y activar un entorno virtual (recomendado):**
   ```bash
   # En Windows:
   python -m venv venv
   .\venv\Scripts\activate

   # En Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Ejecución

Para iniciar la simulación, ejecuta desde la raíz del proyecto:

```bash
python main.py
```

Se abrirán dos ventanas principales:
1. **Visor 3D de MuJoCo:** Simulación física interactiva de la planta y el robot Franka Panda.
2. **Cámaras de la planta (OpenCV):** Mosaico interactivo en tiempo real con las vistas de inspección de cada cinta, etiquetado de formas detectadas y panel de telemetría del sistema.

---

## 🎮 Controles por Teclado

Durante la ejecución, se pueden enviar comandos directos interactuando con la ventana de OpenCV:

| Tecla | Acción / Función |
| :---: | :--- |
| `1` .. `5` | Conmutar marcha / paro de la cinta correspondiente ($C_1$ a $C_5$). |
| `P` | Pausar / Reanudar la simulación global. |
| `R` | Reiniciar la simulación (posiciones de objetos, estado del robot y del horno). |
| `+` | Aumentar manualmente la temperatura del horno (+10 °C). |
| `-` | Disminuir manualmente la temperatura del horno (-10 °C). |
| `S` | Guardar una captura de pantalla del mosaico actual en la carpeta `capturas/`. |
| `Q` / `Esc` | Salir de la aplicación y cerrar ventanas ordenadamente. |

---

## 📐 Diseño Orientado a Objetos

El diseño sigue los principios **SOLID** básicos y la encapsulación estricta de estado:
- **`CintaTransportadora`**: Controla su propio estado de marcha/paro y calcula los vectores de fuerza aplicados a las piezas según el tramo.
- **`CamaraInspeccion`**: Captura frames del renderizador, procesa contornos con OpenCV y actualiza el booleano `objeto_detectado`.
- **`Horno`**: Mantiene la temperatura interna y actualiza dinámicamente las propiedades de color `geom_rgba` de la geometría `oven_interior` en el modelo MuJoCo.
- **`PlantaIndustrial`**: Clase coordinadora que gestiona el bucle a 500 Hz (`timestep = 0.002s`), el teclado y la sincronización general del sistema.

---

## 🎓 Autoría y Asignatura

- **Asignatura:** Programación Modular y Orientada a Objetos (PMOO)
- **Titulacion:** Grado en Ingeniería Informática - Universidad de Burgos (UBU)
- **Curso Académico:** 2026-2027