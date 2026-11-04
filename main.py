#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Punto de entrada de la planta industrial (Práctica I, POO)."""

import argparse

from src.planta import PlantaIndustrial


def main():
    ap = argparse.ArgumentParser(description="Planta industrial MuJoCo (POO)")
    ap.add_argument("--sin-visor", action="store_true", help="no abrir el visor 3D de MuJoCo")
    ap.add_argument("--sin-camaras", action="store_true",
                    help="no abrir el mosaico OpenCV (desactiva también la visión)")
    ap.add_argument("--robot", choices=["ik", "auto"], default="ik",
                    help="ik: prueba de IK (por defecto); auto: pick-and-place")
    args = ap.parse_args()

    planta = PlantaIndustrial(mostrar_visor=not args.sin_visor,
                              mostrar_camaras=not args.sin_camaras,
                              robot_ik_manual=(args.robot == "ik"))
    planta.ejecutar()


if __name__ == "__main__":
    main()
