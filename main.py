#!/usr/bin/env python3
"""Punto de entrada de la planta industrial refactorizada con POO."""

from src.planta import PlantaIndustrial


def main():
    planta = PlantaIndustrial()
    planta.inicializar()
    planta.ejecutar()


if __name__ == "__main__":
    main()