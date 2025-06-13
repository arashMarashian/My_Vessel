import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine_loader import load_engines_from_yaml


def main():
    engines = load_engines_from_yaml("data/engine_data.yaml")
    load_values = np.linspace(0, 100, 100)  # 100 points from 0% to 100% load

    for engine in engines:
        print(f"Plotting SFOC curve for {engine.name}")
        plt.figure(figsize=(8, 5))
        for fuel_type, curve in engine.sfoc_curve.items():
            sfoc_values = curve(load_values)
            plt.plot(load_values, sfoc_values, label=fuel_type)

        plt.title(f"SFOC Curve - {engine.name}")
        plt.xlabel("Load (%)")
        plt.ylabel("SFOC (g/kWh)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
