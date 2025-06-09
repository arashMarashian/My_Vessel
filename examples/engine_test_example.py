import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine_loader import load_engines_from_yaml


def main():
    engines = load_engines_from_yaml("data/engine_data.yaml")
    load_sequence = [0, 50, 75, 85, 0]
    fuel_type = "HFO"

    for engine in engines:
        print(f"\n=== Simulating {engine.name} ===")
        for i, load in enumerate(load_sequence):
            try:
                fuel = engine.step(load, fuel_type)
                print(f"Step {i}: Load = {load}%, Fuel Used = {fuel:.2f} g")
            except ValueError as e:
                print(f"Step {i}: ERROR - {e}")


if __name__ == "__main__":
    main()
