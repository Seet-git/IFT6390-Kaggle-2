import importlib
import os
import sys
import config


def load_hyperparams():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../{config.INPUT_HP_PATH}/"))
    if base_path not in sys.path:
        sys.path.append(base_path)

    # Charger le module spécifié
    return importlib.import_module(config.INPUT_HP_FILENAME)
