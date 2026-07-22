import sys
import os
import copy
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.trainers import train_cycle_model
from src.trainers.train_cycle_model import train



import yaml
from scripts.run_phase4_loss_sweep import VARIANTS, load_config
base_config = load_config("configs/cycle_model.yaml")

run_config = copy.deepcopy(base_config)
run_config["training"]["seed"] = 7
run_config["training"]["loss"] = VARIANTS["huber_rank"]
run_config["training"]["epochs"] = 1

temp_config_path = "temp_train_config.yaml"
with open(temp_config_path, "w") as f:
    yaml.dump(run_config, f)

try:
    train(config_path=temp_config_path)
except Exception as e:
    print(f"FAILED WITH:")
    traceback.print_exc()
