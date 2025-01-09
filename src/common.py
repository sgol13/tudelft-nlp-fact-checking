from typing import List, Dict

import numpy as np
import torch
import json
import datetime
import os

def _build_import_relative_path(path: str):
    this_dir_abs_path = os.path.dirname(os.path.abspath(__file__))
    path_relative_to_this_dir = os.path.join(this_dir_abs_path, path)
    return os.path.abspath(path_relative_to_this_dir)


DATA_PATH = _build_import_relative_path("../data")
MODELS_PATH = _build_import_relative_path("../models")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA: {torch.cuda.device_count()}, use {torch.cuda.get_device_name(0)}")

    elif torch.mps.is_available():
        device = torch.device("mps")
        print(f"MPS: {torch.mps.device_count()}")

    else:
        device = torch.device("cpu")
        print("CPU")

    return device


def read_json(file_path: str) -> Dict[str, str]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def accuracy(output: np.ndarray, labels: np.ndarray) -> float:
    pred_flat = np.argmax(output, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed: float) -> str:
    """"
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_model(model: torch.nn.Module, path: str):
    abs_path = os.path.join(MODELS_PATH, path)
    torch.save(model.state_dict(), abs_path)

def load_model(model: torch.nn.Module, path: str):
    abs_path = os.path.join(MODELS_PATH, path)
    model.load_state_dict(torch.load(abs_path))
    return model