from typing import Dict, Any, Union, List

import torch
import json
import os


def _build_import_relative_path(path: str):
    this_dir_abs_path = os.path.dirname(os.path.abspath(__file__))
    path_relative_to_this_dir = os.path.join(this_dir_abs_path, path)
    return os.path.abspath(path_relative_to_this_dir)


DATA_PATH = _build_import_relative_path("../data")
MODELS_PATH = _build_import_relative_path("../models")

QTClaim = Dict[str, Union[
    str,
    List[str], # decomposed questions / evidences for claim (no decomposition)
    List[Dict[str, Union[str, List[str]]]] # questions with evidences
]]
QTDataset = List[QTClaim]

QT_VERACITY_LABELS = ['Conflicting', 'False', 'True']


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


def read_json(file_path) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(file_path: str, data: Dict[str, Any]) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def cwd_relative_path(abs_path: str) -> str:
    cwd = os.getcwd()
    return os.path.relpath(abs_path, cwd)
