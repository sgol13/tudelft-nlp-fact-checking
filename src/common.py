from typing import Dict, Any, Union, List

import torch
import json
import os
import warnings
import pandas as pd

from sklearn.exceptions import UndefinedMetricWarning

from src.models.gpt2 import GPT2_CONFIG
from src.models.roberta_large_mnli import ROBERTA_LARGE_MNLI_CONFIG

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def _build_import_relative_path(path: str):
    this_dir_abs_path = os.path.dirname(os.path.abspath(__file__))
    path_relative_to_this_dir = os.path.join(this_dir_abs_path, path)
    return os.path.abspath(path_relative_to_this_dir)


DATA_PATH = _build_import_relative_path("../data")
MODELS_PATH = _build_import_relative_path("../models")
OUTPUT_PATH = _build_import_relative_path("../output")

QTClaim = Dict[str, Union[
    str,
    List[str], # subquestions / evidences / top100evidences
]]
QTDataset = List[QTClaim]

QT_VERACITY_LABELS = ['Conflicting', 'False', 'True']

DOC = 'doc'
NO_DECOMPOSITION = 'no_decomposition'
GPT3_5_TURBO = 'gpt3.5-turbo'
FLANT5 = 'flant5'
CUSTOM_DECOMP = 'custom'

DECOMPOSITION_METHODS = [
    DOC,
    NO_DECOMPOSITION,
    GPT3_5_TURBO,
    FLANT5,
    CUSTOM_DECOMP
]

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


def read_data(path) -> Dict[str, Any]:
    abs_path = os.path.join(DATA_PATH, path)
    with open(abs_path, "r") as f:
        return json.load(f)


def save_data(path, data: Dict[str, Any]) -> None:
    abs_path = os.path.join(DATA_PATH, path)
    with open(abs_path, "w") as f:
        json.dump(data, f, indent=4)


def cwd_relative_path(abs_path) -> str:
    cwd = os.getcwd()
    return os.path.relpath(abs_path, cwd)