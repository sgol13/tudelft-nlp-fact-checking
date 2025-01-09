from typing import List, Dict, Callable, Tuple

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
from tqdm import tqdm

QTFact = Dict[str, str]
QTDataset = List[Dict[str, str]]


class QuantempProcessor:
    LABELS = ['Conflicting', 'False', 'True']

    def __init__(self,
                 get_feature: Callable[[QTFact], str],
                 encode: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
                 ):
        self.label_encoder_ = None
        self.get_feature = get_feature
        self.encode = encode

    def fit(self, dataset: QTDataset) -> None:
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(self.LABELS)

        data_labels = self._extract_labels(dataset)
        assert set(data_labels) == set(self.LABELS), f"Labels mismatch: {data_labels} != {self.LABELS}"

    def transform(self, dataset: QTDataset) -> TensorDataset:
        features: List[str] = self._extract_features(dataset)
        labels: List[str] = self._extract_labels(dataset)

        input_tokens, attention_masks = self._encode_features(features)
        encoded_labels = self._encode_labels(labels)

        assert input_tokens.shape[0] == attention_masks.shape[0] == encoded_labels.shape[0] == len(dataset), "Shapes mismatch"

        return TensorDataset(input_tokens, attention_masks, encoded_labels)

    def fit_transform(self, dataset: QTDataset) -> TensorDataset:
        self.fit(dataset)
        return self.transform(dataset)

    @staticmethod
    def _extract_labels(dataset: QTDataset) -> List[str]:
        return [data['label'] for data in dataset]

    def _extract_features(self, dataset: QTDataset) -> List[str]:
        return [self.get_feature(data) for data in dataset]

    def _encode_labels(self, labels: List[str]) -> torch.Tensor:
        return torch.tensor(self.label_encoder_.transform(labels))

    def _encode_features(self, features: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tokens = []
        attention_masks = []

        for feature in tqdm(features):
            input_token, attention_mask = self.encode(feature)
            input_tokens.append(input_token)
            attention_masks.append(attention_mask)

        return torch.cat(input_tokens), torch.cat(attention_masks)