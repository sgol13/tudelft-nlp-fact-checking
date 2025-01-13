from typing import List, Callable, Tuple

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

from src.common import QT_VERACITY_LABELS, QTDataset, QTClaim

qt_veracity_label_encoder = LabelEncoder()
qt_veracity_label_encoder.fit(QT_VERACITY_LABELS)


# don't let anyone tamper with the label encoder
def _disable_method(*args, **kwargs):
    raise RuntimeError("Please, don't try to fit the label encoder again.")


qt_veracity_label_encoder.fit = _disable_method
qt_veracity_label_encoder.fit_transform = _disable_method


def _get_feature_from_doc(claim: QTClaim) -> str:
    return (
            "[Claim]:" + claim["claim"] +
            "[Evidences]:" + claim["doc"]
    )


def _get_feature_no_decomposition(claim: QTClaim) -> str:
    evidences = claim['evidences']
    return (
        "[Claim]:" + claim["claim"] +
        ("[Evidences]:" + " ".join(evidences) if evidences else '')
    )


def _get_feature_claim_decomposition(claim: QTClaim) -> str:
    evidences = claim['evidences']
    return (
        "[Claim]:" + claim["claim"] +
        "[Questions]:" + " ".join(claim['subquestions']) +
        ("[Evidences]:" + " ".join(evidences) if evidences else '')
    )


class QuantempProcessor:
    _MODES = {
        'decomposition': _get_feature_claim_decomposition,
        'no_decomposition': _get_feature_no_decomposition,
        'doc': _get_feature_from_doc
    }

    def __init__(self,
                 tokenizer: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
                 evidence_mode: str
                 ):
        self._tokenizer = tokenizer
        self._evidence_mode = evidence_mode
        assert self._evidence_mode in self._MODES, f"Invalid evidence, must be one of {self._MODES}"

    def transform(self, dataset: QTDataset) -> TensorDataset:
        features: List[str] = self._extract_features(dataset)
        labels: List[str] = self._extract_labels(dataset)

        assert set(labels) == set(QT_VERACITY_LABELS), f"Labels mismatch: {labels} != {QT_VERACITY_LABELS}"

        input_tokens, attention_masks = self._encode_features(features)
        encoded_labels = self._encode_labels(labels)

        assert input_tokens.shape[0] == attention_masks.shape[0] \
               == encoded_labels.shape[0] == len(dataset), "Shapes mismatch"

        return TensorDataset(input_tokens, attention_masks, encoded_labels)

    @staticmethod
    def _extract_labels(dataset: QTDataset) -> List[str]:
        return [claim['label'] for claim in dataset]

    def _extract_features(self, dataset: QTDataset) -> List[str]:
        get_feature = self._MODES[self._evidence_mode]
        return [get_feature(claim) for claim in dataset]

    @staticmethod
    def _encode_labels(labels: List[str]) -> torch.Tensor:
        return torch.tensor(qt_veracity_label_encoder.transform(labels))

    def _encode_features(self, features: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tokens = []
        attention_masks = []

        for feature in tqdm(features):
            input_token, attention_mask = self._tokenizer(feature)
            input_tokens.append(input_token)
            attention_masks.append(attention_mask)

        return torch.cat(input_tokens), torch.cat(attention_masks)
