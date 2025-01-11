from typing import List, Dict, Callable, Tuple, Union

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

QTFact = Dict[str, Union[str, List]]
QTDataset = List[QTFact]

QT_VERACITY_LABELS = ['Conflicting', 'False', 'True']

qt_veracity_label_encoder = LabelEncoder()
qt_veracity_label_encoder.fit(QT_VERACITY_LABELS)

# don't let anyone tamper with the label encoder
def _disable_method(*args, **kwargs):
    raise RuntimeError("Please, don't try to fit the label encoder again.")
qt_veracity_label_encoder.fit = _disable_method
qt_veracity_label_encoder.fit_transform = _disable_method

class QuantempProcessor:
    def __init__(self,
                 tokenizer: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
                 claim_decomposition=False,
                 ):
        self._tokenizer = tokenizer
        self._claim_decomposition = claim_decomposition

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
        return [fact['label'] for fact in dataset]

    def _extract_features(self, dataset: QTDataset) -> List[str]:
        get_feature = self._get_feature_from_claim_decomposition if self._claim_decomposition else self._get_feature_from_doc
        return [get_feature(fact) for fact in dataset]

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

    @staticmethod
    def _get_feature_from_doc(fact: QTFact) -> str:
        claim = fact["claim"]
        feature = "[Claim]:" + claim + "[Evidences]:" + fact["doc"]
        return feature

    @staticmethod
    def _get_feature_from_claim_decomposition(fact: QTFact) -> str:
        claim = fact["claim"]

        evidences = []
        questions = []
        for question in fact["evidences"]:
            if len(question["top_k_doc"]) > 0:
                evidences.append(question["top_k_doc"][0])
            questions.append(question["questions"])

        questions = list(set(questions))
        evidences = list(set(evidences))
        feature = "[Claim]:" + claim + "[Questions]:" + " ".join(questions) + "[Evidences]:" + " ".join(evidences)
        return feature
