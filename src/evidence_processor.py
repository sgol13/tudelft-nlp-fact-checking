from typing import List, Dict, Union

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from tqdm.auto import tqdm
from src.common import QTDataset


class EvidenceProcessor:
    _SIMILARITY_THRESHOLD = 0.5

    def __init__(self, decomposed: bool, top_k: int):
        self._decomposed = decomposed
        self._top_k = top_k
        self._embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def transform(self, dataset: QTDataset) -> QTDataset:
        for claim in tqdm(dataset):
            evidences = claim['top100evidences']
            if self._decomposed:
                assert 'questions' in claim, "No questions found but decomposed=True"
                questions = claim['questions']
                claim['evidences'] = self._find_similar_to_decomposed_questions(questions, evidences)
            else:
                claim['evidences'] = self._find_similar_to_claim(claim['claim'], evidences)

        return dataset

    def _find_similar_to_claim(self, claim: str, evidences: List[str]) -> List[str]:
        claim_emb = self._embedding_model.encode(claim)
        return self._find_similar_evidences(claim, evidences, claim_emb)

    def _find_similar_to_decomposed_questions(self, questions: List[str], evidences: List[str]) -> List[
        Dict[str, Union[str, List[str]]]]:
        questions_with_evidences = []
        for question in questions:
            evidence_embeddings = self._embedding_model.encode(evidences)
            top_evidences = self._find_similar_evidences(question, evidences, evidence_embeddings)

            questions_with_evidences.append({
                "questions": question,
                "top_k_doc": top_evidences
            })

        return questions_with_evidences

    def _find_similar_evidences(self, sentence: str, evidences: List[str], evidence_embeddings: torch.Tensor) -> List[
        str]:
        sentence_emb = self._embedding_model.encode(sentence)

        evidence_similarities = cosine_similarity(evidence_embeddings, [sentence_emb]).tolist()

        numbered_similarities = zip(range(len(evidence_similarities)), evidence_similarities)
        sorted_similarities = sorted(numbered_similarities, key=lambda x: x[1], reverse=True)

        top_evidences = []
        for idx, item in sorted_similarities[:self._top_k]:
            if item[0] > self._SIMILARITY_THRESHOLD:
                top_evidences.append(evidences[idx])

        return top_evidences
