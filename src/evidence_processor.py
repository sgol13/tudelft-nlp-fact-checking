from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from tqdm.auto import tqdm
from src.common import QTDataset, QTClaim


class EvidenceProcessor:
    _THRESHOLD = 0.3

    def __init__(self, decomposed: bool):
        self._decomposed = decomposed
        self._embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def transform(self, dataset: QTDataset) -> QTDataset:
        for claim in tqdm(dataset):
            sentences = claim['subquestions'] if self._decomposed else [claim['claim']]
            claim['evidences'] = self._find_similar_evidences(sentences, claim['top100evidences'])

            del claim['top100evidences']
            if 'doc' in claim:
                del claim['doc']

        return dataset

    def _find_similar_evidences(self, sentences: List[str], all_evidences: List[str]) -> List[str]:
        if not sentences:
            return []

        doc_embs = self._embedding_model.encode(all_evidences)
        sent_embs = self._embedding_model.encode(sentences)

        text_sims = cosine_similarity(doc_embs, sent_embs) # shape (100 x #subquestions) for the similarity between each of 100 sources and each subquestion

        # Get top3 evidences for each subquestion and their scores
        potential_evidences = text_sims.argsort(axis=0)[-3:, :]
        potential_evidences_scored = np.sort(text_sims, axis=0)[-3:, :]

        # Get final 0-5 evidences between all subquestions that are not the same but have a similarity score of higher than 0.5
        final_evidences = list(set(potential_evidences[-1, :][potential_evidences_scored[-1, :] > self._THRESHOLD].tolist()))
        if len(final_evidences) < 3:
            final_evidences += list(set(potential_evidences[-2, :][potential_evidences_scored[-2, :] > self._THRESHOLD].tolist()))
            if len(final_evidences) < 3:
                final_evidences += list(set(potential_evidences[-3, :][potential_evidences_scored[-3, :] > self._THRESHOLD].tolist()))

        # get the actual contents of the final evidences
        selected_evidences = [all_evidences[i] for i in final_evidences]
        return selected_evidences