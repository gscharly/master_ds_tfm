from sentence_transformers import SentenceTransformer
from wmd import WMD

import numpy as np

from typing import Tuple, List, Dict
import math
import logging


class SMS:
    """
    Class used to computed Sentence Movers Similarity between texts
    """
    def __init__(self, sent_emb: str, vocabulary_min: int = 1, k: int = 1, early_stop: float = 1):
        self.sent_emb_model = SentenceTransformer(sent_emb)
        self.vocabulary_min = vocabulary_min
        self.k = k
        self.early_stop = early_stop

    def encode_texts(self, candidate: str, reference: str) -> Tuple[np.array, np.array]:
        # Convert to lists
        processed_candidate_sents = candidate.split('.')
        processed_reference_sents = reference.split('.')
        candidate_embeddings = self.sent_emb_model.encode(processed_candidate_sents)
        reference_embeddings = self.sent_emb_model.encode(processed_reference_sents)
        return candidate_embeddings, reference_embeddings

    @staticmethod
    def _get_weights(candidate_emb: np.array, reference_emb: np.array) -> Tuple[List[int], List[int]]:
        return [len(x) for x in candidate_emb], [len(x) for x in reference_emb]

    @staticmethod
    def _build_reference_lists(candidate_emb: np.array, reference_emb: np.array) ->\
            Tuple[Tuple[List[int], List[int]], Dict[int, np.array]]:
        """
        Returns:
        - Tuple of lists, one for each texts, indicating the ids for each sentence
        - Dictionary, where each key references a sentence id, and each value represents the sentence embedding
        :param candidate_emb:
        :param reference_emb:
        :return:
        """
        rep_map = dict()
        ref_ids = list()
        cand_ids = list()
        for i in range(len(reference_emb)):
            rep_map[i] = reference_emb[i]
            ref_ids.append(i)

        j = len(rep_map)

        for i in range(len(candidate_emb)):
            rep_map[i + j] = candidate_emb[i]
            cand_ids.append(i + j)

        return (ref_ids, cand_ids), rep_map

    @staticmethod
    def _build_doc_dict(ids: Tuple[List[int], List[int]], candidate_w: List[int], reference_w: List[int]) ->\
            Dict[str, Tuple[str, List[int], List[int]]]:
        return {
                "0": ("ref", ids[0], reference_w),
                "1": ("hyp", ids[1], candidate_w)
        }

    def _perform_wmd(self, rep_map, doc_dict):
        calc = WMD(rep_map, doc_dict, vocabulary_min=self.vocabulary_min, vocabulary_optimizer=None,
                   verbosity=logging.ERROR)
        dist = calc.nearest_neighbors("0", k=self.k, early_stop=self.early_stop)[0][1]
        sim = math.exp(-dist)
        return sim

    def calculate_sms(self, candidate: str, reference: str):
        """
        Computes SMS over two texts
        :param candidate:
        :param reference:
        :return:
        """
        candidate_emb, reference_emb = self.encode_texts(candidate, reference)
        candidate_weights, reference_weights = self._get_weights(candidate_emb, reference_emb)
        ids, rep_map = self._build_reference_lists(candidate_emb, reference_emb)
        doc_dict = self._build_doc_dict(ids, candidate_weights, reference_weights)
        sms = self._perform_wmd(rep_map, doc_dict)
        return sms
