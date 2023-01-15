import math
import builtins
import numpy as np
from collections import defaultdict


class BM_25:

    def __init__(self, index, k1=2.1, b=0.1):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.DL)
        self.AVGDL = builtins.sum(index.DL.values()) / self.N

    def calc_idf(self, list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df:
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, query_to_search, N=1000):
        '''
        compute the BM25 score of a query against all relevant documents in the index

        parameters:
        -----------
        query_to_search: list of tokens
        N: number of top documents to return

        ------
        return: a dictionary of {doc_id: score}

        '''
        res_score1 = defaultdict()
        self.idf = self.calc_idf(query_to_search)
        candidates = []
        dict_candidates = {}
        for term in np.unique(query_to_search):
            if term in self.index.df:
                pls = self.index.read_posting_list(
                    term, self.index.df[term])
                dict_candidates.update({term: dict(pls)})
                candidates += [x[0] for x in pls]
        sort_score = [(k, self._score(query_to_search, k, dict_candidates))
             for k in np.unique(candidates)]
        if len(sort_score)< N:
            result = sorted([(doc_id, score) for doc_id, score in sort_score],
                            key=lambda x: x[1], reverse=True)
        else:    
            result = sorted([(doc_id, score) for doc_id, score in sort_score],
                            key=lambda x: x[1], reverse=True)[:N]
        if len(result) != 0:
            max_body = max(result, key=lambda x: x[1])[1]
            res_score1 = {doc_id: bm25 / max_body for doc_id, bm25 in result}
        return res_score1

    def _score(self, query, doc_id, dict_candidates):
        '''
        compute the BM25 score of a query against a single document
        parameters:
        -----------
        query: list of tokens
        doc_id: document id
        candidate_dict: dictionary of {term: {doc_id: frequency}}
        ------
        return: a float number of the score
        '''
        score = 0.0
        doc_len = self.index.DL[doc_id]
        for term in query:
            if term in self.index.df:
                term_frequencies = candidate_dict[term]
                if doc_id in term_frequencies:
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * \
                        (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score
