import math
from itertools import chain
import time
import numpy as np
from backend import *
from inverted_index import *


# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self ,index ,DL,k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.N = len(DL)
        self.AVGDL = sum(DL.values() ) /self.N


    def calc_idf(self ,list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """


        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf


    def search(self, query ,N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

         Parameters:
        -----------
        queries: list of lists. Each inner list is a list of tokens. For example:
                                                                                    [
                                                                                        ['look', 'blue', 'sky'],
                                                                                        ['likes', 'blue', 'sun'],
                                                                                        ['likes', 'diamonds']
                                                                                    ]

        Returns:
        -----------
        list of scores of bm25
        """


        for term in query:
            self.idf = self.calc_idf(query)
            candidates = []
            for term in np.unique(query):
                if term in self.index.df.keys():
                    candidates += (self.read_posting_list(self.index, term))
            a = [(doc_id, self._score(query, doc_id)) for doc_id in np.unique(candidates)]
            n_result = sorted(a, reverse=True, key=lambda x: x[1])[:N]

        return n_result


    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.DL[str(doc_id)]

        for term in query:
            if term in self.index.df.keys():
                term_frequencies = dict(self.read_posting_list(self.index, term))
                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score


    def read_posting_list(inverted, w):
        TUPLE_SIZE = 6
        with closing(MultiFileReader()) as reader:
            locs = inverted.posting_locs[w]
            b = reader.read(locs, inverted.term_total[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.term_total[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list