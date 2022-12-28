import pandas as pd
import nltk
import pickle
import numpy as np
import math
import os
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

pkl_file = "part15_preprocessed.pkl"

with open(pkl_file, 'rb') as f:
    pages = pickle.load(f)

with open("dl/dl.pkl", 'rb') as f:
    DL = pickle.load(f)
    AVGDL = sum(DL.values()) / len(DL)


def tf_idf_scores(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=tfidf_vectorizer.get_feature_names())
    return tfidf_df, tfidf_vectorizer


def cosine_sim_using_sklearn(queries, tfidf):
    cosine_sim_df = pd.DataFrame(cosine_similarity(queries, tfidf))
    return cosine_sim_df


def bm25_preprocess(data):
    dl = []
    tf = []
    df = {}
# df counts only once per document
    for doc in data:
        dl.append(len(doc))
        tf.append({term: doc.count(term)/len(doc) for term in doc})
        for term in set(doc):
            if term in df:
                df[term] += 1
            else:
                df[term] = 1

    return dl, tf, df


class BM25:
    def __init__(self, doc_len, df, tf=None, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.tf_ = tf
        self.doc_len_ = doc_len
        self.df_ = df
        self.N_ = len(doc_len)
        self.avgdl_ = sum(doc_len) / len(doc_len)

    def calc_idf(self, query):
        idf = {}
        for term in query:
            if term not in self.df_:
                idf[term] = 0
            else:
                idf[term] = math.log(
                    1 + (self.N_ - self.df_[term] + 0.5) / (self.df_[term] + 0.5))

        return idf

    def search(self, queries):
        scores = []
        for query in queries:
            scores.append([self._score(query, doc_id)
                          for doc_id in range(self.N_)])
        return scores

    def _score(self, query, doc_id):

        idf = self.calc_idf(query)
        score = 0
        for term in query:
            if term in self.df_ and term in self.tf_[doc_id]:
                score += idf[term] * self.tf_[doc_id][term] * (self.k1 + 1) / (
                    self.tf_[doc_id][term] + self.k1 * (1 - self.b + self.b * self.doc_len_[doc_id] / self.avgdl_))
        return score

    def top_N_documents(df, N):
        top_N = {}
        for query_id in df.index:
            top_N[query_id] = sorted([(doc_id, df[doc_id][query_id])
                                     for doc_id in df.columns], key=lambda x: x[1], reverse=True)[:N]
        return top_N


def generate_query_tfidf_vector(query_to_search, index):

    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        # avoid terms that do not appear in the index.
        if token in index.term_total.keys():
            # term frequency divded by the length of the query
            tf = counter[token]/len(query_to_search)
            df = index.df[token]
            idf = math.log((len(DL))/(df+epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf*idf
            except:
                pass
    return Q


def get_posting_iter(index):
    words, pls = zip(*index.posting_lists_iter())
    return words, pls


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [(doc_id, (freq/DL[str(doc_id)])*math.log(
                len(DL)/index.df[term], 10)) for doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get(
                    (doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    total_vocab_size = len(index.term_total)
    # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    candidates_scores = get_candidate_documents_and_scores(
        query_to_search, index, words, pls)
    unique_candidates = np.unique(
        [doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


def cosine_similarity(D, Q):
    cosine_similarity_scores = {}
    for doc_id in D.index:
        cosine_similarity_scores[doc_id] = np.dot(
            D.loc[doc_id], Q)/(np.linalg.norm(D.loc[doc_id])*np.linalg.norm(Q))
    return cosine_similarity_scores


def get_topN_score_for_queries(queries_to_search, index, N=3):
    def get_top_n(sim_dict, N=3):
        return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]
    words, pls = get_posting_iter(index)
    topN_scores = {}
    for query_id in queries_to_search:
        query_to_search = queries_to_search[query_id]
        D = generate_document_tfidf_matrix(query_to_search, index, words, pls)
        Q = generate_query_tfidf_vector(query_to_search, index)
        sim_dict = cosine_similarity(D, Q)
        topN_scores[query_id] = get_top_n(sim_dict, N)
    return topN_scores


def get_candidate(query_to_search, index, words, pls):
    candidates = []
    for term in np.unique(query_to_search):
        if term in words:
            candidates += (pls[words.index(term)])
    return np.unique(candidates)


def merge_results_pr(scores1, scores2, pr, w1=0.33, w2=0.33, w3=0.33, N=3):

    # Union all docs and add weights, then sort
    merged = [(k, (scores1.get(k, 0) * w1) + (scores2.get(k, 0) * w2) +
               (pr.get(k, 0) * w3)) for k in set(scores1) | set(scores2)]
    return sorted(merged, key=lambda x: x[1], reverse=True)[:N]
