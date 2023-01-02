import pandas as pd
import nltk
import pickle
import numpy as np
import math
import re
import os
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


class Backend:
    index_body = {}
    index_title = {}
    index_anchor = {}

    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    corpus_stopwords = ['category', 'references',
                        'also', 'links', 'extenal', 'see', 'thumb']
    all_stop_words = stop_words.union(corpus_stopwords)
    TUPLE_SIZE = 6

    def __init__(self):
        with open('index_body.pkl', 'rb') as f:
            self.index_body = pickle.load(f)
        with open('index_title.pkl', 'rb') as f:
            self.index_title = pickle.load(f)
        with open('index_anchor.pkl', 'rb') as f:
            self.index_anchor = pickle.load(f)

    def query_preprocess(self, query, index):
        tokens = [token.group()
                  for token in self.RE_WORD.finditer(query.lower())]
        tokens = [self.stemmer.stem(t)
                  for t in tokens if t not in self.all_stop_words]
        ngrams_tokens = []
        try:
            for ngram in list(ngrams(tokens, 2)):
                ngrams_tokens.append(ngram[0] + " " + ngram[1])
        except:
            pass
        tokens = tokens + ngrams_tokens
        tokens = [t for t in tokens if t in index]
        return tokens

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
