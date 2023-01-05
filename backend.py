import math
import re
import numpy as np
from collections import Counter, OrderedDict, defaultdict
from contextlib import closing
# from google.cloud import storage
import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
import itertools

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
from BM_25 import *

class Backend:
    size_vec_len_dict = 0
    index_body = None
    index_title = None
    index_anchor = None
    # dictionnary of {Key : docID, Value : ( doc. len)}
    DL = None
    # dictionnary of PageView -- {Key : docID, Value : Page view score}
    page_view_dict = None
    # dictionnary of PageRank -- {Key : docID, Value : Page rank score}
    page_rank_dict = None
    # dictionnary of docID and title -- {Key : docID, Value : Title}
    title_dict_doc_id = None
    bucket_name = "316048628"
    bm25_body = None
    bm25_title = None
    # preprocess of the query
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    all_stopwords = english_stopwords.union(corpus_stopwords)

    def __init__(self):
        client = storage.Client()
        blobs = client.list_blobs(self.bucket_name)

        # connect to the bucket and read the relevant files.


        for blob in blobs:
            if blob.name == 'postings_body/index.pkl':
                with blob.open("rb") as f:
                    self.InvertedIndex_body = pickle.load(f)
            elif blob.name == 'postings_gcp_anchor/index.pkl':
                with blob.open("rb") as f:
                    self.InvertedIndex_anchor = pickle.load(f)
            elif blob.name == 'postings_gcp_title/index.pkl':
                with blob.open("rb") as f:
                    self.InvertedIndex_title = pickle.load(f)
            elif blob.name == 'page_view_dict.pkl':
                with blob.open("rb") as f:
                    self.page_view_dict = pickle.load(f)
            elif blob.name == 'pagerank_dict.pkl':
                with blob.open("rb") as f:
                    self.page_rank_dict = pickle.load(f)
            elif blob.name == 'dl_dict.pkl':
                with blob.open("rb") as f:
                    self.DL = pickle.load(f)
            elif blob.name == 'title_dict_doc_id.pkl':
                with blob.open("rb") as f:
                    self.title_dict_doc_id = pickle.load(f)

    def get_title_dict(self):
        return self.title_dict_doc_id

    def get_dl(self):
        return self.DL


    def preprocess_query(self, query_as_string, index):
        tokens = [token.group() for token in self.RE_WORD.finditer(query_as_string.lower())]
        tokens_after_filter = [term for term in tokens if term in index.df]

        return tokens_after_filter

    def cosine_sim_using_sklearn(self, D,Q):
        dic = {}
        for row, doc in D.iterrows():
            dic[row] = np.dot(doc, Q) / (np.sqrt(np.dot(doc, doc)) * np.sqrt(np.dot(Q, Q)))

        # return cosine_similarity(queries, tfidf)
        return dic


    def generate_query_tfidf_vector(self, query_to_search, index):
        epsilon = .0000001
        total_vocab_size = len(index.term_total)
        Q = np.zeros((total_vocab_size))
        term_vector = list(index.term_total.keys())
        counter = Counter(query_to_search)
        for token in np.unique(query_to_search):
            if token in index.term_total.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
                df = index.df[token]
                idf = math.log((len(self.get_dl())) / (df + epsilon), 10)  # smoothing

                try:
                    ind = term_vector.index(token)
                    Q[ind] = tf * idf
                except:
                    pass
        return Q

    def generate_document_tfidf_matrix(self, query_to_search, index):
        total_vocab_size = len(index.term_total)
        candidates_scores = get_candidate_documents_and_scores(query_to_search, index)
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        D = np.zeros((len(unique_candidates), total_vocab_size))
        D = pd.DataFrame(D)

        D.index = unique_candidates
        D.columns = index.df.keys()

        for key in candidates_scores:
            tfidf = candidates_scores[key]
            doc_id, term = key
            D.loc[doc_id][term] = tfidf

        return D

    def read_posting_list(self, inverted, w):
        TUPLE_SIZE = 6
        with closing(MultiFileReader()) as reader:
            locs = inverted.posting_locs[w]
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list



    def get_candidate_documents_and_scores(self, query_to_search, index):
        candidates = {}
        for term in np.unique(query_to_search):
            try:
                if term in index.df.keys():
                    list_of_doc = self.read_posting_list(index, term)
                    normlized_tfidf = [(doc_id, (freq / self.get_dl()[doc_id]) * math.log(len(self.get_dl()) / index.df[term], 10)) for
                                       doc_id, freq in list_of_doc]

                    for doc_id, tfidf in normlized_tfidf:
                        candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf
            except:
                continue

        return candidates



    def cosine_similarity(D, Q):

        dic = {}
        for row, doc in D.iterrows():
            dic[row] = np.dot(doc, Q) / (np.sqrt(np.dot(doc, doc)) * np.sqrt(np.dot(Q, Q)))
        return dic

        # cosine_sim_df = pd.DataFrame(cosine_similarity(queries, tfidf))
        # return cosine_sim_df

    def search_title(self,query_lst, index):
        counter_doc = {}
        for term in np.unique(query_lst):
            if term not in index.df.keys():
                continue
            list_of_doc = self.read_posting_list(index, term)

            for dic_id, count in list_of_doc:
                if dic_id in counter_doc:
                    counter_doc[dic_id] += 1
                else:
                    counter_doc[dic_id] = 1
        list_of_dict = sorted([(doc_id, score) for doc_id, score in counter_doc.items()], key=lambda x: x[1],
                              reverse=True)
        result = []
        for doc_id, score in list_of_dict:
            result.append((doc_id, self.get_title_dict()[doc_id]))

        return  result



    def get_top_n(self, sim_dict, N):
        lst =  sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                      reverse=True)[:N]
        return ([doc_id for doc_id, score in lst]), lst

    def BM_25_score(self, query_to_search):
        w1, w2 = 0.4, 0.6
        bm25_title = self.BM_25(self.InvertedIndex_title,self.doc_len_dict )
        bm25_body = self.BM_25(self.InvertedIndex_body,self.doc_len_dict )
        bm25_queries_score_train_title = bm25_title.search(query_to_search)
        bm25_queries_score_train_body = bm25_body.search(query_to_search)
        score_body_title = self.merge_results(bm25_queries_score_train_title, bm25_queries_score_train_body, w1, w2)
        return score_body_title


    def merge_results(self, title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
        title_body_weight = {}

        for key, value in title_scores.items():
            if key not in title_body_weight:
                title_body_weight[key] = {}
            for score in value:
                if score[0] in title_body_weight[key]:
                    title_body_weight[key][score[0]] += score[1] * title_weight
                else:
                    title_body_weight[key][score[0]] = score[1] * title_weight

        for key, value in body_scores.items():
            if key not in title_body_weight:
                title_body_weight[key] = {}
            for score in value:

                if score[0] in title_body_weight[key]:
                    title_body_weight[key][score[0]] += score[1] * text_weight
                else:
                    title_body_weight[key][score[0]] = score[1] * text_weight

        Q_sort = {}
        for Q, dic in title_body_weight.items():
            Q_index = []
            for key, value in dic.items():
                Q_index.append((key, value))
            Q_sort[Q] = sorted(Q_index, key=lambda x: x[1], reverse=True)[:N]

        return Q_sort
    def merge_results_pr(self, scores1, scores2, pr, w1=0.33, w2=0.33, w3=0.33, N=3):

        # Union all docs and add weights, then sort
        merged = [(k, (scores1.get(k, 0) * w1) + (scores2.get(k, 0) * w2) +
                   (pr.get(k, 0) * w3)) for k in set(scores1) | set(scores2)]
        return sorted(merged, key=lambda x: x[1], reverse=True)[:N]



    def get_index_body(self):
        return self.index_body

    def get_index_title(self):
        return self.index_title

    def get_index_anchor(self):
        return self.index_anchor

