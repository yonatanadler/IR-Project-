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
    InvertedIndex_body = None
    InvertedIndex_title = None
    InvertedIndex_anchor = None
    # dictionnary of {Key : docID, Value : title}
    title_dict_dic_id = None
    # dictionnary of {Key : docID, Value : (tfidf vector size, doc. len)}
    doc_len_dict = None
    # dictionnary of PageView -- {Key : docID, Value : Page view score}
    page_view_dict = None
    # dictionnary of PageRank -- {Key : docID, Value : Page rank score}
    page_rank_dict = None
    # dictionnary of docID and title -- {Key : docID, Value : Title}
    id_title_dict = None
    bucket_name = "316048628"
    bm25_body = None
    bm25_title = None
    # preprocess of the query
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

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
                    self.doc_len_dict = pickle.load(f)
            elif blob.name == 'title_dict_dic_id.pkl':
                with blob.open("rb") as f:
                    self.title_dict_dic_id = pickle.load(f)



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

    def cosine_sim_using_sklearn(queries,tfidf):
        """
        Parameters:
        -----------
          queries: sparse matrix represent the queries after transformation of tfidfvectorizer.
          documents: sparse matrix represent the documents.

        Returns:
        --------
          DataFrame: This function will return a DataFrame in the following shape: (# of queries, # of documents).
          Each value in the DataFrame will represent the cosine_similarity between given query and document.

        """

        return cosine_similarity(queries, tfidf)

    def tf_idf_scores(data):
        """
        This function calculates the tfidf for each word in a single document utilizing TfidfVectorizer via sklearn.

        Parameters:
        -----------
          data: list of strings.

        Returns:
        --------
          Two objects as follows:
                                    a) DataFrame, documents as rows (i.e., 0,1,2,3, etc'), terms as columns ('bird','bright', etc').
                                    b) TfidfVectorizer object.

        """
        # YOUR CODE HERE
        # YOUR CODE HERE
        df = self.df_
        lendoc = self.N_
        dic = {}

        for term in query:
            if term in df:
                cal_idf = np.log((lendoc - df[term] + 0.5) / (df[term] + 0.5) + 1)
            else:
                cal_idf = 0
            dic[term] = cal_idf
        return dic

    def cosine_similarity(D, Q):
        """
            Parameters:
            -----------
            D: DataFrame of tfidf scores.
            Q: vectorized query with tfidf scores
            Returns:
            -----------
            dictionary of cosine similarity score as follows:
             key: document id (e.g., doc_id)
             value: cosine similarty score.
            """
        dic = {}
        for row, doc in D.iterrows():
            dic[row] = np.dot(doc, Q) / (np.sqrt(np.dot(doc, doc)) * np.sqrt(np.dot(Q, Q)))
        return dic

        # cosine_sim_df = pd.DataFrame(cosine_similarity(queries, tfidf))
        # return cosine_sim_df
    def read_posting_list(inverted, w):
        TUPLE_SIZE = 6
        with closing(backend.MultiFileReader()) as reader:
            locs = inverted.posting_locs[w]
            b = reader.read(locs, inverted.term_total[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.term_total[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list


    def search_title(query_lst, index):
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








    def get_top_n(sim_dict, N):
        """
        Sort and return the highest N documents according to the cosine similarity score.
        Generate a dictionary of cosine similarity scores

        Parameters:
        -----------
        sim_dict: a dictionary of similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

        N: Integer (how many documents to retrieve). By default N = 3

        Returns:
        -----------
        a ranked list of pairs (doc_id, score) in the length of N.
        """

        lst =  sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                      reverse=True)[:N]
        return ([doc_id for doc_id, score in lst]), lst

    def BM_25_score(self ,query_to_search):
        w1, w2 = 0.4, 0.6
        bm25_title = self.BM_25(self.InvertedIndex_title,self.doc_len_dict )
        bm25_body = self.BM_25(self.InvertedIndex_body,self.doc_len_dict )
        bm25_queries_score_train_title = bm25_title.search(query_to_search)
        bm25_queries_score_train_body = bm25_body.search(query_to_search)
        score_body_title = self.merge_results(bm25_queries_score_train_title, bm25_queries_score_train_body, w1, w2)
        return score_body_title




    def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
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
    def merge_results_pr(scores1, scores2, pr, w1=0.33, w2=0.33, w3=0.33, N=3):

        # Union all docs and add weights, then sort
        merged = [(k, (scores1.get(k, 0) * w1) + (scores2.get(k, 0) * w2) +
                   (pr.get(k, 0) * w3)) for k in set(scores1) | set(scores2)]
        return sorted(merged, key=lambda x: x[1], reverse=True)[:N]

    def tf_idf_scores(data):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(data)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                                columns=tfidf_vectorizer.get_feature_names())
        return tfidf_df, tfidf_vectorizer


    def get_index_body(self):
        return self.index_body

    def get_index_title(self):
        return self.index_title

    def get_index_anchor(self):
        return self.index_anchor

    def get_title_dict(self):
        return self.title_dict_dic_id
