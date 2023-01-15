import re
import os
import nltk
import json
import math
import time
import pickle
import struct
import builtins
import itertools
import numpy as np
import pandas as pd
from collections import Counter
from google.cloud import storage
from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter, OrderedDict, defaultdict

import hashlib


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


nltk.download('stopwords')

#########################################################################################
client3 = storage.Client()
bucket3 = client3.get_bucket('pkl_207044777')

id_title = pickle.loads(bucket3.get_blob('id2title.pkl').download_as_string())

id2tf = pickle.loads(bucket3.get_blob('id2tf.pkl').download_as_string())

pr = pickle.loads(bucket3.get_blob('page_rank.pickle').download_as_string())

pv = pickle.loads(bucket3.get_blob('pageviews.pkl').download_as_string())

index_title = pickle.loads(bucket3.get_blob(
    'title_index.pkl').download_as_string())
index_title.bucket_name = "title_207044777"

index_title_stem = pickle.loads(bucket3.get_blob(
    'title_stem_index.pkl').download_as_string())
index_title_stem.bucket_name = "title_stem_207044777"

index_body = pickle.loads(bucket3.get_blob(
    'body_index.pkl').download_as_string())
index_body.bucket_name = "body_207044777"

index_body_stem = pickle.loads(bucket3.get_blob(
    'body_stem_index.pkl').download_as_string())
index_body_stem.bucket_name = "body_stem_207044777"

index_anchor = pickle.loads(bucket3.get_blob(
    'anchor_index.pkl').download_as_string())
index_anchor.bucket_name = "anchor_207044777"


def get_id2tf():
    return id2tf


def get_pv():
    return pv


def get_pr():
    return pr


def len_DL():
    return 6348910


def preprocess_query(query_as_string):
    '''
    Preprocess the query string and return a list of tokens.

    Parameters
    ----------
    query_as_string : str
        The query string.
    ----------
    Returns : list
        A list of tokens.
    '''
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ['category', 'references',
                        'also', 'links', 'extenal', 'see', 'thumb', 'make']
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    all_stopwords = english_stopwords.union(corpus_stopwords)
    tokens = [token.group() for token in RE_WORD.finditer(
        query_as_string.lower()) if token.group() not in all_stopwords]
    # tokens_after_filter = [term for term in tokens if term in index.df]
    return tokens


def tf_idf_and_cosine(query_tokens, index):
    '''
    Compute the tf-idf score and than cosine similarity between the quary and the docs for each document in the index and return a dictionary with there score.

    Parameters
    ----------
    query_tokens : list
        A list of tokens.
    index : Index
        The inverted index object.
    ----------
    Returns : dict
        A dictionary key:doc_id, value:score.
        '''  
    query_lst = np.ones(len(query_tokens))
    result = {}
    # read all the posting lists at once and store them in memory
    posting_lists = {term: index.read_posting_list(
        term, index.df[term]) for term in np.unique(query_tokens) if term in index.df}
    # Create lists of document ids and term frequencies at once
    candidates_dict = {term: dict(posting_list)
                       for term, posting_list in posting_lists.items()}
    # Extract the document ids from the candidates_dict
    candi = {doc_id for term_freq in candidates_dict.values()
             for doc_id in term_freq}
    for doc in candi:
        tf_score = np.zeros(len(query_tokens))
        i = 0
        for w in query_tokens:
            if w in index.df:
                if doc in candidates_dict[w]:
                    tf_score[i] = (candidates_dict[w][doc] /
                                       index.DL[doc])*np.log2(len_DL() / index.df[w])
            else:
                tf_score[i] = 0
            i+= 1    
        result[doc] = np.dot(tf_score, query_lst) / \
            (np.linalg.norm(query_lst) * id2tf[doc])
    return result


def merge_results_pr(scores1, scores2, pr, w1=0.33, w2=0.33, w3=0.33, N=10):
    '''
    Merge the scores of two methods and the pr scores and return a list of the top 10 documents.

    Parameters
    ----------
    scores1 : dict
        A dictionary of scores with BM25.
    scores2 : dict
         A dictionary of scores with binary(using search title).
    pr : dict
        A dictionary of the pr scores.
    w1 : float
        The weight of the first methods.
    w2 : float
        The weight of the second methods.
    w3 : float
        The weight of the pr scores.
     N : int
        The number of documents to return.
    ----------
    Returns : list
        A list of the top 10 documents.''
    '''
    # Union all docs and add weights, then sort
    pr_highest = max(pr.values())
    merged = [(k, (scores1.get(k, 0) * w1) + (scores2.get(k, 0) * w2) +
               ((pr.get(int(k), 0)/pr_highest) * w3)) for k in set(scores1) | set(scores2)]
    return sorted(merged, key=lambda x: x[1], reverse=True)[:N]


def merge_results_pr_withtitle(scores1, pr, w1=0.666, w3=0.33, N=10):
    '''
    Merge the scores of bm25 and the pr scores and return a list of the top 10 documents.

    Parameters
    ----------
    scores1 : dict
        A dictionary of the top 10 documents for the first query.
    pr : dict
        A dictionary of the pr scores.
    w1 : float
        The weight of the first methods.
    w3 : float
        The weight of the pr scores.
    N : int
        The number of documents to return.
        ----------
    Returns : list
        A list of the top 10 documents.'
    '''
    # Union all docs and add weights, then sort
    pr_highest = max(pr.values())
    merged = [(int(k), (scores1.get(k, 0) * w1) + ((pr.get(k, 0)/pr_highest) * w3))
              for k in scores1]
    return sorted(merged, key=lambda x: x[1], reverse=True)[:N]


def stemmer_query(query):
    '''
    Stem the query.

    Parameters
    ----------
    query : str
        The query.
    ----------
    Returns : list
        A list of the stemmed query.
    '''
    stemmer = PorterStemmer()
    query_lst = preprocess_query(query)
    query_stemmed = [stemmer.stem(token) for token in query_lst]
    return query_stemmed


def merge(score_title_res, score_body):
    '''
    check if title has score and send to merge_results_pr or merge_results_pr_withtitle.

    Parameters
    ----------
    score_title_res : dict
        A dictionary with scores for title.
    score_body : dict
        A dictionary with scores for body.
    ----------
    Returns : list
        A list of the top 10 documents.
    '''
    if len(score_title_res) != 0:
        max_title = max(score_title_res.values())
        score_title = {doc_id: score / max_title for doc_id,
                       score in score_title_res.items()}
        result = merge_results_pr(
            score_body, score_title, pr, 0.3333, 0.3333, 0.3333)
    else:
        result = merge_results_pr_withtitle(score_body, pr)
    return result


def id_title_search(result):
    '''
    Get the title of the document.

    Parameters
    ----------
    result : list
        A list of documents.
    ----------
    Returns : list
        A list of documents with their title.
    '''
    res = []
    for doc_id, score in result:
        if doc_id not in id_title.keys():
            res.append((str(doc_id), 'None'))
        else:
            res.append((str(doc_id), id_title[doc_id]))
    return res


def id_title_dict(result):
    '''
    Get the title of the document.

    Parameters
    ----------
    result : list
        A list of documents.
    ----------
    Returns : list
        A list of documents with their title.
    '''
    res = []
    for doc_id, score in result:
        if doc_id not in id_title.keys():
            res.append((doc_id, 'None'))
        else:
            res.append((doc_id, id_title[doc_id]))
    return res


def get_sorted_docs(dict):
    '''
    Sort the documents by their score.

    Parameters
    ----------
    dict : dict
        A dictionary of documents.
    ----------
    Returns : list
        A list of docs sorted by their score.
    '''
    result = sorted([(doc_id, score) for doc_id,
                    score in dict.items()], key=lambda x: x[1], reverse=True)
    return result


def title_score(query_lst, index_title):
    '''
    binary scoring for the title index if the word is in the title or not.

    Parameters
    ----------
    query_to_search : list
        A list of tokens.
    index_title : Index
        The inverted index object.
        ----------
    Returns : dict
        A dictionary key: doc_id, value: score title.'''
    count_quary_word = {}
    for term in np.unique(query_lst):
        if term not in index_title.df.keys():
            continue
        plss = index_title.read_posting_list(term, index_title.df[term])
        for doc_id, count in plss:
            if doc_id in count_quary_word.keys():
                count_quary_word[doc_id] += 1
            else:
                count_quary_word[doc_id] = 1
    return count_quary_word


def search__anchor(query_lst, index_anchor):
    '''
    binary scoring for the anchor index if the word is in the anchor or not.

    Parameters
    ----------
    query_lst : list
         A list of tokens.
    index_anchor : Index
        The anchor index.
    ----------
    Returns : dict
         A dictionary key: doc_id, value: score title.
    '''
    count_quary_word = {}
    for term in np.unique(query_lst):
        if term not in index_anchor.df.keys():
            continue
        plss = index_anchor.read_posting_list(term, index_anchor.df[term])
        for doc_id, count in plss:
            if doc_id in count_quary_word.keys():
                count_quary_word[doc_id] += 1
            else:
                count_quary_word[doc_id] = 1
    return count_quary_word


def id_title_dict_anchor(result):
    '''
    Get the title of the document.

    Parameters
    ----------
    result : list
        A list of the top 10 documents.
    ----------
    Returns : list
        A list of the top 10 documents with their title.
    '''
    res = []
    for doc_id, score in result:
        if doc_id in id_title.keys():
            res.append((doc_id, id_title[doc_id]))
    return res
