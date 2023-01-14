from inverted_index import *
from backendd import Backend
from nitzan import *
from flask import Flask, request, jsonify
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords')
# from backend import Backendd


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(
            host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # query_lst = backend.query_preprocess(query, backend.get_index_body())
    # query_lst = backend.query_preprocess(query, backend.get_index_title())
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query_list = Backend.query_preprocess(query, Backend.get_index_body())
    q_vec = Backend.generate_query_tfidf_vector(
        query_list, Backend.get_index_title())
    tfidf = Backend.generate_document_tfidf_matrix(
        query_list, Backend.index_body())
    cos_sim = Backend.cosine_similarity(tfidf, q_vec)

    # Sort Documents by Cos sim score and retrieve top 100 only
    res_score = sorted([(doc_id, score) for doc_id, score in cos_sim.items(
    )], key=lambda x: x[1], reverse=True)[:100]

    for doc_id, score in res_score:
        if doc_id not in Backend.get_title_dict().keys():
            res.append((doc_id, 'None'))
        else:
            res.append((doc_id, Backend.get_title_dict()[doc_id]))

    return jsonify(res)

    # BEGIN SOLUTION
    # preprocess the query and get a list of "clean" terms
    # dfidf_body = backend.tf_idf_scores(backend.get_index_body())
    # query_lst = backend.preprocess_query(query, backend.get_index_body())
    # dfidf_query = backend.tf_idf_scores(query_lst)
    # cosin = backend.cosine_sim_using_sklearn(dfidf_query, dfidf_body)
    # top_n_id, list_scores = backend.get_top_n_docs(cosin, 100)
    # for doc_id in top_n_id:
    #     res.append((doc_id, backend.get_title_dict()[doc_id]))

    # END SOLUTION

    return jsonify(res)

    # END SOLUTION
    # return jsonify(res)


@app.route("/search_title")
def search_title():
    print("2")
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    # with open("index.pkl", "rb") as f:
    #     index_title = pickle.load(f)

    res = []
    # query = request.args.get('query', '')
    query = "apple"
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # query_list = Backend.query_preprocess(query, query)
    index_title = get_index()
    query_lst = Backend.preprocess_query(query, index_title)
    # index_title = Backend.get_index_title()

    count_quary_word = {}
    for term in np.unique(query_lst):
        if term not in index_title.df.keys():
            continue
        pls = Backend.read_posting_list(index_title, term, term)
        for doc_id, count in pls:
            if doc_id in count_quary_word.keys():
                count_quary_word[doc_id] += 1
            else:
                count_quary_word[doc_id] = 1
    print("2")
    result = sorted([(doc_id, score) for doc_id, score in count_quary_word.items(
    )], key=lambda x: x[1], reverse=True)

    # for doc_id, score in result:
    #     res.append((doc_id, Backend.get_title_dict()[doc_id]))

    # return jsonify(res)
    print(result)
    return result
    return res

    # END SOLUTION

    #
    #
    # def search_titlee(query_lst, index):
    #     counter_doc = {}
    #
    #     for term in np.unique(query_lst):
    #         if term not in index.df.keys():
    #             continue
    #         list_of_doc = read_posting_list(index, term)
    #
    #         for dic_id, count in list_of_doc:
    #             if dic_id in counter_doc:
    #                 counter_doc[dic_id] += 1
    #             else:
    #                 counter_doc[dic_id] = 1
    #     list_of_dict = sorted([(doc_id, score) for doc_id, score in counter_doc.items()], key=lambda x: x[1],
    #                           reverse=True)
    #     # result = []
    #     # for doc_id, score in list_of_dict:
    #     #     result.append((doc_id, self.get_title_dict()[doc_id]))
    #     print("list_of_dict")
    #     return list_of_dict
    #
search_title()


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    query_list = Backend.query_preprocess(query)
    index_title = Backend.get_index_anchor()
    count_quary_word = {}
    for term in np.unique(query_list):
        if term not in index_title.df.keys():
            continue
        pls = Backend.read_posting_list(index_title, term)
        for doc_id, count in pls:
            if doc_id in count_quary_word.keys():
                count_quary_word[doc_id] += 1
            else:
                count_quary_word[doc_id] = 1
    result = sorted([(doc_id, score) for doc_id, score in count_quary_word.items(
    )], key=lambda x: x[1], reverse=True)

    for doc_id, score in result:
        res.append((doc_id, Backend.get_title_dict()[doc_id]))

    # return jsonify(res)
    return res
    # END SOLUTION


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for docID in wiki_ids:
        try:
            item = float(Backend.page_rank_dict[str(docID)])
        except:
            item = 0.005
        res.append(item)
    return res
    #  return jsonify(res)
    # END SOLUTION


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    for docID in wiki_ids:
        item = Backend.page_view_dict[docID]
        res.append(item)
    return res
    #     return jsonify(res)
    # END SOLUTION


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
