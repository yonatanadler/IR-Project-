import math


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
