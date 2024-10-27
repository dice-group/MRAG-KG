""" Implementation of OKapi BM25 with sklearn's TfidfVectorizer
Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import rdflib

class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1



#------------ End of library impl. Followings are the example -----------------
from sklearn.datasets import fetch_20newsgroups
# documents = fetch_20newsgroups(subset='train').data
g = rdflib.Graph()
g.parse("fashionpedia-third-generation.owl", format="xml")

# Extract triplets
triplets = []
cn = 0
for subj, pred, obj in g:
    triplets.append((str(subj), str(pred), str(obj)))
    # cn += 1
    # if cn > 10000:
    #     break
print(len(triplets))
# Index the data (convert triplets to text format)
documents = ["\n".join(triplet) for triplet in triplets]

bm25 = BM25()
bm25.fit(documents)
# Find the similar documents given  query
query = "What are some clothes containing blue tshirt with long sleeves?"
scores = bm25.transform(query, documents)
print(documents[np.argmax(scores)])
