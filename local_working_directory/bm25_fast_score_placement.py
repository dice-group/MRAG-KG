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


# ------------ End of library impl. Followings are the example -----------------
from sklearn.datasets import fetch_20newsgroups

# documents = fetch_20newsgroups(subset='train').data
g = rdflib.Graph()
g.parse("fashionpedia-third-generation.owl", format="xml")

# Extract triplets
documents = []
for subject in g.subjects():
    txt = f"{str(subject)} \n"
    # exclude uneccesary information
    for predicate, obj in g.predicate_objects(subject):
        txt += f"{obj} \n"
    documents.append(txt)

bm25 = BM25()
bm25.fit(documents)
# Find the similar documents given  query
query = "What elegant and traditional bridal gowns are available?"
target_iri = "http://example.org/image_47178"
scores = bm25.transform(query, documents)
storage = dict()
for i, val in enumerate(documents):
    iri = val.split()[0]
    storage[iri] = scores[i]

storage_sorted = {k: v for k, v in sorted(storage.items(), key=lambda item: item[1], reverse=True)}

placement = list(storage_sorted.keys()).index(target_iri)

print(f"target_iri placement: {placement} \n target_iri score: {storage_sorted[target_iri]}")
print(f" Top scorer: {storage_sorted.items()}")

# print(documents[np.argmax(scores)])
