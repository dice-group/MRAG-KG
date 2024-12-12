import rdflib
from declarations import BM25

g = rdflib.Graph()
g.parse("fashionpedia-third-generation.owl", format="xml")

# Extract triplets
documents = []
for subject in g.subjects():
    txt = f"{str(subject)} \n"

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
