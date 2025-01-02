import rdflib
from declarations import BM25, evaluation_samples
import json

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
results = dict()
count = 0
# Find the similar documents given  query
for target_iri, query in evaluation_samples.items():
    try:
        scores = bm25.transform(query, documents)
        storage = dict()
        for i, val in enumerate(documents):
            iri = val.split()[0]
            storage[iri] = scores[i]
        storage_sorted = {k: v for k, v in sorted(storage.items(), key=lambda item: item[1], reverse=True)}
        placement = list(storage_sorted.keys()).index(target_iri)
        results[target_iri] = placement
        count += 1
        print(f"Done {target_iri}: {count}/100")
    except Exception as e:
        print(e)
        print(target_iri)
        print(query)
        with open("evaluation_results_bm25_uncompleted.json", "w") as outfile:
            json.dump(results, outfile)

print(results)
with open("evaluation_results_bm25.json", "w") as outfile:
    json.dump(results, outfile)

# print(documents[np.argmax(scores)])
