import json
from declarations import BM25, evaluation_samples
import rdflib
import pandas as pd

g = rdflib.Graph()
g.parse("fashionpedia-third-generation.owl", format="xml")

# Extract triplets
documents = []
for subject in g.subjects():
    txt = f"{str(subject)} \n"

    for predicate, obj in g.predicate_objects(subject):
        txt += f"{obj} \n"
    documents.append(txt)

df = pd.read_csv("benchmark_knn.csv", index_col=0, nrows=None)
iris = df.index.values.tolist()
knn = df.values.tolist()

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

        top_placement = 45623
        sorted_storage_iris = list(storage_sorted.keys())
        target_iri_index = iris.index(target_iri)
        # find the top placement of relevant instances.
        for neighbor in knn[target_iri_index]:
            neighbor_iri = iris[neighbor]
            p = sorted_storage_iris.index(neighbor_iri)
            if p < top_placement:
                top_placement = p
        results[target_iri] = top_placement

        count += 1
        print(f"Done {count}/100 {target_iri} with placement: {top_placement:,}/45,623")
    except Exception as e:
        print(e)
        print(target_iri)
        print(query)
        with open("evaluation_results2_bm25_uncompleted.json", "w") as outfile:
            json.dump(results, outfile)

with open("evaluation_results2_bm25_on_third_KG.json", "w") as outfile:
    json.dump(results, outfile)

