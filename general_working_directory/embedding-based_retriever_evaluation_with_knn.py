import json
from openai import OpenAI
import pandas as pd
import numpy as np
from numpy.linalg import norm
from declarations import evaluation_samples

df = pd.read_csv("embeddings_third_kg.csv", index_col=0, nrows=None)
iris = df.index.values.tolist()
client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8502/v1", api_key="token-tentris-upb")
docs = np.array(df.values)
docs_norms = docs / norm(docs, axis=1, keepdims=True)

df = pd.read_csv("benchmark_knn.csv", index_col=0, nrows=None)
knn = df.values.tolist()

results = dict()
count = 0

for target_iri, query in evaluation_samples.items():
    try:
        qr = np.array(client.embeddings.create(input=[query], model="tentris").data[0].embedding)
        qr_norms = qr / norm(qr)
        cosine_similarities = (docs_norms @ qr_norms).flatten()

        best_score = -1
        target_iri_index = iris.index(target_iri)

        for neighbor in knn[target_iri_index]:
            neighbor_score = cosine_similarities[neighbor]
            if neighbor_score > best_score:
                best_score = neighbor_score

        cosine_similarities = cosine_similarities.tolist()
        cosine_similarities.sort(reverse=True)
        placement = cosine_similarities.index(best_score)
        results[target_iri] = placement

        count += 1
        print(f"Done {count}/100 {target_iri} with placement: {placement:,}/45,623")
    except Exception as e:
        print(e)
        print(target_iri)
        print(query)
        with open("evaluation_results2_embedding-based_uncompleted.json", "w") as outfile:
            json.dump(results, outfile)

with open("evaluation_results2_embedding-based_on_third_KG.json", "w") as outfile:
    json.dump(results, outfile)