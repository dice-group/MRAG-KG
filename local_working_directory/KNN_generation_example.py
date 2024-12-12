import numpy as np
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from declarations import evaluation_samples


client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8502/v1", api_key="token-tentris-upb")


def embed(txt):
    em = client.embeddings.create(input=[txt], model="tentris").data[0].embedding
    assert type(em) is list, f"{type(em)} is not a list"
    return em


X = np.array([embed(q) for iri, q in evaluation_samples.items()])

assert X.shape == (100, 4096)

neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(X)

labels = neigh.kneighbors(X, return_distance=False)

results = list()
questions = list(evaluation_samples.values())
print(labels.shape)
for idx, label in enumerate(labels):
    results.append(label)

for i in results[2]:
    print(questions[i])
