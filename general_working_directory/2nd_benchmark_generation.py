import numpy as np
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
import json
import math
import re
import random
import csv

with open('benchmark_dataset.json') as json_file:
    benchmark_data = json.load(json_file)

client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8502/v1", api_key="token-tentris-upb")


def embed(txt):
    em = client.embeddings.create(input=[txt], model="tentris").data[0].embedding
    assert type(em) is list, f"{type(em)} is not a list"
    return em


def get_random_question(questions):
    q = questions.strip()
    if q.startswith("*"):
        splits = q.split("*")
        del splits[0]  # del empty string
    elif q.startswith("1."):
        splits = re.split(r'\d+\.\s*', q)
        del splits[0]  # del empty string
    else:
        splits = q.split("?")
    random_i = random.randint(0, len(splits) - 1)
    q = splits[random_i]
    # if splits are done in "?" basis, remove possible trailing "*" and add a "?" in the end.
    if "*" in q:
        q.replace("*", "")
    q = q.strip()
    if "?" not in q:
        q = q + "?"

    return q


iris = list(benchmark_data.keys())
single_questions = list([get_random_question(qs) for qs in benchmark_data.values()])

X = np.array([embed(q) for q in single_questions])
N = (len(benchmark_data))  # N = 45618
D = 4096
K = int(math.sqrt(N))  # k ≈ 213
assert X.shape == (N, D)
assert K == 213

with open('benchmark_embeddings.csv', mode='a', newline='') as file:
    print("Saving embeddings...")
    writer = csv.writer(file)
    writer.writerow(["IRI"] + [f"{i}" for i in range(D)])
    for idx, emb in enumerate(X):
        iri = iris[idx]
        writer.writerow([iri] + list(emb))
    print("Done saving embeddings!")

knn = NearestNeighbors(n_neighbors=K)
knn.fit(X)

labels = knn.kneighbors(X, return_distance=False)

with open('benchmark_knn.csv', mode='a', newline='') as file:
    print("Saving KNN...")
    writer = csv.writer(file)
    writer.writerow(["IRI"] + [f"{i}" for i in range(K)])
    for idx, label in enumerate(labels):
        iri = iris[idx]
        writer.writerow([iri] + list(label))
    print("Done saving KNN!")
