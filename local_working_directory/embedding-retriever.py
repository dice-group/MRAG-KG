from openai import OpenAI
import pandas as pd
import numpy as np
from numpy.linalg import norm

query = 'I like a dress with wide neckline'

df = pd.read_csv("embeddings_short2.csv", index_col=0, nrows=None)
iris = df.index.values.tolist()

client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8502/v1", api_key="token-tentris-upb")

docs = np.array(df.values)
qr = np.array(client.embeddings.create(input=[query], model="tentris").data[0].embedding)

docs_norms = docs / norm(docs, axis=1, keepdims=True)
qr_norms = qr / norm(qr)

cosine_similarities = (docs_norms @ qr_norms).flatten()

best_match_index = np.argmax(cosine_similarities)
best_similarity = cosine_similarities[best_match_index]

print(cosine_similarities)
print(f"The best scoring image is the image with iri: {iris[best_match_index]} and score: {best_similarity}")