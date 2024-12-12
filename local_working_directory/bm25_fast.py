import numpy as np
import rdflib
from declarations import BM25

g = rdflib.Graph()
g.parse("fashionpedia-third-generation.owl", format="xml")

# Extract triplets
triplets = []
cn = 0
for subj, pred, obj in g:
    triplets.append((str(subj), str(pred), str(obj)))

print(len(triplets))
# Index the data (convert triplets to text format)
documents = ["\n".join(triplet) for triplet in triplets]

bm25 = BM25()
bm25.fit(documents)
# Find the similar documents given  query
query = "What are some clothes containing blue tshirt with long sleeves?"
scores = bm25.transform(query, documents)
print(documents[np.argmax(scores)])
