from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel
import json
import torch
import pandas as pd
import torch.nn.functional as F
import polars as pl

k = 3
# Set to None if you want to read all
nrows = None

with open('../image-filename-mappings.json', 'r') as file:
    filename_of = json.load(file)

# (1) Load the csv file fashionpedia-embeddings.csv". 
print("Reading embeddings", end="\t")
df = pd.read_csv("../fashionpedia-embeddings.csv", index_col=0, nrows=nrows)
print(df.shape)
# (2) D a matrix each row represents an embedding vector
document_embeddings = F.normalize(torch.from_numpy(df.values).type(torch.float32), p=2, dim=1)

# (3)
document_ordered_names = df.index.values.tolist()
# (4) Initialize the embedder
print("Loading embedding model", end="\t")
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-de', trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)
# model = model
while True:
    query = input('What do you like to wear?\n')  # "I like a dress with wide neckline"
    print(f"QUERY:{query}")
    # query_embeddings: np.ndarray: torchFloatTensor: dim x 1
    query_embeddings = torch.from_numpy(model.encode(query))
    query_embeddings = F.normalize(query_embeddings.reshape(len(query_embeddings), -1), p=2, dim=0)

    similarities = (document_embeddings @ query_embeddings).flatten()

    top_scores, top_k_indices = torch.topk(similarities, k)
    top_k_indices = top_k_indices.cpu().numpy()
    # Plot k images given user's query.
    seen_set = set()
    pos_set = set()
    neg_set = set()
    for i in top_k_indices:
        # Text Preprocess
        try:
            image = document_ordered_names[i][:-2]
            filename = filename_of[image]
            if filename not in seen_set:
                seen_set.add(filename)
                img = np.asarray(Image.open(f"../images/{filename}"))
                plt.imshow(img)
                plt.show()
                feedback = input('Does this image contain something that fit your preferences? (y/n)\n')
                if feedback == "y":
                    pos_set.add(image)
                elif feedback == "n":
                    neg_set.add(image)
                else:
                    print('Neutral selected')

        except KeyError:
            print(f"{i} not found")

    print(f"Positive examples: {pos_set}")
    print(f"Negative examples: {neg_set}")