import numpy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel
from numpy.linalg import norm
import json


def sort(d):
    # Sort the dictionary by value in descending order
    sorted_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict

intent = input('What do you like to wear?\n')  # "I like a dress with wide neckline"

cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-de', trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)

with open("fashionpedia-embeddings", 'r') as f:
    data = json.load(f)

with open('image-filename-mappings', 'r') as file:
    filename_of = json.load(file)

image_cos = {}
top_cos_sin_value = 0
top_image = ""
for image in data:
    embeddings = model.encode(intent)
    embeddings2 = numpy.array(data.get(image))
    v = cos_sim(embeddings, embeddings2)
    image_cos[image] = v
    if v > top_cos_sin_value:
        top_cos_sin_value = v
        top_image = image

print(sort(image_cos))
# print(f"Top Image: {top_image} with Cosine Similarity: {top_cos_sin_value}")
# top_image_filename = filename_of[top_image[:-2]]
#
# img = np.asarray(Image.open(f"images/{top_image_filename}"))
# plt.imshow(img)
# plt.show()

seen_set = set()
pos_set = set()
neg_set = set()
for image in sort(image_cos).keys():
    filename = filename_of[image[:-2]]
    if filename not in seen_set:
        seen_set.add(filename)
        img = np.asarray(Image.open(f"images/{filename}"))
        plt.imshow(img)
        plt.show()
        feedback = input('Does this image contain something that fit your preferences? (y/n)\n')
        if feedback == "y":
            pos_set.add(image[:-2])
        elif feedback == "n":
            neg_set.add(image[:-2])
        else:
            print('Neutral selected')
    if len(seen_set) == 4:
        break

print(f"Positive examples: {pos_set}")
print(f"Negative examples: {neg_set}")
