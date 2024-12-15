import numpy as np
from openai import OpenAI
from declarations import evaluation_samples
import matplotlib.pyplot as plt
from sklearn import manifold

client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8502/v1", api_key="token-tentris-upb")


def embed(txt):
    em = client.embeddings.create(input=[txt], model="tentris").data[0].embedding
    assert type(em) is list, f"{type(em)} is not a list"
    return em


X = np.array([embed(q) for iri, q in evaluation_samples.items()])

assert X.shape == (100, 4096)

n_components = 2
t_sne = manifold.TSNE(
    n_components=n_components,
    perplexity=2,
    init="random",
    max_iter=500,
    random_state=0,
)
(embeddings_2d) = t_sne.fit_transform(X)
# print(type(embeddings_2d))
# print(embeddings_2d)

def plot_2d(points, title):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    x, y = points.T
    ax.scatter(x, y, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    # plt.axis([-50, 50, -50, 50])
    plt.savefig("tsne_plot.png", dpi=300, format="png")
    plt.show()


plot_2d(embeddings_2d, "T-distributed Stochastic Neighbor Embedding (TSNE)")