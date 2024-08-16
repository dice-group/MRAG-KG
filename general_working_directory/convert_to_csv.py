import pandas as pd

df = pd.read_json("../fashionpedia-embeddings", orient='index')
print(df.head())
df.to_csv("../fashionpedia-embeddings.csv")
