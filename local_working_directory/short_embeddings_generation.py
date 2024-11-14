import csv
from openai import OpenAI

document1 = 'This is a example about a dress for a formal event with a nice floral design. The dress has a slim fit size and its made of a light material.'
document2 = 'The image contains a dress with a open neckline and it has a casual style. It is mainly used in warm weather and it is very convenient for every day use'
client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8502/v1", api_key="token-tentris-upb")
with open("embeddings_short2.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    embd1 = client.embeddings.create(input=[document1], model="tentris").data[0].embedding
    embd2 = client.embeddings.create(input=[document2], model="tentris").data[0].embedding

    writer.writerow(["IRI"] + [f"{i}" for i in range(len(embd1))])
    writer.writerow(["https://example/iri1"] + embd1)
    writer.writerow(["https://example/iri2"] + embd2)
