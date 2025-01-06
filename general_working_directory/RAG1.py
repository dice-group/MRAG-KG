from openai import OpenAI
import pandas as pd
import numpy as np
from numpy.linalg import norm
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_reasoner import StructuralReasoner

manager = OntologyManager()
ontology = manager.load_ontology(IRI.create("file://fashionpedia-third-generation.owl"))
reasoner = StructuralReasoner(ontology=ontology)
dprop1 = OWLDataProperty(IRI.create("http://example.org/hasDescription"))
dprop2 = OWLDataProperty(IRI.create("http://example.org/hasLLMDescription"))

llm_client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8501/v1", api_key="token-tentris-upb")

def get_result(query, docs):
    return llm_client.chat.completions.create(
    model="tentris",
    messages=[
        {"role": "system", "content": "You are a apparel-loving AI and your focus is to give information about apparels. You should find similar points on the assisting information provided to you and present them in a short paragraph tailored to the user's query."},
        {"role": "user", "content": [{ "type": "text", "text": f"{query}"}]},
        {"role": "assistant", "content": f"{docs}"},
    ],
    temperature=0.1,
    seed=1
).choices[0].message.content

def get_result2(query, docs):
    return llm_client.chat.completions.create(
        model="tentris",
        messages=[
            {
                "role": "user",
                "content":
                    [
                        {
                            "type": "text",
                            "text": "You are a apparel-loving AI and your focus is to give information about apparels. You should find similar points on the information provided to you and present them in a short paragraph tailored to the following query: "
                                    f"'{query}'"
                                    f"The information is as follows: {docs}"
                        }
                    ]
            }
        ],
        temperature=0.1,
        seed=1
    ).choices[0].message.content


query = input("What would you like to wear?\n")

df = pd.read_csv("embeddings_third_kg.csv", index_col=0, nrows=None)
iris = df.index.values.tolist()

embbeding_client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8502/v1", api_key="token-tentris-upb")

docs = np.array(df.values)
qr = np.array(embbeding_client.embeddings.create(input=[query], model="tentris").data[0].embedding)

docs_norms = docs / norm(docs, axis=1, keepdims=True)
qr_norms = qr / norm(qr)

cosine_similarities = (docs_norms @ qr_norms).flatten()

best_match_index = np.argmax(cosine_similarities)
best_similarity = cosine_similarities[best_match_index]

indexes = np.argpartition(cosine_similarities, -10)[-10:]
merged_documents = ""
iris_result = set()
for index in indexes:

    iri = iris[index]
    iris_result.add(iri)
    image_ind = OWLNamedIndividual(iri)

    llm_description = str(list(reasoner.data_property_values(image_ind, dprop2))[0].get_literal())
    if len(llm_description) > 800:
        llm_description = llm_description[:800]

    all_descriptions = ""
    for d in list(reasoner.data_property_values(image_ind, dprop1)):
        all_descriptions = all_descriptions + d.get_literal() + "\n"

    merged_documents += llm_description + " \n" + all_descriptions

result = get_result(query, merged_documents)
print(iris_result)
print(result)
