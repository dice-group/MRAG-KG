import csv
from openai import OpenAI
import pandas as pd
import numpy as np
from numpy.linalg import norm
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_reasoner import StructuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import TDL
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer

manager = OntologyManager()
ontology = manager.load_ontology(IRI.create("file://fashionpedia-third-generation.owl"))
reasoner = StructuralReasoner(ontology=ontology)
dprop1 = OWLDataProperty(IRI.create("http://example.org/hasDescription"))
dprop2 = OWLDataProperty(IRI.create("http://example.org/hasLLMDescription"))
kb = KnowledgeBase(path="fashionpedia-second-generation-v2.owl")

llm_client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8501/v1", api_key="token-tentris-upb")


def verbalize(expression):
    return llm_client.chat.completions.create(
        model="tentris",
        messages=[
            {"role": "system",
             "content": "You are a description logics translator. Your task is to explain in natural language the description logic expression that is given to you."},
            {"role": "user", "content": [{"type": "text",
                                          "text": f"Explain the following expression in natural language: {expression}. Provide only the explenation in a single sentence."}]}
        ],
        temperature=0.1,
        seed=1
    ).choices[0].message.content


def get_result(query, docs):
    return llm_client.chat.completions.create(
        model="tentris",
        messages=[
            {"role": "system",
             "content": "You are a apparel-loving AI and your focus is to give information about apparels. You should find similar points on the assisting information provided to you and present them in a short paragraph tailored to the user's query."},
            {"role": "user", "content": [{"type": "text", "text": f"{query}"}]},
            {"role": "assistant", "content": f"{docs}"},
        ],
        temperature=0.1,
        seed=1
    ).choices[0].message.content


def find_cosine_similarities(doc, docs):
    docs_norms = docs / norm(docs, axis=1, keepdims=True)
    doc_norms = doc / norm(doc)
    cosine_similarities = (docs_norms @ doc_norms).flatten()
    return cosine_similarities


query = input("What would you like to wear?\n")

df = pd.read_csv("embeddings_third_kg.csv", index_col=0, nrows=None)
iris = df.index.values

embbeding_client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8502/v1", api_key="token-tentris-upb")

docs = np.array(df.values)
qr = np.array(embbeding_client.embeddings.create(input=[query], model="tentris").data[0].embedding)

cosine_similarities = find_cosine_similarities(qr, docs)

# best_match_index = np.argmax(cosine_similarities)
# best_similarity = cosine_similarities[best_match_index]

top_scores_indexes = np.argpartition(cosine_similarities, -20)[-20:]
bottom_scores_indexes = np.argpartition(cosine_similarities, 100)[:100]

top_iris = {iris[i] for i in top_scores_indexes}
bottom_iris = {iris[i] for i in top_scores_indexes}

typed_pos = set(map(OWLNamedIndividual, top_iris))
typed_neg = set(map(OWLNamedIndividual, bottom_iris))

lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

render = DLSyntaxObjectRenderer()
model = TDL(knowledge_base=kb, use_nominals=True, max_runtime=10)
h = model.fit(learning_problem=lp).best_hypotheses()
str_concept = render.render(h)
print("Verbalized concept", verbalize(str_concept))

classified_instances = np.array([i.str for i in list(kb.individuals(h)) if i.str not in top_iris])
indexes_of_classified_instances = np.where(np.isin(iris, classified_instances))[0]

classified_instances_docs = docs[indexes_of_classified_instances]
retrieved_docs = docs[top_scores_indexes]
indexxxes = set()
for dc in retrieved_docs:
    cs = find_cosine_similarities(dc, classified_instances_docs)
    index = np.argmax(cs)
    indexxxes.add(index)
results = {iris[i] for i in indexxxes}
print(results)

merged_documents = ""
for iri in results:
    image_ind = OWLNamedIndividual(iri)
    llm_description = str(list(reasoner.data_property_values(image_ind, dprop2))[0].get_literal())
    if len(llm_description) > 800:
        llm_description = llm_description[:800]

    all_descriptions = ""
    for d in list(reasoner.data_property_values(image_ind, dprop1)):
        all_descriptions = all_descriptions + d.get_literal() + "\n"

    merged_documents += llm_description + " \n" + all_descriptions

textual_result = get_result(query, merged_documents)
print(textual_result)
